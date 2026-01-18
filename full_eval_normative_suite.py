"""full_eval_normative_suite.py
Unified orchestration script to train and evaluate normative DDPMs with MLP/SAINT backbones.

Features:
- Trains MLP or SAINT diffusion models (data-space), reusing existing training utils.
- Generates conditional samples and computes evaluation metrics, including:
  - Absolute Centile Error (ACE) with optional LOWESS smoothing for both Real and Gen centiles
  - Empirical Coverage Probability (ECP)
  - KS per age-bin
  - Z-score style deviation metrics
  - Joint heatmaps (Real vs Gen) and baseline product-of-marginals panels with Energy/MMD^2
  - Pair-of-pair density-shape correlation matrices and Mantel test (no dCor)
- Learning curves (vary train fractions) and dimensional scaling (vary IDP size) with bootstrap CIs.

Outputs:
Structured results under:
  <results_dir>/results_full_eval/<run_group>/<dataset>/<backbone>/<suite_tag>/D<idp_size>/frac_<f>/seed_<k>/
where `--results_dir` selects the base directory that will contain `results_full_eval/`.

Note:
- We intentionally do not modify the existing scripts. We reuse their utilities and mirror multiprocessing patterns.
"""
from __future__ import annotations

import argparse
import os
import json
import math
from typing import List, Tuple, Dict, Optional, Any
import re
import random
import time

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from statsmodels.nonparametric.smoothers_lowess import lowess

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    import yaml
except ImportError:
    yaml = None

import multiprocessing as mp

# -----------------------------------------------------------------------------
# Path tweaks so we can import shared training utilities and saint builder
# -----------------------------------------------------------------------------
import sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# All modules are in the same directory in the bundle
DDPM_SRC = THIS_DIR
if DDPM_SRC not in sys.path:
    sys.path.append(DDPM_SRC)

# MLP training utils
from diffusion_models import GeneralDiffusionModel  # type: ignore
from train_diffusion import train_diffusion_model  # type: ignore

# SAINT training utils
from train_diffusion_saint import build_saint_diffusion_model, train_diffusion_saint_model  # type: ignore


# -----------------------------------------------------------------------------
# Diffusion schedule helpers (mirror baseline scripts)
# -----------------------------------------------------------------------------

def get_diffusion_parameters(
    num_steps: int,
    beta_start: float = 0.02,
    beta_end: float = 0.95,
    device: torch.device = torch.device('cpu'),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32, device=device)
    alphas = 1.0 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_cumprod


# -----------------------------------------------------------------------------
# Sampling implementations for both backbones
# -----------------------------------------------------------------------------

def sample_mlp_full_ddpm(
    model: torch.nn.Module,
    num_samples: int,
    num_steps: int,
    cond_values: np.ndarray,
    latent_dim: int,
    betas: torch.Tensor,
    alphas: torch.Tensor,
    alpha_cumprod: torch.Tensor,
    device: torch.device,
    diffusion_space: str = 'data',
) -> torch.Tensor:
    """Iterative denoising following DDPM reverse process (mirrors MLP script)."""
    model.eval()
    z = torch.randn(num_samples, latent_dim, device=device)
    cond = torch.as_tensor(cond_values, dtype=torch.float32, device=device)
    if cond.dim() == 1:
        cond = cond.unsqueeze(-1)
    if cond.shape[0] != num_samples:
        cond = cond.expand(num_samples, -1)
    for t in reversed(range(num_steps)):
        t_tensor = torch.full((num_samples, 1), float(t), device=device)
        current_alpha = alphas[t]
        current_alpha_bar = alpha_cumprod[t]
        sigma_t = torch.sqrt(betas[t]) if t > 0 else 0.0
        with torch.no_grad():
            predicted_noise = model.denoiser(z, t_tensor, cond)
        eps_denom = torch.sqrt(current_alpha + 1e-8)
        noise_denom = torch.sqrt(torch.clamp(1 - current_alpha_bar, min=1e-12))
        mean_pred = (z - ((1 - current_alpha) / noise_denom) * predicted_noise) / eps_denom
        if t > 0:
            z = mean_pred + sigma_t * torch.randn_like(z)
        else:
            z = mean_pred
    if diffusion_space == 'latent':
        with torch.no_grad():
            x_samples = model.decoder(z, cond)
    else:
        x_samples = z
    return torch.cat((x_samples, cond), dim=1)


def sample_saint_full_ddpm(
    model: Any,
    num_samples: int,
    num_steps: int,
    betas: torch.Tensor,
    alphas: torch.Tensor,
    alpha_cumprod: torch.Tensor,
    cond_values: np.ndarray,
    device: torch.device,
    idp_dim: int,
) -> torch.Tensor:
    """Sampling loop matching SAINT script implementation (data-space)."""
    model.eval()
    cond_t = torch.as_tensor(cond_values, device=device, dtype=torch.float32)
    x = torch.randn(num_samples, idp_dim, device=device)
    with torch.no_grad():
        for t in reversed(range(num_steps)):
            t_tensor = torch.full((num_samples, 1), float(t), device=device)
            alpha_t, alpha_bar_t = alphas[t], alpha_cumprod[t]
            sigma_t = torch.sqrt(betas[t]) if t > 0 else 0.0
            eps_hat = model.denoiser(x, t_tensor, cond_t)
            x0_pred = (x - ((1 - alpha_t) / torch.sqrt(torch.clamp(1 - alpha_bar_t, min=1e-12))) * eps_hat) / torch.sqrt(alpha_t + 1e-8)
            x = x0_pred + sigma_t * torch.randn_like(x) if t > 0 else x0_pred
    return torch.cat([x, torch.as_tensor(cond_values, device=device)], dim=1)


# -----------------------------------------------------------------------------
# Evaluation utilities (ACE, ECP, KS per-bin, z-score)
# -----------------------------------------------------------------------------

def _bin_age(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    age_min, age_max = df['age'].min(), df['age'].max()
    age_bins = np.arange(math.floor(age_min) - 0.5, math.ceil(age_max) + 0.5, 1.0)
    dfx = df.copy()
    dfx['age_bin'] = pd.cut(dfx['age'], bins=age_bins)
    dfx['bin_mid'] = dfx['age_bin'].apply(lambda b: b.mid if pd.notna(b) else np.nan)
    return dfx, age_bins


def _smooth_series_by_lowess(x_vals: np.ndarray, y_vals: np.ndarray, frac: float) -> np.ndarray:
    if len(x_vals) < 3:
        return y_vals
    try:
        sm = lowess(y_vals, x_vals, frac=frac, it=0, return_sorted=False)
        return sm
    except Exception:
        return y_vals


def _enforce_monotone_quantiles(vals: np.ndarray) -> np.ndarray:
    """Make quantile function values non-decreasing in quantile index via cumulative max."""
    if vals.size == 0:
        return vals
    out = np.asarray(vals, dtype=float).copy()
    for i in range(1, len(out)):
        if not np.isfinite(out[i - 1]):
            continue
        if not np.isfinite(out[i]) or out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def compute_ace(
    real_df: pd.DataFrame,
    gen_df: pd.DataFrame,
    idp_cols: List[str],
    percentiles: List[float],
    lowess_frac: float = 0.2,
    use_smoothing: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Absolute Centile Error (ACE) per IDP and percentile.

    For each age bin and IDP:
      - compute Gen quantile at q: g_q(b)
      - estimate the Real percentile p_hat(b) = F_real_bin(g_q(b)) using the empirical CDF
        over all real values in the bin (fine-grained near tails; no clamping to quantile grid)
      - ACE(q) accumulates |p_hat(b) - q|
    Returns dict[idp][f"ace_{int(q*100)}"] = mean |p_hat - q| across valid bins.
    """
    rdf, _ = _bin_age(real_df)
    gdf, _ = _bin_age(gen_df)

    # Build per-IDP per-bin quantiles and (optional) smoothing across bin centres
    bins_sorted = sorted([b for b in rdf['age_bin'].dropna().unique()], key=lambda x: x.left)
    bin_mids = np.array([b.mid for b in bins_sorted], dtype=float)
    out: Dict[str, Dict[str, float]] = {}

    for col in idp_cols:
        # real quantiles per bin for each percentile
        rq_table = []  # shape: (n_bins, n_perc)
        gq_table = []
        for b in bins_sorted:
            rvals = rdf.loc[rdf['age_bin'] == b, col].dropna().values
            gvals = gdf.loc[gdf['age_bin'] == b, col].dropna().values
            if rvals.size == 0 or gvals.size == 0:
                rq_table.append([np.nan] * len(percentiles))
                gq_table.append([np.nan] * len(percentiles))
            else:
                rq_table.append([np.quantile(rvals, q) for q in percentiles])
                gq_table.append([np.quantile(gvals, q) for q in percentiles])

        rq = np.asarray(rq_table, dtype=float)  # (B, P)
        gq = np.asarray(gq_table, dtype=float)  # (B, P)

        if use_smoothing and np.isfinite(bin_mids).sum() >= 3:
            rq_sm = np.zeros_like(rq)
            gq_sm = np.zeros_like(gq)
            for j in range(len(percentiles)):
                y_r = rq[:, j]
                y_g = gq[:, j]
                mask_r = np.isfinite(y_r) & np.isfinite(bin_mids)
                mask_g = np.isfinite(y_g) & np.isfinite(bin_mids)
                yr_sm = y_r.copy()
                yg_sm = y_g.copy()
                if mask_r.sum() >= 3:
                    yr_sm[mask_r] = _smooth_series_by_lowess(bin_mids[mask_r], y_r[mask_r], frac=lowess_frac)
                if mask_g.sum() >= 3:
                    yg_sm[mask_g] = _smooth_series_by_lowess(bin_mids[mask_g], y_g[mask_g], frac=lowess_frac)
                rq_sm[:, j] = yr_sm
                gq_sm[:, j] = yg_sm
            rq = rq_sm
            gq = gq_sm

        # enforce monotonicity across quantiles for Real per bin
        for i in range(rq.shape[0]):
            rq[i, :] = _enforce_monotone_quantiles(rq[i, :])

        # invert Real percentile via empirical CDF within each bin (not quantile-grid interpolation)
        ace_vals: Dict[str, float] = {}
        for j, q in enumerate(percentiles):
            errs = []
            for i in range(rq.shape[0]):
                g_val = gq[i, j]
                if not np.isfinite(g_val):
                    continue
                rvals = rdf.loc[rdf['age_bin'] == bins_sorted[i], col].dropna().values
                if rvals.size == 0:
                    continue
                # empirical CDF percentile of generated value under the real-bin distribution
                p_hat = float(np.mean(rvals <= g_val))
                p_hat = float(np.clip(p_hat, 0.0, 1.0))
                errs.append(abs(p_hat - q))
            ace_vals[f"ace_{int(round(q*100))}"] = float(np.mean(errs)) if errs else float('nan')
        out[col] = ace_vals
    return out


def compute_ecp(
    real_df: pd.DataFrame,
    gen_df: pd.DataFrame,
    idp_cols: List[str],
    q_low: float = 0.025,
    q_high: float = 0.975,
) -> Dict[str, Dict[str, float]]:
    rdf, _ = _bin_age(real_df)
    gdf, _ = _bin_age(gen_df)
    out: Dict[str, Dict[str, float]] = {}
    for col in idp_cols:
        ql = gdf.groupby('age_bin')[col].quantile(q_low)
        qh = gdf.groupby('age_bin')[col].quantile(q_high)
        merged = rdf[['age_bin', col]].join(ql.rename('ql'), on='age_bin').join(qh.rename('qh'), on='age_bin').dropna()
        cov = float(((merged[col] >= merged['ql']) & (merged[col] <= merged['qh'])).mean())
        out[col] = {'ecp': cov, 'nominal': q_high - q_low}
    return out


def ks_per_bin(
    real_df: pd.DataFrame,
    gen_df: pd.DataFrame,
    idp_cols: List[str],
) -> Dict[str, Dict[str, float]]:
    from scipy.stats import ks_2samp
    rdf, _ = _bin_age(real_df)
    gdf, _ = _bin_age(gen_df)
    out: Dict[str, Dict[str, float]] = {}
    for col in idp_cols:
        stats_list = []
        # iterate bins in order
        bins_sorted = sorted([b for b in rdf['age_bin'].dropna().unique()], key=lambda x: x.left)
        for b in bins_sorted:
            r = rdf.loc[rdf['age_bin'] == b, col].dropna().values
            g = gdf.loc[gdf['age_bin'] == b, col].dropna().values
            if len(r) > 0 and len(g) > 0:
                stats_list.append(float(ks_2samp(r, g).statistic))
        if stats_list:
            out[col] = {'ks_mean': float(np.mean(stats_list)), 'ks_max': float(np.max(stats_list))}
    return out


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR control. Returns q-values in original order.
    NaNs are preserved.
    """
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    mask = np.isfinite(p)
    if not np.any(mask):
        return q
    ps = p[mask]
    m = ps.size
    order = np.argsort(ps)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, m + 1)
    q_raw = ps * m / ranks
    # enforce monotonicity from the end
    q_sorted = np.minimum.accumulate(q_raw[order[::-1]])[::-1]
    q_vals = np.minimum(q_sorted, 1.0)
    q[mask] = q_vals
    return q


def ks_per_bin_with_pvalues(
    real_df: pd.DataFrame,
    gen_df: pd.DataFrame,
    idp_cols: List[str],
    min_per_bin: int = 10,
    fdr_alpha: float = 0.05,
    use_permutation: bool = False,
    B: int = 1000,
) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame]:
    """Compute per-bin KS statistics and p-values, FDR-adjust within IDP, and
    aggregate per-IDP summaries including Fisher/Stouffer combined p-values.

    Returns (summary_dict, per_bin_df).
    """
    from scipy.stats import ks_2samp, combine_pvalues, norm
    rdf, _ = _bin_age(real_df)
    gdf, _ = _bin_age(gen_df)
    records = []
    # ensure deterministic bin order
    bins_sorted = sorted([b for b in rdf['age_bin'].dropna().unique()], key=lambda x: x.left)
    for col in idp_cols:
        for b in bins_sorted:
            r = rdf.loc[rdf['age_bin'] == b, col].dropna().values
            g = gdf.loc[gdf['age_bin'] == b, col].dropna().values
            n_r = int(r.size); n_g = int(g.size)
            if n_r >= min_per_bin and n_g >= min_per_bin:
                try:
                    stat, pval = ks_2samp(r, g, method='auto')
                    stat = float(stat); pval = float(pval)
                    if use_permutation and np.isfinite(stat):
                        # permutation p-value: shuffle labels many times and recompute KS
                        try:
                            rng = np.random.default_rng(12345)
                            pooled = np.concatenate([r, g])
                            n = pooled.size
                            nr = n_r
                            perm_stats = []
                            for _ in range(int(B)):
                                perm = rng.permutation(n)
                                r_idx = perm[:nr]; g_idx = perm[nr:]
                                stat_p, _ = ks_2samp(pooled[r_idx], pooled[g_idx], method='auto')
                                perm_stats.append(float(stat_p))
                            perm_stats = np.asarray(perm_stats, dtype=float)
                            p_perm = float((1.0 + np.sum(perm_stats >= stat)) / (1.0 + perm_stats.size))
                            pval = p_perm
                        except Exception:
                            pass
                except Exception:
                    stat, pval = float('nan'), float('nan')
            else:
                stat, pval = float('nan'), float('nan')
            records.append({
                'idp': col,
                'age_bin_left': float(b.left),
                'age_bin_right': float(b.right),
                'n_real': n_r,
                'n_gen': n_g,
                'ks_stat': stat,
                'p_value': pval,
            })
    df = pd.DataFrame.from_records(records)
    # FDR adjust within each IDP
    q_vals = []
    for col, grp in df.groupby('idp', sort=False):
        q = _bh_fdr(grp['p_value'].values)
        q_vals.append(pd.Series(q, index=grp.index))
    if q_vals:
        df['q_value_bh'] = pd.concat(q_vals).sort_index()
    else:
        df['q_value_bh'] = np.nan

    # Aggregate per-IDP
    out: Dict[str, Dict[str, float]] = {}
    for col, grp in df.groupby('idp', sort=False):
        ks_vals = grp['ks_stat'].values
        ks_vals = ks_vals[np.isfinite(ks_vals)]
        pvals = grp['p_value'].values
        qvals = grp['q_value_bh'].values
        finite_p = np.isfinite(pvals)
        # Fisher/Stouffer
        fisher_p = float('nan')
        stouffer_p = float('nan')
        if np.any(finite_p):
            try:
                fisher_p = float(combine_pvalues(pvals[finite_p], method='fisher')[1])
            except Exception:
                fisher_p = float('nan')
            try:
                # weights ~ sqrt(effective n)
                n_r = grp['n_real'].values[finite_p]
                n_g = grp['n_gen'].values[finite_p]
                n_eff = (n_r * n_g) / np.maximum(n_r + n_g, 1)
                w = np.sqrt(np.maximum(n_eff, 1.0))
                stouffer_p = float(combine_pvalues(pvals[finite_p], method='stouffer', weights=w)[1])
            except Exception:
                stouffer_p = float('nan')
        frac_sig = float(np.mean(qvals[finite_p] <= fdr_alpha)) if np.any(finite_p) else float('nan')
        d: Dict[str, float] = {}
        if ks_vals.size:
            d['ks_mean'] = float(np.mean(ks_vals))
            d['ks_max'] = float(np.max(ks_vals))
        d['ks_p_fisher'] = fisher_p
        d['ks_p_stouffer'] = stouffer_p
        d['ks_frac_sig_bh'] = frac_sig
        out[col] = d
    return out, df


def zscore_eval(
    real_df: pd.DataFrame,
    gen_df: pd.DataFrame,
    idp_cols: List[str],
    label_col: Optional[str] = None,
    positive_class: int = 1,
) -> Dict[str, Dict[str, float]]:
    rdf, _ = _bin_age(real_df)
    gdf, _ = _bin_age(gen_df)
    mu = gdf.groupby('age_bin')[idp_cols].mean()
    sd = gdf.groupby('age_bin')[idp_cols].std().replace(0, np.nan)
    joined = rdf.join(mu, on='age_bin', rsuffix='_mu').join(sd, on='age_bin', rsuffix='_sd')
    out: Dict[str, Dict[str, float]] = {}
    for c in idp_cols:
        mu_c = joined[f"{c}_mu"]; sd_c = joined[f"{c}_sd"]
        z = (joined[c] - mu_c) / sd_c
        out[c] = {
            'z_mean_abs': float(np.nanmean(np.abs(z))),
            'z_frac_gt_1.96': float(np.nanmean(np.abs(z) > 1.96)),
        }
    if label_col and (label_col in real_df.columns):
        try:
            from scipy.stats import ttest_ind
            from sklearn.metrics import roc_auc_score
            lbl = real_df[label_col].values
            pos = (lbl == positive_class)
            for c in idp_cols:
                mu_c = joined[f"{c}_mu"]; sd_c = joined[f"{c}_sd"]
                z = (joined[c] - mu_c) / sd_c
                z0 = z[~pos]; z1 = z[pos]
                t_p = float(ttest_ind(z0.dropna(), z1.dropna(), equal_var=False).pvalue) if (np.sum(~pos)>1 and np.sum(pos)>1) else float('nan')
                auc = float(roc_auc_score(lbl[~np.isnan(z)], np.abs(z[~np.isnan(z)]))) if np.sum(pos)>0 and np.sum(~pos)>0 else float('nan')
                out[c].update({'ttest_p': t_p, 'auroc_absz': auc})
        except Exception:
            pass
    return out


# -----------------------------------------------------------------------------
# Additional calibration and generalization diagnostics
#   - PIT histograms (pooled across age-bins; optional QQ)
#   - Coverage-vs-nominal curves
#   - Nearest-neighbour memorisation checks (train vs hold-out)
# -----------------------------------------------------------------------------

# Short-name labels for UKB base 20-IDP centile plots (applied only for D20 base runs)
UKB_20_IDP_SHORT: Dict[str, str] = {
    # Hippocampus
    "26562": "Vol Hipp (L)",
    "26593": "Vol Hipp (R)",
    # Amygdala
    "26563": "Vol Amyg (L)",
    "26594": "Vol Amyg (R)",
    # Lateral ventricles
    "26554": "Vol Lat Vent (L)",
    "26585": "Vol Lat Vent (R)",
    # Inferior lateral ventricles
    "26555": "Vol InfLatVent (L)",
    "26586": "Vol InfLatVent (R)",
    # Cortical thickness measures
    "26760": "Entorhinal Thk (L)",
    "26861": "Entorhinal Thk (R)",
    "26770": "Parahippo Thk (L)",
    "26871": "Parahippo Thk (R)",
    "26763": "InfTemp Thk (L)",
    "26864": "InfTemp Thk (R)",
    "26769": "MidTemp Thk (L)",
    "26870": "MidTemp Thk (R)",
    "26777": "PostCing Thk (L)",
    "26878": "PostCing Thk (R)",
    # Global volumes
    "25781": "WMH Vol",
    "26521": "eTIV Vol",
}


def _ukb_short_label_from_col(col: str) -> str:
    m = re.search(r"(\d+)", str(col))
    if m is None:
        return col
    key = m.group(1)
    return UKB_20_IDP_SHORT.get(key, col)

def _empirical_pit(real_vals: np.ndarray, gen_vals: np.ndarray) -> np.ndarray:
    """
    For each real value y, return PIT u = F_gen(y) where F_gen is the
    empirical CDF formed from generated samples (same conditioning bin).
    """
    if gen_vals.size == 0 or real_vals.size == 0:
        return np.array([], dtype=float)
    gen_sorted = np.sort(gen_vals)
    idx = np.searchsorted(gen_sorted, real_vals, side='right')
    u = idx / float(gen_sorted.size)
    eps = 1.0 / (gen_sorted.size * 10.0)
    return np.clip(u, eps, 1.0 - eps)


def pit_histograms(
    real_df: pd.DataFrame,
    gen_df: pd.DataFrame,
    idp_cols: List[str],
    out_dir: str,
    bins: int = 20,
    dpi: int = 300,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    rdf, _ = _bin_age(real_df)
    gdf, _ = _bin_age(gen_df)
    all_rows = []
    for col in idp_cols:
        pit_vals = []
        bins_sorted = sorted([b for b in rdf['age_bin'].dropna().unique()], key=lambda x: x.left)
        for b in bins_sorted:
            r = rdf.loc[rdf['age_bin'] == b, col].dropna().values
            g = gdf.loc[gdf['age_bin'] == b, col].dropna().values
            if r.size and g.size:
                pit = _empirical_pit(r, g)
                if pit.size:
                    pit_vals.append(pd.DataFrame({
                        'idp': col,
                        'age_bin_left': b.left, 'age_bin_right': b.right,
                        'pit': pit
                    }))
        if pit_vals:
            dfc = pd.concat(pit_vals, axis=0, ignore_index=True)
            all_rows.append(dfc)
            plt.figure(figsize=(8, 4))
            u_all = dfc['pit'].values
            plt.hist(u_all, bins=bins, density=True, alpha=0.6)
            plt.hlines(1.0, 0.0, 1.0, linewidth=2)
            plt.title(f'PIT histogram (pooled age-bins): {col}', fontsize=13)
            plt.xlabel('u', fontsize=12); plt.ylabel('density', fontsize=12)
            ax = plt.gca(); ax.tick_params(axis='both', which='major', labelsize=11)
            plt.tight_layout()
            out_png = os.path.join(out_dir, f'pit_{col}.png')
            plt.savefig(out_png, dpi=dpi); plt.close()
    if not all_rows:
        return ""
    df_all = pd.concat(all_rows, axis=0, ignore_index=True)
    df_all.to_csv(os.path.join(out_dir, 'pit_values.csv'), index=False)
    # Pooled across all IDPs
    try:
        u_all_pooled = df_all['pit'].values
        if u_all_pooled.size >= 5:
            plt.figure(figsize=(8, 4))
            plt.hist(u_all_pooled, bins=bins, density=True, alpha=0.6)
            plt.hlines(1.0, 0.0, 1.0, linewidth=2)
            plt.title('PIT histogram (pooled across all IDPs)', fontsize=13)
            plt.xlabel('u', fontsize=12); plt.ylabel('density', fontsize=12)
            ax = plt.gca(); ax.tick_params(axis='both', which='major', labelsize=11)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'pit_pooled.png'), dpi=dpi)
            plt.close()
            u_sorted_all = np.sort(u_all_pooled)
            q_unif_all = np.linspace(0, 1, u_sorted_all.size, endpoint=False) + 0.5 / u_sorted_all.size
            plt.figure(figsize=(4.5, 4.5))
            plt.plot(q_unif_all, u_sorted_all, '.', ms=2)
            plt.plot([0, 1], [0, 1], '--')
            plt.title('PIT QQ vs Uniform (pooled)', fontsize=13)
            plt.xlabel('Uniform quantile', fontsize=12); plt.ylabel('Empirical PIT quantile', fontsize=12)
            ax = plt.gca(); ax.tick_params(axis='both', which='major', labelsize=11)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'pit_qq_pooled.png'), dpi=dpi)
            plt.close()
    except Exception:
        pass
    for col in idp_cols:
        u = df_all.loc[df_all['idp'] == col, 'pit'].values
        if u.size < 5:
            continue
        u_sorted = np.sort(u)
        q_unif = np.linspace(0, 1, u_sorted.size, endpoint=False) + 0.5 / u_sorted.size
        plt.figure(figsize=(4.5, 4.5))
        plt.plot(q_unif, u_sorted, '.', ms=2)
        plt.plot([0, 1], [0, 1], '--')
        plt.title(f'PIT QQ vs Uniform: {col}', fontsize=13)
        plt.xlabel('Uniform quantile', fontsize=12); plt.ylabel('Empirical PIT quantile', fontsize=12)
        ax = plt.gca(); ax.tick_params(axis='both', which='major', labelsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'pit_qq_{col}.png'), dpi=dpi)
        plt.close()
    return out_dir


def coverage_vs_nominal(
    real_df: pd.DataFrame,
    gen_df: pd.DataFrame,
    idp_cols: List[str],
    out_dir: str,
    nominals: Optional[List[float]] = None,
    dpi: int = 300,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    if nominals is None:
        nominals = [x / 100.0 for x in range(50, 100, 5)] + [0.99]
    rdf, _ = _bin_age(real_df)
    gdf, _ = _bin_age(gen_df)
    rows = []
    for col in idp_cols:
        for a in nominals:
            ql = gdf.groupby('age_bin')[col].quantile((1.0 - a) / 2.0)
            qh = gdf.groupby('age_bin')[col].quantile(1.0 - (1.0 - a) / 2.0)
            merged = rdf[['age_bin', col]].join(ql.rename('ql'), on='age_bin').join(qh.rename('qh'), on='age_bin').dropna()
            if merged.shape[0] == 0:
                cov = float('nan')
            else:
                cov = float(((merged[col] >= merged['ql']) & (merged[col] <= merged['qh'])).mean())
            rows.append({'idp': col, 'nominal': a, 'empirical': cov})
        dfc = pd.DataFrame([r for r in rows if r['idp'] == col])
        if dfc.shape[0]:
            plt.figure(figsize=(4.8, 4.2))
            plt.plot(dfc['nominal'], dfc['empirical'], marker='o')
            plt.plot([0, 1], [0, 1], '--')
            plt.xlabel('Nominal coverage'); plt.ylabel('Empirical coverage')
            plt.title(f'Coverage curve: {col}')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'coverage_curve_{col}.png'), dpi=dpi)
            plt.close()
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'coverage_curves.csv'), index=False)
    try:
        agg = pd.DataFrame(rows).groupby('nominal')['empirical'].median().reset_index()
        plt.figure(figsize=(5, 4.2))
        plt.plot(agg['nominal'], agg['empirical'], marker='o')
        plt.plot([0, 1], [0, 1], '--')
        plt.title('Coverage curve (median across IDPs)')
        plt.xlabel('Nominal'); plt.ylabel('Empirical (median IDP)')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'coverage_curve_median.png'), dpi=dpi)
        plt.close()
    except Exception:
        pass
    return out_dir


def coverage_vs_nominal_diff(
    real_df: pd.DataFrame,
    gen_df: pd.DataFrame,
    idp_cols: List[str],
    out_dir: str,
    nominals: Optional[List[float]] = None,
    dpi: int = 300,
) -> str:
    """Plot coverage difference curves: (empirical - nominal) vs nominal.

    Saves per-IDP plots, a median-across-IDPs plot, and coverage_diff_curves.csv.
    """
    os.makedirs(out_dir, exist_ok=True)
    if nominals is None:
        nominals = [x / 100.0 for x in range(50, 100, 5)] + [0.99]
    rdf, _ = _bin_age(real_df)
    gdf, _ = _bin_age(gen_df)
    rows = []
    for col in idp_cols:
        diffs = []
        for a in nominals:
            ql = gdf.groupby('age_bin')[col].quantile((1.0 - a) / 2.0)
            qh = gdf.groupby('age_bin')[col].quantile(1.0 - (1.0 - a) / 2.0)
            merged = rdf[['age_bin', col]].join(ql.rename('ql'), on='age_bin').join(qh.rename('qh'), on='age_bin').dropna()
            if merged.shape[0] == 0:
                emp = float('nan')
            else:
                emp = float(((merged[col] >= merged['ql']) & (merged[col] <= merged['qh'])).mean())
            diff = emp - a if np.isfinite(emp) else float('nan')
            rows.append({'idp': col, 'nominal': a, 'empirical': emp, 'diff': diff})
        # Per-IDP plot
        dfc = pd.DataFrame([r for r in rows if r['idp'] == col])
        if dfc.shape[0]:
            plt.figure(figsize=(4.8, 4.2))
            plt.axhline(0.0, color='k', linestyle='--', linewidth=1.5)
            plt.plot(dfc['nominal'], dfc['diff'], marker='o')
            plt.xlabel('Nominal coverage'); plt.ylabel('Empirical - nominal')
            plt.title(f'Coverage difference: {col}')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'coverage_diff_{col}.png'), dpi=dpi)
            plt.close()
    df_rows = pd.DataFrame(rows)
    df_rows.to_csv(os.path.join(out_dir, 'coverage_diff_curves.csv'), index=False)
    # Median-across-IDPs
    try:
        agg = df_rows.groupby('nominal')['diff'].median().reset_index()
        plt.figure(figsize=(5, 4.2))
        plt.axhline(0.0, color='k', linestyle='--', linewidth=1.5)
        plt.plot(agg['nominal'], agg['diff'], marker='o')
        plt.title('Coverage difference (median across IDPs)')
        plt.xlabel('Nominal'); plt.ylabel('Empirical - nominal (median IDP)')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'coverage_diff_median.png'), dpi=dpi)
        plt.close()
    except Exception:
        pass
    return out_dir


def _coverage_diff_overlay_diffusion_models(
    suite_root: str,
    dataset_tag: str,
    suite_tag: str,
    idp_size_tag: str,
    frac: float,
    seed_idx: int,
    dpi: int = 300,
) -> Optional[str]:
    """Create overlay plot of median coverage differences for MLP and SAINT only.
    Returns output path or None.
    """
    try:
        def _csv_path(backbone: str) -> str:
            return os.path.join(
                suite_root, dataset_tag, backbone, suite_tag,
                f"D{idp_size_tag}", f"frac_{frac:.2f}", f"seed_{seed_idx}",
                'results', 'coverage_curves_diff', 'coverage_diff_curves.csv'
            )
        
        csv_mlp = _csv_path('mlp')
        csv_saint = _csv_path('saint')
        
        # Require both MLP and SAINT to exist
        if (not os.path.exists(csv_mlp)) or (not os.path.exists(csv_saint)):
            return None
        
        df_mlp = pd.read_csv(csv_mlp)
        df_saint = pd.read_csv(csv_saint)
        med_mlp = df_mlp.groupby('nominal')['diff'].median().reset_index()
        med_saint = df_saint.groupby('nominal')['diff'].median().reset_index()
        
        out_dir = os.path.join(suite_root, dataset_tag, 'combined', suite_tag, f"D{idp_size_tag}")
        os.makedirs(out_dir, exist_ok=True)
        
        plt.figure(figsize=(5.2, 4.4))
        plt.axhline(0.0, color='k', linestyle='--', linewidth=1.5)
        
        # faint per-IDP points
        try:
            # Plot all IDPs points for MLP
            pts_mlp = df_mlp[['nominal', 'diff']].dropna()
            plt.scatter(pts_mlp['nominal'], pts_mlp['diff'], s=10, alpha=0.15, color='C0')
            # Plot all IDPs points for SAINT
            pts_saint = df_saint[['nominal', 'diff']].dropna()
            plt.scatter(pts_saint['nominal'], pts_saint['diff'], s=10, alpha=0.15, color='C1')
        except Exception:
            pass
        
        # Median lines
        plt.plot(med_mlp['nominal'], med_mlp['diff'], marker='D', mfc='none', mec='C0', color='C0', label='MLP (median)', linewidth=1.5)
        plt.plot(med_saint['nominal'], med_saint['diff'], marker='o', mfc='none', mec='C1', color='C1', label='SAINT (median)', linewidth=1.5)
        
        plt.xlabel('Nominal coverage'); plt.ylabel('Empirical - nominal (median IDP)')
        plt.title(f'Coverage difference: MLP vs SAINT ({dataset_tag.upper()} D{idp_size_tag})')
        plt.legend()
        plt.tight_layout()
        out_png = os.path.join(out_dir, 'coverage_diff_overlay_diffusion_models.png')
        plt.savefig(out_png, dpi=dpi); plt.close()
        return out_png
    except Exception:
        return None


def _coverage_diff_overlay_gamlss(
    suite_root: str,
    dataset_tag: str,
    suite_tag: str,
    idp_size_tag: str,
    dpi: int = 300,
) -> Optional[str]:
    """Create overlay plot showing GAMLSS coverage differences with median line and per-IDP points.
    Returns output path or None.
    """
    try:
        # GAMLSS has different path structure (no frac/seed subdirs)
        csv_gamlss = os.path.join(
            suite_root, dataset_tag, 'gamlss', suite_tag,
            f"D{idp_size_tag}",
            'results', 'coverage_curves_diff', 'coverage_diff_curves.csv'
        )
        
        if not os.path.exists(csv_gamlss):
            return None
        
        df_gamlss = pd.read_csv(csv_gamlss)
        med_gamlss = df_gamlss.groupby('nominal')['diff'].median().reset_index()
        
        out_dir = os.path.join(suite_root, dataset_tag, 'combined', suite_tag, f"D{idp_size_tag}")
        os.makedirs(out_dir, exist_ok=True)
        
        plt.figure(figsize=(5.2, 4.4))
        plt.axhline(0.0, color='k', linestyle='--', linewidth=1.5)
        
        # Per-IDP points with higher opacity to show variability
        try:
            pts_gamlss = df_gamlss[['nominal', 'diff']].dropna()
            plt.scatter(pts_gamlss['nominal'], pts_gamlss['diff'], s=12, alpha=0.35, color='C2', label='Per-IDP')
        except Exception:
            pass
        
        # Median line with markers
        plt.plot(med_gamlss['nominal'], med_gamlss['diff'], marker='s', mfc='none', mec='C2', 
                 color='C2', label='GAMLSS (median)', linewidth=2.0, markersize=6)
        
        plt.xlabel('Nominal coverage'); plt.ylabel('Empirical - nominal (median IDP)')
        plt.title(f'Coverage difference: GAMLSS ({dataset_tag.upper()} D{idp_size_tag})')
        plt.legend()
        plt.tight_layout()
        out_png = os.path.join(out_dir, 'coverage_diff_overlay_gamlss.png')
        plt.savefig(out_png, dpi=dpi); plt.close()
        return out_png
    except Exception:
        return None

def _build_nn_structure(X: np.ndarray) -> Tuple[str, object]:
    """Build a nearest-neighbour data structure with best available backend.
    Returns a tuple (backend, obj) where backend in {'sklearn','scipy','numpy'}.
    """
    try:
        from sklearn.neighbors import KDTree as SK_KDTree  # type: ignore
        return 'sklearn', SK_KDTree(X, leaf_size=40, metric='euclidean')
    except Exception:
        try:
            from scipy.spatial import cKDTree  # type: ignore
            return 'scipy', cKDTree(X)
        except Exception:
            return 'numpy', X.astype(np.float64, copy=False)


def _nn_distances_generic(Xq: np.ndarray, nn_struct: Tuple[str, object]) -> np.ndarray:
    backend, obj = nn_struct
    if backend == 'sklearn':
        d, _ = obj.query(Xq, k=1, return_distance=True)  # type: ignore[attr-defined]
        return d.ravel()
    if backend == 'scipy':
        d, _ = obj.query(Xq, k=1)  # type: ignore[attr-defined]
        return np.atleast_1d(d).ravel()
    # numpy fallback: brute-force in chunks to limit memory
    Xref = obj  # type: ignore[assignment]
    qn = Xq.shape[0]; rn = Xref.shape[0]
    bs = max(1, int(1e6 // max(1, Xq.shape[1])))
    out = np.empty(qn, dtype=np.float64)
    for i in range(0, qn, bs):
        j = min(qn, i + bs)
        block = Xq[i:j]
        # (b, d) vs (rn, d) -> (b, rn)
        d2 = (
            np.sum(block * block, axis=1, keepdims=True)
            + np.sum(Xref * Xref, axis=1, keepdims=True).T
            - 2.0 * (block @ Xref.T)
        )
        d2 = np.maximum(d2, 0.0)
        out[i:j] = np.sqrt(np.min(d2, axis=1))
    return out


def nn_memorisation_checks(
    gen_df: pd.DataFrame,
    train_df: pd.DataFrame,
    hold_df: pd.DataFrame,
    idp_cols: List[str],
    out_dir: str,
    standardize: bool = True,
    dpi: int = 300,
    near_dup_eps: float = 1e-6,
    cov2_col: str = 'sex',
    num_age_bins: int = 10,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    # Balance train and holdout by (age_bin, cov2) strata to ensure matched density
    df_tr = train_df.copy(); df_ho = hold_df.copy()
    # Build common age bins
    a_min = float(min(df_tr['age'].min(), df_ho['age'].min()))
    a_max = float(max(df_tr['age'].max(), df_ho['age'].max()))
    if not np.isfinite(a_min) or not np.isfinite(a_max) or a_max <= a_min:
        age_edges = np.array([0.0, 1.0])
    else:
        # aim for approx yearly bins within [min, max]
        left = math.floor(a_min) - 0.5
        right = math.ceil(a_max) + 0.5
        step = max(1.0, (right - left) / max(1, num_age_bins))
        age_edges = np.arange(left, right + 1e-9, step)
    df_tr['_age_bin'] = pd.cut(df_tr['age'], bins=age_edges, include_lowest=True)
    df_ho['_age_bin'] = pd.cut(df_ho['age'], bins=age_edges, include_lowest=True)
    # Determine target counts per stratum
    strata = sorted(set(df_tr['_age_bin'].dropna().unique()).union(set(df_ho['_age_bin'].dropna().unique())), key=lambda x: x.left if hasattr(x, 'left') else -np.inf)
    def _sample_by_strata(df: pd.DataFrame, targets: Dict[Tuple[Any, Any], int]) -> pd.DataFrame:
        parts = []
        rng = np.random.default_rng(12345)
        for (ab, c2), n_tgt in targets.items():
            if n_tgt <= 0:
                continue
            m = df[(df['_age_bin'] == ab) & (df[cov2_col] == c2)]
            if m.shape[0] == 0:
                continue
            take = min(n_tgt, m.shape[0])
            idx = rng.choice(m.index.values, size=take, replace=False)
            parts.append(df.loc[idx])
        if parts:
            out_df = pd.concat(parts, axis=0)
            return out_df.sample(frac=1.0, random_state=12345).reset_index(drop=True)
        return df.iloc[0:0].copy()
    # Build joint level combinations
    combos = sorted(set([(ab, c2) for ab in strata for c2 in pd.concat([df_tr[cov2_col], df_ho[cov2_col]]).dropna().unique()]), key=lambda t: (str(t[0]), t[1]))
    targets: Dict[Tuple[Any, Any], int] = {}
    for (ab, c2) in combos:
        n_tr = int(df_tr[(df_tr['_age_bin'] == ab) & (df_tr[cov2_col] == c2)].shape[0])
        n_ho = int(df_ho[(df_ho['_age_bin'] == ab) & (df_ho[cov2_col] == c2)].shape[0])
        targets[(ab, c2)] = min(n_tr, n_ho)
    df_tr_b = _sample_by_strata(df_tr, targets)
    df_ho_b = _sample_by_strata(df_ho, targets)
    # Remove helper cols
    df_tr_b = df_tr_b.drop(columns=['_age_bin'], errors='ignore')
    df_ho_b = df_ho_b.drop(columns=['_age_bin'], errors='ignore')

    G = gen_df[idp_cols].to_numpy(dtype=np.float64, copy=True)
    T = df_tr_b[idp_cols].to_numpy(dtype=np.float64, copy=True)
    H = df_ho_b[idp_cols].to_numpy(dtype=np.float64, copy=True)

    if standardize:
        mu = T.mean(axis=0)
        sd = T.std(axis=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        G = (G - mu) / sd
        T = (T - mu) / sd
        H = (H - mu) / sd

    tree_T = _build_nn_structure(T)
    tree_H = _build_nn_structure(H)
    dT = _nn_distances_generic(G, tree_T)
    dH = _nn_distances_generic(G, tree_H)
    ratio = dT / np.maximum(dH, 1e-12)

    out_csv = os.path.join(out_dir, 'nn_memorisation_metrics.csv')
    pd.DataFrame({'d_train': dT, 'd_holdout': dH, 'ratio': ratio}).to_csv(out_csv, index=False)

    # Helper for binomial Wilson CI (95%)
    def _wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float, float]:
        if n <= 0:
            return float('nan'), float('nan'), float('nan')
        p = k / float(n)
        z = 1.959963984540054  # approx 97.5th percentile
        den = 1.0 + (z * z) / n
        center = (p + (z * z) / (2.0 * n)) / den
        half = z * math.sqrt((p * (1.0 - p) / n) + (z * z) / (4.0 * n * n)) / den
        return p, max(0.0, center - half), min(1.0, center + half)

    plt.figure(figsize=(5, 4))
    r_ok = ratio[np.isfinite(ratio)]
    plt.hist(r_ok, bins=50, density=True, alpha=0.8)
    plt.title('Nearest-neighbour ratio: dNN(train) / dNN(holdout)')
    plt.xlabel('ratio'); plt.ylabel('density')
    # Annotate expectation indicators; remove CI and n
    n_r = int(np.isfinite(ratio).sum())
    k_lt1 = int(np.sum(r_ok < 1.0))
    k_lt08 = int(np.sum(r_ok < 0.8))
    p_lt1, _lo1, _hi1 = _wilson_ci(k_lt1, n_r)
    p_lt08 = (k_lt08 / float(n_r)) if n_r > 0 else float('nan')
    txt = (
        f"E[𝟙(r<1)] = {p_lt1:.3f}\n"
        f"E[𝟙(r<0.8)] = {p_lt08:.3f}"
    )
    ax = plt.gca()
    ax.text(0.98, 0.98, txt, ha='right', va='top', transform=ax.transAxes, fontsize=9,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'nn_ratio_hist.png'), dpi=dpi); plt.close()

    plt.figure(figsize=(5, 4))
    eps = 1e-12
    logH = np.log10(dH + eps)
    logT = np.log10(dT + eps)
    plt.plot(logH, logT, '.', ms=2)
    try:
        both = np.concatenate([logH[np.isfinite(logH)], logT[np.isfinite(logT)]])
        if both.size > 0:
            lo, hi = np.nanpercentile(both, [0.5, 99.5])
            # Ensure a reasonable numeric span
            if not np.isfinite(lo):
                lo = float(np.nanmin(both))
            if not np.isfinite(hi):
                hi = float(np.nanmax(both))
            if not np.isfinite(lo) or not np.isfinite(hi):
                lo, hi = -8.0, 1.0
            margin = 0.05 * max(1e-6, (hi - lo))
            lo -= margin; hi += margin
            xs = np.linspace(lo, hi, 100)
            plt.plot(xs, xs, '--')
            ax = plt.gca()
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
    except Exception:
        pass
    plt.xlabel('log10 dNN(holdout)'); plt.ylabel('log10 dNN(train)')
    plt.title('NN distances: train vs hold-out (log-log)')

    # Annotate median ratio and Pr[dT<dH] without CI and n
    mask = np.isfinite(dT) & np.isfinite(dH)
    n = int(np.sum(mask))
    med_r = float(np.nanmedian(ratio)) if np.isfinite(ratio).any() else float('nan')
    k = int(np.sum(dT[mask] < dH[mask]))
    p_hat, _lo_p, _hi_p = _wilson_ci(k, n)
    ann = (
        f"median r={med_r:.3f}\n"
        f"Pr[dT<dH]={p_hat:.3f}"
    )
    ax = plt.gca()
    ax.text(0.98, 0.98, ann, ha='right', va='top', transform=ax.transAxes, fontsize=9,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'nn_train_vs_hold_scatter.png'), dpi=dpi); plt.close()

    try:
        near_dups = float(np.mean(dT <= near_dup_eps))
        with open(os.path.join(out_dir, 'nn_summary.json'), 'w') as f:
            json.dump({
                'near_duplicate_rate_train': near_dups,
                'ratio_median': float(np.nanmedian(ratio)),
                'ratio_p05': float(np.nanpercentile(ratio, 5)),
                'ratio_p95': float(np.nanpercentile(ratio, 95)),
                'n_train_balanced': int(T.shape[0]),
                'n_holdout_balanced': int(H.shape[0]),
                'covariate': cov2_col,
                'num_age_bins': int(num_age_bins),
            }, f, indent=2)
    except Exception:
        pass
    return out_csv

# -----------------------------------------------------------------------------
# Joint analysis utilities (ported from SAINT script with robust color scaling)
# -----------------------------------------------------------------------------

_GLOBAL_JOINT_CTX: Dict[str, object] = {}


def _init_joint_worker(ctx: Dict[str, object]) -> None:
    global _GLOBAL_JOINT_CTX
    _GLOBAL_JOINT_CTX = ctx


def _hist2d_density(xy: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray) -> np.ndarray:
    H, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[x_edges, y_edges], density=True)
    return H


def _standardize_pair(real_xy: np.ndarray, gen_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    both = np.vstack([real_xy, gen_xy])
    mean_vals = np.nanmean(both, axis=0)
    std_vals = np.nanstd(both, axis=0)
    std_vals = np.where(std_vals < 1e-12, 1.0, std_vals)
    real_z = (real_xy - mean_vals) / std_vals
    gen_z = (gen_xy - mean_vals) / std_vals
    return real_z, gen_z


def _compute_two_sample_metrics(
    real_xy: np.ndarray,
    gen_xy: np.ndarray,
    num_permutations: int = 300,
    random_state: int = 42,
) -> dict:
    rng = np.random.default_rng(random_state)
    n_r = real_xy.shape[0]
    n_g = gen_xy.shape[0]
    n_use = min(n_r, n_g)
    if n_use <= 1:
        return {
            'energy_stat': float('nan'), 'energy_p': float('nan'),
            'mmd2_stat': float('nan'), 'mmd2_p': float('nan'),
            'n_used': int(n_use * 2)
        }
    idx_r = rng.choice(n_r, size=n_use, replace=False)
    idx_g = rng.choice(n_g, size=n_use, replace=False)
    X = np.vstack([real_xy[idx_r], gen_xy[idx_g]])
    y = np.concatenate([np.zeros(n_use, dtype=int), np.ones(n_use, dtype=int)])[:, None]
    try:
        from scipy.spatial.distance import pdist, squareform
        D = squareform(pdist(X, metric='euclidean'))
        D2 = D * D
    except Exception:
        diff = X[:, None, :] - X[None, :, :]
        D2 = np.sum(diff * diff, axis=-1)
        D = np.sqrt(D2)
    triu_idx = np.triu_indices(D2.shape[0], k=1)
    med_sq = np.median(D2[triu_idx]) if D2.shape[0] > 1 else 1.0
    med_sq = float(med_sq if np.isfinite(med_sq) and med_sq > 1e-12 else 1.0)
    gamma = 1.0 / (2.0 * med_sq)
    K = np.exp(-gamma * D2)
    np.fill_diagonal(K, 0.0)

    def _energy_stat_from_labels(lbl: np.ndarray) -> float:
        lbl = lbl.ravel().astype(int)
        idx0 = np.where(lbl == 0)[0]
        idx1 = np.where(lbl == 1)[0]
        n = len(idx0); m = len(idx1)
        if n <= 1 or m <= 1:
            return float('nan')
        cross = D[np.ix_(idx0, idx1)].sum() / (n * m)
        within0 = D[np.ix_(idx0, idx0)].sum() / (n * (n - 1))
        within1 = D[np.ix_(idx1, idx1)].sum() / (m * (m - 1))
        return float(2.0 * cross - within0 - within1)

    def _mmd2_stat_from_labels(lbl: np.ndarray) -> float:
        lbl = lbl.ravel().astype(int)
        idx0 = np.where(lbl == 0)[0]
        idx1 = np.where(lbl == 1)[0]
        n = len(idx0); m = len(idx1)
        if n <= 1 or m <= 1:
            return float('nan')
        k00 = K[np.ix_(idx0, idx0)].sum() / (n * (n - 1))
        k11 = K[np.ix_(idx1, idx1)].sum() / (m * (m - 1))
        k01 = K[np.ix_(idx0, idx1)].sum() / (n * m)
        return float(k00 + k11 - 2.0 * k01)

    energy_stat = _energy_stat_from_labels(np.vstack([np.zeros((n_use, 1), dtype=int), np.ones((n_use, 1), dtype=int)]))
    mmd2_stat = _mmd2_stat_from_labels(np.vstack([np.zeros((n_use, 1), dtype=int), np.ones((n_use, 1), dtype=int)]))

    perm_stats_energy = []
    perm_stats_mmd = []
    if num_permutations and num_permutations > 0:
        for _ in range(num_permutations):
            perm_lbl = y.copy()
            rng.shuffle(perm_lbl)
            perm_stats_energy.append(_energy_stat_from_labels(perm_lbl))
            perm_stats_mmd.append(_mmd2_stat_from_labels(perm_lbl))
        perm_stats_energy = np.asarray(perm_stats_energy, dtype=float)
        perm_stats_mmd = np.asarray(perm_stats_mmd, dtype=float)
        energy_p = float((1.0 + np.sum(perm_stats_energy >= energy_stat)) / (1.0 + len(perm_stats_energy))) if np.isfinite(energy_stat) else float('nan')
        mmd2_p = float((1.0 + np.sum(perm_stats_mmd >= mmd2_stat)) / (1.0 + len(perm_stats_mmd))) if np.isfinite(mmd2_stat) else float('nan')
    else:
        energy_p = float('nan'); mmd2_p = float('nan')

    return {
        'energy_stat': float(energy_stat), 'energy_p': float(energy_p),
        'mmd2_stat': float(mmd2_stat), 'mmd2_p': float(mmd2_p),
        'n_used': int(n_use * 2),
    }


def _joint_worker(pair: Tuple[str, str]) -> Optional[Dict[str, object]]:
    try:
        cx, cy = pair
        ctx = _GLOBAL_JOINT_CTX
        real_df: pd.DataFrame = ctx['real_df']  # type: ignore[assignment]
        gen_df: pd.DataFrame = ctx['gen_df']  # type: ignore[assignment]
        bins: int = int(ctx['bins'])
        standardize_axes: bool = bool(ctx['standardize_axes'])
        max_samples_per_group: int = int(ctx['max_samples_per_group'])
        num_permutations: int = int(ctx['num_permutations'])
        random_state: int = int(ctx['random_state'])

        rng_local = np.random.default_rng((hash(cx + "|" + cy) + random_state) % (2**32 - 1))
        real_xy = real_df[[cx, cy]].dropna().values
        gen_xy = gen_df[[cx, cy]].dropna().values
        if len(real_xy) == 0 or len(gen_xy) == 0:
            return None
        if len(real_xy) > max_samples_per_group:
            real_xy = real_xy[rng_local.choice(len(real_xy), size=max_samples_per_group, replace=False)]
        if len(gen_xy) > max_samples_per_group:
            gen_xy = gen_xy[rng_local.choice(len(gen_xy), size=max_samples_per_group, replace=False)]
        if standardize_axes:
            real_xy, gen_xy = _standardize_pair(real_xy, gen_xy)
            x_edges = np.linspace(-3.0, 3.0, bins + 1)
            y_edges = np.linspace(-3.0, 3.0, bins + 1)
        else:
            both = np.vstack([real_xy, gen_xy])
            x_low, x_high = np.percentile(both[:, 0], [1.0, 99.0])
            y_low, y_high = np.percentile(both[:, 1], [1.0, 99.0])
            if not np.isfinite(x_high - x_low) or (x_high - x_low) < 1e-6:
                x_low, x_high = float(both[:, 0].min()), float(both[:, 0].max())
                if x_high - x_low < 1e-6:
                    x_low, x_high = -3.0, 3.0
            if not np.isfinite(y_high - y_low) or (y_high - y_low) < 1e-6:
                y_low, y_high = float(both[:, 1].min()), float(both[:, 1].max())
                if y_high - y_low < 1e-6:
                    y_low, y_high = -3.0, 3.0
            x_edges = np.linspace(x_low, x_high, bins + 1)
            y_edges = np.linspace(y_low, y_high, bins + 1)
        real_H = _hist2d_density(real_xy, x_edges, y_edges)
        gen_H = _hist2d_density(gen_xy, x_edges, y_edges)
        diff_H = gen_H - real_H
        m = _compute_two_sample_metrics(real_xy, gen_xy, num_permutations=num_permutations, random_state=random_state)
        m.update({'idp_x': cx, 'idp_y': cy})
        return {
            'idp_x': cx,
            'idp_y': cy,
            'real_H': real_H,
            'gen_H': gen_H,
            'diff_H': diff_H,
            'x_edges': x_edges,
            'y_edges': y_edges,
            'metrics': m,
        }
    except Exception:
        return None


def make_joint_heatmap_pages(
    real_df: pd.DataFrame,
    gen_df: pd.DataFrame,
    idp_cols: List[str],
    output_dir: str,
    rows_per_page: int = 10,
    bins: int = 60,
    standardize_axes: bool = True,
    log_scale: bool = False,
    max_samples_per_group: int = 3000,
    num_permutations: int = 300,
    random_state: int = 42,
    n_jobs: int = 1,
    color_quantile: Optional[float] = 0.995,
    dpi: int = 300,
) -> str:
    global _GLOBAL_JOINT_CTX
    os.makedirs(output_dir, exist_ok=True)
    include_p = bool(num_permutations) and int(num_permutations) > 0
    pairs = [(idp_cols[i], idp_cols[j]) for i in range(len(idp_cols)) for j in range(i + 1, len(idp_cols))]
    if not pairs:
        return ""
    total_pairs = len(pairs)
    n_workers = os.cpu_count() if n_jobs in (-1, 0, None) else max(1, int(n_jobs))
    ctx: Dict[str, object] = {
        'real_df': real_df,
        'gen_df': gen_df,
        'bins': int(bins),
        'standardize_axes': bool(standardize_axes),
        'max_samples_per_group': int(max_samples_per_group),
        'num_permutations': int(num_permutations),
        'random_state': int(random_state),
    }
    results_list = []
    if n_workers == 1:
        _GLOBAL_JOINT_CTX = ctx
        iterator = pairs
        if tqdm is not None:
            iterator = tqdm(pairs, total=total_pairs, desc='Computing joint pairs', leave=False)
        for p in iterator:
            r = _joint_worker(p)
            if r is not None:
                results_list.append(r)
    else:
        pbar = tqdm(total=int(math.ceil(total_pairs / float(n_workers))), desc=f'Computing joint pairs (x{n_workers})', leave=False) if tqdm is not None else None
        processed_in_batch = 0
        try:
            start_method = mp.get_start_method()
        except Exception:
            start_method = None
        if start_method == 'fork':
            _GLOBAL_JOINT_CTX = ctx
            with mp.Pool(processes=n_workers) as pool:
                for r in pool.imap_unordered(_joint_worker, pairs):
                    processed_in_batch += 1
                    if r is not None:
                        results_list.append(r)
                    if processed_in_batch >= n_workers:
                        processed_in_batch = 0
                        if pbar is not None:
                            pbar.update(1)
        else:
            with mp.Pool(processes=n_workers, initializer=_init_joint_worker, initargs=(ctx,)) as pool:
                for r in pool.imap_unordered(_joint_worker, pairs):
                    processed_in_batch += 1
                    if r is not None:
                        results_list.append(r)
                    if processed_in_batch >= n_workers:
                        processed_in_batch = 0
                        if pbar is not None:
                            pbar.update(1)
        if pbar is not None and processed_in_batch > 0:
            pbar.update(1)
        if pbar is not None:
            pbar.close()

    key_map = {(r['idp_x'], r['idp_y']): r for r in results_list}
    ordered_results = [key_map[p] for p in pairs if p in key_map]

    pages = int(math.ceil(len(ordered_results) / float(rows_per_page)))
    page_prefix = os.path.join(output_dir, 'joint_pairs_page')

    metrics_rows = []
    eps = 1e-12
    for page_idx in range(pages):
        start = page_idx * rows_per_page
        end = min((page_idx + 1) * rows_per_page, len(ordered_results))
        page_res = ordered_results[start:end]
        if not page_res:
            continue

        # robust page-level density scaling
        if color_quantile is not None and 0.0 < float(color_quantile) < 1.0:
            dens_vals = []
            for r in page_res:
                if r['real_H'].size:
                    dens_vals.append(np.maximum(r['real_H'], eps).ravel())
                if r['gen_H'].size:
                    dens_vals.append(np.maximum(r['gen_H'], eps).ravel())
            if dens_vals:
                dens_all = np.concatenate(dens_vals)
                max_density = float(np.nanquantile(dens_all, float(color_quantile)))
            else:
                max_density = 1.0
        else:
            max_density = 0.0
            for r in page_res:
                if r['real_H'].size:
                    max_density = max(max_density, float(np.nanmax(r['real_H'])))
                if r['gen_H'].size:
                    max_density = max(max_density, float(np.nanmax(r['gen_H'])))
            if not np.isfinite(max_density) or max_density <= 0:
                max_density = 1.0
        norm_realgen = LogNorm(vmin=eps, vmax=max_density) if log_scale else Normalize(vmin=0.0, vmax=max_density)

        max_abs_diff = 0.0
        for r in page_res:
            if r['diff_H'].size:
                mad = float(np.nanmax(np.abs(r['diff_H'])))
                if np.isfinite(mad):
                    max_abs_diff = max(max_abs_diff, mad)
        if not np.isfinite(max_abs_diff) or max_abs_diff <= 0:
            max_abs_diff = 1.0
        norm_diff = Normalize(vmin=-max_abs_diff, vmax=max_abs_diff)

        n_rows = len(page_res)
        fig, axs = plt.subplots(n_rows, 3, figsize=(3 * 4.2, n_rows * 3.6), squeeze=False)

        for row_idx, item in enumerate(page_res):
            cx = item['idp_x']; cy = item['idp_y']
            real_H = item['real_H']; gen_H = item['gen_H']; diff_H = item['diff_H']
            x_edges = item['x_edges']; y_edges = item['y_edges']
            m = item['metrics']
            extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
            im0 = axs[row_idx, 0].imshow(np.maximum(real_H.T, eps), origin='lower', extent=extent, aspect='auto', cmap='inferno', norm=norm_realgen)
            axs[row_idx, 0].set_title(f"Real: {cx} vs {cy}", fontsize=9)
            im1 = axs[row_idx, 1].imshow(np.maximum(gen_H.T, eps), origin='lower', extent=extent, aspect='auto', cmap='inferno', norm=norm_realgen)
            axs[row_idx, 1].set_title("Generated", fontsize=9)
            im2 = axs[row_idx, 2].imshow(diff_H.T, origin='lower', extent=extent, aspect='auto', cmap='magma', norm=norm_diff)
            axs[row_idx, 2].set_title("Gen - Real", fontsize=9)
            axs[row_idx, 0].set_ylabel(cy, fontsize=8)
            for c in range(3):
                if row_idx == n_rows - 1:
                    axs[row_idx, c].set_xlabel(cx, fontsize=8)
                axs[row_idx, c].tick_params(axis='both', which='major', labelsize=7)
            try:
                r_flat = real_H.ravel(); g_flat = gen_H.ravel()
                if np.all((r_flat == 0)) and np.all((g_flat == 0)):
                    pear_r = float('nan')
                else:
                    pear_r = float(np.corrcoef(r_flat + 1e-12, g_flat + 1e-12)[0, 1])
            except Exception:
                pear_r = float('nan')
            try:
                if include_p:
                    txt = (
                        f"MMD^2={m['mmd2_stat']:.4f} (p={m['mmd2_p']:.3g})\n"
                        f"Energy={m['energy_stat']:.4f} (p={m['energy_p']:.3g})\n"
                        f"Pearson(r)={pear_r:.3f}  N={m['n_used']}"
                    )
                else:
                    txt = (
                        f"MMD^2={m['mmd2_stat']:.4f}\n"
                        f"Energy={m['energy_stat']:.4f}\n"
                        f"Pearson(r)={pear_r:.3f}  N={m['n_used']}"
                    )
            except Exception:
                txt = f"N={m.get('n_used', 0)}"
            axs[row_idx, 2].text(0.02, 0.98, txt, ha='left', va='top', transform=axs[row_idx, 2].transAxes, fontsize=8,
                                  bbox=dict(facecolor='white', edgecolor='none', alpha=0.6))

        fig.tight_layout(rect=[0, 0, 1, 0.9])
        cbar_ax1 = fig.add_axes([0.12, 0.92, 0.35, 0.015])
        cbar_ax2 = fig.add_axes([0.55, 0.92, 0.35, 0.015])
        cbar0 = fig.colorbar(im0, cax=cbar_ax1, orientation='horizontal')
        cbar0.ax.set_xlabel('Density' + (' (log)' if log_scale else ''), fontsize=8)
        cbar1 = fig.colorbar(im2, cax=cbar_ax2, orientation='horizontal')
        cbar1.ax.set_xlabel('Density diff (gen - real)', fontsize=8)

        fig.suptitle('Joint IDP distributions: Real vs Generated (rows: IDP pairs; cols: Real | Gen | Diff)', fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.9])
        out_path = os.path.join(output_dir, f"joint_pairs_page_{page_idx+1:03d}.png")
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)

        for item in page_res:
            metrics_rows.append(item['metrics'])

    metrics_df = pd.DataFrame(metrics_rows)
    if not include_p:
        drop_cols = [c for c in ['energy_p', 'mmd2_p'] if c in metrics_df.columns]
        if drop_cols:
            metrics_df = metrics_df.drop(columns=drop_cols)
    metrics_csv = os.path.join(output_dir, 'joint_pairs_metrics.csv')
    metrics_df.to_csv(metrics_csv, index=False)
    return os.path.join(output_dir, 'joint_pairs_page')


def _joint_vs_marginal_worker(pair: Tuple[str, str]) -> Optional[Dict[str, object]]:
    try:
        cx, cy = pair
        ctx = _GLOBAL_JOINT_CTX
        real_df: pd.DataFrame = ctx['real_df']  # type: ignore[assignment]
        gen_df: pd.DataFrame = ctx['gen_df']  # type: ignore[assignment]
        bins: int = int(ctx['bins'])
        standardize_axes: bool = bool(ctx['standardize_axes'])
        max_samples_per_group: int = int(ctx['max_samples_per_group'])
        num_permutations: int = int(ctx['num_permutations'])
        random_state: int = int(ctx['random_state'])

        rng_local = np.random.default_rng((hash('MP|' + cx + '|' + cy) + random_state) % (2**32 - 1))
        real_xy = real_df[[cx, cy]].dropna().values
        gen_xy = gen_df[[cx, cy]].dropna().values
        if len(real_xy) == 0 or len(gen_xy) == 0:
            return None
        if len(real_xy) > max_samples_per_group:
            real_xy = real_xy[rng_local.choice(len(real_xy), size=max_samples_per_group, replace=False)]
        if len(gen_xy) > max_samples_per_group:
            gen_xy = gen_xy[rng_local.choice(len(gen_xy), size=max_samples_per_group, replace=False)]

        gen_cx = gen_df[cx].dropna().values
        gen_cy = gen_df[cy].dropna().values
        if gen_cx.size == 0 or gen_cy.size == 0:
            return None
        n_baseline = min(max_samples_per_group, gen_cx.size, gen_cy.size)
        # Optionally repeat baseline resampling for more stable metrics
        repeats = int(_GLOBAL_JOINT_CTX.get('baseline_resample_repeats', 1))
        baseline_samples = []
        for rep in range(max(1, repeats)):
            idx_x = rng_local.choice(gen_cx.size, size=n_baseline, replace=True)
            idx_y = rng_local.choice(gen_cy.size, size=n_baseline, replace=True)
            baseline_samples.append(np.column_stack([gen_cx[idx_x], gen_cy[idx_y]]))
        baseline_xy = baseline_samples[0]

        if standardize_axes:
            both_rg = np.vstack([real_xy, gen_xy])
            mean_vals = np.nanmean(both_rg, axis=0)
            std_vals = np.nanstd(both_rg, axis=0)
            std_vals = np.where(std_vals < 1e-12, 1.0, std_vals)
            real_xy = (real_xy - mean_vals) / std_vals
            gen_xy = (gen_xy - mean_vals) / std_vals
            baseline_xy = (baseline_xy - mean_vals) / std_vals
            x_edges = np.linspace(-3.0, 3.0, bins + 1)
            y_edges = np.linspace(-3.0, 3.0, bins + 1)
        else:
            both = np.vstack([real_xy, gen_xy, baseline_xy])
            x_low, x_high = np.percentile(both[:, 0], [1.0, 99.0])
            y_low, y_high = np.percentile(both[:, 1], [1.0, 99.0])
            if not np.isfinite(x_high - x_low) or (x_high - x_low) < 1e-6:
                x_low, x_high = float(both[:, 0].min()), float(both[:, 0].max())
                if x_high - x_low < 1e-6:
                    x_low, x_high = -3.0, 3.0
            if not np.isfinite(y_high - y_low) or (y_high - y_low) < 1e-6:
                y_low, y_high = float(both[:, 1].min()), float(both[:, 1].max())
                if y_high - y_low < 1e-6:
                    y_low, y_high = -3.0, 3.0
            x_edges = np.linspace(x_low, x_high, bins + 1)
            y_edges = np.linspace(y_low, y_high, bins + 1)

        base_H = _hist2d_density(baseline_xy, x_edges, y_edges)
        gen_H = _hist2d_density(gen_xy, x_edges, y_edges)
        real_H = _hist2d_density(real_xy, x_edges, y_edges)
        # Differences following new spec
        diff_gen_vs_base = gen_H - base_H
        diff_real_vs_gen = real_H - gen_H

        # Metrics: product-of-marginals vs gen; and real vs gen
        # Compute baseline-vs-gen stats over repeats if requested
        if len(baseline_samples) > 1:
            m_gen_list = []
            for bxy in baseline_samples:
                m_gen_list.append(_compute_two_sample_metrics(bxy, gen_xy, num_permutations=num_permutations, random_state=random_state))
            # aggregate mean and std
            def _agg(key: str) -> Tuple[float, float]:
                vals = np.array([d.get(key, float('nan')) for d in m_gen_list], dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    return float('nan'), float('nan')
                return float(np.mean(vals)), float(np.std(vals))
            m_gen = {
                'mmd2_stat': _agg('mmd2_stat')[0],
                'mmd2_stat_std': _agg('mmd2_stat')[1],
                'mmd2_p': _agg('mmd2_p')[0],
                'energy_stat': _agg('energy_stat')[0],
                'energy_stat_std': _agg('energy_stat')[1],
                'energy_p': _agg('energy_p')[0],
                'n_used': int(np.mean([d.get('n_used', 0) for d in m_gen_list])),
            }
        else:
            m_gen = _compute_two_sample_metrics(baseline_xy, gen_xy, num_permutations=num_permutations, random_state=random_state)
        m_real_vs_gen = _compute_two_sample_metrics(real_xy, gen_xy, num_permutations=num_permutations, random_state=random_state)

        try:
            pear_gen = float(np.corrcoef(base_H.ravel() + 1e-12, gen_H.ravel() + 1e-12)[0, 1])
        except Exception:
            pear_gen = float('nan')
        try:
            pear_real = float(np.corrcoef(base_H.ravel() + 1e-12, real_H.ravel() + 1e-12)[0, 1])
        except Exception:
            pear_real = float('nan')

        return {
            'idp_x': cx,
            'idp_y': cy,
            'base_H': base_H,
            'gen_H': gen_H,
            'real_H': real_H,
            'diff_gen_vs_base': diff_gen_vs_base,
            'diff_real_vs_gen': diff_real_vs_gen,
            'x_edges': x_edges,
            'y_edges': y_edges,
        'metrics_gen': {**m_gen, 'pearson_density_corr': pear_gen},
            'metrics_real_vs_gen': {**m_real_vs_gen, 'pearson_density_corr': pear_real},
        }
    except Exception:
        return None


def make_joint_vs_marginal_product_pages(
    real_df: pd.DataFrame,
    gen_df: pd.DataFrame,
    idp_cols: List[str],
    output_dir: str,
    rows_per_page: int = 10,
    bins: int = 60,
    standardize_axes: bool = True,
    log_scale: bool = False,
    max_samples_per_group: int = 3000,
    num_permutations: int = 300,
    random_state: int = 42,
    n_jobs: int = 1,
    color_quantile: Optional[float] = 0.995,
    dpi: int = 300,
    produce_ranked_panels: bool = False,
    selected_k: int = 2,
    rank_by: str = 'mmd2_stat_gen',
    stat_overlay_kde: bool = False,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    global _GLOBAL_JOINT_CTX
    include_p = bool(num_permutations) and int(num_permutations) > 0
    pairs = [(idp_cols[i], idp_cols[j]) for i in range(len(idp_cols)) for j in range(i + 1, len(idp_cols))]
    if not pairs:
        return ""
    total_pairs = len(pairs)
    n_workers = os.cpu_count() if n_jobs in (-1, 0, None) else max(1, int(n_jobs))
    # Thread baseline repeats from CLI
    baseline_repeats = int(max(1, getattr(make_joint_vs_marginal_product_pages, '_baseline_repeats', 1)))
    ctx: Dict[str, object] = {
        'real_df': real_df,
        'gen_df': gen_df,
        'bins': int(bins),
        'standardize_axes': bool(standardize_axes),
        'max_samples_per_group': int(max_samples_per_group),
        'num_permutations': int(num_permutations),
        'random_state': int(random_state),
        'baseline_resample_repeats': baseline_repeats,
    }
    results_list = []
    if n_workers == 1:
        _GLOBAL_JOINT_CTX = ctx
        iterator = pairs
        if tqdm is not None:
            iterator = tqdm(pairs, total=total_pairs, desc='Computing joint vs product-of-marginals', leave=False)
        for p in iterator:
            r = _joint_vs_marginal_worker(p)
            if r is not None:
                results_list.append(r)
    else:
        pbar = tqdm(total=int(math.ceil(total_pairs / float(n_workers))), desc=f'Computing joint vs marginals (x{n_workers})', leave=False) if tqdm is not None else None
        processed_in_batch = 0
        try:
            start_method = mp.get_start_method()
        except Exception:
            start_method = None
        if start_method == 'fork':
            _GLOBAL_JOINT_CTX = ctx
            with mp.Pool(processes=n_workers) as pool:
                for r in pool.imap_unordered(_joint_vs_marginal_worker, pairs):
                    processed_in_batch += 1
                    if r is not None:
                        results_list.append(r)
                    if processed_in_batch >= n_workers:
                        processed_in_batch = 0
                        if pbar is not None:
                            pbar.update(1)
        else:
            with mp.Pool(processes=n_workers, initializer=_init_joint_worker, initargs=(ctx,)) as pool:
                for r in pool.imap_unordered(_joint_vs_marginal_worker, pairs):
                    processed_in_batch += 1
                    if r is not None:
                        results_list.append(r)
                    if processed_in_batch >= n_workers:
                        processed_in_batch = 0
                        if pbar is not None:
                            pbar.update(1)
        if pbar is not None and processed_in_batch > 0:
            pbar.update(1)
        if pbar is not None:
            pbar.close()

    key_map = {(r['idp_x'], r['idp_y']): r for r in results_list}
    ordered_results = [key_map[p] for p in pairs if p in key_map]

    pages = int(math.ceil(len(ordered_results) / float(rows_per_page)))
    page_prefix = os.path.join(output_dir, 'joint_vs_marginal_page')

    metrics_rows = []
    eps = 1e-12
    for page_idx in range(pages):
        start = page_idx * rows_per_page
        end = min((page_idx + 1) * rows_per_page, len(ordered_results))
        page_res = ordered_results[start:end]
        if not page_res:
            continue

        if color_quantile is not None and 0.0 < float(color_quantile) < 1.0:
            dens_vals = []
            for r in page_res:
                for key in ['base_H', 'gen_H', 'real_H']:
                    if r[key].size:
                        dens_vals.append(np.maximum(r[key], eps).ravel())
            if dens_vals:
                dens_all = np.concatenate(dens_vals)
                max_density = float(np.nanquantile(dens_all, float(color_quantile)))
            else:
                max_density = 1.0
        else:
            max_density = 0.0
            for r in page_res:
                for key in ['base_H', 'gen_H', 'real_H']:
                    if r[key].size:
                        max_density = max(max_density, float(np.nanmax(r[key])))
            if not np.isfinite(max_density) or max_density <= 0:
                max_density = 1.0
        norm_dens = LogNorm(vmin=eps, vmax=max_density) if log_scale else Normalize(vmin=0.0, vmax=max_density)

        max_abs_diff = 0.0
        for r in page_res:
            for k in ['diff_gen_vs_base', 'diff_real_vs_gen']:
                if r[k].size:
                    mad = float(np.nanmax(np.abs(r[k])))
                    if np.isfinite(mad):
                        max_abs_diff = max(max_abs_diff, mad)
        if not np.isfinite(max_abs_diff) or max_abs_diff <= 0:
            max_abs_diff = 1.0
        norm_diff = Normalize(vmin=-max_abs_diff, vmax=max_abs_diff)

        n_rows = len(page_res)
        fig, axs = plt.subplots(n_rows, 5, figsize=(5 * 4.2, n_rows * 3.6), squeeze=False)

        for row_idx, item in enumerate(page_res):
            cx = item['idp_x']; cy = item['idp_y']
            base_H = item['base_H']; gen_H = item['gen_H']; real_H = item['real_H']
            diff_gen = item['diff_gen_vs_base']; diff_real = item['diff_real_vs_gen']
            x_edges = item['x_edges']; y_edges = item['y_edges']
            m_gen = item['metrics_gen']; m_real = item['metrics_real_vs_gen']
            extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

            im0 = axs[row_idx, 0].imshow(np.maximum(base_H.T, eps), origin='lower', extent=extent, aspect='auto', cmap='inferno', norm=norm_dens)
            axs[row_idx, 0].set_title(f"Baseline (Prod. Marginals): {cx} vs {cy}", fontsize=9)
            im1 = axs[row_idx, 1].imshow(np.maximum(gen_H.T, eps), origin='lower', extent=extent, aspect='auto', cmap='inferno', norm=norm_dens)
            axs[row_idx, 1].set_title("Generated", fontsize=9)
            im2 = axs[row_idx, 2].imshow(diff_gen.T, origin='lower', extent=extent, aspect='auto', cmap='magma', norm=norm_diff)
            axs[row_idx, 2].set_title("Gen - Baseline", fontsize=9)
            im3 = axs[row_idx, 3].imshow(np.maximum(real_H.T, eps), origin='lower', extent=extent, aspect='auto', cmap='inferno', norm=norm_dens)
            axs[row_idx, 3].set_title("Real", fontsize=9)
            im4 = axs[row_idx, 4].imshow(diff_real.T, origin='lower', extent=extent, aspect='auto', cmap='magma', norm=norm_diff)
            axs[row_idx, 4].set_title("Real - Gen", fontsize=9)
            axs[row_idx, 0].set_ylabel(cy, fontsize=8)
            for c in range(5):
                if row_idx == n_rows - 1:
                    axs[row_idx, c].set_xlabel(cx, fontsize=8)
                axs[row_idx, c].tick_params(axis='both', which='major', labelsize=7)
            # Per new spec: remove stats text overlays on difference maps

        fig.tight_layout(rect=[0, 0, 1, 0.9])
        cbar_ax1 = fig.add_axes([0.12, 0.92, 0.35, 0.015])
        cbar_ax2 = fig.add_axes([0.55, 0.92, 0.35, 0.015])
        cbar0 = fig.colorbar(im1, cax=cbar_ax1, orientation='horizontal')
        cbar0.ax.set_xlabel('Density' + (' (log)' if log_scale else ''), fontsize=8)
        cbar1 = fig.colorbar(im2, cax=cbar_ax2, orientation='horizontal')
        cbar1.ax.set_xlabel('Density diff (X - baseline)', fontsize=8)

        fig.suptitle('Joint vs Product-of-Marginals: Baseline | Gen | Gen-Baseline | Real | Real-Baseline', fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.9])
        out_path = os.path.join(output_dir, f"joint_vs_marginal_page_{page_idx+1:03d}.png")
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)

        for item in page_res:
            row = {
                'idp_x': item['idp_x'], 'idp_y': item['idp_y'],
                'mmd2_stat_gen': item['metrics_gen'].get('mmd2_stat', float('nan')),
                'mmd2_p_gen': item['metrics_gen'].get('mmd2_p', float('nan')),
                'energy_stat_gen': item['metrics_gen'].get('energy_stat', float('nan')),
                'energy_p_gen': item['metrics_gen'].get('energy_p', float('nan')),
                'pearson_density_corr_gen': item['metrics_gen'].get('pearson_density_corr', float('nan')),
                'n_used_gen': item['metrics_gen'].get('n_used', 0),
                'mmd2_stat_real_vs_gen': item['metrics_real_vs_gen'].get('mmd2_stat', float('nan')),
                'mmd2_p_real_vs_gen': item['metrics_real_vs_gen'].get('mmd2_p', float('nan')),
                'energy_stat_real_vs_gen': item['metrics_real_vs_gen'].get('energy_stat', float('nan')),
                'energy_p_real_vs_gen': item['metrics_real_vs_gen'].get('energy_p', float('nan')),
                'pearson_density_corr_real_vs_gen': item['metrics_real_vs_gen'].get('pearson_density_corr', float('nan')),
                'n_used_real_vs_gen': item['metrics_real_vs_gen'].get('n_used', 0),
            }
            metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)
    if not include_p:
        drop_cols = [c for c in ['mmd2_p_gen', 'energy_p_gen', 'mmd2_p_real_vs_gen', 'energy_p_real_vs_gen'] if c in metrics_df.columns]
        if drop_cols:
            metrics_df = metrics_df.drop(columns=drop_cols)
    metrics_csv = os.path.join(output_dir, 'joint_vs_marginal_metrics.csv')
    metrics_df.to_csv(metrics_csv, index=False)
    # Distribution plots of statistics
    try:
        stat_dir = os.path.join(output_dir, 'stat_distributions')
        os.makedirs(stat_dir, exist_ok=True)
        def _overlay_hist_kde(x_a: np.ndarray, x_b: np.ndarray, xlabel: str, out_name: str) -> None:
            a = x_a[np.isfinite(x_a)]
            b = x_b[np.isfinite(x_b)]
            if a.size == 0 and b.size == 0:
                return
            lo = float(np.nanmin(np.concatenate([a, b])) if a.size and b.size else (np.nanmin(a) if a.size else np.nanmin(b)))
            hi = float(np.nanmax(np.concatenate([a, b])) if a.size and b.size else (np.nanmax(a) if a.size else np.nanmax(b)))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = 0.0, 1.0
            # Truncate x-axis to half-range for clearer presentation
            x_hi_trunc = lo + 0.5 * (hi - lo)
            bins_edges = np.linspace(lo, x_hi_trunc, 40)
            plt.figure(figsize=(5, 4))
            if a.size:
                plt.hist(a, bins=bins_edges, density=True, alpha=0.5, label='Prod marginals vs Gen')
            if b.size:
                plt.hist(b, bins=bins_edges, density=True, alpha=0.5, label='Real vs Gen')
            plt.xlim(lo, x_hi_trunc)
            plt.xlabel(xlabel, fontsize=12); plt.ylabel('density', fontsize=12)
            ax = plt.gca(); ax.tick_params(axis='both', which='major', labelsize=11)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(stat_dir, out_name), dpi=dpi); plt.close()

        # MMD^2
        mmd_gen = metrics_df['mmd2_stat_gen'].values if 'mmd2_stat_gen' in metrics_df.columns else np.array([])
        mmd_rg = metrics_df['mmd2_stat_real_vs_gen'].values if 'mmd2_stat_real_vs_gen' in metrics_df.columns else np.array([])
        _overlay_hist_kde(mmd_gen, mmd_rg, 'MMD^2', 'mmd2_overlay.png')
        # Energy (E^2 per spec naming)
        e_gen = metrics_df['energy_stat_gen'].values if 'energy_stat_gen' in metrics_df.columns else np.array([])
        e_rg = metrics_df['energy_stat_real_vs_gen'].values if 'energy_stat_real_vs_gen' in metrics_df.columns else np.array([])
        _overlay_hist_kde(e_gen, e_rg, 'Energy', 'energy_overlay.png')
    except Exception:
        pass

    # Optional ranked panels (top/mid/bottom by rank_by metric)
    if produce_ranked_panels:
        try:
            metric_vals = metrics_df[rank_by].values if rank_by in metrics_df.columns else None
            if metric_vals is not None and np.isfinite(metric_vals).any():
                order_idx = np.argsort(metric_vals)  # ascending
                n = len(order_idx)
                groups = {
                    'bottom': order_idx[:selected_k],
                    'middle': order_idx[(n // 2 - selected_k // 2):(n // 2 - selected_k // 2 + selected_k)],
                    'top': order_idx[-selected_k:],
                }
                # Map idp pairs to item dicts for easy retrieval
                key_map = {(r['idp_x'], r['idp_y']): r for r in results_list}
                pair_list = [(row['idp_x'], row['idp_y']) for _, row in metrics_df.iterrows()]
                for grp_name, idxs in groups.items():
                    count = 0
                    for i in idxs:
                        p = pair_list[int(i)]
                        item = key_map.get(p)
                        if item is None:
                            continue
                        count += 1
                        cx = item['idp_x']; cy = item['idp_y']
                        base_H = item['base_H']; gen_H = item['gen_H']; real_H = item['real_H']
                        diff_gen = item['diff_gen_vs_base']; diff_real = item['diff_real_vs_gen']
                        x_edges = item['x_edges']; y_edges = item['y_edges']
                        extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
                        # Per-plot scaling
                        dens_vals = []
                        for key in ['base_H', 'gen_H', 'real_H']:
                            if item[key].size:
                                dens_vals.append(np.maximum(item[key], eps).ravel())
                        if dens_vals:
                            dens_all = np.concatenate(dens_vals)
                            max_density = float(np.nanquantile(dens_all, 0.995)) if np.isfinite(dens_all).any() else 1.0
                            if not np.isfinite(max_density) or max_density <= 0:
                                max_density = 1.0
                        else:
                            max_density = 1.0
                        norm_dens = LogNorm(vmin=eps, vmax=max_density) if log_scale else Normalize(vmin=0.0, vmax=max_density)
                        max_abs_diff = 0.0
                        for k in ['diff_gen_vs_base', 'diff_real_vs_gen']:
                            if item[k].size:
                                mad = float(np.nanmax(np.abs(item[k])))
                                if np.isfinite(mad):
                                    max_abs_diff = max(max_abs_diff, mad)
                        if not np.isfinite(max_abs_diff) or max_abs_diff <= 0:
                            max_abs_diff = 1.0
                        norm_diff = Normalize(vmin=-max_abs_diff, vmax=max_abs_diff)
                        # Build single-row, 5-column plot; larger fonts; no colorbars
                        fig, axs = plt.subplots(1, 5, figsize=(5 * 4.2, 1 * 3.6), squeeze=False)
                        axs = axs[0]
                        axs[0].imshow(np.maximum(base_H.T, eps), origin='lower', extent=extent, aspect='auto', cmap='inferno', norm=norm_dens)
                        axs[0].set_title(f"Baseline: {cx} vs {cy}", fontsize=11)
                        axs[1].imshow(np.maximum(gen_H.T, eps), origin='lower', extent=extent, aspect='auto', cmap='inferno', norm=norm_dens)
                        axs[1].set_title("Generated", fontsize=11)
                        axs[2].imshow(diff_gen.T, origin='lower', extent=extent, aspect='auto', cmap='magma', norm=norm_diff)
                        axs[2].set_title("Gen - Baseline", fontsize=11)
                        axs[3].imshow(np.maximum(real_H.T, eps), origin='lower', extent=extent, aspect='auto', cmap='inferno', norm=norm_dens)
                        axs[3].set_title("Real", fontsize=11)
                        axs[4].imshow(diff_real.T, origin='lower', extent=extent, aspect='auto', cmap='magma', norm=norm_diff)
                        axs[4].set_title("Real - Gen", fontsize=11)
                        axs[0].set_ylabel(cy, fontsize=10)
                        for c in range(5):
                            axs[c].set_xlabel(cx, fontsize=10)
                            axs[c].tick_params(axis='both', which='major', labelsize=10)
                        fig.tight_layout()
                        out_path = os.path.join(output_dir, f"ranked_{grp_name}_{count}_k{selected_k:02d}.png")
                        fig.savefig(out_path, dpi=dpi)
                        plt.close(fig)
        except Exception:
            pass
    return page_prefix


# -----------------------------------------------------------------------------
# Pair-of-pair density-shape correlation utilities (no dCor in this suite)
# -----------------------------------------------------------------------------

_GLOBAL_PAIR_CTX: Dict[str, object] = {}


def _init_pair_worker(ctx: Dict[str, object]) -> None:
    global _GLOBAL_PAIR_CTX
    _GLOBAL_PAIR_CTX = ctx


def _pair_hist_vector_worker(pair: Tuple[str, str]) -> Optional[Dict[str, object]]:
    try:
        cx, cy = pair
        ctx = _GLOBAL_PAIR_CTX
        df: pd.DataFrame = ctx['df']  # type: ignore[assignment]
        bins: int = int(ctx['bins'])
        standardize_axes: bool = bool(ctx['standardize_axes'])
        max_samples_per_group: int = int(ctx['max_samples_per_group'])
        random_state: int = int(ctx['random_state'])

        rng_local = np.random.default_rng((hash('PHV|' + cx + '|' + cy) + random_state) % (2**32 - 1))
        xy = df[[cx, cy]].dropna().values
        if xy.size == 0:
            return None
        if len(xy) > max_samples_per_group:
            xy = xy[rng_local.choice(len(xy), size=max_samples_per_group, replace=False)]
        if standardize_axes:
            m = np.nanmean(xy, axis=0)
            s = np.nanstd(xy, axis=0)
            s = np.where(s < 1e-12, 1.0, s)
            xy = (xy - m) / s
        lo = np.percentile(xy, 1.0, axis=0)
        hi = np.percentile(xy, 99.0, axis=0)
        span = hi - lo
        if not np.all(np.isfinite(span)) or np.any(span < 1e-6):
            lo = np.array([-3.0, -3.0])
            hi = np.array([3.0, 3.0])
        x_edges = np.linspace(lo[0], hi[0], bins + 1)
        y_edges = np.linspace(lo[1], hi[1], bins + 1)
        H, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[x_edges, y_edges], density=True)
        v = H.astype(np.float64).ravel()
        denom = np.linalg.norm(v) + 1e-12
        v = v / denom
        return {'idp_x': cx, 'idp_y': cy, 'vec': v}
    except Exception:
        return None


def build_pair_density_corr_matrix(
    df: pd.DataFrame,
    idp_cols: List[str],
    bins: int = 60,
    standardize_axes: bool = True,
    max_samples_per_group: int = 3000,
    random_state: int = 42,
    n_jobs: int = 1,
) -> Tuple[List[Tuple[str, str]], np.ndarray]:
    global _GLOBAL_PAIR_CTX
    pairs = [(idp_cols[i], idp_cols[j]) for i in range(len(idp_cols)) for j in range(i + 1, len(idp_cols))]
    if not pairs:
        return [], np.zeros((0, 0), dtype=np.float32)
    ctx: Dict[str, object] = {
        'df': df,
        'bins': int(bins),
        'standardize_axes': bool(standardize_axes),
        'max_samples_per_group': int(max_samples_per_group),
        'random_state': int(random_state),
    }
    results_list: List[Dict[str, object]] = []
    n_workers = os.cpu_count() if n_jobs in (-1, 0, None) else max(1, int(n_jobs))
    if n_workers == 1:
        _GLOBAL_PAIR_CTX = ctx
        iterator = pairs
        if tqdm is not None:
            iterator = tqdm(pairs, total=len(pairs), desc='Pair density vectors', leave=False)
        for p in iterator:
            r = _pair_hist_vector_worker(p)
            if r is not None:
                results_list.append(r)
    else:
        try:
            start_method = mp.get_start_method()
        except Exception:
            start_method = None
        if start_method == 'fork':
            _GLOBAL_PAIR_CTX = ctx
            with mp.Pool(processes=n_workers) as pool:
                for r in pool.imap_unordered(_pair_hist_vector_worker, pairs):
                    if r is not None:
                        results_list.append(r)
        else:
            with mp.Pool(processes=n_workers, initializer=_init_pair_worker, initargs=(ctx,)) as pool:
                for r in pool.imap_unordered(_pair_hist_vector_worker, pairs):
                    if r is not None:
                        results_list.append(r)
    key_map = {(r['idp_x'], r['idp_y']): r for r in results_list}
    vecs = []
    present_pairs = []
    for p in pairs:
        r = key_map.get(p)
        if r is None:
            continue
        vecs.append(r['vec'])
        present_pairs.append(p)
    if not vecs:
        return [], np.zeros((0, 0), dtype=np.float32)
    V = np.stack(vecs, axis=0).astype(np.float64)
    Vc = V - V.mean(axis=1, keepdims=True)
    sd = Vc.std(axis=1, ddof=1, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Z = Vc / sd
    den = max(1, Z.shape[1] - 1)
    C = (Z @ Z.T) / den
    C = C.astype(np.float32)
    np.fill_diagonal(C, 1.0)
    return present_pairs, C


def _reorder_by_clustering(C: np.ndarray) -> np.ndarray:
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list  # type: ignore
        from scipy.spatial.distance import squareform  # type: ignore
        D = 1.0 - np.clip(C, -1.0, 1.0)
        D = np.where(np.isfinite(D), D, 1.0)
        D_sym = (D + D.T) * 0.5
        Z = linkage(squareform(D_sym, checks=False), method='average')
        order = leaves_list(Z)
        return order
    except Exception:
        return np.arange(C.shape[0])


def plot_pair_matrix(C: np.ndarray, pairs: List[Tuple[str, str]], out_png: str, title: str,
                     vmin: Optional[float] = None, vmax: Optional[float] = None, cmap: str = 'viridis',
                     order: Optional[np.ndarray] = None, origin: str = 'upper', dpi: int = 300,
                     show_axis_labels: bool = False, title_fontsize: int = 12, colorbar_tick_fontsize: int = 10) -> str:
    labels = [f"{a}×{b}" for (a, b) in pairs]
    if C.size == 0:
        return ''
    if order is None:
        order = _reorder_by_clustering(C)
    C2 = C[np.ix_(order, order)]
    plt.figure(figsize=(10, 8))
    im = plt.imshow(C2, origin=origin, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title, fontsize=title_fontsize)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    try:
        cbar.ax.tick_params(labelsize=colorbar_tick_fontsize)
    except Exception:
        pass
    if show_axis_labels:
        step = max(1, len(labels) // 30)
        idxs = np.arange(0, len(labels), step)
        plt.xticks(idxs, [labels[o] for o in order[idxs]], rotation=90, fontsize=7)
        plt.yticks(idxs, [labels[o] for o in order[idxs]], fontsize=7)
    else:
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=dpi)
    plt.close()
    lab_csv = out_png.replace('.png', '_labels.csv')
    pd.DataFrame({'rowcol_index': np.arange(len(order)), 'pair': [labels[o] for o in order]}).to_csv(lab_csv, index=False)
    npy_path = out_png.replace('.png', '.npy')
    np.save(npy_path, C)
    csv_path = out_png.replace('.png', '.csv')
    pd.DataFrame(C, index=labels, columns=labels).to_csv(csv_path)
    return lab_csv


def plot_pair_absdiff_matrix(CA: np.ndarray, CB: np.ndarray, pairs: List[Tuple[str, str]], out_png: str, title: str,
                             order: Optional[np.ndarray] = None, cmap: str = 'magma', dpi: int = 300,
                             fixed_vmax: Optional[float] = None, title_fontsize: int = 12, colorbar_tick_fontsize: int = 10) -> str:
    if CA.shape != CB.shape or CA.size == 0:
        return ''
    D = np.abs(CA - CB)
    vmax = float(np.nanmax(D)) if fixed_vmax is None else float(fixed_vmax)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    return plot_pair_matrix(D, pairs, out_png, title, vmin=0.0, vmax=vmax, cmap=cmap, order=order, origin='upper', dpi=dpi, title_fontsize=title_fontsize, colorbar_tick_fontsize=colorbar_tick_fontsize)


def mantel_test(A: np.ndarray, B: np.ndarray, num_permutations: int = 1000, random_state: int = 42) -> Dict[str, float]:
    if A.shape != B.shape or A.size == 0:
        return {'pearson': float('nan'), 'p_value': float('nan')}
    mask = np.triu(np.ones_like(A, dtype=bool), k=1)
    a = A[mask]; b = B[mask]
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return {'pearson': float('nan'), 'p_value': float('nan')}
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    obs = float(np.dot(a, b) / len(a))
    if num_permutations is None or num_permutations <= 0:
        return {'pearson': obs, 'p_value': float('nan')}
    rng = np.random.default_rng(random_state)
    P = A.shape[0]
    perm_vals = []
    for _ in range(num_permutations):
        perm = rng.permutation(P)
        Bp = B[np.ix_(perm, perm)]
        bp = Bp[mask]
        bp = bp[np.isfinite(bp)]
        bp = (bp - bp.mean()) / (bp.std() + 1e-12)
        perm_vals.append(float(np.dot(a, bp) / len(a)))
    perm_vals = np.asarray(perm_vals, dtype=float)
    p = float((1.0 + np.sum(perm_vals >= obs)) / (1.0 + len(perm_vals)))
    return {'pearson': obs, 'p_value': p}


# -----------------------------------------------------------------------------
# Orchestration helpers
# -----------------------------------------------------------------------------

def _derive_seed(base_seed: int, *tags: Any) -> int:
    s = str((base_seed,) + tuple(tags))
    return abs(hash(s)) % (2**31 - 1)


def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _save_json(obj: Any, path: str) -> None:
    dirn = os.path.dirname(path)
    if dirn:
        os.makedirs(dirn, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def _aggregate_metric_for_learning(value_dict: Dict[str, Dict[str, float]], prefer_keys: List[str]) -> float:
    """Aggregate a nested metric dict into a single scalar for learning/dim fits.
    Preference order among keys; fallback to overall mean of all finite floats.
    """
    # Try preferred keys
    vals = []
    for _, d in value_dict.items():
        for k in prefer_keys:
            if k in d and np.isfinite(d[k]):
                vals.append(float(d[k]))
    if not vals:
        # fallback: collect all finite floats
        for _, d in value_dict.items():
            for v in d.values():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    vals.append(float(v))
    return float(np.mean(vals)) if vals else float('nan')


def _fit_slope(xs: np.ndarray, ys: np.ndarray) -> float:
    # OLS slope of y = a + b x
    if xs.size < 2 or ys.size < 2:
        return float('nan')
    X = np.vstack([np.ones_like(xs), xs]).T
    try:
        beta, *_ = np.linalg.lstsq(X, ys, rcond=None)
        return float(beta[1])
    except Exception:
        return float('nan')


def _bootstrap_slope(seeds_to_points: Dict[int, Tuple[np.ndarray, np.ndarray]], B: int, logx: bool = False) -> Dict[str, float]:
    seed_keys = list(seeds_to_points.keys())
    if not seed_keys:
        return {'slope': float('nan'), 'ci_low': float('nan'), 'ci_high': float('nan')}
    rng = np.random.default_rng(12345)
    slopes = []
    for _ in range(B):
        # resample seeds with replacement
        chosen = rng.choice(seed_keys, size=len(seed_keys), replace=True)
        xs_all = []
        ys_all = []
        for s in chosen:
            xs, ys = seeds_to_points[int(s)]
            xs_all.append(xs)
            ys_all.append(ys)
        xs_all = np.concatenate(xs_all)
        ys_all = np.concatenate(ys_all)
        if logx:
            xs_use = np.log(xs_all)
        else:
            xs_use = xs_all
        slopes.append(_fit_slope(xs_use, ys_all))
    slopes = np.asarray(slopes, dtype=float)
    slopes = slopes[np.isfinite(slopes)]
    if slopes.size == 0:
        return {'slope': float('nan'), 'ci_low': float('nan'), 'ci_high': float('nan')}
    return {
        'slope': float(np.mean(slopes)),
        'ci_low': float(np.percentile(slopes, 2.5)),
        'ci_high': float(np.percentile(slopes, 97.5)),
    }


# -----------------------------------------------------------------------------
# CLI and main
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("Full evaluation suite for normative DDPMs (MLP/SAINT) and GAMLSS baseline")
    
    # Config file support
    p.add_argument('--config', type=str, help='Path to YAML config file. CLI arguments override config file values.')
    
    # Data IO (base): allow single case or maps for dimensional scaling
    p.add_argument('--backbone', choices=['mlp', 'saint', 'both'], default='both', help='Which backbone(s) to run.')
    
    # GAMLSS baseline options
    p.add_argument('--run_gamlss_baseline', action='store_true', help='Run GAMLSS univariate baseline (one model per IDP).')
    p.add_argument('--gamlss_only', action='store_true', help='Run ONLY GAMLSS baseline, skip diffusion models.')
    p.add_argument('--gamlss_family', type=str, default='SHASH', help='GAMLSS family (SHASH, BCT, BCPE, etc.).')
    p.add_argument('--gamlss_mu_formula', type=str, default='~ cs(age, df=3) + cov2', help='GAMLSS mu formula (R syntax). "cov2" is automatically replaced with sex/state.')
    p.add_argument('--gamlss_sigma_formula', type=str, default='~ cs(age, df=3) + cov2', help='GAMLSS sigma formula (R syntax). "cov2" is automatically replaced with sex/state.')
    p.add_argument('--gamlss_nu_formula', type=str, default='~ cs(age, df=3) + cov2', help='GAMLSS nu formula (R syntax).')
    p.add_argument('--gamlss_tau_formula', type=str, default='~ cs(age, df=3) + cov2', help='GAMLSS tau formula (R syntax).')
    p.add_argument('--gamlss_samples_per_grid', type=int, default=512, help='Number of samples to draw per covariate grid point for KS tests.')
    p.add_argument('--gamlss_n_cyc', type=int, default=100, help='GAMLSS max number of cycles for outer iteration (default: 100).')
    p.add_argument('--gamlss_c_crit', type=float, default=0.001, help='GAMLSS convergence criterion for outer iteration (default: 0.001).')
    p.add_argument('--gamlss_trace_fit', action='store_true', help='Print GAMLSS fitting progress for each IDP.')
    
    p.add_argument('--csv_path', type=str, help='Single CSV to split (base case).')
    p.add_argument('--train_csv', type=str, help='Training CSV (base case).')
    p.add_argument('--holdout_csv', type=str, help='Hold-out CSV (base case).')
    p.add_argument('--train_csv_map', type=str, help='JSON path mapping IDP size -> training CSV for dim scaling.')
    p.add_argument('--holdout_csv_map', type=str, help='JSON path mapping IDP size -> holdout CSV for dim scaling.')
    p.add_argument('--min_age', type=int, default=55)
    p.add_argument('--max_age', type=int, default=75)
    p.add_argument('--split_ratio', type=float, default=0.8)
    p.add_argument('--split_seed', type=int, default=42)
    p.add_argument('--save_splits_dir', type=str, default=os.path.join(THIS_DIR, 'data'))
    # Synthetic dataset support
    p.add_argument('--synth_csv', type=str, help='Synthetic dataset CSV (age + state). Evaluated alongside UKB if provided.')
    p.add_argument('--synth_split_ratio', type=float, default=0.8)
    p.add_argument('--synth_split_seed', type=int, default=42)
    p.add_argument('--synth_min_age', type=float, default=None)
    p.add_argument('--synth_max_age', type=float, default=None)
    # Covariate-2 configuration (UKB: sex, Synth: state)
    p.add_argument('--ukb_cov2_col', type=str, default='sex')
    p.add_argument('--synth_cov2_col', type=str, default='state')

    # Diffusion + backbones (union)
    p.add_argument('--num_steps', type=int, default=100)
    p.add_argument('--beta_start', type=float, default=2e-6)
    p.add_argument('--beta_end', type=float, default=0.6)
    # MLP denoiser
    p.add_argument('--denoiser_depth', type=int, default=16)
    p.add_argument('--denoiser_width', type=int, default=1024)
    p.add_argument('--denoiser_dropout', type=float, default=0.05)
    # SAINT specifics
    p.add_argument('--d_model', type=int, default=64)
    p.add_argument('--depth', type=int, default=6)
    p.add_argument('--n_heads', type=int, default=8)
    p.add_argument('--row_interval', type=int, default=2)

    # Training
    p.add_argument('--run_group', type=str, default=None, help='Group name for this multi-run suite (used for directory structure).')
    p.add_argument(
        '--results_dir',
        type=str,
        default=os.path.abspath(os.path.join(THIS_DIR, '..')),
        help=(
            "Directory that will contain the `results_full_eval/` folder. "
            "Default matches the historical `ukb_rap/` layout (i.e., bundle lives in a subfolder "
            "and results are written next to it)."
        ),
    )
    p.add_argument('--run_id', type=str, default='baseline', help='Run id stem (will be augmented per condition).')
    p.add_argument('--epochs', type=int, default=1500)
    p.add_argument('--mlp_epochs', type=int, default=None, help='Override epochs for MLP (default: --epochs).')
    p.add_argument('--saint_epochs', type=int, default=None, help='Override epochs for SAINT (default: --epochs).')
    # Separate default learning rates per backbone
    p.add_argument('--mlp_lr', type=float, default=1e-3)
    p.add_argument('--saint_lr', type=float, default=3e-4)
    # Separate batch sizes per backbone (defaults: MLP=1024, SAINT=512). Optional legacy global override.
    p.add_argument('--mlp_batch_size', type=int, default=None)
    p.add_argument('--saint_batch_size', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=None, help='Legacy global batch size; applies to both backbones if specific overrides are not provided.')
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--skip_if_exists', action='store_true', help='Skip training if a best checkpoint exists for the condition.')
    # Checkpoint selection for evaluation
    p.add_argument('--eval_checkpoint', choices=['best', 'final'], default='best', help='Which checkpoint to load for evaluation.')
    p.add_argument('--checkpoint_path', type=str, help='Optional explicit checkpoint path to load for evaluation.')

    # Sampling
    p.add_argument('--num_samples', type=int, default=1000)
    p.add_argument('--times_to_sample', type=int, default=200)

    # Evaluation controls
    p.add_argument('--label_col', type=str, default=None)
    p.add_argument('--positive_class', type=int, default=1)
    # New diagnostics
    p.add_argument('--eval_pit', action='store_true')
    p.add_argument('--eval_coverage_curve', action='store_true')
    p.add_argument('--eval_nn_mem', action='store_true')
    # Joint analysis flags
    p.add_argument('--joint_analysis', action='store_true')
    p.add_argument('--joint_rows_per_page', type=int, default=10)
    p.add_argument('--joint_bins', type=int, default=15)
    p.add_argument('--joint_standardize_axes', action='store_true')
    p.add_argument('--joint_log_scale', action='store_true')
    p.add_argument('--joint_max_samples_per_group', type=int, default=5000)
    p.add_argument('--joint_num_permutations', type=int, default=300)
    p.add_argument('--joint_random_state', type=int, default=42)
    p.add_argument('--joint_n_jobs', type=int, default=-1)
    p.add_argument('--joint_color_quantile', type=float, default=0.995)
    # Product-of-marginals analysis
    p.add_argument('--marginal_product_analysis', action='store_true')
    p.add_argument('--disable_permutation_tests', action='store_true')
    p.add_argument('--stat_overlay_kde', action='store_true', help='Overlay kernel density estimates on metric distributions for joint-vs-marginals.')
    # Pair-of-pair analyses (density-shape only)
    p.add_argument('--pair_density_shape_corr', action='store_true')
    p.add_argument('--pair_bins', type=int, default=15)
    p.add_argument('--pair_standardize_axes', action='store_true')
    p.add_argument('--pair_max_samples_per_pair', type=int, default=3000)
    p.add_argument('--pair_random_state', type=int, default=42)
    p.add_argument('--pair_n_jobs', type=int, default=-1)
    p.add_argument('--pair_mantel', action='store_true')
    p.add_argument('--pair_idp_limit', type=int, default=20, help='Limit IDP count for pair-of-pair analyses to avoid combinatorial blow-up.')

    # New: ACE and smoothing
    p.add_argument('--ace_percentiles', type=str, default='0.02,0.25,0.5,0.75,0.98')
    p.add_argument('--ace_lowess_frac', type=float, default=0.2)
    p.add_argument('--ace_use_smoothing', action='store_true')
    p.add_argument('--save_plots_dpi', type=int, default=300)

    # KS p-values options
    p.add_argument('--ks_perm_per_bin', action='store_true', help='Use permutation-based p-values for KS in each age bin.')
    p.add_argument('--ks_perm_B', type=int, default=1000, help='Number of permutations for per-bin KS p-values if enabled.')

    # New: Learning curves and dimensional scaling
    p.add_argument('--run_learning_curves', action='store_true')
    p.add_argument('--train_fracs', type=str, default='0.1,0.25,0.5,1.0')
    p.add_argument('--num_seeds', type=int, default=3)
    p.add_argument('--bootstrap_B', type=int, default=100)
    p.add_argument('--run_dim_scaling', action='store_true')
    p.add_argument('--idp_sizes', type=str, default='2,20,100,200')
    p.add_argument('--global_seed', type=int, default=42)

    # Product-of-marginals baseline repeats
    p.add_argument('--baseline_resample_repeats', type=int, default=1, help='Number of independent product-of-marginals resamples per pair for baseline-vs-gen metrics (>=1).')

    return p.parse_args()


def load_config_and_merge_args(args: argparse.Namespace) -> argparse.Namespace:
    """Load YAML config file if specified and merge with CLI arguments.
    
    CLI arguments take precedence over config file values.
    Config file values are only used if the CLI argument was not explicitly provided
    (i.e., it's still at its default value).
    """
    if not args.config:
        return args
    
    if yaml is None:
        print("Warning: PyYAML not installed. Install it with 'pip install pyyaml' to use config files.")
        print("Proceeding with CLI arguments only.")
        return args
    
    config_path = args.config
    if not os.path.isabs(config_path):
        # Make relative to script directory
        config_path = os.path.join(THIS_DIR, config_path)
    
    if not os.path.exists(config_path):
        print(f"Warning: Config file not found: {config_path}")
        print("Proceeding with CLI arguments only.")
        return args
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            config = {}
        
        print(f"Loaded config from: {config_path}")

        # Merge strategy: CLI arguments override config values.
        # We detect CLI-provided args by scanning `sys.argv` for `--<flag>` occurrences.
        cli_args_set = set()
        for i, arg in enumerate(sys.argv):
            if arg.startswith('--'):
                # Extract argument name (remove -- and convert to underscore format)
                arg_name = arg[2:].replace('-', '_')
                cli_args_set.add(arg_name)
        
        # Apply config values for arguments that weren't set via CLI
        for key, value in config.items():
            arg_name = key
            # Convert to the format argparse uses (with underscores)
            if arg_name in cli_args_set:
                # CLI argument was explicitly provided, skip
                continue
            
            if hasattr(args, arg_name):
                # Only set if current value is None or default
                current_value = getattr(args, arg_name)
                
                # For paths, make them absolute relative to config file directory
                if value is not None and isinstance(value, str) and arg_name in [
                    'csv_path', 'train_csv', 'holdout_csv', 'train_csv_map', 
                    'holdout_csv_map', 'synth_csv', 'save_splits_dir', 'results_dir'
                ]:
                    if not os.path.isabs(value):
                        config_dir = os.path.dirname(config_path)
                        value = os.path.join(config_dir, value)
                
                setattr(args, arg_name, value)
        
        print(f"Config loaded and merged with CLI arguments.")
        
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        print("Proceeding with CLI arguments only.")
    
    return args


def _load_and_preprocess(csv_path: str, min_age: Optional[float], max_age: Optional[float], cov2_col: str = 'sex') -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Standardize age and second covariate names
    if 'age' not in df.columns and 'p21003_i2' in df.columns:
        df.rename(columns={'p21003_i2': 'age'}, inplace=True)
    if cov2_col not in df.columns:
        if cov2_col == 'sex' and 'p31' in df.columns:
            df.rename(columns={'p31': 'sex'}, inplace=True)
    if 'age' not in df.columns or cov2_col not in df.columns:
        raise ValueError(f"CSV must contain 'age' and '{cov2_col}' (or accepted aliases).")
    # Apply optional min/max age filters only if provided
    if min_age is not None:
        df = df[df['age'] >= float(min_age)]
    if max_age is not None:
        df = df[df['age'] <= float(max_age)]
    df = df.copy()
    if 'eid' in df.columns:
        df.drop(columns=['eid'], inplace=True)
    return df.dropna().reset_index(drop=True)


def _normalize_train_hold(train_df: pd.DataFrame, hold_df: pd.DataFrame, idp_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, float]]]:
    stats: Dict[str, Dict[str, float]] = {}
    for c in idp_cols + ['age']:
        mu = float(train_df[c].mean())
        sd = float(train_df[c].std())
        if not np.isfinite(sd) or sd <= 1e-12:
            sd = 1.0
        stats[c] = {'mean': mu, 'std': sd}
    tr = train_df.copy(); ho = hold_df.copy()
    for c in idp_cols + ['age']:
        tr[c] = (tr[c] - stats[c]['mean']) / stats[c]['std']
        ho[c] = (ho[c] - stats[c]['mean']) / stats[c]['std']
    return tr, ho, stats


def _stratified_split_by_age_and_cov2(
    df: pd.DataFrame,
    train_ratio: float,
    seed: int,
    age_col: str = 'age',
    cov2_col: str = 'state',
    num_age_bins: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split preserving joint distribution over age bins and a second covariate.

    - Age is binned into approximately equal-frequency bins (qcut) with up to num_age_bins.
    - Strata are formed as (age_bin, cov2) combinations and split proportionally per stratum.
    """
    if age_col not in df.columns or cov2_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{age_col}' and '{cov2_col}' columns for stratified split.")
    df2 = df.copy().reset_index(drop=True)
    df2 = df2.dropna(subset=[age_col, cov2_col])

    # Build age bins: prefer equal-frequency bins; fall back to equal-width if necessary
    n_unique_age = int(pd.Series(df2[age_col]).nunique())
    q = max(1, min(int(num_age_bins), n_unique_age))
    try:
        df2['_age_bin'] = pd.qcut(df2[age_col], q=q, duplicates='drop')
    except Exception:
        a_min = float(df2[age_col].min()); a_max = float(df2[age_col].max())
        if not np.isfinite(a_min) or not np.isfinite(a_max) or a_max <= a_min:
            df2['_age_bin'] = pd.Series(['all'] * len(df2))
        else:
            edges = np.linspace(a_min, a_max, q + 1)
            df2['_age_bin'] = pd.cut(df2[age_col], bins=edges, include_lowest=True)

    df2['_strat'] = df2['_age_bin'].astype(str) + '|' + df2[cov2_col].astype(str)

    rng = np.random.default_rng(int(seed))
    train_idx_parts: List[np.ndarray] = []
    hold_idx_parts: List[np.ndarray] = []

    for _, idxs in df2.groupby('_strat').indices.items():
        idx_arr = np.fromiter(idxs, dtype=int)
        if idx_arr.size == 0:
            continue
        perm = rng.permutation(idx_arr.size)
        idx_arr = idx_arr[perm]
        n = idx_arr.size
        k = int(np.floor(float(train_ratio) * n))
        if n >= 2:
            if k <= 0 and train_ratio > 0.0:
                k = 1
            if k >= n and train_ratio < 1.0:
                k = n - 1
        elif n == 1:
            k = 1 if train_ratio >= 0.5 else 0
        train_idx_parts.append(idx_arr[:k])
        hold_idx_parts.append(idx_arr[k:])

    train_idx = np.concatenate(train_idx_parts) if train_idx_parts else np.array([], dtype=int)
    hold_idx = np.concatenate(hold_idx_parts) if hold_idx_parts else np.array([], dtype=int)

    if train_idx.size:
        train_idx = train_idx[rng.permutation(train_idx.size)]
    if hold_idx.size:
        hold_idx = hold_idx[rng.permutation(hold_idx.size)]

    df_train = df2.iloc[train_idx].drop(columns=['_age_bin', '_strat'], errors='ignore').reset_index(drop=True)
    df_hold = df2.iloc[hold_idx].drop(columns=['_age_bin', '_strat'], errors='ignore').reset_index(drop=True)
    return df_train, df_hold


def _train_and_sample_condition(
    backbone: str,
    args: argparse.Namespace,
    idp_cols: List[str],
    df_train: pd.DataFrame,
    df_hold: pd.DataFrame,
    run_dir: str,
    condition_tag: str,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]], Dict[str, float]]:
    # Per-run deterministic seeding: derive a run-specific seed from global_seed and identifiers
    t_start_total = time.time()
    timings: Dict[str, float] = {}
    run_seed = _derive_seed(args.global_seed, 'run', backbone, condition_tag)
    random.seed(run_seed)
    np.random.seed(run_seed)
    try:
        import torch
        torch.manual_seed(run_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_seed)
        try:
            import torch.backends.cudnn as cudnn
            cudnn.deterministic = True
            cudnn.benchmark = False
        except Exception:
            pass
    except Exception:
        pass
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    df_train_n, df_hold_n, stats = _normalize_train_hold(df_train, df_hold, idp_cols)
    nb_feats = len(idp_cols)
    nb_cond = 2
    # build diffusion schedule
    betas, alphas, alpha_bar = get_diffusion_parameters(args.num_steps, args.beta_start, args.beta_end, device)

    models_root = _ensure_dir(os.path.join(run_dir, 'models'))
    results_root = _ensure_dir(os.path.join(run_dir, 'results'))

    # setup model
    if backbone == 'mlp':
        denoiser_units = [args.denoiser_width] * args.denoiser_depth
        denoiser_kwargs = {
            'nb_units': denoiser_units,
            'bias': True,
            'use_dropout': True,
            'dropout_rate': args.denoiser_dropout,
            'weight_init': 'kaiming_normal_',
            'activation': 'PReLU',
            'cond_method': 'film',
            'nb_experts': 2,
            'batch_norm': True,
            'use_fan': False,
            'dp_ratio': 0.25,
        }
        vae_kwargs = {
            'encoder_hidden_dims': [256, 256],
            'decoder_hidden_dims': [256, 256],
            'activation': 'PReLU',
            'batch_norm': True,
            'use_dropout': True,
            'dropout_rate': 0.1,
            'vae_cond_method': 'film',
            'nb_experts': 2,
            'use_variational': False,
            'encoder_conditioning': True,
            'use_fan': False,
            'dp_ratio': 0.1,
        }
        model = GeneralDiffusionModel(
            data_dim=nb_feats,
            latent_dim=nb_feats,
            cond_dim=nb_cond,
            denoiser_kwargs=denoiser_kwargs,
            vae_kwargs=vae_kwargs,
            diffusion_space='data',
            combine_t_cond=False,
        ).to(device)
    else:
        saint_kwargs = {'d_model': args.d_model, 'depth': args.depth, 'nhead': args.n_heads, 'row_attention_interval': args.row_interval}
        model = build_saint_diffusion_model(nb_feats, nb_cond, saint_kwargs, device)

    # train (skip if requested and checkpoint exists)
    t_train_start = time.time()
    best_epoch = None
    did_train = False
    # Resolve per-backbone batch sizes with precedence: specific -> legacy global -> default
    global_bs = getattr(args, 'batch_size', None)
    mlp_bs = args.mlp_batch_size if args.mlp_batch_size is not None else (global_bs if global_bs is not None else 1024)
    saint_bs = args.saint_batch_size if args.saint_batch_size is not None else (global_bs if global_bs is not None else 512)
    # Resolve per-backbone epochs with precedence: specific -> global
    mlp_epochs = args.mlp_epochs if getattr(args, 'mlp_epochs', None) is not None else args.epochs
    saint_epochs = args.saint_epochs if getattr(args, 'saint_epochs', None) is not None else args.epochs
    if args.skip_if_exists:
        # simple heuristic: look for any best checkpoint matching steps
        existing = [f for f in os.listdir(models_root) if f.startswith('best_model_epoch_') and f.endswith(f"_steps_{args.num_steps}.pt")]
        if existing:
            pass
        else:
            if backbone == 'mlp':
                best_loss, best_epoch = train_diffusion_model(
                    data=df_train_n[idp_cols + ['age', getattr(args, 'cov2_col_active', 'sex')]].values,
                    nb_conditions=nb_cond,
                    model=model,
                    num_steps=args.num_steps,
                    alpha_cumprod=alpha_bar,
                    epochs=mlp_epochs,
                    lr=args.mlp_lr,
                    batch_size=mlp_bs,
                    run_id=args.run_id + '_' + condition_tag,
                    swa=False,
                    swa_start_ratio=0.85,
                    swa_lr=args.mlp_lr,
                    pretrain_epochs=0,
                    device=device,
                    use_variational=False,
                    diffusion_space='data',
                    model_save_dir=models_root,
                )[:2]
                did_train = True
            else:
                _best_loss, best_epoch, _model = train_diffusion_saint_model(
                    data=df_train_n[idp_cols + ['age', getattr(args, 'cov2_col_active', 'sex')]].values.astype(np.float32),
                    nb_conditions=nb_cond,
                    num_steps=args.num_steps,
                    alpha_cumprod=alpha_bar,
                    saint_kwargs={'d_model': args.d_model, 'depth': args.depth, 'nhead': args.n_heads, 'row_attention_interval': args.row_interval},
                    model=model,
                    epochs=saint_epochs,
                    lr=args.saint_lr,
                    batch_size=saint_bs,
                    run_id=args.run_id + '_' + condition_tag,
                    device=device,
                    diffusion_space='data',
                    pretrain_epochs=0,
                    model_save_dir=models_root,
                )
                did_train = True
    else:
        if backbone == 'mlp':
            best_loss, best_epoch = train_diffusion_model(
                data=df_train_n[idp_cols + ['age', getattr(args, 'cov2_col_active', 'sex')]].values,
                nb_conditions=nb_cond,
                model=model,
                num_steps=args.num_steps,
                alpha_cumprod=alpha_bar,
                epochs=mlp_epochs,
                lr=args.mlp_lr,
                batch_size=mlp_bs,
                run_id=args.run_id + '_' + condition_tag,
                swa=False,
                swa_start_ratio=0.85,
                swa_lr=args.mlp_lr,
                pretrain_epochs=0,
                device=device,
                use_variational=False,
                diffusion_space='data',
                model_save_dir=models_root,
            )[:2]
            did_train = True
        else:
            _best_loss, best_epoch, _model = train_diffusion_saint_model(
                data=df_train_n[idp_cols + ['age', getattr(args, 'cov2_col_active', 'sex')]].values.astype(np.float32),
                nb_conditions=nb_cond,
                num_steps=args.num_steps,
                alpha_cumprod=alpha_bar,
                saint_kwargs={'d_model': args.d_model, 'depth': args.depth, 'nhead': args.n_heads, 'row_attention_interval': args.row_interval},
                model=model,
                epochs=saint_epochs,
                lr=args.saint_lr,
                batch_size=saint_bs,
                run_id=args.run_id + '_' + condition_tag,
                device=device,
                diffusion_space='data',
                pretrain_epochs=0,
                model_save_dir=models_root,
            )
            did_train = True
    
    timings['training_seconds'] = time.time() - t_train_start

    # Save final checkpoint if we trained in this call (to ensure availability)
    final_ckpt_path = os.path.join(models_root, f"final_model_steps_{args.num_steps}.pt")
    if did_train:
        try:
            torch.save(model.state_dict(), final_ckpt_path)
        except Exception:
            pass
    else:
        # If we skipped training but only a best exists, ensure a final file is present by cloning best
        if not os.path.exists(final_ckpt_path):
            try:
                bests = sorted([f for f in os.listdir(models_root) if f.startswith('best_model_epoch_') and f.endswith(f"_steps_{args.num_steps}.pt")])
                if bests:
                    best_path = os.path.join(models_root, bests[-1])
                    try:
                        state_dict = torch.load(best_path, map_location='cpu', weights_only=True)
                    except TypeError:
                        state_dict = torch.load(best_path, map_location='cpu')
                    torch.save(state_dict, final_ckpt_path)
            except Exception:
                pass

    # load checkpoint for eval
    ckpt_to_load = None
    # explicit path has highest priority
    if getattr(args, 'checkpoint_path', None):
        if os.path.isfile(args.checkpoint_path):
            ckpt_to_load = args.checkpoint_path
    if ckpt_to_load is None:
        if args.eval_checkpoint == 'best':
            if best_epoch is not None:
                candidate = os.path.join(models_root, f"best_model_epoch_{best_epoch}_steps_{args.num_steps}.pt")
                if os.path.exists(candidate):
                    ckpt_to_load = candidate
            if ckpt_to_load is None:
                bests = sorted([f for f in os.listdir(models_root) if f.startswith('best_model_epoch_') and f.endswith(f"_steps_{args.num_steps}.pt")])
                if bests:
                    ckpt_to_load = os.path.join(models_root, bests[-1])
            if ckpt_to_load is None:
                cand_final = os.path.join(models_root, f"final_model_steps_{args.num_steps}.pt")
                if os.path.exists(cand_final):
                    ckpt_to_load = cand_final
        else:  # final
            cand_final = os.path.join(models_root, f"final_model_steps_{args.num_steps}.pt")
            if os.path.exists(cand_final):
                ckpt_to_load = cand_final
            if ckpt_to_load is None:
                bests = sorted([f for f in os.listdir(models_root) if f.startswith('best_model_epoch_') and f.endswith(f"_steps_{args.num_steps}.pt")])
                if bests:
                    ckpt_to_load = os.path.join(models_root, bests[-1])
    if ckpt_to_load is None:
        raise RuntimeError('No checkpoint found for evaluation.')
    # Informative print about which checkpoint will be used for sampling
    used_msg = "final"
    m = re.search(r"best_model_epoch_(\d+)_steps_", os.path.basename(ckpt_to_load))
    if m is not None:
        used_msg = f"best (epoch {int(m.group(1))})"
    print(f"Using checkpoint for sampling: {used_msg} -> {ckpt_to_load}")
    try:
        state_dict = torch.load(ckpt_to_load, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(ckpt_to_load, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # sampling
    t_sampling_start = time.time()
    print("Sampling …")
    samps = []
    for _ in range(args.times_to_sample):
        rand_age = np.random.uniform(df_train['age'].min(), df_train['age'].max(), args.num_samples)
        rand_age_norm = (rand_age - stats['age']['mean']) / stats['age']['std']
        # derive cov2 column from df_train
        cov2_col = 'sex'
        if hasattr(args, 'cov2_col_active') and args.cov2_col_active in df_train.columns:
            cov2_col = args.cov2_col_active
        elif hasattr(args, 'ukb_cov2_col') and args.ukb_cov2_col in df_train.columns:
            cov2_col = args.ukb_cov2_col
        cov_vals = df_train[cov2_col].dropna().values
        if cov_vals.size == 0:
            cov2_sample = np.zeros(args.num_samples, dtype=np.float32)
        else:
            uniq, cnt = np.unique(cov_vals, return_counts=True)
            probs = cnt / cnt.sum()
            cov2_sample = np.random.choice(uniq, size=args.num_samples, p=probs).astype(np.float32)
        cond_arr = np.stack([rand_age_norm, cov2_sample], 1).astype(np.float32)
        if backbone == 'mlp':
            samples = sample_mlp_full_ddpm(
                model=model,
                num_samples=args.num_samples,
                num_steps=args.num_steps,
                cond_values=cond_arr,
                latent_dim=nb_feats,
                betas=betas,
                alphas=alphas,
                alpha_cumprod=alpha_bar,
                device=device,
                diffusion_space='data',
            ).cpu().numpy()
        else:
            samples = sample_saint_full_ddpm(
                model=model,
                num_samples=args.num_samples,
                num_steps=args.num_steps,
                betas=betas,
                alphas=alphas,
                alpha_cumprod=alpha_bar,
                cond_values=cond_arr,
                device=device,
                idp_dim=nb_feats,
            ).cpu().numpy()
        samps.append(samples)
    samples_all = np.concatenate(samps, axis=0)
    timings['sampling_seconds'] = time.time() - t_sampling_start
    
    cov2_col_used = 'sex'
    if hasattr(args, 'cov2_col_active') and args.cov2_col_active:
        cov2_col_used = args.cov2_col_active
    samp_df = pd.DataFrame(samples_all, columns=idp_cols + ['age', cov2_col_used])
    for c in idp_cols + ['age']:
        samp_df[c] = samp_df[c] * stats[c]['std'] + stats[c]['mean']

    timings['total_train_and_sample_seconds'] = time.time() - t_start_total
    
    meta = {
        'idp_cols': idp_cols,
        'stats': stats,
        'checkpoint': ckpt_to_load,
        'condition_tag': condition_tag,
        'cov2_col': cov2_col_used,
    }
    _save_json(meta, os.path.join(results_root, 'meta.json'))
    return samp_df, stats, timings


def _run_gamlss_baseline(
    args: argparse.Namespace,
    suite_root: str,
    train_csv: str,
    holdout_csv: str,
    idp_size_tag: str,
    suite_tag: str = 'gamlss_baseline',
    cov2_col: str = 'sex',
    dataset_tag: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Run GAMLSS univariate baseline: fit one model per IDP, compute ACE/ECP/PIT/KS.
    
    Args:
        args: CLI arguments
        suite_root: root directory for all results
        train_csv: path to training CSV
        holdout_csv: path to holdout CSV
        idp_size_tag: IDP size tag for directory structure
        suite_tag: suite tag for directory structure
        cov2_col: second covariate column name (e.g., 'sex', 'state')
        dataset_tag: optional dataset tag for nested structure
        
    Returns:
        Tuple of (scalar metrics dict, timings dict)
    """
    t_start_total = time.time()
    timings: Dict[str, float] = {}
    
    try:
        from gamlss_adapter import GAMLSSBaseline
    except ImportError:
        raise ImportError("gamlss_adapter not found. Ensure rpy2 and R gamlss packages are installed.")
    
    # Build output directory
    if dataset_tag is not None and str(dataset_tag).strip() != '':
        run_root = _ensure_dir(os.path.join(
            suite_root,
            dataset_tag, 'gamlss', suite_tag, f"D{idp_size_tag}"
        ))
    else:
        run_root = _ensure_dir(os.path.join(
            suite_root,
            'gamlss', suite_tag, f"D{idp_size_tag}"
        ))
    
    results_root = _ensure_dir(os.path.join(run_root, 'results'))
    
    # Load data
    df_train = _load_and_preprocess(train_csv, args.min_age if hasattr(args, 'min_age') else None, 
                                     args.max_age if hasattr(args, 'max_age') else None, cov2_col)
    df_hold = _load_and_preprocess(holdout_csv, args.min_age if hasattr(args, 'min_age') else None,
                                    args.max_age if hasattr(args, 'max_age') else None, cov2_col)
    
    idp_cols = [c for c in df_train.columns if c not in ['age', cov2_col]]
    covariates = ['age', cov2_col]
    
    print(f"GAMLSS baseline: fitting {len(idp_cols)} IDPs...")
    
    # Replace "cov2" placeholder with actual covariate name (sex for UKB, state for synth)
    mu_formula = args.gamlss_mu_formula.replace('cov2', cov2_col)
    sigma_formula = args.gamlss_sigma_formula.replace('cov2', cov2_col) if args.gamlss_sigma_formula else None
    nu_formula = args.gamlss_nu_formula.replace('cov2', cov2_col) if args.gamlss_nu_formula else None
    tau_formula = args.gamlss_tau_formula.replace('cov2', cov2_col) if args.gamlss_tau_formula else None
    
    # Initialize GAMLSS baseline
    gamlss = GAMLSSBaseline(
        family=args.gamlss_family,
        mu_formula=mu_formula,
        sigma_formula=sigma_formula,
        nu_formula=nu_formula,
        tau_formula=tau_formula,
        verbose=bool(getattr(args, 'gamlss_trace_fit', False)),
        n_cyc=int(getattr(args, 'gamlss_n_cyc', 200)),
        c_crit=float(getattr(args, 'gamlss_c_crit', 0.0001))
    )
    
    # Fit all IDPs
    t_fit_start = time.time()
    gamlss.fit(df_train, idp_cols, covariates)
    timings['fitting_seconds'] = time.time() - t_fit_start
    
    # Build evaluation grid (age bins × cov2 levels)
    age_min, age_max = df_train['age'].min(), df_train['age'].max()
    age_bins = np.arange(math.floor(age_min) - 0.5, math.ceil(age_max) + 0.5, 1.0)
    age_bin_centers = (age_bins[:-1] + age_bins[1:]) / 2.0
    cov2_levels = sorted(df_train[cov2_col].dropna().unique())
    
    # Create evaluation grid
    grid_rows = []
    for age_c in age_bin_centers:
        for cov2 in cov2_levels:
            grid_rows.append({'age': age_c, cov2_col: cov2})
    eval_grid_df = pd.DataFrame(grid_rows)
    
    # Get quantiles for ACE/ECP
    ace_ps = [float(x) for x in str(args.ace_percentiles).split(',') if x]
    Q = gamlss.quantiles(eval_grid_df, probs=tuple(ace_ps))
    
    # Sample for KS tests
    print("GAMLSS baseline: sampling for KS tests...")
    t_sample_start = time.time()
    S = gamlss.sample(eval_grid_df, M=args.gamlss_samples_per_grid)
    timings['sampling_seconds'] = time.time() - t_sample_start
    
    # Construct synthetic samples dataframe for metric computation
    # For each grid point, take samples and assign age/cov2
    samp_rows = []
    for idx, row in eval_grid_df.iterrows():
        for idp in idp_cols:
            samples_idp = S[idp][idx]
            for s in samples_idp:
                samp_rows.append({
                    'age': row['age'],
                    cov2_col: row[cov2_col],
                    idp: s
                })
    
    # Pivot to wide format
    samp_df_list = []
    grid_idx = 0
    for idx, row in eval_grid_df.iterrows():
        chunk_size = args.gamlss_samples_per_grid
        start = grid_idx * chunk_size
        end = (grid_idx + 1) * chunk_size
        chunk_df = pd.DataFrame(samp_rows[start:end])
        samp_df_list.append(chunk_df)
        grid_idx += 1
    
    # Simpler approach: create samples per IDP and concatenate
    samp_dict = {'age': [], cov2_col: []}
    for idp in idp_cols:
        samp_dict[idp] = []
    
    for idx, row in eval_grid_df.iterrows():
        for idp in idp_cols:
            samples_idp = S[idp][idx]
            samp_dict[idp].extend(samples_idp)
        samp_dict['age'].extend([row['age']] * args.gamlss_samples_per_grid)
        samp_dict[cov2_col].extend([row[cov2_col]] * args.gamlss_samples_per_grid)
    
    samp_df = pd.DataFrame(samp_dict)
    
    print("GAMLSS baseline: computing metrics...")
    t_eval_start = time.time()
    
    # Compute metrics (reuse existing functions)
    real_unscaled = df_hold.copy()
    ace = compute_ace(real_unscaled, samp_df, idp_cols, percentiles=ace_ps, 
                      lowess_frac=args.ace_lowess_frac, use_smoothing=args.ace_use_smoothing)
    ecp = compute_ecp(real_unscaled, samp_df, idp_cols)
    ksbin = ks_per_bin(real_unscaled, samp_df, idp_cols)
    ks_p_summary, ks_perbin_df = ks_per_bin_with_pvalues(
        real_unscaled, samp_df, idp_cols,
        min_per_bin=10, fdr_alpha=0.05,
        use_permutation=bool(getattr(args, 'ks_perm_per_bin', False)),
        B=int(getattr(args, 'ks_perm_B', 1000))
    )
    zmet = zscore_eval(real_unscaled, samp_df, idp_cols, label_col=args.label_col, positive_class=args.positive_class)
    timings['evaluation_metrics_seconds'] = time.time() - t_eval_start
    
    # KS raw p-value summaries
    ks_raw_per_idp: Dict[str, Dict[str, float]] = {}
    run_ks_raw_summary: Dict[str, float] = {}
    try:
        if ('p_value' in ks_perbin_df.columns) and (ks_perbin_df.shape[0] > 0):
            pv_all = ks_perbin_df['p_value'].values
            mask_all = np.isfinite(pv_all)
            total_all = int(np.sum(mask_all))
            cnt_005_all = int(np.sum(pv_all[mask_all] < 0.05)) if total_all > 0 else 0
            cnt_01_all = int(np.sum(pv_all[mask_all] < 0.10)) if total_all > 0 else 0
            prop_005_all = float(cnt_005_all / total_all) if total_all > 0 else float('nan')
            prop_01_all = float(cnt_01_all / total_all) if total_all > 0 else float('nan')
            run_ks_raw_summary = {
                'ks_p_raw_total': float(total_all),
                'ks_p_raw_count_lt_0p05': float(cnt_005_all),
                'ks_p_raw_count_lt_0p1': float(cnt_01_all),
                'ks_p_raw_proportion_lt_0p05': prop_005_all,
                'ks_p_raw_proportion_lt_0p1': prop_01_all,
            }
            for col, grp in ks_perbin_df.groupby('idp', sort=False):
                pv = grp['p_value'].values
                mask = np.isfinite(pv)
                total = int(np.sum(mask))
                cnt_005 = int(np.sum(pv[mask] < 0.05)) if total > 0 else 0
                cnt_01 = int(np.sum(pv[mask] < 0.10)) if total > 0 else 0
                prop_005 = float(cnt_005 / total) if total > 0 else float('nan')
                prop_01 = float(cnt_01 / total) if total > 0 else float('nan')
                ks_raw_per_idp[col] = {
                    'ks_p_raw_total': float(total),
                    'ks_p_raw_count_lt_0p05': float(cnt_005),
                    'ks_p_raw_count_lt_0p1': float(cnt_01),
                    'ks_p_raw_proportion_lt_0p05': prop_005,
                    'ks_p_raw_proportion_lt_0p1': prop_01,
                }
    except Exception:
        pass
    
    merged = {}
    for c in idp_cols:
        merged[c] = {
            **ace.get(c, {}),
            **ecp.get(c, {}),
            **ksbin.get(c, {}),
            **ks_p_summary.get(c, {}),
            **zmet.get(c, {}),
            **ks_raw_per_idp.get(c, {}),
        }
    if run_ks_raw_summary:
        merged['_run_ks_p_raw_summary'] = run_ks_raw_summary
    
    _save_json(merged, os.path.join(results_root, 'evaluation_summary.json'))
    try:
        ks_perbin_df.to_csv(os.path.join(results_root, 'ks_per_bin.csv'), index=False)
    except Exception:
        pass
    
    # PIT histograms
    try:
        if args.eval_pit:
            print("GAMLSS baseline: computing PIT...")
            pit_dir = os.path.join(results_root, 'pit')
            U = gamlss.pit(df_hold, covariates)
            os.makedirs(pit_dir, exist_ok=True)
            
            # Collect all PIT values for pooled histogram
            all_pit_vals = []
            
            for col in idp_cols:
                u_vals = U.get(col, np.array([]))
                if u_vals.size > 0:
                    all_pit_vals.append(u_vals)
                    # Per-IDP histogram
                    plt.figure(figsize=(8, 4))
                    plt.hist(u_vals, bins=20, density=True, alpha=0.6)
                    plt.hlines(1.0, 0.0, 1.0, linewidth=2)
                    plt.title(f'PIT histogram (GAMLSS): {col}', fontsize=13)
                    plt.xlabel('u', fontsize=12)
                    plt.ylabel('density', fontsize=12)
                    plt.tight_layout()
                    plt.savefig(os.path.join(pit_dir, f'pit_{col}.png'), dpi=args.save_plots_dpi)
                    plt.close()
            
            # Pooled PIT histogram across all IDPs
            if all_pit_vals:
                u_all_pooled = np.concatenate(all_pit_vals)
                if u_all_pooled.size >= 5:
                    plt.figure(figsize=(8, 4))
                    plt.hist(u_all_pooled, bins=20, density=True, alpha=0.6)
                    plt.hlines(1.0, 0.0, 1.0, linewidth=2)
                    plt.title('PIT histogram (pooled across all IDPs, GAMLSS)', fontsize=13)
                    plt.xlabel('u', fontsize=12)
                    plt.ylabel('density', fontsize=12)
                    ax = plt.gca()
                    ax.tick_params(axis='both', which='major', labelsize=11)
                    plt.tight_layout()
                    plt.savefig(os.path.join(pit_dir, 'pit_pooled.png'), dpi=args.save_plots_dpi)
                    plt.close()
                    
                    # QQ plot for pooled PIT
                    u_sorted_all = np.sort(u_all_pooled)
                    q_unif_all = np.linspace(0, 1, u_sorted_all.size, endpoint=False) + 0.5 / u_sorted_all.size
                    plt.figure(figsize=(4.5, 4.5))
                    plt.plot(q_unif_all, u_sorted_all, '.', ms=2)
                    plt.plot([0, 1], [0, 1], '--')
                    plt.title('PIT QQ vs Uniform (pooled, GAMLSS)', fontsize=13)
                    plt.xlabel('Uniform quantile', fontsize=12)
                    plt.ylabel('Empirical PIT quantile', fontsize=12)
                    ax = plt.gca()
                    ax.tick_params(axis='both', which='major', labelsize=11)
                    plt.tight_layout()
                    plt.savefig(os.path.join(pit_dir, 'pit_qq_pooled.png'), dpi=args.save_plots_dpi)
                    plt.close()
    except Exception:
        pass
    
    # Coverage curves
    try:
        if args.eval_coverage_curve:
            print("GAMLSS baseline: computing coverage curves...")
            cov_dir = os.path.join(results_root, 'coverage_curves')
            coverage_vs_nominal(real_unscaled, samp_df, idp_cols, cov_dir, nominals=None, dpi=args.save_plots_dpi)
            covd_dir = os.path.join(results_root, 'coverage_curves_diff')
            coverage_vs_nominal_diff(real_unscaled, samp_df, idp_cols, covd_dir, nominals=None, dpi=args.save_plots_dpi)
    except Exception:
        pass
    
    # Centile plots
    # Note: Curves are marginalized over cov2 (sex for UKB, state for synth), similar to diffusion models.
    # The groupby('age_bin').quantile() automatically pools across all cov2 values within each age bin,
    # producing centile curves that show variation with age only.
    try:
        print("GAMLSS baseline: creating centile plots...")
        real_df = real_unscaled.copy()
        gen_df = samp_df.copy()
        bin_w = 1
        real_df['age_bin'] = pd.cut(real_df['age'], bins=np.arange(real_df['age'].min()-0.5, real_df['age'].max()+0.5, bin_w))
        gen_df['age_bin'] = pd.cut(gen_df['age'], bins=np.arange(gen_df['age'].min()-0.5, gen_df['age'].max()+0.5, bin_w))
        percs = ace_ps
        colours_r = ['blue'] * len(percs)
        colours_g = ['red'] * len(percs)
        linestyles = [':', '--', '-', '--', ':'][:len(percs)]
        n_cols = 5
        rows_per_page = 10
        max_per_page = n_cols * rows_per_page
        total_pages = int(math.ceil(len(idp_cols)/float(max_per_page)))
        title_fs = 12
        tick_fs = 10
        for page in range(total_pages):
            start = page*max_per_page
            end = min(len(idp_cols), (page+1)*max_per_page)
            subset = idp_cols[start:end]
            n_rows = int(math.ceil(len(subset)/float(n_cols)))
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*4.6, n_rows*3.5), sharex=True)
            axs = axs.flatten()
            for idx, col in enumerate(subset):
                ax = axs[idx]
                for p, c_r, c_g, ls in zip(percs, colours_r, colours_g, linestyles):
                    vr = real_df.groupby('age_bin')[col].quantile(p).dropna()
                    vg = gen_df.groupby('age_bin')[col].quantile(p).dropna()
                    if len(vr)==0 or len(vg)==0:
                        continue
                    sm_r = lowess(vr.values, vr.index.map(lambda b: b.mid), frac=args.ace_lowess_frac)
                    sm_g = lowess(vg.values, vg.index.map(lambda b: b.mid), frac=args.ace_lowess_frac)
                    ax.plot(vr.index.map(lambda b: b.mid), sm_r[:,1], color=c_r, linestyle=ls)
                    ax.plot(vg.index.map(lambda b: b.mid), sm_g[:,1], color=c_g, linestyle=ls)
                try:
                    use_ukb_short = (str(idp_size_tag) == '20' and dataset_tag == 'ukb')
                except Exception:
                    use_ukb_short = False
                if use_ukb_short:
                    label_text = _ukb_short_label_from_col(col)
                else:
                    label_text = col
                ax.set_title(label_text, fontsize=title_fs)
                ax.tick_params(axis='both', labelsize=tick_fs)
            for j in range(idx+1, len(axs)):
                axs[j].axis('off')
            fig.tight_layout(rect=[0,0,1,0.96])
            out_png = os.path.join(results_root, f'centile_curves_page_{page+1}.png')
            fig.savefig(out_png, dpi=args.save_plots_dpi)
            plt.close(fig)
    except Exception:
        pass
    
    # Aggregate scalars
    ace_scalar = _aggregate_metric_for_learning(ace, prefer_keys=[f"ace_{int(100*x)}" for x in ace_ps])
    ks_scalar = _aggregate_metric_for_learning(ksbin, prefer_keys=['ks_mean'])
    ecp_err = {}
    for k, v in ecp.items():
        if 'ecp' in v and 'nominal' in v:
            ecp_err[k] = {'ecp_abs_error': abs(float(v['ecp']) - float(v['nominal']))}
    ecp_scalar = _aggregate_metric_for_learning(ecp_err, prefer_keys=['ecp_abs_error'])
    scalars = {'ace': ace_scalar, 'ks_mean': ks_scalar, 'ecp_abs_error': ecp_scalar}
    _save_json(scalars, os.path.join(results_root, 'scalar_metrics.json'))
    
    timings['total_gamlss_seconds'] = time.time() - t_start_total
    _save_json(timings, os.path.join(results_root, 'runtimes.json'))
    
    # Save metadata
    meta = {
        'method': 'GAMLSS',
        'family': args.gamlss_family,
        'mu_formula': args.gamlss_mu_formula,
        'sigma_formula': args.gamlss_sigma_formula,
        'nu_formula': args.gamlss_nu_formula,
        'tau_formula': args.gamlss_tau_formula,
        'idp_cols': idp_cols,
        'covariates': covariates,
    }
    _save_json(meta, os.path.join(results_root, 'meta.json'))
    
    print(f"GAMLSS baseline complete. Results saved to {results_root}")
    return scalars, timings


def main():
    t_main_start = time.time()
    all_run_timings: Dict[str, Any] = {}
    
    args = parse_args()
    # Load config file and merge with CLI args (CLI takes precedence)
    args = load_config_and_merge_args(args)
    
    # parse lists
    ace_ps = [float(x) for x in str(args.ace_percentiles).split(',') if x]
    fracs = [float(x) for x in str(args.train_fracs).split(',') if x]
    idp_sizes = [int(x) for x in str(args.idp_sizes).split(',') if x]
    perm_effective = 0 if args.disable_permutation_tests else args.joint_num_permutations

    if not getattr(args, 'run_group', None):
        raise ValueError("Missing required `run_group`. Provide `--run_group ...` or set `run_group:` in the YAML config.")

    # Prepare results root
    suite_root = _ensure_dir(os.path.abspath(os.path.join(str(args.results_dir), 'results_full_eval', args.run_group)))

    # Persist full CLI configuration (including defaults) next to summary.json
    try:
        args_dict = vars(args).copy()
        args_dict['_argv'] = sys.argv
        with open(os.path.join(suite_root, 'cli_args.json'), 'w') as f:
            json.dump(args_dict, f, indent=2)
    except Exception:
        pass

    # Helper to execute one configuration and run evaluations
    def run_eval(backbone: str, idp_size_tag: str, train_csv: str, holdout_csv: str, frac: float, seed_idx: int, heavy_analyses: bool = True, suite_tag: str = 'base_full', cov2_col: str = 'sex', dataset_tag: Optional[str] = None) -> Dict[str, Any]:
        t_run_eval_start = time.time()
        eval_timings: Dict[str, float] = {}
        
        # Build output directory; if dataset_tag provided, nest under it to avoid collisions
        if dataset_tag is not None and str(dataset_tag).strip() != '':
            run_root = _ensure_dir(os.path.join(suite_root, dataset_tag, backbone, suite_tag, f"D{idp_size_tag}", f"frac_{frac:.2f}", f"seed_{seed_idx}"))
        else:
            run_root = _ensure_dir(os.path.join(suite_root, backbone, suite_tag, f"D{idp_size_tag}", f"frac_{frac:.2f}", f"seed_{seed_idx}"))
        # Load data
        df_tr = _load_and_preprocess(train_csv, args.min_age, args.max_age, cov2_col)
        df_ho = _load_and_preprocess(holdout_csv, args.min_age, args.max_age, cov2_col)
        # Subsample fraction
        if frac < 1.0:
            rng = np.random.default_rng(_derive_seed(args.global_seed, 'subsample', backbone, idp_size_tag, frac, seed_idx))
            n_keep = max(10, int(len(df_tr) * frac))
            idx = rng.choice(len(df_tr), size=n_keep, replace=False)
            df_tr = df_tr.iloc[idx].reset_index(drop=True)
        # ensure downstream uses correct covariate
        args.cov2_col_active = cov2_col
        idp_cols = [c for c in df_tr.columns if c not in ['age', cov2_col]]

        cond_tag = f"D{idp_size_tag}_f{frac:.2f}_s{seed_idx}"
        samp_df, _stats, train_sample_timings = _train_and_sample_condition(backbone, args, idp_cols, df_tr, df_ho, run_root, cond_tag)
        eval_timings.update(train_sample_timings)

        # Metrics
        t_metrics_start = time.time()
        real_unscaled = df_ho.copy()
        ace = compute_ace(real_unscaled, samp_df, idp_cols, percentiles=ace_ps, lowess_frac=args.ace_lowess_frac, use_smoothing=args.ace_use_smoothing)
        ecp = compute_ecp(real_unscaled, samp_df, idp_cols)
        # KS with p-values and FDR, plus legacy KS stats
        ksbin = ks_per_bin(real_unscaled, samp_df, idp_cols)
        ks_p_summary, ks_perbin_df = ks_per_bin_with_pvalues(
            real_unscaled, samp_df, idp_cols,
            min_per_bin=10, fdr_alpha=0.05,
            use_permutation=bool(getattr(args, 'ks_perm_per_bin', False)),
            B=int(getattr(args, 'ks_perm_B', 1000))
        )
        zmet = zscore_eval(real_unscaled, samp_df, idp_cols, label_col=args.label_col, positive_class=args.positive_class)
        eval_timings['core_metrics_seconds'] = time.time() - t_metrics_start

        # KS raw p-value summaries (per-IDP and run-level) from ks_perbin_df['p_value']
        ks_raw_per_idp: Dict[str, Dict[str, float]] = {}
        run_ks_raw_summary: Dict[str, float] = {}
        try:
            if ('p_value' in ks_perbin_df.columns) and (ks_perbin_df.shape[0] > 0):
                # Run-level
                pv_all = ks_perbin_df['p_value'].values
                mask_all = np.isfinite(pv_all)
                total_all = int(np.sum(mask_all))
                cnt_005_all = int(np.sum(pv_all[mask_all] < 0.05)) if total_all > 0 else 0
                cnt_01_all = int(np.sum(pv_all[mask_all] < 0.10)) if total_all > 0 else 0
                prop_005_all = float(cnt_005_all / total_all) if total_all > 0 else float('nan')
                prop_01_all = float(cnt_01_all / total_all) if total_all > 0 else float('nan')
                run_ks_raw_summary = {
                    'ks_p_raw_total': float(total_all),
                    'ks_p_raw_count_lt_0p05': float(cnt_005_all),
                    'ks_p_raw_count_lt_0p1': float(cnt_01_all),
                    'ks_p_raw_proportion_lt_0p05': prop_005_all,
                    'ks_p_raw_proportion_lt_0p1': prop_01_all,
                }
                # Per-IDP
                for col, grp in ks_perbin_df.groupby('idp', sort=False):
                    pv = grp['p_value'].values
                    mask = np.isfinite(pv)
                    total = int(np.sum(mask))
                    cnt_005 = int(np.sum(pv[mask] < 0.05)) if total > 0 else 0
                    cnt_01 = int(np.sum(pv[mask] < 0.10)) if total > 0 else 0
                    prop_005 = float(cnt_005 / total) if total > 0 else float('nan')
                    prop_01 = float(cnt_01 / total) if total > 0 else float('nan')
                    ks_raw_per_idp[col] = {
                        'ks_p_raw_total': float(total),
                        'ks_p_raw_count_lt_0p05': float(cnt_005),
                        'ks_p_raw_count_lt_0p1': float(cnt_01),
                        'ks_p_raw_proportion_lt_0p05': prop_005,
                        'ks_p_raw_proportion_lt_0p1': prop_01,
                    }
        except Exception:
            ks_raw_per_idp = {}
            run_ks_raw_summary = {}

        merged = {}
        for c in idp_cols:
            merged[c] = {
                **ace.get(c, {}),
                **ecp.get(c, {}),
                **ksbin.get(c, {}),
                **ks_p_summary.get(c, {}),
                **zmet.get(c, {}),
                **ks_raw_per_idp.get(c, {}),
            }
        results_root = os.path.join(run_root, 'results')
        # Attach run-level KS raw p-value summary at the top level
        if run_ks_raw_summary:
            merged['_run_ks_p_raw_summary'] = run_ks_raw_summary
        _save_json(merged, os.path.join(results_root, 'evaluation_summary.json'))
        try:
            ks_perbin_df.to_csv(os.path.join(results_root, 'ks_per_bin.csv'), index=False)
        except Exception:
            pass

        # Additional diagnostics
        try:
            if args.eval_pit:
                t_pit_start = time.time()
                pit_dir = os.path.join(results_root, 'pit')
                pit_histograms(real_unscaled, samp_df, idp_cols, pit_dir, bins=20, dpi=args.save_plots_dpi)
                eval_timings['pit_seconds'] = time.time() - t_pit_start
        except Exception:
            pass
        try:
            if args.eval_coverage_curve:
                t_cov_start = time.time()
                cov_dir = os.path.join(results_root, 'coverage_curves')
                coverage_vs_nominal(real_unscaled, samp_df, idp_cols, cov_dir, nominals=None, dpi=args.save_plots_dpi)
                covd_dir = os.path.join(results_root, 'coverage_curves_diff')
                coverage_vs_nominal_diff(real_unscaled, samp_df, idp_cols, covd_dir, nominals=None, dpi=args.save_plots_dpi)
                eval_timings['coverage_curves_seconds'] = time.time() - t_cov_start
        except Exception:
            pass
        try:
            if heavy_analyses and args.eval_nn_mem:
                t_nn_start = time.time()
                nn_dir = os.path.join(results_root, 'nn_memorisation')
                nn_memorisation_checks(samp_df, df_tr, df_ho, idp_cols, nn_dir, standardize=True, dpi=args.save_plots_dpi, cov2_col=cov2_col, num_age_bins=10)
                eval_timings['nn_memorisation_seconds'] = time.time() - t_nn_start
        except Exception:
            pass

        # Centile plots using existing style (LOWESS for both)
        try:
            # build per-IDP plot pages (reuse same paginate style as existing centile plot)
            # Here we simply save one page per 50 IDPs
            from copy import deepcopy
            real_df = real_unscaled.copy(); gen_df = samp_df.copy()
            bin_w = 1
            real_df['age_bin'] = pd.cut(real_df['age'], bins=np.arange(real_df['age'].min()-0.5, real_df['age'].max()+0.5, bin_w))
            gen_df['age_bin'] = pd.cut(gen_df['age'], bins=np.arange(gen_df['age'].min()-0.5, gen_df['age'].max()+0.5, bin_w))
            percs = ace_ps
            colours_r = ['blue'] * len(percs)
            colours_g = ['red'] * len(percs)
            linestyles = [':', '--', '-', '--', ':'][:len(percs)]
            n_cols = 5; rows_per_page = 10; max_per_page = n_cols * rows_per_page
            total_pages = int(math.ceil(len(idp_cols)/float(max_per_page)))
            # Larger fonts for headings and ticks; increase figure size slightly
            title_fs = 12
            tick_fs = 10
            for page in range(total_pages):
                start = page*max_per_page
                end = min(len(idp_cols), (page+1)*max_per_page)
                subset = idp_cols[start:end]
                n_rows = int(math.ceil(len(subset)/float(n_cols)))
                fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*4.6, n_rows*3.5), sharex=True)
                axs = axs.flatten()
                for idx, col in enumerate(subset):
                    ax = axs[idx]
                    for p, c_r, c_g, ls in zip(percs, colours_r, colours_g, linestyles):
                        vr = real_df.groupby('age_bin')[col].quantile(p).dropna()
                        vg = gen_df.groupby('age_bin')[col].quantile(p).dropna()
                        if len(vr)==0 or len(vg)==0:
                            continue
                        sm_r = lowess(vr.values, vr.index.map(lambda b: b.mid), frac=args.ace_lowess_frac)
                        sm_g = lowess(vg.values, vg.index.map(lambda b: b.mid), frac=args.ace_lowess_frac)
                        ax.plot(vr.index.map(lambda b: b.mid), sm_r[:,1], color=c_r, linestyle=ls)
                        ax.plot(vg.index.map(lambda b: b.mid), sm_g[:,1], color=c_g, linestyle=ls)
                    # Apply short labels for UKB 20-IDP base models only
                    # Detect via directory context: this block runs inside run_eval; base_full UKB D20 has suite_tag 'base_full', idp_size_tag '20'
                    try:
                        use_ukb_short = (str(idp_size_tag) == '20' and dataset_tag == 'ukb')
                    except Exception:
                        use_ukb_short = False
                    if use_ukb_short:
                        label_text = _ukb_short_label_from_col(col)
                    else:
                        label_text = col
                    ax.set_title(label_text, fontsize=title_fs)
                    ax.tick_params(axis='both', labelsize=tick_fs)
                for j in range(idx+1, len(axs)):
                    axs[j].axis('off')
                fig.tight_layout(rect=[0,0,1,0.96])
                out_png = os.path.join(results_root, f'centile_curves_page_{page+1}.png')
                fig.savefig(out_png, dpi=args.save_plots_dpi)
                plt.close(fig)
        except Exception:
            pass

        # Joint analyses (only when heavy analyses are enabled)
        if heavy_analyses and args.joint_analysis:
            t_joint_start = time.time()
            joint_dir = os.path.join(results_root, 'joint_pairs')
            make_joint_heatmap_pages(
                real_df=real_unscaled,
                gen_df=samp_df,
                idp_cols=idp_cols,
                output_dir=joint_dir,
                rows_per_page=args.joint_rows_per_page,
                bins=args.joint_bins,
                standardize_axes=args.joint_standardize_axes,
                log_scale=args.joint_log_scale,
                max_samples_per_group=args.joint_max_samples_per_group,
                num_permutations=perm_effective,
                random_state=args.joint_random_state,
                n_jobs=args.joint_n_jobs,
                color_quantile=args.joint_color_quantile,
                dpi=args.save_plots_dpi,
            )
            eval_timings['joint_analysis_seconds'] = time.time() - t_joint_start

        if heavy_analyses and args.marginal_product_analysis:
            t_marginal_start = time.time()
            mp_dir = os.path.join(results_root, 'joint_vs_marginals')
            # set baseline repeats from CLI
            make_joint_vs_marginal_product_pages._baseline_repeats = int(max(1, getattr(args, 'baseline_resample_repeats', 1)))
            make_joint_vs_marginal_product_pages(
                real_df=real_unscaled,
                gen_df=samp_df,
                idp_cols=idp_cols,
                output_dir=mp_dir,
                rows_per_page=args.joint_rows_per_page,
                bins=args.joint_bins,
                standardize_axes=args.joint_standardize_axes,
                log_scale=args.joint_log_scale,
                max_samples_per_group=args.joint_max_samples_per_group,
                num_permutations=perm_effective,
                random_state=args.joint_random_state,
                n_jobs=args.joint_n_jobs,
                color_quantile=args.joint_color_quantile,
                dpi=args.save_plots_dpi,
                produce_ranked_panels=(dataset_tag == 'ukb' and str(idp_size_tag) == '20'),
                selected_k=2,
                rank_by='mmd2_stat_gen',
                stat_overlay_kde=bool(getattr(args, 'stat_overlay_kde', False)),
            )
            eval_timings['marginal_product_analysis_seconds'] = time.time() - t_marginal_start

        # Pair-of-pair shape correlation limited to <= pair_idp_limit
        if heavy_analyses and args.pair_density_shape_corr and len(idp_cols) <= int(args.pair_idp_limit):
            t_pair_start = time.time()
            pair_dir = os.path.join(results_root, 'pair_of_pair')
            os.makedirs(pair_dir, exist_ok=True)
            prs_R, Cdens_R = build_pair_density_corr_matrix(
                df=real_unscaled,
                idp_cols=idp_cols,
                bins=args.pair_bins,
                standardize_axes=args.pair_standardize_axes,
                max_samples_per_group=args.pair_max_samples_per_pair,
                random_state=args.pair_random_state,
                n_jobs=args.pair_n_jobs,
            )
            prs_G, Cdens_G = build_pair_density_corr_matrix(
                df=samp_df,
                idp_cols=idp_cols,
                bins=args.pair_bins,
                standardize_axes=args.pair_standardize_axes,
                max_samples_per_group=args.pair_max_samples_per_pair,
                random_state=args.pair_random_state,
                n_jobs=args.pair_n_jobs,
            )
            # align by intersection
            setR = {p: i for i, p in enumerate(prs_R)}
            common = [p for p in prs_R if p in {q for q in prs_G}]
            idxR = [setR[p] for p in common]
            setG = {p: i for i, p in enumerate(prs_G)}
            idxG = [setG[p] for p in common]
            Cdens_Ra = Cdens_R[np.ix_(idxR, idxR)].astype(np.float32)
            Cdens_Ga = Cdens_G[np.ix_(idxG, idxG)].astype(np.float32)
            order_density = _reorder_by_clustering(Cdens_Ra)
            plot_pair_matrix(Cdens_Ra, common, os.path.join(pair_dir, 'pair_density_corr_real.png'),
                             title='Pairwise joint density similarity (Real)', cmap='magma', vmin=-1.0, vmax=1.0, order=order_density, origin='upper', dpi=args.save_plots_dpi, show_axis_labels=False, title_fontsize=13, colorbar_tick_fontsize=11)
            plot_pair_matrix(Cdens_Ga, common, os.path.join(pair_dir, 'pair_density_corr_gen.png'),
                             title='Pairwise joint density similarity (Generated)', cmap='magma', vmin=-1.0, vmax=1.0, order=order_density, origin='upper', dpi=args.save_plots_dpi, show_axis_labels=False, title_fontsize=13, colorbar_tick_fontsize=11)
            # Enforce shared colorbar scale across base models by persisting vmax
            try:
                # Compute local max abs difference
                local_max = float(np.nanmax(np.abs(Cdens_Ra - Cdens_Ga))) if np.isfinite(np.abs(Cdens_Ra - Cdens_Ga)).any() else 0.0
            except Exception:
                local_max = 0.0
            # Default baseline scale if no persisted scale exists
            default_scale = 0.6
            # Build path under combined to share across backbones
            try:
                scale_dir = os.path.join(suite_root, (dataset_tag if dataset_tag is not None else 'dataset'), 'combined', 'base_full', f"D{idp_size_tag}")
            except Exception:
                scale_dir = os.path.join(results_root)
            os.makedirs(scale_dir, exist_ok=True)
            scale_meta_path = os.path.join(scale_dir, 'pair_absdiff_scale.json')
            prev = None
            if os.path.exists(scale_meta_path):
                try:
                    with open(scale_meta_path, 'r') as f:
                        prev = json.load(f).get('absdiff_vmax', None)
                except Exception:
                    prev = None
            used_vmax = float(max(default_scale if prev is None else prev, local_max))
            plot_pair_absdiff_matrix(Cdens_Ra, Cdens_Ga, common, os.path.join(pair_dir, 'pair_density_corr_absdiff.png'),
                                     title='Abs difference |density-shape| (Real vs Gen)', order=order_density, cmap='magma', dpi=args.save_plots_dpi, fixed_vmax=used_vmax, title_fontsize=13, colorbar_tick_fontsize=11)
            # Persist updated scale
            try:
                with open(scale_meta_path, 'w') as f:
                    json.dump({'absdiff_vmax': used_vmax}, f, indent=2)
            except Exception:
                pass
            if args.pair_mantel:
                res = mantel_test(Cdens_Ra, Cdens_Ga, num_permutations=perm_effective, random_state=args.pair_random_state)
                if perm_effective is None or int(perm_effective) <= 0:
                    # omit p-value from outputs when permutation tests are disabled
                    if 'p_value' in res:
                        res = {k: v for k, v in res.items() if k != 'p_value'}
                _save_json(res, os.path.join(pair_dir, 'pair_density_corr_mantel.json'))
            eval_timings['pair_density_shape_corr_seconds'] = time.time() - t_pair_start

        # Aggregate scalars for learning/dim analyses
        ace_scalar = _aggregate_metric_for_learning(ace, prefer_keys=[f"ace_{int(100*x)}" for x in ace_ps])
        ks_scalar = _aggregate_metric_for_learning(ksbin, prefer_keys=['ks_mean'])
        ecp_err = {}
        for k, v in ecp.items():
            if 'ecp' in v and 'nominal' in v:
                ecp_err[k] = {'ecp_abs_error': abs(float(v['ecp']) - float(v['nominal']))}
        ecp_scalar = _aggregate_metric_for_learning(ecp_err, prefer_keys=['ecp_abs_error'])
        scalars = {'ace': ace_scalar, 'ks_mean': ks_scalar, 'ecp_abs_error': ecp_scalar}
        _save_json(scalars, os.path.join(results_root, 'scalar_metrics.json'))
        
        eval_timings['total_run_eval_seconds'] = time.time() - t_run_eval_start
        _save_json(eval_timings, os.path.join(results_root, 'runtimes.json'))
        
        return scalars

    # Determine which backbones to run (skip if gamlss_only)
    if args.gamlss_only:
        backbones = []
    elif args.backbone == 'both':
        # When running both backbones, execute SAINT before MLP
        backbones = ['saint', 'mlp']
    else:
        backbones = [args.backbone]

    # Build CSV maps (UKB) only if any UKB inputs are provided. Otherwise, if only --synth_csv
    # is supplied, we skip UKB entirely and run synthetic-only pipeline.
    csv_train_map = {}
    csv_hold_map = {}
    has_ukb_inputs = bool(args.train_csv_map and args.holdout_csv_map) or bool(args.train_csv and args.holdout_csv) or bool(args.csv_path)
    
    # =========================================================================
    # PHASE 1: PREPARE CSV MAPS AND BASE DATASETS
    # =========================================================================
    
    # Build the CSV maps and determine base datasets
    if has_ukb_inputs:
        if args.train_csv_map and args.holdout_csv_map:
            with open(args.train_csv_map, 'r') as f:
                csv_train_map = json.load(f)
            with open(args.holdout_csv_map, 'r') as f:
                csv_hold_map = json.load(f)
        else:
            # base case only
            if args.train_csv and args.holdout_csv:
                csv_train_map = {str(idp_sizes[0]): args.train_csv}
                csv_hold_map = {str(idp_sizes[0]): args.holdout_csv}
            elif args.csv_path:
                # split single CSV deterministically once and use base size
                df_all = _load_and_preprocess(args.csv_path, args.min_age, args.max_age, args.ukb_cov2_col)
                df_train, df_hold = _stratified_split_by_age_and_cov2(
                    df=df_all,
                    train_ratio=float(args.split_ratio),
                    seed=int(args.split_seed),
                    age_col='age',
                    cov2_col=args.ukb_cov2_col,
                    num_age_bins=10,
                )
                os.makedirs(args.save_splits_dir, exist_ok=True)
                base = os.path.splitext(os.path.basename(args.csv_path))[0]
                tr_out = os.path.join(args.save_splits_dir, f"{base}_train_seed{args.split_seed}_r{args.split_ratio}.csv")
                ho_out = os.path.join(args.save_splits_dir, f"{base}_holdout_seed{args.split_seed}_r{args.split_ratio}.csv")
                df_train.to_csv(tr_out, index=False)
                df_hold.to_csv(ho_out, index=False)
                print(f"Saved splits -> {tr_out} | {ho_out}")
                csv_train_map = {str(idp_sizes[0]): tr_out}
                csv_hold_map = {str(idp_sizes[0]): ho_out}
    else:
        # No UKB inputs. If no synthetic CSV either, then error out; otherwise proceed with synth-only.
        if not getattr(args, 'synth_csv', None):
            raise ValueError('Provide either train/holdout CSVs or a single csv_path to split, or JSON maps for dimensional scaling, or set --synth_csv for synthetic-only run.')

    # Run suites
    summary_records = []
    # 1) Always run a single base evaluation (full data) for the selected backbones on the main CSVs
    base_train_csv = None
    base_holdout_csv = None
    base_idp_tag = 'base'
    _idp_size_int_base = -1
    if args.train_csv and args.holdout_csv:
        base_train_csv = args.train_csv
        base_holdout_csv = args.holdout_csv
        try:
            _df_tmp = _load_and_preprocess(base_train_csv, args.min_age, args.max_age, args.ukb_cov2_col)
            _idp_size_int_base = len([c for c in _df_tmp.columns if c not in ['age', args.ukb_cov2_col]])
            base_idp_tag = str(_idp_size_int_base)
        except Exception:
            base_idp_tag = 'base'
    elif len(csv_train_map) == 1:
        only_key = list(csv_train_map.keys())[0]
        base_train_csv = csv_train_map[only_key]
        base_holdout_csv = csv_hold_map[only_key]
        base_idp_tag = only_key
        try:
            _df_tmp = _load_and_preprocess(base_train_csv, args.min_age, args.max_age, args.ukb_cov2_col)
            _idp_size_int_base = len([c for c in _df_tmp.columns if c not in ['age', args.ukb_cov2_col]])
            base_idp_tag = str(_idp_size_int_base)
        except Exception:
            pass
    else:
        # If neither explicit base CSVs nor a single-map entry is available, base full analyses cannot run
        pass
    
    # GAMLSS baseline helper (can be invoked in GAMLSS-only mode or after diffusion models)
    def run_gamlss_phase() -> None:
        print("\n" + "="*80)
        print("Running GAMLSS baseline")
        print("="*80 + "\n")

        # Run GAMLSS for UKB base dataset
        if has_ukb_inputs and base_train_csv is not None and base_holdout_csv is not None:
            print("Running GAMLSS baseline for UKB dataset...")
            try:
                scalars_gamlss, gamlss_timings_ukb = _run_gamlss_baseline(
                    args, suite_root, base_train_csv, base_holdout_csv, base_idp_tag,
                    suite_tag='base_full', cov2_col=args.ukb_cov2_col, dataset_tag='ukb'
                )
                all_run_timings['gamlss_ukb_base_full'] = gamlss_timings_ukb
                summary_records.append({
                    'method': 'gamlss',
                    'dataset': 'ukb',
                    'analysis': 'base_full',
                    'idp_size': (_idp_size_int_base if _idp_size_int_base and _idp_size_int_base > 0 else None),
                    'scalars': scalars_gamlss,
                })
                print(f"✓ GAMLSS baseline for UKB completed successfully")
                # Create GAMLSS-only coverage overlay for UKB (uses GAMLSS outputs only)
                try:
                    if str(base_idp_tag) == '20':
                        _coverage_diff_overlay_gamlss(suite_root, 'ukb', 'base_full', str(base_idp_tag), dpi=args.save_plots_dpi)
                except Exception:
                    pass
            except Exception as e:
                print(f"✗ GAMLSS baseline failed for UKB base: {e}")
                import traceback
                traceback.print_exc()
                if args.gamlss_only:
                    raise  # If GAMLSS-only mode, fail fast

        # Run GAMLSS for synthetic dataset (prepare splits first if needed)
        if getattr(args, 'synth_csv', None):
            print("Running GAMLSS baseline for synthetic dataset...")
            try:
                # Deterministic split
                df_s_all = _load_and_preprocess(args.synth_csv, args.synth_min_age, args.synth_max_age, args.synth_cov2_col)
                df_s_tr, df_s_ho = _stratified_split_by_age_and_cov2(
                    df=df_s_all,
                    train_ratio=float(args.synth_split_ratio),
                    seed=int(args.synth_split_seed),
                    age_col='age',
                    cov2_col=args.synth_cov2_col,
                    num_age_bins=10,
                )
                os.makedirs(args.save_splits_dir, exist_ok=True)
                base_s = os.path.splitext(os.path.basename(args.synth_csv))[0]
                tr_s_out = os.path.join(args.save_splits_dir, f"{base_s}_train_seed{args.synth_split_seed}_r{args.synth_split_ratio}.csv")
                ho_s_out = os.path.join(args.save_splits_dir, f"{base_s}_holdout_seed{args.synth_split_seed}_r{args.synth_split_ratio}.csv")
                df_s_tr.to_csv(tr_s_out, index=False)
                df_s_ho.to_csv(ho_s_out, index=False)
                print(f"Saved synthetic splits -> {tr_s_out} | {ho_s_out}")
                
                # Determine IDP size
                try:
                    _df_tmp_s = _load_and_preprocess(tr_s_out, args.synth_min_age, args.synth_max_age, args.synth_cov2_col)
                    size_s = len([c for c in _df_tmp_s.columns if c not in ['age', args.synth_cov2_col]])
                except Exception:
                    size_s = 4
                idp_tag_s = str(size_s)
                
                scalars_gamlss_s, gamlss_timings_synth = _run_gamlss_baseline(
                    args, suite_root, tr_s_out, ho_s_out, idp_tag_s,
                    suite_tag='base_full', cov2_col=args.synth_cov2_col, dataset_tag='synth'
                )
                all_run_timings['gamlss_synth_base_full'] = gamlss_timings_synth
                summary_records.append({
                    'method': 'gamlss',
                    'dataset': 'synth',
                    'analysis': 'base_full',
                    'idp_size': size_s,
                    'scalars': scalars_gamlss_s,
                })
                print(f"✓ GAMLSS baseline for synthetic dataset completed successfully")
                # Create GAMLSS-only coverage overlay for synthetic data (uses GAMLSS outputs only)
                try:
                    if str(idp_tag_s) == '4':
                        _coverage_diff_overlay_gamlss(suite_root, 'synth', 'base_full', str(idp_tag_s), dpi=args.save_plots_dpi)
                except Exception:
                    pass
            except Exception as e:
                print(f"✗ GAMLSS baseline failed for synthetic dataset: {e}")
                import traceback
                traceback.print_exc()
                if args.gamlss_only:
                    raise  # If GAMLSS-only mode, fail fast

        print("\n" + "="*80)
        print("GAMLSS baseline complete")
        print("="*80 + "\n")

    # In GAMLSS-only mode, run the baseline now and exit before diffusion models
    if args.gamlss_only:
        run_gamlss_phase()
        _save_json({'records': summary_records}, os.path.join(suite_root, 'summary.json'))
        all_run_timings['total_script_seconds'] = time.time() - t_main_start
        _save_json(all_run_timings, os.path.join(suite_root, 'all_runtimes.json'))
        print('GAMLSS-only mode: Skipping diffusion model training.')
        print(f'Results saved to: {suite_root}')
        return

    # =========================================================================
    # PHASE 2: RUN DIFFUSION MODELS (MLP/SAINT)
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 2: Running diffusion models (MLP/SAINT)")
    print("="*80 + "\n")
    
    if has_ukb_inputs:
        for backbone in backbones:
            # If we have a base dataset, run heavy analyses ONCE on full data before sweeps
            if base_train_csv is not None and base_holdout_csv is not None:
                scalars_base = run_eval(
                    backbone,
                    base_idp_tag,
                    base_train_csv,
                    base_holdout_csv,
                    1.0,
                    0,
                    heavy_analyses=(str(base_idp_tag) == '20'),
                    suite_tag='base_full',
                    cov2_col=args.ukb_cov2_col,
                    dataset_tag='ukb'
                )
                rec = {
                    'backbone': backbone,
                    'analysis': 'base_full',
                    'idp_size': (_idp_size_int_base if _idp_size_int_base and _idp_size_int_base > 0 else None),
                    'scalars': scalars_base,
                }
                summary_records.append(rec)

            # learning curves: use ONLY the base train/holdout CSVs, not the CSV maps
            if args.run_learning_curves:
                # Resolve base train/holdout CSVs for learning curves
                if args.train_csv and args.holdout_csv:
                    lc_train_csv = args.train_csv
                    lc_holdout_csv = args.holdout_csv
                    # determine IDP size from the base train CSV
                    try:
                        _df_tmp = _load_and_preprocess(lc_train_csv, args.min_age, args.max_age, args.ukb_cov2_col)
                        _idp_size_int = len([c for c in _df_tmp.columns if c not in ['age', args.ukb_cov2_col]])
                    except Exception:
                        _idp_size_int = -1
                else:
                    # Fallback: if exactly one entry exists in the maps (e.g., created from csv_path split), use it.
                    map_keys = list(csv_train_map.keys())
                    if len(map_keys) == 1:
                        only_key = map_keys[0]
                        lc_train_csv = csv_train_map[only_key]
                        lc_holdout_csv = csv_hold_map[only_key]
                        try:
                            _df_tmp = _load_and_preprocess(lc_train_csv, args.min_age, args.max_age, args.ukb_cov2_col)
                            _idp_size_int = len([c for c in _df_tmp.columns if c not in ['age', args.ukb_cov2_col]])
                        except Exception:
                            try:
                                _idp_size_int = int(only_key)
                            except Exception:
                                _idp_size_int = -1
                    else:
                        # When multiple CSVs are provided via maps, require explicit base train/holdout CSVs
                        raise ValueError('run_learning_curves requires --train_csv and --holdout_csv when multiple CSV maps are provided. Maps are only used for run_dim_scaling.')

                # Tag for directory structure
                lc_idp_tag = str(_idp_size_int) if _idp_size_int and _idp_size_int > 0 else 'base'

                # collect points per seed
                seeds_points_ace: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
                seeds_points_ks: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
                seeds_points_ecp: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
                num_seeds_eff = max(1, int(args.num_seeds))
                for seed_idx in range(num_seeds_eff):
                    xs = []
                    ys_ace = []
                    ys_ks = []
                    ys_ecp = []
                    for f in fracs:
                        scalars = run_eval(backbone, lc_idp_tag, lc_train_csv, lc_holdout_csv, f, seed_idx, heavy_analyses=False, suite_tag='learning_curves', cov2_col=args.ukb_cov2_col, dataset_tag='ukb')
                        # N_f = f * N_train (we need N_train of the original; load it once)
                        if 'N_train_full' in scalars:
                            Nf = float(scalars['N_train_full']) * float(f)
                        else:
                            # approximate by using current df size (post subsample) / f
                            try:
                                # approximate original size by N_current / f
                                Nf = 1.0 / max(f, 1e-12)
                            except Exception:
                                Nf = 1.0
                        xs.append(Nf)
                        ys_ace.append(float(scalars.get('ace', float('nan'))))
                        ys_ks.append(float(scalars.get('ks_mean', float('nan'))))
                        ys_ecp.append(float(scalars.get('ecp_abs_error', float('nan'))))
                    seeds_points_ace[seed_idx] = (np.asarray(xs, dtype=float), np.asarray(ys_ace, dtype=float))
                    seeds_points_ks[seed_idx] = (np.asarray(xs, dtype=float), np.asarray(ys_ks, dtype=float))
                    seeds_points_ecp[seed_idx] = (np.asarray(xs, dtype=float), np.asarray(ys_ecp, dtype=float))
                # bootstrap slopes vs log N
                res_ace = _bootstrap_slope(seeds_points_ace, args.bootstrap_B, logx=True)
                res_ks = _bootstrap_slope(seeds_points_ks, args.bootstrap_B, logx=True)
                res_ecp = _bootstrap_slope(seeds_points_ecp, args.bootstrap_B, logx=True)
                suite = {
                    'backbone': backbone,
                    'analysis': 'learning_curves',
                    'idp_size': (_idp_size_int if _idp_size_int and _idp_size_int > 0 else None),
                    'slopes': {'ACE': res_ace, 'KS': res_ks, 'ECP_abs_error': res_ecp},
                }
                summary_records.append(suite)
                _save_json(suite, os.path.join(suite_root, 'ukb', backbone, 'learning_curves', f'D{lc_idp_tag}', 'summary.json'))

            # dimensional scaling across provided sizes
            if args.run_dim_scaling:
                seeds_points_ace: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
                seeds_points_ks: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
                seeds_points_ecp: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
                num_seeds_eff = max(1, int(args.num_seeds))
                for seed_idx in range(num_seeds_eff):
                    xs = []
                    ys_ace = []
                    ys_ks = []
                    ys_ecp = []
                    for D in idp_sizes:
                        key = str(D)
                        if key not in csv_train_map or key not in csv_hold_map:
                            continue
                        scalars = run_eval(backbone, key, csv_train_map[key], csv_hold_map[key], 1.0, seed_idx, heavy_analyses=False, suite_tag='dim_scaling', cov2_col=args.ukb_cov2_col, dataset_tag='ukb')
                        xs.append(float(D))
                        ys_ace.append(float(scalars.get('ace', float('nan'))))
                        ys_ks.append(float(scalars.get('ks_mean', float('nan'))))
                        ys_ecp.append(float(scalars.get('ecp_abs_error', float('nan'))))
                    if xs:
                        seeds_points_ace[seed_idx] = (np.asarray(xs, dtype=float), np.asarray(ys_ace, dtype=float))
                        seeds_points_ks[seed_idx] = (np.asarray(xs, dtype=float), np.asarray(ys_ks, dtype=float))
                        seeds_points_ecp[seed_idx] = (np.asarray(xs, dtype=float), np.asarray(ys_ecp, dtype=float))
                # bootstrap slopes vs D
                res_ace = _bootstrap_slope(seeds_points_ace, args.bootstrap_B, logx=False)
                res_ks = _bootstrap_slope(seeds_points_ks, args.bootstrap_B, logx=False)
                res_ecp = _bootstrap_slope(seeds_points_ecp, args.bootstrap_B, logx=False)
                suite = {
                    'backbone': backbone,
                    'analysis': 'dim_scaling',
                    'slopes': {'ACE': res_ace, 'KS': res_ks, 'ECP_abs_error': res_ecp},
                    'idp_sizes': idp_sizes,
                }
                summary_records.append(suite)
                _save_json(suite, os.path.join(suite_root, 'ukb', backbone, 'dim_scaling', 'summary.json'))

            # Base single-condition run when no sweep flags are provided
            if (not args.run_learning_curves) and (not args.run_dim_scaling):
                # Only run the explicit base_full once (already executed above when base CSVs are available).
                # Avoid running multiple IDP sizes under base_full; use --run_dim_scaling to evaluate multiple sizes.
                if base_train_csv is None or base_holdout_csv is None:
                    print('Skipping base-only runs: no explicit base train/holdout CSVs. Use --run_dim_scaling for multiple IDP sizes.')

        # After both backbones base runs, create overlay plots for UKB D20 if present
        try:
            if str(base_idp_tag) == '20':
                # Create diffusion models overlay (MLP + SAINT)
                _coverage_diff_overlay_diffusion_models(suite_root, 'ukb', 'base_full', str(base_idp_tag), 1.0, 0, dpi=args.save_plots_dpi)
        except Exception:
            pass

    # 2) Synthetic dataset (optional)
    if getattr(args, 'synth_csv', None):
        # Deterministic split
        df_s_all = _load_and_preprocess(args.synth_csv, args.synth_min_age, args.synth_max_age, args.synth_cov2_col)
        df_s_tr, df_s_ho = _stratified_split_by_age_and_cov2(
            df=df_s_all,
            train_ratio=float(args.synth_split_ratio),
            seed=int(args.synth_split_seed),
            age_col='age',
            cov2_col=args.synth_cov2_col,
            num_age_bins=10,
        )
        os.makedirs(args.save_splits_dir, exist_ok=True)
        base_s = os.path.splitext(os.path.basename(args.synth_csv))[0]
        tr_s_out = os.path.join(args.save_splits_dir, f"{base_s}_train_seed{args.synth_split_seed}_r{args.synth_split_ratio}.csv")
        ho_s_out = os.path.join(args.save_splits_dir, f"{base_s}_holdout_seed{args.synth_split_seed}_r{args.synth_split_ratio}.csv")
        df_s_tr.to_csv(tr_s_out, index=False)
        df_s_ho.to_csv(ho_s_out, index=False)
        print(f"Saved synthetic splits -> {tr_s_out} | {ho_s_out}")
        # Determine IDP columns size
        try:
            _df_tmp_s = _load_and_preprocess(tr_s_out, args.synth_min_age, args.synth_max_age, args.synth_cov2_col)
            size_s = len([c for c in _df_tmp_s.columns if c not in ['age', args.synth_cov2_col]])
        except Exception:
            size_s = 4
        idp_tag_s = str(size_s)
        for backbone in backbones:
            # Heavy analyses single base run
            scalars_base_s = run_eval(
                backbone,
                idp_tag_s,
                tr_s_out,
                ho_s_out,
                1.0,
                0,
                heavy_analyses=False,  # Restrict heavy analyses to UKB D20 only
                suite_tag='base_full',
                cov2_col=args.synth_cov2_col,
                dataset_tag='synth'
            )
            summary_records.append({
                'dataset': 'synth',
                'backbone': backbone,
                'analysis': 'base_full',
                'idp_size': size_s,
                'scalars': scalars_base_s,
            })
        # After both backbones base runs on synth, create overlay plots for D4 if present
        try:
            if str(idp_tag_s) == '4':
                # Create diffusion models overlay (MLP + SAINT)
                _coverage_diff_overlay_diffusion_models(suite_root, 'synth', 'base_full', str(idp_tag_s), 1.0, 0, dpi=args.save_plots_dpi)
        except Exception:
            pass

    # =========================================================================
    # PHASE 3: RUN GAMLSS BASELINE LAST (if requested)
    # =========================================================================
    if args.run_gamlss_baseline:
        run_gamlss_phase()

    # overall summary
    _save_json({'records': summary_records}, os.path.join(suite_root, 'summary.json'))
    
    # Save all runtime timings
    all_run_timings['total_script_seconds'] = time.time() - t_main_start
    _save_json(all_run_timings, os.path.join(suite_root, 'all_runtimes.json'))
    
    # Print summary
    diffusion_runs = [r for r in summary_records if r.get('backbone') in ['mlp', 'saint']]
    gamlss_runs = [r for r in summary_records if r.get('method') == 'gamlss']
    print('\n' + '='*80)
    print('Full evaluation suite complete.')
    print(f'  Diffusion model runs: {len(diffusion_runs)}')
    print(f'  GAMLSS baseline runs: {len(gamlss_runs)}')
    print(f'  Total runtime: {all_run_timings["total_script_seconds"]:.2f} seconds ({all_run_timings["total_script_seconds"]/60:.2f} minutes)')
    print(f'  Results saved to: {suite_root}')
    print(f'  Runtime details saved to: {os.path.join(suite_root, "all_runtimes.json")}')
    print('='*80 + '\n')


if __name__ == '__main__':
    main()


