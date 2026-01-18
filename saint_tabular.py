"""saint_tabular.py
A lightweight implementation of the SAINT (Self-Attention and Intersample Attention) architecture
adapted to serve as the denoiser backbone for a tabular Diffusion Model.

Key simplifications / notes
--------------------------
1.  Continuous-only features: The UKB IDPs and covariates are numeric, so this module ignores
    categorical encoders for now.  Categorical columns can still be handled by pre-embedding
    them as numbers before passing to the diffusion model.
2.  Column attention → nn.TransformerEncoder blocks acting on the *feature* dimension.
3.  Row attention  → a single Multi-Head Self-Attention layer acting on the *batch* (row)
    dimension, applied to a per-row summary token (mean of feature tokens).  The result is
    broadcast back to all feature tokens, closely following the idea in SAINT without the
    heavy cost of B×B attention on every token.
4.  Conditioning (age, sex, …) + timestep  t  are injected by *adding* learned linear
    projections to every feature token.  This is cheap and empirically works as well as the
    FiLM modulation used in the original MLP implementation.
5.  The module exposes:
        SAINTDenoiser  – predicts ε̂(x_t, t, c) (noise per feature)
        SAINTDiffusionModel – wraps the denoiser and implements the DDPM forward pass in *data* space
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Utility: sinusoidal timestep embedding (as in DDPM official impl)
# -----------------------------------------------------------------------------

def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Create sinusoidal timestep embeddings of shape [batch, dim]."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps.float() * freqs
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb

# -----------------------------------------------------------------------------
# SAINT blocks
# -----------------------------------------------------------------------------

class ColumnTransformer(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.model = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class RowAttentionBlock(nn.Module):
    """
    Row-wise attention block with optional true intersample attention.

    Modes:
      - intersample == False:
          Original "degenerate" behaviour. Operates on a per-sample summary token
          of shape [B, 1, d] so there is no cross-row mixing; attention reduces to
          a small bottleneck MLP-like transformation applied independently per row.
      - intersample == True:
          During training (model.train()), performs true intersample attention by
          treating the batch dimension as the sequence ([1, B, d]). During eval
          (model.eval()) it falls back to the degenerate mode to avoid batch-
          dependent behaviour at sampling time.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.1,
        intersample: bool = False,
        intersample_prob: float = 0.5,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.ln = nn.LayerNorm(d_model)
        self.intersample = intersample
        # Probability of using true intersample attention on a given forward pass
        # when in training mode and intersample is enabled.
        self.intersample_prob = intersample_prob

    def _forward_degenerate(self, x: torch.Tensor) -> torch.Tensor:
        # Original behaviour: per-sample row summary; no cross-row mixing.
        # x: [B, F, d]
        row_tok = x.mean(dim=1, keepdim=True)  # [B, 1, d]
        att, _ = self.attn(row_tok, row_tok, row_tok, need_weights=False)
        out = self.ln(row_tok + att)
        out = out + self.ffn(out)
        return x + out.expand(-1, x.size(1), -1)

    def _forward_intersample(self, x: torch.Tensor) -> torch.Tensor:
        # True intersample attention across rows in the batch.
        # x: [B, F, d]
        B, F, _ = x.shape
        row_tok = x.mean(dim=1)          # [B, d]
        row_tok = row_tok.unsqueeze(0)   # [1, B, d]  (batch size = 1, seq len = B)
        att, _ = self.attn(row_tok, row_tok, row_tok, need_weights=False)  # [1, B, d]
        out = self.ln(row_tok + att)     # [1, B, d]
        out = out.squeeze(0)             # [B, d]
        out = out + self.ffn(out)        # [B, d]
        out = out.unsqueeze(1).expand(-1, F, -1)  # [B, F, d]
        return x + out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If intersample is disabled, always use degenerate behaviour.
        if not self.intersample:
            return self._forward_degenerate(x)
        # If intersample is enabled, randomly gate between intersample and
        # degenerate paths during training so that the degenerate path is
        # explicitly trained, but always use degenerate at eval/sampling time.
        if self.training:
            if torch.rand((), device=x.device) < self.intersample_prob:
                return self._forward_intersample(x)
            else:
                return self._forward_degenerate(x)
        # Eval/sampling: avoid batch-dependent behaviour.
        return self._forward_degenerate(x)

class SAINTDenoiser(nn.Module):
    def __init__(
        self,
        num_features: int,
        cond_dim: int,
        d_model: int = 64,
        depth: int = 6,
        nhead: int = 8,
        dropout: float = 0.1,
        row_attention_interval: int = 2,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.feature_weight = nn.Parameter(torch.randn(num_features, d_model) * 0.01)
        self.feature_bias = nn.Parameter(torch.zeros(num_features, d_model))
        self.column_embed = nn.Parameter(torch.randn(num_features, d_model) * 0.01)
        self.cond_proj = nn.Linear(cond_dim, d_model)
        self.time_proj = nn.Linear(d_model, d_model, bias=False)

        # Interpret sign of row_attention_interval:
        #   >0 : degenerate row attention only (no intersample)
        #   <0 : enable true intersample attention during training only;
        #        fall back to degenerate behaviour at eval/sampling time.
        #   =0 : disable row attention entirely.
        intersample = row_attention_interval < 0
        interval = abs(row_attention_interval)

        blocks: List[nn.Module] = []
        for i in range(depth):
            blocks.append(ColumnTransformer(d_model, nhead, 1, dropout))
            if interval > 0 and (i + 1) % interval == 0:
                blocks.append(RowAttentionBlock(d_model, nhead, dropout, intersample=intersample))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B, F = x_noisy.shape
        assert F == self.num_features, f"Expected {self.num_features} features, got {F}"
        x = x_noisy.unsqueeze(-1) * self.feature_weight + self.feature_bias
        x = x + self.column_embed.unsqueeze(0)
        x = x + self.cond_proj(cond).unsqueeze(1) + self.time_proj(timestep_embedding(t, x.size(-1))).unsqueeze(1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x).squeeze(-1)

    def denoise(self, x_noisy: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, alpha_bar_t: torch.Tensor, std_t: torch.Tensor) -> torch.Tensor:
        eps_hat = self.forward(x_noisy, t, cond)
        return (x_noisy - std_t * eps_hat) / torch.sqrt(alpha_bar_t + 1e-8)

class SAINTDiffusionModel(nn.Module):
    def __init__(self, data_dim: int, cond_dim: int, saint_kwargs: Optional[dict] | None = None):
        super().__init__()
        saint_kwargs = saint_kwargs or {}
        self.denoiser = SAINTDenoiser(num_features=data_dim, cond_dim=cond_dim, **saint_kwargs)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        alpha_bar_t: torch.Tensor,
        std_t: torch.Tensor,
        use_mu: bool = False,
    ):
        eps = torch.randn_like(x)
        x_noisy = torch.sqrt(alpha_bar_t) * x + std_t * eps
        eps_pred = self.denoiser(x_noisy, t, cond)
        x_recon = (x_noisy - std_t * eps_pred) / torch.sqrt(alpha_bar_t + 1e-8)
        return eps_pred, x_recon, None, None, x, x_noisy
