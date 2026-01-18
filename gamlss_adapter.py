# -*- coding: utf-8 -*-
"""
GAMLSS adapter: univariate baseline per-IDP for ACE/ECP/PIT/KS.

Usage sketch:
    from gamlss_adapter import GAMLSSBaseline

    g = GAMLSSBaseline(family="SHASH",
                       mu_formula="~ cs(age, df=3) + sex",
                       sigma_formula="~ cs(age, df=3) + sex",
                       nu_formula=None, tau_formula=None)

    g.fit(train_df, idp_cols=IDP_LIST, covariates=["age","sex"])
    Q = g.quantiles(eval_grid_df, probs=[0.02,0.25,0.5,0.75,0.98])  # dict[idp] -> (ngrid x nprob)
    U = g.pit(holdout_df)                                          # dict[idp] -> (n_holdout,)
    S = g.sample(eval_grid_df, M=512)                               # dict[idp] -> list of length ngrid of (M,) samples
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

# rpy2 bridge
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter

# Load required R packages once
_gamlss      = importr("gamlss")
_gamlss_dist = importr("gamlss.dist")
_base        = importr("base")
_stats       = importr("stats")

# R helper: function to get centiles for a fitted gamlss object at newdata
R_HELPER = ro.r("""
get_percentiles_newdata <- function(obj, newdata, cents=c(2,25,50,75,98)) {
  if (!gamlss::is.gamlss(obj)) stop("Not a gamlss object")
  fam   <- obj$family[1]
  qfun  <- get(paste0("q", fam), asNamespace("gamlss.dist"))
  pars  <- gamlss::predictAll(obj, newdata=newdata)  # list of mu, sigma, nu, tau as needed
  L     <- length(pars)
  # Order params into a list in qfun(..., mu=, sigma=, nu=, tau=) order
  make_args <- function(p) {
    if (L==1) list(mu=pars$mu[p])
    else if (L==2) list(mu=pars$mu[p], sigma=pars$sigma[p])
    else if (L==3) list(mu=pars$mu[p], sigma=pars$sigma[p], nu=pars$nu[p])
    else           list(mu=pars$mu[p], sigma=pars$sigma[p], nu=pars$nu[p], tau=pars$tau[p])
  }
  out <- matrix(NA_real_, nrow=nrow(newdata), ncol=length(cents))
  for (i in seq_len(nrow(newdata))) {
    args <- make_args(i)
    out[i, ] <- sapply(cents/100, function(pr) do.call(qfun, c(list(p=pr), args)))
  }
  colnames(out) <- paste0("q", cents)
  out
}

# PIT values for observed y at their covariates (using predicted params)
pit_values <- function(obj, newdata, y) {
  fam  <- obj$family[1]
  pfun <- get(paste0("p", fam), asNamespace("gamlss.dist"))
  pars <- gamlss::predictAll(obj, newdata=newdata)
  L    <- length(pars)
  make_args <- function(i) {
    if (L==1) list(q=y[i], mu=pars$mu[i])
    else if (L==2) list(q=y[i], mu=pars$mu[i], sigma=pars$sigma[i])
    else if (L==3) list(q=y[i], mu=pars$mu[i], sigma=pars$sigma[i], nu=pars$nu[i])
    else           list(q=y[i], mu=pars$mu[i], sigma=pars$sigma[i], nu=pars$nu[i], tau=pars$tau[i])
  }
  u <- numeric(length(y))
  for (i in seq_along(y)) u[i] <- do.call(pfun, make_args(i))
  u
}

# Draw M samples at each row of newdata (for KS etc.), returning list of numeric vectors
sample_M <- function(obj, newdata, M=512) {
  fam  <- obj$family[1]
  rfun <- get(paste0("r", fam), asNamespace("gamlss.dist"))
  pars <- gamlss::predictAll(obj, newdata=newdata)
  L    <- length(pars)
  make_args <- function(i) {
    if (L==1) list(n=M, mu=pars$mu[i])
    else if (L==2) list(n=M, mu=pars$mu[i], sigma=pars$sigma[i])
    else if (L==3) list(n=M, mu=pars$mu[i], sigma=pars$sigma[i], nu=pars$nu[i])
    else           list(n=M, mu=pars$mu[i], sigma=pars$sigma[i], nu=pars$nu[i], tau=pars$tau[i])
  }
  out <- vector("list", nrow(newdata))
  for (i in seq_len(nrow(newdata))) out[[i]] <- do.call(rfun, make_args(i))
  out
}
""")

get_percentiles_newdata = ro.globalenv["get_percentiles_newdata"]
pit_values              = ro.globalenv["pit_values"]
sample_M                = ro.globalenv["sample_M"]

class GAMLSSBaseline:
    def __init__(self,
                 family: str = "SHASH",
                 mu_formula: str = "~ cs(age, df=3)",
                 sigma_formula: Optional[str] = "~ cs(age, df=3)",
                 nu_formula: Optional[str] = None,
                 tau_formula: Optional[str] = None,
                 verbose: bool = False,
                 n_cyc: int = 100,
                 c_crit: float = 0.001):
        """
        family: one of the gamlss families, e.g., 'SHASH','BCT','BCPE' (from gamlss.dist)
        *_formula: R right-hand-side strings; include covariates like age, sex, eTIV, etc.
        verbose: whether to print progress during fitting
        n_cyc: maximum number of cycles for outer iteration (default 100)
        c_crit: convergence criterion for outer iteration (default 0.001)
        """
        self.family = family
        self.mu_formula = mu_formula
        self.sigma_formula = sigma_formula
        self.nu_formula = nu_formula
        self.tau_formula = tau_formula
        self.verbose = verbose
        self.n_cyc = n_cyc
        self.c_crit = c_crit
        self._fits = {}  # idp -> R gamlss object

    def _fit_one(self, df: pd.DataFrame, ycol: str):
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_df = ro.conversion.py2rpy(df)
        fam  = getattr(_gamlss_dist, self.family)
        # Build the GAMLSS call with optional formulas
        # gamlss(y ~ mu_form, sigma.formula=..., nu.formula=..., tau.formula=..., family=fam, data=df, n.cyc=..., c.crit=...)
        # Wrap column name in backticks if it starts with a digit or contains special chars
        ycol_safe = f"`{ycol}`" if (ycol[0].isdigit() or not ycol.replace('_', '').isalnum()) else ycol
        kwargs = {
            "formula": ro.Formula(f"{ycol_safe} {self.mu_formula}"),
            "family": fam,
            "data": r_df,
            "trace": ro.BoolVector([self.verbose]),
            "n.cyc": ro.IntVector([self.n_cyc]),
            "c.crit": ro.FloatVector([self.c_crit])
        }
        if self.sigma_formula is not None:
            kwargs["sigma.formula"] = ro.Formula(self.sigma_formula)
        if self.nu_formula is not None:
            kwargs["nu.formula"] = ro.Formula(self.nu_formula)
        if self.tau_formula is not None:
            kwargs["tau.formula"] = ro.Formula(self.tau_formula)

        fit = _gamlss.gamlss(**kwargs)
        return fit

    def fit(self, train_df: pd.DataFrame, idp_cols: List[str], covariates: List[str]):
        """Fit one GAMLSS per IDP using train_df[idp + covariates]."""
        cols = covariates[:]  # copy
        for idx, y in enumerate(idp_cols):
            if self.verbose:
                print(f"Fitting GAMLSS for IDP {idx+1}/{len(idp_cols)}: {y}")
            sub = train_df[cols + [y]].dropna()
            self._fits[y] = self._fit_one(sub, y)

    def quantiles(self, grid_df: pd.DataFrame, probs: Tuple[float, ...] = (0.02, 0.25, 0.5, 0.75, 0.98)) -> Dict[str, np.ndarray]:
        """Return dict[idp] -> (n_grid x n_probs) centiles at grid covariates."""
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_grid = ro.conversion.py2rpy(grid_df)
        cents  = ro.FloatVector([p*100 for p in probs])
        out = {}
        for y, fit in self._fits.items():
            mat = get_percentiles_newdata(fit, r_grid, cents)
            out[y] = np.asarray(mat, dtype=float)
        return out

    def pit(self, holdout_df: pd.DataFrame, covariates: List[str]) -> Dict[str, np.ndarray]:
        """Return dict[idp] -> PIT values for each held-out row (uses its covariates)."""
        out = {}
        for y, fit in self._fits.items():
            sub = holdout_df[covariates + [y]].dropna()
            if sub.shape[0] == 0:
                out[y] = np.array([], dtype=float)
                continue
            with localconverter(ro.default_converter + pandas2ri.converter):
                r_sub = ro.conversion.py2rpy(sub[covariates])
            u = pit_values(fit, r_sub, ro.FloatVector(sub[y].to_numpy(dtype=float)))
            out[y] = np.asarray(u, dtype=float)
        return out

    def sample(self, grid_df: pd.DataFrame, M: int = 512) -> Dict[str, List[np.ndarray]]:
        """Return dict[idp] -> list over grid rows of np.array shape (M,)."""
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_grid = ro.conversion.py2rpy(grid_df)
        out = {}
        for y, fit in self._fits.items():
            lst = sample_M(fit, r_grid, M=M)
            out[y] = [np.asarray(v, dtype=float) for v in lst]
        return out

