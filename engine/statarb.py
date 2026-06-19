"""Statistical-arbitrage primitives: cointegration testing and spread math.

The chain is Engle-Granger two-step:
  1. OLS-regress one leg on the other to get the hedge ratio (the long-run
     equilibrium relationship).
  2. Run an Augmented Dickey-Fuller test on the regression residual (the
     spread). If the spread is stationary, the legs are cointegrated and the
     spread is tradeable as a mean-reverting series.

We implement ADF from scratch (numpy/scipy only -- no statsmodels dependency)
so the test is auditable and the project stays lean. Critical values are the
standard asymptotic tables: MacKinnon for the plain ADF (constant, no trend)
and the Engle-Granger residual-based values for the cointegration step (more
negative, because the residual comes from an *estimated* relationship).
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

# Asymptotic critical values, model with constant and no trend.
# Plain unit-root test on an observed series (MacKinnon 2010).
_ADF_CRIT: Dict[float, float] = {0.01: -3.43, 0.05: -2.86, 0.10: -2.57}
# Residual-based test for a cointegrating regression with one regressor
# (Engle-Granger). Shifted more negative to account for the estimated beta.
_EG_CRIT: Dict[float, float] = {0.01: -3.90, 0.05: -3.34, 0.10: -3.04}


@dataclass
class ADFResult:
    """Outcome of an Augmented Dickey-Fuller test.

    `stat` is the t-statistic on the lagged-level coefficient; more negative
    means stronger evidence of stationarity (mean reversion). Compare against
    `crit` -- which table is used depends on whether this is a plain ADF or an
    Engle-Granger residual test."""
    stat: float
    used_lag: int
    nobs: int
    crit: Dict[float, float]

    def reject(self, alpha: float = 0.05) -> bool:
        """True if we reject the unit-root null at `alpha` (i.e. stationary)."""
        if alpha not in self.crit:
            raise ValueError(f"no critical value for alpha={alpha}; "
                             f"have {sorted(self.crit)}")
        return self.stat < self.crit[alpha]

    @property
    def is_stationary(self) -> bool:
        return self.reject(0.05)


@dataclass
class CointResult:
    """Engle-Granger cointegration outcome for legs (y, x)."""
    beta: float
    alpha: float
    adf: ADFResult
    spread: FloatArray

    @property
    def cointegrated(self) -> bool:
        return self.adf.is_stationary


def adf_test(y: npt.ArrayLike, max_lag: Optional[int] = None,
             crit: Optional[Dict[float, float]] = None) -> ADFResult:
    """Augmented Dickey-Fuller test with a constant term.

    Regression:  Δy_t = a + ρ·y_{t-1} + Σ_{i=1..p} γ_i·Δy_{t-i} + e_t
    The test statistic is ρ̂ / se(ρ̂). Under the unit-root null ρ = 0; a
    sufficiently negative statistic rejects it in favour of stationarity.

    `max_lag` defaults to the Schwert rule ⌊12·(n/100)^¼⌋, capped by length.
    """
    arr = np.asarray(y, dtype=np.float64)
    n = arr.size
    if n < 8:
        raise ValueError(f"ADF needs at least 8 observations, got {n}")
    dy = np.diff(arr)
    ndy = dy.size

    if max_lag is None:
        max_lag = int(np.floor(12.0 * (n / 100.0) ** 0.25))
    max_lag = max(0, min(max_lag, ndy // 2 - 2))

    lag = max_lag
    m = ndy - lag
    if m <= lag + 2:
        raise ValueError(f"series too short for ADF with lag={lag}")

    # Δy_t for t = lag .. ndy-1, with y_{t-1} = arr[t] (since dy[j]=arr[j+1]-arr[j]).
    target = dy[lag:]
    y_lag1 = arr[lag:ndy]
    cols = [np.ones(m), y_lag1]
    for i in range(1, lag + 1):
        cols.append(dy[lag - i:ndy - i])
    design = np.column_stack(cols)

    coef, _res, _rank, _sv = np.linalg.lstsq(design, target, rcond=None)
    resid = target - design @ coef
    dof = m - design.shape[1]
    if dof <= 0:
        raise ValueError("not enough degrees of freedom for ADF regression")
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.inv(design.T @ design)
    se = np.sqrt(np.diag(sigma2 * xtx_inv))
    rho_idx = 1  # column order: [const, y_{t-1}, Δy lags...]
    stat = float(coef[rho_idx] / se[rho_idx])
    return ADFResult(stat=stat, used_lag=lag, nobs=m,
                     crit=dict(crit if crit is not None else _ADF_CRIT))


def hedge_ratio(y: npt.ArrayLike, x: npt.ArrayLike) -> Tuple[float, float]:
    """OLS of y on x with intercept. Returns (beta, alpha) so that
    y ≈ alpha + beta·x is the estimated long-run relationship."""
    ya = np.asarray(y, dtype=np.float64)
    xa = np.asarray(x, dtype=np.float64)
    if ya.size != xa.size:
        raise ValueError(f"length mismatch: y={ya.size}, x={xa.size}")
    design = np.column_stack([np.ones(xa.size), xa])
    coef, _r, _rk, _sv = np.linalg.lstsq(design, ya, rcond=None)
    return float(coef[1]), float(coef[0])


def spread(y: npt.ArrayLike, x: npt.ArrayLike, beta: float,
           alpha: float = 0.0) -> FloatArray:
    """The cointegrating residual y - (alpha + beta·x). Mean-reverting iff the
    legs are cointegrated with these coefficients."""
    ya = np.asarray(y, dtype=np.float64)
    xa = np.asarray(x, dtype=np.float64)
    return ya - (alpha + beta * xa)


def zscore(s: npt.ArrayLike, mean: Optional[float] = None,
           std: Optional[float] = None) -> FloatArray:
    """Standardise a spread. Pass `mean`/`std` estimated on training data to
    avoid look-ahead when scoring out-of-sample bars."""
    sa = np.asarray(s, dtype=np.float64)
    mu = float(sa.mean()) if mean is None else mean
    sd = float(sa.std(ddof=0)) if std is None else std
    if sd == 0.0:
        sd = 1.0
    return (sa - mu) / sd


def engle_granger(y: npt.ArrayLike, x: npt.ArrayLike,
                  max_lag: Optional[int] = None) -> CointResult:
    """Two-step Engle-Granger cointegration test of legs (y, x).

    Fits the hedge ratio by OLS, then ADF-tests the residual spread using the
    Engle-Granger residual critical values."""
    beta, alpha = hedge_ratio(y, x)
    sp = spread(y, x, beta, alpha)
    adf = adf_test(sp, max_lag=max_lag, crit=_EG_CRIT)
    return CointResult(beta=beta, alpha=alpha, adf=adf, spread=sp)
