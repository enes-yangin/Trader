"""Causal (leak-free) smoothing features: EMA and Savitzky-Golay.

Both methods here are deliberately one-sided. The Savitzky-Golay filter is
*centered* in its textbook form -- it fits a polynomial to a window straddling
each sample, which means the value at bar t depends on bars t+1..t+k. That is
future information and would leak straight into the model. We avoid it by
fitting the polynomial to the *trailing* window only and evaluating it at the
most recent bar (pos = window-1), so the estimate at bar t uses only bars
<= t. EMA is causal by construction (an exponentially weighted average of the
past) when computed recursively with adjust=False.

Every feature produced here is a price-normalised ratio, not a price level, so
it stays stationary and scale-invariant -- the same discipline as the rest of
data/indicators.py.
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_coeffs
from utils.config import FEATURES, INDICATORS


def _as_series(series):
    return series if isinstance(series, pd.Series) else pd.Series(np.asarray(series, dtype=float))


def causal_savgol(series, window=INDICATORS.savgol_window,
                  polyorder=INDICATORS.savgol_polyorder, deriv=0):
    """One-sided Savitzky-Golay applied at the trailing edge of each window.

    Fits a degree-`polyorder` polynomial to the last `window` samples and
    returns its value (deriv=0) or derivative (deriv=1, per-bar) at the most
    recent sample. Because the estimate at index i depends only on
    x[i-window+1 .. i], it carries no look-ahead: computing it on a truncated
    prefix gives identical values on the overlap.
    """
    if polyorder >= window:
        raise ValueError(f"polyorder ({polyorder}) must be < window ({window})")
    s = _as_series(series)
    coeffs = savgol_coeffs(window, polyorder, deriv=deriv, delta=1.0,
                           pos=window - 1, use="dot")
    return s.rolling(window).apply(lambda w: float(np.dot(coeffs, w)), raw=True)


def causal_ema(series, span=INDICATORS.ema_smooth_span):
    """Exponentially weighted moving average over past samples only."""
    return _as_series(series).ewm(span=span, adjust=False).mean()


def add_smoothing_features(df, window=INDICATORS.savgol_window,
                           polyorder=INDICATORS.savgol_polyorder,
                           span=INDICATORS.ema_smooth_span):
    """Add causal smoothing-derived features, all price-normalised.

    savgol_slope: trailing Savitzky-Golay first derivative / price -- a denoised
                  momentum estimate.
    ema_slope:    one-bar change of a causal EMA / price -- a smoothed momentum.
    """
    df = df.copy()
    close = df[FEATURES.close_col]
    sg_slope = causal_savgol(close, window=window, polyorder=polyorder, deriv=1)
    df["savgol_slope"] = sg_slope / close
    ema = causal_ema(close, span=span)
    df["ema_slope"] = ema.diff() / close
    return df
