import warnings
import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from tests.realistic_data import make_realistic_ohlcv
from data.indicators import engineer, get_features
from data.smoothing import causal_savgol, causal_ema, add_smoothing_features
from utils.types import FeatureSpec


# ------------------------------------------------------------------ #
# Causality / no-lookahead                                            #
# ------------------------------------------------------------------ #

def test_causal_savgol_truncation_invariant():
    # The value at index i must depend only on x[i-window+1 .. i], so computing
    # on a truncated prefix yields identical values on the overlap. This is the
    # core proof that the (normally centered) Savitzky-Golay filter is leak-free.
    rng = np.random.default_rng(0)
    x = pd.Series(np.cumsum(rng.standard_normal(500)) + 100.0)
    k = 300
    full = causal_savgol(x, window=11, polyorder=3, deriv=1).values
    trunc = causal_savgol(x.iloc[:k], window=11, polyorder=3, deriv=1).values
    assert np.allclose(full[10:k], trunc[10:k], atol=1e-9, equal_nan=True)


def test_causal_ema_truncation_invariant():
    rng = np.random.default_rng(1)
    x = pd.Series(np.cumsum(rng.standard_normal(500)) + 100.0)
    k = 300
    full = causal_ema(x, span=10).values
    trunc = causal_ema(x.iloc[:k], span=10).values
    assert np.allclose(full[:k], trunc[:k], atol=1e-9)


def test_savgol_slope_no_lookahead_end_to_end():
    # Same property through the full engineer() pipeline: smoothing features at a
    # given date are identical whether computed on the full series or a prefix.
    df = make_realistic_ohlcv(n=900, seed=0)
    full = engineer(df, with_news=False, with_micro=False, with_smoothing=True)
    trunc = engineer(df.iloc[:600], with_news=False, with_micro=False,
                     with_smoothing=True)
    common = full.index.intersection(trunc.index)
    assert len(common) > 100
    for col in ("savgol_slope", "ema_slope"):
        a = full.loc[common, col].values
        b = trunc.loc[common, col].values
        assert np.allclose(a, b, atol=1e-9), f"{col} leaks future data"


# ------------------------------------------------------------------ #
# Correctness                                                         #
# ------------------------------------------------------------------ #

def test_causal_savgol_slope_recovers_linear_trend():
    # On a clean ramp x = a + b*t, the trailing-window 1st derivative must be b.
    b = 0.37
    x = pd.Series(5.0 + b * np.arange(200, dtype=float))
    slope = causal_savgol(x, window=11, polyorder=2, deriv=1).values
    assert np.allclose(slope[11:], b, atol=1e-8)


def test_causal_savgol_level_recovers_value():
    # deriv=0 on a ramp returns the value at the current (most recent) bar.
    x = pd.Series(np.arange(100, dtype=float) * 2.0)
    smoothed = causal_savgol(x, window=11, polyorder=2, deriv=0).values
    assert np.allclose(smoothed[11:], x.values[11:], atol=1e-8)


def test_causal_savgol_rejects_high_polyorder():
    with pytest.raises(ValueError):
        causal_savgol(pd.Series(np.arange(50.0)), window=5, polyorder=5)


# ------------------------------------------------------------------ #
# Stationarity + price-scale independence                             #
# ------------------------------------------------------------------ #

def test_smoothing_features_scale_invariant():
    df_low = make_realistic_ohlcv(n=900, seed=0)
    df_high = df_low.copy()
    df_high[["open", "high", "low", "close"]] *= 1000.0
    spec = FeatureSpec(micro=False, smooth=True)
    el = engineer(df_low, with_news=False, with_micro=False, with_smoothing=True)
    eh = engineer(df_high, with_news=False, with_micro=False, with_smoothing=True)
    Xl = get_features(el, spec=spec)
    Xh = get_features(eh, spec=spec)
    n = min(len(Xl), len(Xh))
    for col in ("savgol_slope", "ema_slope"):
        assert np.allclose(Xl[col].values[:n], Xh[col].values[:n], atol=1e-6), (
            f"{col} depends on absolute price scale"
        )


def test_smoothing_features_are_stationary():
    df = engineer(make_realistic_ohlcv(n=900, seed=2), with_news=False,
                  with_micro=False, with_smoothing=True)
    X = get_features(df, spec=FeatureSpec(micro=False, smooth=True))
    for col in ("savgol_slope", "ema_slope"):
        vals = X[col].values
        assert np.all(np.isfinite(vals)), f"{col} has NaN/inf"
        assert abs(np.mean(vals)) < 5, f"{col} mean far from 0"
        assert np.std(vals) < 50, f"{col} std too large (price-scale leak?)"


# ------------------------------------------------------------------ #
# Feature registration / toggle                                       #
# ------------------------------------------------------------------ #

def test_smoothing_columns_included_only_when_enabled():
    df = engineer(make_realistic_ohlcv(n=500, seed=3), with_news=False,
                  with_micro=False, with_smoothing=True)
    on = get_features(df, spec=FeatureSpec(micro=False, smooth=True))
    assert "savgol_slope" in on.columns and "ema_slope" in on.columns
    off = get_features(df, spec=FeatureSpec(micro=False, smooth=False))
    assert "savgol_slope" not in off.columns
    assert "ema_slope" not in off.columns


def test_smoothing_disabled_by_default():
    # Default engineer() (smoothing off) must not add the columns to features.
    df = engineer(make_realistic_ohlcv(n=500, seed=4), with_news=False,
                  with_micro=True, with_smoothing=False)
    X = get_features(df, spec=FeatureSpec(micro=True, smooth=False))
    assert "savgol_slope" not in X.columns
    assert "ema_slope" not in X.columns


def test_add_smoothing_features_writes_both_columns():
    df = make_realistic_ohlcv(n=200, seed=5)
    out = add_smoothing_features(df.copy())
    assert "savgol_slope" in out.columns
    assert "ema_slope" in out.columns
