import warnings
import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from tests.realistic_data import make_realistic_ohlcv
from data.indicators import engineer, get_features
from utils.types import FeatureSpec
from utils.config import REFERENCE
import data.reference_series as rs
from data.reference_series import (
    add_reference_features, generate_sample_series, FEATURE_COLS,
    _t_fear_greed, _t_dxy, SERIES,
)


def _price(n=120, seed=0):
    return make_realistic_ohlcv(n=n, seed=seed)


def _synth_ref(index):
    # Raw level series for all 7 sources (untransformed), injected as ref_data
    # so tests never touch the network.
    return {name: generate_sample_series(name, index) for name in SERIES}


@pytest.fixture(autouse=True)
def clean_reference_cache():
    rs.clear_cache()


# ------------------------------------------------------------------ #
# Leakage: no same-day                                                #
# ------------------------------------------------------------------ #

def test_reference_shift_prevents_same_day_leak():
    df = _price(80)
    idx = df.index
    # Distinct daily fear&greed levels so we can pinpoint which day leaks.
    fg = pd.Series(np.linspace(10.0, 90.0, len(idx)), index=idx)
    out = add_reference_features(df, "BTC/USDT", ref_data={"fear_greed": fg}, shift=1)

    d = 40
    same_day = (fg.iloc[d] - 50.0) / 50.0
    prev_day = (fg.iloc[d - 1] - 50.0) / 50.0
    got = out["fear_greed"].iloc[d]
    assert np.isclose(got, prev_day), "row d must carry day d-1's value (shift=1)"
    assert not np.isclose(got, same_day), "row d must NOT see same-day value (leak)"


def test_reference_default_shift_is_at_least_one():
    assert REFERENCE.shift_days >= 1, "shift_days < 1 would allow same-day leak"


# ------------------------------------------------------------------ #
# Stationarity + price-scale independence                             #
# ------------------------------------------------------------------ #

def test_reference_features_are_price_scale_independent():
    df_low = _price(150, seed=1)
    df_high = df_low.copy()
    df_high[["open", "high", "low", "close"]] *= 1000.0
    ref = _synth_ref(df_low.index)
    low = add_reference_features(df_low, "BTC/USDT", ref_data=ref)
    high = add_reference_features(df_high, "BTC/USDT", ref_data=ref)
    for col in FEATURE_COLS:
        assert np.allclose(low[col].values, high[col].values, atol=1e-9), (
            f"{col} must not depend on price scale"
        )


def test_reference_features_are_stationary_and_bounded():
    df = _price(200, seed=2)
    out = add_reference_features(df, "BTC/USDT", ref_data=_synth_ref(df.index))
    for col in FEATURE_COLS:
        v = out[col].values
        assert np.all(np.isfinite(v)), f"{col} has NaN/inf"
        assert abs(np.mean(v)) < 5, f"{col} mean far from 0"
        assert np.std(v) < 50, f"{col} std too large"
    assert out["fear_greed"].abs().max() <= 1.0 + 1e-9
    assert out["funding_rate"].abs().max() <= REFERENCE.funding_clip + 1e-9


# ------------------------------------------------------------------ #
# Transforms                                                          #
# ------------------------------------------------------------------ #

def test_fear_greed_of_neutral_is_zero():
    s = pd.Series(50.0, index=pd.date_range("2022-01-01", periods=20, freq="D"))
    assert np.allclose(_t_fear_greed(s).values, 0.0)


def test_dxy_change_is_clipped():
    # A huge jump must be clipped to the configured bound.
    s = pd.Series([100.0, 100.0, 1000.0, 1000.0],
                  index=pd.date_range("2022-01-01", periods=4, freq="D"))
    out = _t_dxy(s)
    assert out.abs().max() <= REFERENCE.dxy_clip + 1e-9


# ------------------------------------------------------------------ #
# Fetch fallback (hermetic: patch the network boundary)               #
# ------------------------------------------------------------------ #

def test_fetch_falls_back_to_synthetic(monkeypatch):
    monkeypatch.setattr(rs, "_get_json", lambda url: None)
    monkeypatch.setattr(rs, "_fetch_ccxt_history", lambda sym, index, kind: None)
    df = _price(120)
    out = add_reference_features(df, "BTC/USDT", ref_data=None, allow_sample=True)
    for col in FEATURE_COLS:
        assert out[col].notna().all(), f"{col} should be synthetic-filled, not NaN"
    # At least one series is non-constant (synthetic data actually populated).
    assert any(out[col].std() > 0 for col in FEATURE_COLS)


def test_fetch_failure_without_sample_yields_zeros(monkeypatch):
    monkeypatch.setattr(rs, "_get_json", lambda url: None)
    monkeypatch.setattr(rs, "_fetch_ccxt_history", lambda sym, index, kind: None)
    df = _price(120)
    out = add_reference_features(df, "BTC/USDT", ref_data=None, allow_sample=False)
    for col in FEATURE_COLS:
        assert (out[col] == 0.0).all(), f"{col} should degrade to 0.0, not raise"


def test_onchain_cache_is_symbol_specific(monkeypatch):
    # On-chain is per-asset (CoinMetrics keys on the base symbol); the cache must
    # not hand BTC's on-chain series to ETH. Regression for cross-contamination.
    idx = _price(60).index

    def fake_cm(sym, index, metric):
        val = 100.0 if sym.split("/")[0] == "BTC" else 999.0
        return pd.Series(val, index=pd.to_datetime(index).normalize())

    monkeypatch.setattr(rs, "_fetch_coinmetrics", fake_cm)
    btc = rs.fetch_reference_series("onchain_z", "BTC/USDT", idx)
    eth = rs.fetch_reference_series("onchain_z", "ETH/USDT", idx)
    assert btc.iloc[0] == 100.0
    assert eth.iloc[0] == 999.0, "ETH must not reuse BTC's cached on-chain data"
    assert "onchain_z:BTC/USDT" in rs._series_cache
    assert "onchain_z:ETH/USDT" in rs._series_cache


def test_synthetic_not_cached_across_date_ranges(monkeypatch):
    # Synthetic fallback is index-dependent, so it must be recomputed per call
    # (not cached by name) -- otherwise a second df with a different date range
    # gets the first df's series. Regression for the index-cache bug.
    monkeypatch.setattr(rs, "_get_json", lambda url: None)
    monkeypatch.setattr(rs, "_fetch_ccxt_history", lambda sym, index, kind: None)
    idx_a = pd.date_range("2020-01-01", periods=60, freq="D")
    idx_b = pd.date_range("2023-01-01", periods=60, freq="D")
    a = rs.fetch_reference_series("dxy_chg", "BTC/USDT", idx_a, allow_sample=True)
    b = rs.fetch_reference_series("dxy_chg", "BTC/USDT", idx_b, allow_sample=True)
    assert a.index.max() >= idx_a.max()
    assert b.index.max() >= idx_b.max()
    assert b.index.max() > a.index.max(), "second range must get its own series"
    # Only an empty failure-marker may be cached -- never the synthetic series,
    # which is index-dependent and would contaminate other date ranges.
    # B7: Cache now stores (series, timestamp) tuples.
    entry = rs._series_cache.get("dxy_chg")
    cached = entry[0] if entry is not None else None
    assert cached is None or cached.empty, "synthetic series must not be cached"


# ------------------------------------------------------------------ #
# Registration / toggle through engineer + get_features               #
# ------------------------------------------------------------------ #

def test_reference_columns_included_only_when_enabled():
    df = _price(300)
    ref = _synth_ref(df.index)
    eng = engineer(df, with_news=False, with_micro=False, with_reference=True,
                   reference_ref_data=ref)
    on = get_features(eng, spec=FeatureSpec(micro=False, reference=True))
    for col in FEATURE_COLS:
        assert col in on.columns
    off = get_features(eng, spec=FeatureSpec(micro=False, reference=False))
    for col in FEATURE_COLS:
        assert col not in off.columns


def test_reference_disabled_by_default():
    df = engineer(_price(300), with_news=False, with_micro=True, with_reference=False)
    X = get_features(df, spec=FeatureSpec(micro=True, reference=False))
    for col in FEATURE_COLS:
        assert col not in X.columns


def test_reference_features_survive_engineer_stationarity():
    # End-to-end: reference features pass the same scale-independence contract
    # through the full engineer() pipeline.
    df_low = _price(300, seed=3)
    df_high = df_low.copy()
    df_high[["open", "high", "low", "close"]] *= 1000.0
    ref = _synth_ref(df_low.index)
    el = engineer(df_low, with_news=False, with_micro=False, with_reference=True,
                  reference_ref_data=ref)
    eh = engineer(df_high, with_news=False, with_micro=False, with_reference=True,
                  reference_ref_data=ref)
    spec = FeatureSpec(micro=False, reference=True)
    Xl = get_features(el, spec=spec)
    Xh = get_features(eh, spec=spec)
    n = min(len(Xl), len(Xh))
    for col in FEATURE_COLS:
        assert np.allclose(Xl[col].values[:n], Xh[col].values[:n], atol=1e-6)
