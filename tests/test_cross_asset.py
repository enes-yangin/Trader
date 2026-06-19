import numpy as np
import pandas as pd
from data.cross_asset import add_cross_asset_features, _align
from data.indicators import engineer, get_features
from utils.config import CROSS_ASSET, FEATURES


def _ref_df(idx, base_price, vol, seed):
    rng = np.random.RandomState(seed)
    p = base_price + np.cumsum(rng.randn(len(idx)) * base_price * 0.01)
    p = np.abs(p) + 1
    v = vol + np.abs(rng.randn(len(idx))) * vol * 0.2
    return pd.DataFrame({"close": p, "volume": v}, index=idx)


def _synthetic_ref_data(idx):
    return {
        "BTC/USDT": _ref_df(idx, 30000, 1_000_000, seed=1),
        "ETH/USDT": _ref_df(idx, 2000, 500_000, seed=2),
        "BNB/USDT": _ref_df(idx, 300, 100_000, seed=3),
        CROSS_ASSET.eth_btc_symbol: _ref_df(idx, 0.065, 1000, seed=4),
    }


def test_cross_asset_features_added(synthetic_ohlcv):
    ref_data = _synthetic_ref_data(synthetic_ohlcv.index)
    out = add_cross_asset_features(synthetic_ohlcv.copy(), "SOL/USDT", ref_data=ref_data)

    for col in FEATURES.cross_asset_feature_cols:
        assert col in out.columns
        assert np.all(np.isfinite(out[col].values)), f"{col} has non-finite values"


def test_cross_asset_features_no_ref_data_fallback_zero(synthetic_ohlcv):
    out = add_cross_asset_features(synthetic_ohlcv.copy(), "SOL/USDT", ref_data={})
    for col in FEATURES.cross_asset_feature_cols:
        assert col in out.columns
        assert (out[col] == 0).all()


def test_btc_vol_share_is_zscore_like(synthetic_ohlcv):
    ref_data = _synthetic_ref_data(synthetic_ohlcv.index)
    out = add_cross_asset_features(synthetic_ohlcv.copy(), "SOL/USDT", ref_data=ref_data)
    vals = out["btc_vol_share"].values
    assert abs(np.mean(vals)) < 3
    assert np.std(vals) < 10


def test_mkt_corr_in_valid_range(synthetic_ohlcv):
    ref_data = _synthetic_ref_data(synthetic_ohlcv.index)
    out = add_cross_asset_features(synthetic_ohlcv.copy(), "SOL/USDT", ref_data=ref_data)
    vals = out["mkt_corr"].values
    assert np.all(vals >= -1.0001)
    assert np.all(vals <= 1.0001)


def test_align_handles_missing_dates():
    idx1 = pd.date_range("2024-01-01", periods=10, freq="D")
    idx2 = pd.date_range("2024-01-03", periods=10, freq="D")
    s = pd.Series(np.arange(10, dtype=float), index=idx1)
    aligned = _align(s, idx2)
    assert len(aligned) == len(idx2)
    assert not aligned.isna().any()


def test_engineer_with_cross_asset_disabled_by_default(synthetic_ohlcv):
    df = engineer(synthetic_ohlcv, with_news=False, with_micro=True)
    X = get_features(df, with_news=False, with_micro=True)
    for col in FEATURES.cross_asset_feature_cols:
        assert col not in X.columns


def test_engineer_with_cross_asset_enabled_no_sym_falls_back(synthetic_ohlcv):
    df = engineer(synthetic_ohlcv, with_news=False, with_micro=True,
                  with_cross_asset=True, sym=None)
    X = get_features(df, with_news=False, with_micro=True, with_cross_asset=True)
    for col in FEATURES.cross_asset_feature_cols:
        assert col not in X.columns or col not in df.columns


def test_train_all_with_cross_asset_enabled(mock_ccxt_exchange):
    from unittest.mock import patch
    from engine.trainer import train_all
    from data.indicators import get_features

    with patch("ccxt.binance", return_value=mock_ccxt_exchange):
        bundle = train_all("SOL/USDT", with_news=False, with_cross_asset=True)

        assert bundle["with_cross_asset"] is True
        X = get_features(bundle["df"], with_news=False, with_micro=True, with_cross_asset=True)
        for col in FEATURES.cross_asset_feature_cols:
            assert col in X.columns

        n_expected = 12 + 6 + 3
        assert X.shape[1] == n_expected
        assert bundle["split"]["X_tr"].shape[1] == n_expected
