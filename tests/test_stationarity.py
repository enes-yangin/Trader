import numpy as np
from data.indicators import engineer, get_features


def test_features_are_stationary(synthetic_ohlcv):
    df = engineer(synthetic_ohlcv, with_news=False, with_micro=True)
    X = get_features(df, with_news=False, with_micro=True)

    for col in X.columns:
        vals = X[col].values
        assert np.all(np.isfinite(vals)), f"{col} contains NaN/inf"
        assert abs(np.mean(vals)) < 5, f"{col} mean far from 0: {np.mean(vals)}"
        assert np.std(vals) < 50, f"{col} std too large (likely price-scale): {np.std(vals)}"


def test_features_independent_of_price_scale(synthetic_ohlcv):
    df_low = synthetic_ohlcv.copy()
    df_high = synthetic_ohlcv.copy()
    price_cols = ["open", "high", "low", "close"]
    df_high[price_cols] = df_high[price_cols] * 1000

    eng_low = engineer(df_low, with_news=False, with_micro=True)
    eng_high = engineer(df_high, with_news=False, with_micro=True)

    X_low = get_features(eng_low, with_news=False, with_micro=True)
    X_high = get_features(eng_high, with_news=False, with_micro=True)

    n = min(len(X_low), len(X_high))
    for col in X_low.columns:
        a = X_low[col].values[:n]
        b = X_high[col].values[:n]
        assert np.allclose(a, b, atol=1e-4), (
            f"{col} depends on absolute price scale (not stationary)"
        )


def test_no_raw_price_columns_in_features(synthetic_ohlcv):
    df = engineer(synthetic_ohlcv, with_news=False, with_micro=True)
    X = get_features(df, with_news=False, with_micro=True)
    forbidden = {"close", "open", "high", "low", "bb_high", "bb_low",
                  "ema_20", "sma_50", "atr", "volume_sma", "macd", "macd_signal"}
    assert forbidden.isdisjoint(set(X.columns))


def test_cross_asset_features_are_bounded(synthetic_ohlcv):
    import pandas as pd
    from data.cross_asset import add_cross_asset_features
    from utils.config import CROSS_ASSET

    idx = synthetic_ohlcv.index
    rng = np.random.RandomState(9)
    ref = {CROSS_ASSET.eth_btc_symbol: pd.DataFrame({
        "close": np.abs(rng.randn(len(idx))) + 0.05,
        "volume": np.abs(rng.randn(len(idx))) * 1000 + 500,
    }, index=idx)}
    out = add_cross_asset_features(synthetic_ohlcv.copy(), "SOL/USDT", ref_data=ref)

    assert out["eth_btc_chg"].abs().max() <= 0.5 + 1e-9, (
        "eth_btc_chg must be clipped to keep it stationary and outlier-robust"
    )
    assert np.all(np.isfinite(out["eth_btc_chg"].values))
    assert np.all(out["mkt_corr"].abs() <= 1.0001)
