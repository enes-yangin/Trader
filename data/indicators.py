import numpy as np
import ta
from utils.config import FEATURES, INDICATORS, MODEL
from utils.types import FeatureSpec


def add_returns(df):
    df["ret_1d"] = df[FEATURES.close_col].pct_change(1)
    df["ret_5d"] = df[FEATURES.close_col].pct_change(5)
    df["ret_10d"] = df[FEATURES.close_col].pct_change(10)
    return df


def add_rsi(df, w=INDICATORS.rsi_window):
    df["rsi"] = ta.momentum.RSIIndicator(df[FEATURES.close_col], window=w).rsi() / 100.0
    return df


def add_macd(df, fw=INDICATORS.macd_fast, sw=INDICATORS.macd_slow, sig=INDICATORS.macd_signal):
    m = ta.trend.MACD(df[FEATURES.close_col], window_fast=fw, window_slow=sw, window_sign=sig)
    df["macd_diff"] = m.macd_diff() / df[FEATURES.close_col]
    return df


def add_bollinger(df, w=INDICATORS.bb_window, d=INDICATORS.bb_dev):
    bb = ta.volatility.BollingerBands(df[FEATURES.close_col], window=w, window_dev=d)
    df["bb_pband"] = bb.bollinger_pband()
    return df


def add_atr(df, w=INDICATORS.atr_window):
    atr = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df[FEATURES.close_col], window=w
    ).average_true_range()
    df["atr_pct"] = atr / df[FEATURES.close_col]
    return df


def add_volume_feats(df, w=INDICATORS.volume_sma_window):
    df["vol_chg"] = df["volume"].pct_change(1)
    vol_sma = df["volume"].rolling(window=w).mean()
    df["vol_ratio"] = df["volume"] / vol_sma
    return df


def add_ma_distances(df, ema_w=INDICATORS.ema_window, sma_w=INDICATORS.sma_window):
    ema = ta.trend.EMAIndicator(df[FEATURES.close_col], window=ema_w).ema_indicator()
    sma = ta.trend.SMAIndicator(df[FEATURES.close_col], window=sma_w).sma_indicator()
    df["ema20_dist"] = (df[FEATURES.close_col] - ema) / ema
    df["sma50_dist"] = (df[FEATURES.close_col] - sma) / sma
    df["ema_sma_dist"] = (ema - sma) / sma
    return df


def add_target(df, h=MODEL.pred_horizon):
    df[FEATURES.target_col] = df[FEATURES.close_col].pct_change(periods=h).shift(-h)
    return df


def clean_inf(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def engineer(df, horizon=MODEL.pred_horizon, sym=None, spec=None, with_news=False,
             sample_news=False, precomputed_news=False, with_micro=FEATURES.use_micro,
             with_cross_asset=FEATURES.use_cross_asset, cross_asset_ref_data=None):
    if spec is None:
        spec = FeatureSpec.from_bools(with_news, with_micro, with_cross_asset)

    df = df.copy()
    df = add_returns(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger(df)
    df = add_atr(df)
    df = add_volume_feats(df)
    df = add_ma_distances(df)
    if spec.micro:
        from data.microstructure import add_micro
        df = add_micro(df)
    if spec.cross_asset and sym is not None:
        from data.cross_asset import add_cross_asset_features
        df = add_cross_asset_features(df, sym, ref_data=cross_asset_ref_data,
                                       allow_sample=sample_news)
    if spec.news and not precomputed_news and sym is not None:
        from data.news_features import add_news_features
        df = add_news_features(df, sym, sample=sample_news)
    df = add_target(df, h=horizon)
    df = clean_inf(df)
    df.dropna(inplace=True)

    from data.validation import validate_features
    feat_cols = get_features(df, spec=spec).columns
    validate_features(df, feat_cols)
    return df


def get_features(df, spec=None, with_news=False, with_micro=FEATURES.use_micro,
                 with_cross_asset=FEATURES.use_cross_asset):
    if spec is None:
        spec = FeatureSpec.from_bools(with_news, with_micro, with_cross_asset)
    cols = spec.feature_columns()
    available = [c for c in cols if c in df.columns]
    return df[available]
