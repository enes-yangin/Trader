import pandas as pd
import ta
from utils.config import (
    FEATURE_COLS, NEWS_FEATURE_COLS, TARGET_COL,
    PRED_HORIZON, CLOSE_COL, USE_NEWS
)


def add_rsi(df, w=14):
    df["rsi"] = ta.momentum.RSIIndicator(df[CLOSE_COL], window=w).rsi()
    return df


def add_macd(df, fw=12, sw=26, sig=9):
    m = ta.trend.MACD(df[CLOSE_COL], window_fast=fw, window_slow=sw, window_sign=sig)
    df["macd"] = m.macd()
    df["macd_signal"] = m.macd_signal()
    df["macd_diff"] = m.macd_diff()
    return df


def add_bollinger(df, w=20, d=2):
    bb = ta.volatility.BollingerBands(df[CLOSE_COL], window=w, window_dev=d)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_pband"] = bb.bollinger_pband()
    return df


def add_atr(df, w=14):
    df["atr"] = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df[CLOSE_COL], window=w
    ).average_true_range()
    return df


def add_moving_avgs(df):
    df["ema_20"] = ta.trend.EMAIndicator(df[CLOSE_COL], window=20).ema_indicator()
    df["sma_50"] = ta.trend.SMAIndicator(df[CLOSE_COL], window=50).sma_indicator()
    df["volume_sma"] = df["volume"].rolling(window=20).mean()
    return df


def add_target(df, h=PRED_HORIZON):
    df[TARGET_COL] = df[CLOSE_COL].pct_change(periods=h).shift(-h)
    return df


def engineer(df, horizon=PRED_HORIZON, sym=None, with_news=False, sample_news=False):
    df = df.copy()
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger(df)
    df = add_atr(df)
    df = add_moving_avgs(df)
    if with_news and sym is not None:
        from data.news_features import add_news_features
        df = add_news_features(df, sym, sample=sample_news)
    df = add_target(df, h=horizon)
    df.dropna(inplace=True)
    return df


def get_features(df, with_news=False):
    cols = list(FEATURE_COLS)
    if with_news:
        cols = cols + NEWS_FEATURE_COLS
    available = [c for c in cols if c in df.columns]
    return df[available]
