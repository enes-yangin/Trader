import numpy as np
from utils.config import FEATURES, INDICATORS


def volume_delta(df):
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    clv = ((df[FEATURES.close_col] - df["low"]) - (df["high"] - df[FEATURES.close_col])) / rng
    clv = clv.fillna(0)
    return clv * df["volume"]


def amihud(df, w=INDICATORS.amihud_window):
    ret = df[FEATURES.close_col].pct_change().abs()
    dollar_vol = (df[FEATURES.close_col] * df["volume"]).replace(0, np.nan)
    illiq = (ret / dollar_vol) * 1e9
    return illiq.rolling(w).mean()


def amihud_zscore(df, w=INDICATORS.amihud_window, zw=INDICATORS.amihud_zscore_window):
    raw = amihud(df, w=w)
    mu = raw.rolling(zw).mean()
    sd = raw.rolling(zw).std().replace(0, np.nan)
    return ((raw - mu) / sd).fillna(0)


def roll_spread(df, w=INDICATORS.roll_spread_window):
    dp = df[FEATURES.close_col].diff()
    cov = dp.rolling(w).apply(
        lambda x: np.cov(x[:-1], x[1:])[0, 1] if len(x) > 2 else 0.0,
        raw=True,
    )
    spread = 2 * np.sqrt(np.abs(cov.clip(upper=0)))
    return spread / df[FEATURES.close_col]


def kyle_lambda(df, w=INDICATORS.kyle_lambda_window):
    ret = df[FEATURES.close_col].pct_change()
    signed_vol = np.sign(ret) * df["volume"]
    vol_var = df["volume"].rolling(w).var().replace(0, np.nan)
    lam = ret.rolling(w).cov(signed_vol) / vol_var
    return lam.replace([np.inf, -np.inf], np.nan)


def vwap_dist(df, w=INDICATORS.vwap_window):
    tp = (df["high"] + df["low"] + df[FEATURES.close_col]) / 3
    pv = (tp * df["volume"]).rolling(w).sum()
    vv = df["volume"].rolling(w).sum().replace(0, np.nan)
    vwap = pv / vv
    return (df[FEATURES.close_col] - vwap) / vwap


def hl_range(df):
    return (df["high"] - df["low"]) / df[FEATURES.close_col]


def add_micro(df, vd_window=INDICATORS.volume_sma_window):
    df = df.copy()
    vd = volume_delta(df)
    vd_sma = vd.rolling(vd_window).mean()
    vd_std = vd.rolling(vd_window).std().replace(0, np.nan)
    df["vol_delta"] = ((vd - vd_sma) / vd_std).fillna(0)
    df["amihud_illiq"] = amihud_zscore(df)
    df["roll_spread"] = roll_spread(df)
    df["kyle_lambda"] = kyle_lambda(df)
    df["vwap_dist"] = vwap_dist(df)
    df["hl_range"] = hl_range(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df
