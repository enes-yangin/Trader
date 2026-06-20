import numpy as np
import pandas as pd
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


def add_rsi_divergence(df, rsi_col="rsi", k=3, hold_bars=5):
    """
    Computes causal RSI divergence.
    A peak/trough is confirmed with a lag of k bars (using a window of 2k+1 bars).
    Saves binary signal indicator columns 'rsi_bull_div' and 'rsi_bear_div'.
    """
    df = df.copy()
    n = len(df)
    
    bull_div = np.zeros(n)
    bear_div = np.zeros(n)
    
    last_peaks = []
    last_troughs = []
    
    close_vals = df[FEATURES.close_col].values
    high_vals = df["high"].values
    low_vals = df["low"].values
    rsi_vals = df[rsi_col].values
    
    for t in range(2 * k, n):
        # 1. Bearish Divergence (peak detection)
        window_highs = high_vals[t - 2 * k : t + 1]
        window_rsi = rsi_vals[t - 2 * k : t + 1]
        if np.isnan(window_highs).any() or np.isnan(window_rsi).any():
            continue
            
        if np.argmax(window_highs) == k:
            peak_idx = t - k
            peak_price = high_vals[peak_idx]
            peak_rsi = rsi_vals[peak_idx]
            
            last_peaks.append((peak_price, peak_rsi, peak_idx))
            if len(last_peaks) > 2:
                last_peaks.pop(0)
                
            if len(last_peaks) == 2:
                p2_price, p2_rsi, _ = last_peaks[0]
                p1_price, p1_rsi, _ = last_peaks[1]
                if p1_price > p2_price and p1_rsi < p2_rsi:
                    bear_div[t] = 1.0
                    
        # 2. Bullish Divergence (trough detection)
        window_lows = low_vals[t - 2 * k : t + 1]
        if np.isnan(window_lows).any():
            continue
            
        if np.argmin(window_lows) == k:
            trough_idx = t - k
            trough_price = low_vals[trough_idx]
            trough_rsi = rsi_vals[trough_idx]
            
            last_troughs.append((trough_price, trough_rsi, trough_idx))
            if len(last_troughs) > 2:
                last_troughs.pop(0)
                
            if len(last_troughs) == 2:
                tr2_price, tr2_rsi, _ = last_troughs[0]
                tr1_price, tr1_rsi, _ = last_troughs[1]
                if tr1_price < tr2_price and tr1_rsi > tr2_rsi:
                    bull_div[t] = 1.0
                    
    # Hold the signal for `hold_bars`
    if hold_bars > 1:
        bull_series = pd.Series(bull_div)
        bear_series = pd.Series(bear_div)
        df["rsi_bull_div"] = bull_series.rolling(hold_bars, min_periods=1).max().values
        df["rsi_bear_div"] = bear_series.rolling(hold_bars, min_periods=1).max().values
    else:
        df["rsi_bull_div"] = bull_div
        df["rsi_bear_div"] = bear_div
        
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
             with_cross_asset=FEATURES.use_cross_asset, cross_asset_ref_data=None,
             with_smoothing=FEATURES.use_smoothing,
             with_reference=FEATURES.use_reference, reference_ref_data=None):
    if spec is None:
        spec = FeatureSpec.from_bools(with_news, with_micro, with_cross_asset,
                                       with_smoothing, with_reference)

    df = df.copy()
    df = add_returns(df)
    df = add_rsi(df)
    df = add_rsi_divergence(df)
    df = add_macd(df)
    df = add_bollinger(df)
    df = add_atr(df)
    df = add_volume_feats(df)
    df = add_ma_distances(df)
    if spec.smooth:
        from data.smoothing import add_smoothing_features
        df = add_smoothing_features(df)
    if spec.micro:
        from data.microstructure import add_micro
        df = add_micro(df)
    if spec.cross_asset and sym is not None:
        from data.cross_asset import add_cross_asset_features
        df = add_cross_asset_features(df, sym, ref_data=cross_asset_ref_data,
                                       allow_sample=sample_news)
    if spec.reference:
        from data.reference_series import add_reference_features
        # Real data only: if a live source is unavailable the feature stays 0,
        # never synthetic (no fabricated macro/derivatives series in the pipeline).
        df = add_reference_features(df, sym or "", ref_data=reference_ref_data,
                                     allow_sample=False)
    if spec.orderbook:
        from data.orderbook_features import add_orderbook_features
        from data.orderbook_recorder import OrderBookRecorder
        recorder = OrderBookRecorder(sym or "BTC/USDT")
        ob_hist = recorder.load_history()
        df = add_orderbook_features(df, ob_hist, allow_sample=sample_news)
    if spec.macro_events:
        from data.event_features import add_event_features
        df = add_event_features(df, allow_sample=sample_news)
    if spec.social:
        from data.social_features import add_social_features
        df = add_social_features(df, allow_sample=sample_news)
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
