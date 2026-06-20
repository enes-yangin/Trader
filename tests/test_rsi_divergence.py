import numpy as np
import pandas as pd
import pytest
from data.indicators import add_rsi_divergence, engineer
from utils.config import FEATURES
from utils.types import FeatureSpec


def test_rsi_divergence_columns_added():
    n = 50
    df = pd.DataFrame({
        "open": np.full(n, 10.0),
        "high": np.full(n, 10.0),
        "low": np.full(n, 10.0),
        "close": np.full(n, 10.0),
        "volume": np.full(n, 1000.0),
        "rsi": np.linspace(0.3, 0.7, n),
    }, index=pd.date_range("2024-01-01", periods=n, freq="D"))

    out = add_rsi_divergence(df, rsi_col="rsi", k=2, hold_bars=3)
    assert "rsi_bull_div" in out.columns
    assert "rsi_bear_div" in out.columns


def test_rsi_divergence_is_causal():
    # Make a mock series that should trigger a divergence
    n = 60
    # Price peaks: at t=20 (high=15) and t=40 (high=20) -> Higher High
    # RSI peaks: at t=20 (rsi=0.8) and t=40 (rsi=0.6) -> Lower High (Bearish Divergence)
    prices = np.full(n, 10.0)
    prices[20] = 15.0
    prices[40] = 20.0
    
    rsi = np.full(n, 0.5)
    rsi[20] = 0.8
    rsi[40] = 0.6
    
    df = pd.DataFrame({
        "open": prices,
        "high": prices,
        "low": prices,
        "close": prices,
        "volume": np.full(n, 1000.0),
        "rsi": rsi,
    }, index=pd.date_range("2024-01-01", periods=n, freq="D"))
    
    # Run full
    out_full = add_rsi_divergence(df, rsi_col="rsi", k=3, hold_bars=5)
    
    # Run on subset up to t=45 (which is after confirmation at t=40+3=43)
    subset = df.iloc[:45]
    out_subset = add_rsi_divergence(subset, rsi_col="rsi", k=3, hold_bars=5)
    
    # Causal check: subset output at t=44 must be equal to full output at t=44
    assert out_subset["rsi_bear_div"].iloc[44] == out_full["rsi_bear_div"].iloc[44]
    assert out_subset["rsi_bull_div"].iloc[44] == out_full["rsi_bull_div"].iloc[44]


def test_bullish_rsi_divergence_detected():
    n = 60
    # Troughs: t=15 (low=8, rsi=0.2) and t=35 (low=6, rsi=0.3)
    # Price: 8 -> 6 (Lower Low)
    # RSI: 0.2 -> 0.3 (Higher Low) => Bullish Divergence
    lows = np.full(n, 10.0)
    lows[15] = 8.0
    lows[35] = 6.0
    
    rsi = np.full(n, 0.5)
    rsi[15] = 0.2
    rsi[35] = 0.3
    
    df = pd.DataFrame({
        "open": lows,
        "high": np.full(n, 12.0),
        "low": lows,
        "close": lows,
        "volume": np.full(n, 1000.0),
        "rsi": rsi,
    }, index=pd.date_range("2024-01-01", periods=n, freq="D"))
    
    # k=2 means confirmed at trough + 2 = 37
    out = add_rsi_divergence(df, rsi_col="rsi", k=2, hold_bars=5)
    
    # At t=37, trough is confirmed, so bull_div should be 1.0
    assert out["rsi_bull_div"].iloc[37] == 1.0
    # At t=30, trough 35 is not yet reached/confirmed, so it should be 0.0
    assert out["rsi_bull_div"].iloc[30] == 0.0


def test_bearish_rsi_divergence_detected():
    n = 60
    # Peaks: t=15 (high=12, rsi=0.8) and t=35 (high=14, rsi=0.7)
    # Price: 12 -> 14 (Higher High)
    # RSI: 0.8 -> 0.7 (Lower High) => Bearish Divergence
    highs = np.full(n, 10.0)
    highs[15] = 12.0
    highs[35] = 14.0
    
    rsi = np.full(n, 0.5)
    rsi[15] = 0.8
    rsi[35] = 0.7
    
    df = pd.DataFrame({
        "open": highs,
        "high": highs,
        "low": np.full(n, 8.0),
        "close": highs,
        "volume": np.full(n, 1000.0),
        "rsi": rsi,
    }, index=pd.date_range("2024-01-01", periods=n, freq="D"))
    
    # k=2 means confirmed at peak + 2 = 37
    out = add_rsi_divergence(df, rsi_col="rsi", k=2, hold_bars=5)
    
    # At t=37, peak is confirmed, so bear_div should be 1.0
    assert out["rsi_bear_div"].iloc[37] == 1.0
    # At t=30, peak 35 is not yet reached/confirmed, so it should be 0.0
    assert out["rsi_bear_div"].iloc[30] == 0.0
