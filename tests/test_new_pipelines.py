"""Integration tests for new feature pipelines: Orderbook, Macro Events, and Social Sentiment."""
import pytest
import pandas as pd
import numpy as np
from data.indicators import engineer
from utils.types import FeatureSpec
from data.event_features import add_event_features
from data.social_features import add_social_features

def test_pipeline_integration_and_leak_prevention(synthetic_ohlcv):
    # Set attrs
    df = synthetic_ohlcv.copy()
    df.attrs["symbol"] = "BTC/USDT"
    
    # Create a spec with orderbook, macro_events, and social
    spec = FeatureSpec(
        smooth=False,
        micro=False,
        cross_asset=False,
        reference=False,
        news=False,
        orderbook=True,
        macro_events=True,
        social=True
    )
    
    # Run engineer
    engineered_df = engineer(df, spec=spec, sample_news=True)
    
    # Assert columns exist
    expected_cols = [
        "depth_imbalance", "wall_persistence", "microprice_drift", "spread_regime",
        "bars_to_next_event", "last_event_surprise", "event_importance", "event_day_flag",
        "social_volume_z", "social_sentiment_avg", "social_polarity_shift"
    ]
    for col in expected_cols:
        assert col in engineered_df.columns
        assert not engineered_df[col].isna().any()

def test_causal_integrity_leaks():
    # We will pass specific custom mock inputs for events and social posts to verify causality.
    # 1. Macro Calendar
    # Event on day 10 with actual = 5.0, forecast = 2.0 (surprise = 3.0)
    # Event on day 20 with actual = 10.0, forecast = 4.0 (surprise = 6.0)
    dates = pd.date_range("2019-01-01", periods=30, freq="D")
    price_df = pd.DataFrame({
        "open": 100.0, "high": 105.0, "low": 95.0, "close": 100.0, "volume": 1000.0, "volume_sma": 1000.0
    }, index=dates)
    price_df.attrs["symbol"] = "BTC/USDT"
    
    calendar = pd.DataFrame({
        "date": [dates[10], dates[20]],
        "name": ["FOMC", "CPI"],
        "country": ["US", "US"],
        "importance": [3, 2],
        "forecast": [2.0, 4.0],
        "actual": [5.0, 10.0],
        "description": ["FOMC test", "CPI test"]
    })
    
    # 2. Social Posts
    # Posts on day 15
    posts_df = pd.DataFrame({
        "datetime": [dates[15] + pd.Timedelta("10h")],
        "text": ["BTC is super bullish today!"],
        "source": ["reddit"],
        "symbol": ["BTC"],
        "engagement": [10]
    })
    
    event_features_df = add_event_features(price_df, calendar=calendar, shift_surprise=1)
    
    # Assert surprise is 0 on day 10 (event day), and only visible on day 11
    assert event_features_df.loc[dates[10], "last_event_surprise"] == 0.0
    assert event_features_df.loc[dates[11], "last_event_surprise"] == 3.0
    
    # Assert bars_to_next_event is forward-looking but not leaky about surprise
    assert event_features_df.loc[dates[5], "bars_to_next_event"] == 5
    assert event_features_df.loc[dates[10], "bars_to_next_event"] == 0
    
    social_features_df = add_social_features(price_df, posts_df=posts_df, shift_days=1)
    # Post on day 15. Assert sentiment is 0 on day 15, and non-zero on day 16
    assert social_features_df.loc[dates[15], "social_sentiment_avg"] == 0.0
    assert social_features_df.loc[dates[16], "social_sentiment_avg"] > 0.0
