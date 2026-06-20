"""Test Social Fetcher + Social Features (Feature 3)."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from data.social_fetcher import (
    XAdapter, RedditAdapter, fetch_social_posts, POST_COLUMNS,
)
from data.social_features import (
    social_metrics, add_social_features, SOCIAL_FEATURE_COLS,
)


class TestSocialAdapters:
    def test_x_adapter_name(self):
        adapter = XAdapter()
        assert adapter.name == "x"

    def test_reddit_adapter_name(self):
        adapter = RedditAdapter()
        assert adapter.name == "reddit"

    def test_x_adapter_sample_fallback(self):
        adapter = XAdapter(api_key="", allow_synthetic=True)  # Key yok → sample
        posts = adapter.fetch("BTC", since=datetime.utcnow() - timedelta(days=7))
        assert len(posts) > 0
        for p in posts:
            assert "datetime" in p
            assert "text" in p
            assert p["source"] == "x"

    def test_x_adapter_no_key_no_synthetic_returns_empty(self):
        """B2: Without API key and without allow_synthetic, return empty list."""
        adapter = XAdapter(api_key="", allow_synthetic=False)
        posts = adapter.fetch("BTC", since=datetime.utcnow() - timedelta(days=7))
        assert len(posts) == 0

    def test_reddit_adapter_sample_fallback(self):
        adapter = RedditAdapter(api_key="", allow_synthetic=True)
        posts = adapter.fetch("ETH", since=datetime.utcnow() - timedelta(days=3))
        assert len(posts) > 0

    def test_fetch_social_posts_returns_dataframe(self):
        df = fetch_social_posts("BTC", days=7, allow_synthetic=True)
        assert isinstance(df, pd.DataFrame)
        for col in POST_COLUMNS:
            assert col in df.columns
        assert len(df) > 0

    def test_fetch_social_posts_date_range(self):
        since = datetime.utcnow() - timedelta(days=7)
        df = fetch_social_posts("SOL", since=since, allow_synthetic=True)
        if len(df) > 0:
            assert df["datetime"].min() >= since


class TestSocialFeatures:
    @staticmethod
    def _make_price():
        idx = pd.date_range("2024-01-01", periods=60, freq="D")
        return pd.DataFrame({
            "open": 30000.0, "high": 31000.0, "low": 29000.0,
            "close": np.linspace(29000, 35000, 60),
            "volume": 1000.0,
        }, index=idx)

    def _make_posts(self):
        return fetch_social_posts("BTC", days=30, allow_synthetic=True)

    def test_social_metrics_returns_all_cols(self):
        price = self._make_price()
        posts = self._make_posts()
        metrics = social_metrics(posts, pd.to_datetime(price.index), shift_days=1)
        for col in SOCIAL_FEATURE_COLS:
            assert col in metrics.columns
        assert len(metrics) == 60

    def test_social_metrics_empty_posts_zero(self):
        price = self._make_price()
        empty = pd.DataFrame(columns=POST_COLUMNS)
        metrics = social_metrics(empty, pd.to_datetime(price.index))
        assert (metrics["social_volume_z"] == 0).all()
        assert (metrics["social_sentiment_avg"] == 0).all()

    def test_social_shift_no_lookahead(self):
        """Sosyal duygu shift'lenmeli: bugünkü post ancak yarınki bar'da görünür."""
        price = self._make_price()

        # Price tarihleriyle eşleşen post'lar oluştur
        from datetime import datetime
        posts = pd.DataFrame({
            "datetime": [
                datetime(2024, 1, 15, 10, 0), datetime(2024, 1, 15, 14, 0),
                datetime(2024, 1, 16, 9, 0), datetime(2024, 1, 20, 11, 0),
            ],
            "text": [
                "BTC looking very bullish today", "Might see a correction soon",
                "Just bought the dip on BTC", "BTC accumulation phase ending",
            ],
            "source": ["test"] * 4,
            "symbol": ["BTC"] * 4,
            "engagement": [10, 5, 20, 15],
        })

        metrics_no_shift = social_metrics(
            posts, pd.to_datetime(price.index), shift_days=0
        )
        metrics_shifted = social_metrics(
            posts, pd.to_datetime(price.index), shift_days=1
        )

        # Shift'li ve shiftsiz değerler farklı olmalı
        assert not (metrics_no_shift["social_sentiment_avg"].values
                    == metrics_shifted["social_sentiment_avg"].values).all()

    def test_add_social_features_integrates(self):
        price = self._make_price()
        posts = self._make_posts()
        result = add_social_features(price, posts, shift_days=1)
        for col in SOCIAL_FEATURE_COLS:
            assert col in result.columns
        assert len(result) == len(price)
        assert not result["social_volume_z"].isna().any()
