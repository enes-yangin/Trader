import numpy as np
import pandas as pd
from engine.news_analysis import sentiment_price_corr, top_sentiment_days, sentiment_buckets, report


def _df_with_sentiment(n=300, seed=1, with_news_volume=True):
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    close = 100 + np.cumsum(rng.randn(n))
    sentiment = rng.uniform(-1, 1, n)
    news_volume = rng.randint(0, 5, n) if with_news_volume else np.zeros(n)
    return pd.DataFrame({
        "close": close,
        "sentiment_avg": sentiment,
        "news_volume": news_volume,
    }, index=idx)


def test_sentiment_price_corr_returns_expected_shape():
    df = _df_with_sentiment()
    out = sentiment_price_corr(df)
    assert list(out["horizon"]) == [1, 3, 5, 10]
    assert {"sent_vs_return", "volume_vs_volatility"}.issubset(out.columns)
    for v in out["sent_vs_return"]:
        assert -1.0001 <= v <= 1.0001


def test_sentiment_price_corr_no_sentiment_column_returns_empty():
    df = pd.DataFrame({"close": [1, 2, 3]})
    out = sentiment_price_corr(df)
    assert out.empty


def test_top_sentiment_days_pos_and_neg():
    df = _df_with_sentiment()
    pos = top_sentiment_days(df, n=5, kind="pos")
    neg = top_sentiment_days(df, n=5, kind="neg")
    assert len(pos) == 5
    assert len(neg) == 5
    assert pos["sentiment_avg"].iloc[0] >= pos["sentiment_avg"].iloc[-1]
    assert neg["sentiment_avg"].iloc[0] <= neg["sentiment_avg"].iloc[-1]


def test_top_sentiment_days_no_sentiment_returns_empty():
    df = pd.DataFrame({"close": [1, 2, 3]})
    assert top_sentiment_days(df).empty


def test_sentiment_buckets_groups_correctly():
    df = _df_with_sentiment()
    out = sentiment_buckets(df)
    assert {"avg_fwd_5d", "win_rate", "count"}.issubset(out.columns)
    assert out["count"].sum() <= len(df)
    for w in out["win_rate"].dropna():
        assert 0.0 <= w <= 1.0


def test_sentiment_buckets_no_news_volume_returns_empty_or_no_crash():
    df = _df_with_sentiment(with_news_volume=False)
    out = sentiment_buckets(df)
    assert out["count"].sum() == 0


def test_report_no_crash_with_data():
    df = _df_with_sentiment()
    out = report(df)
    assert "NEWS-PRICE CORRELATION ANALYSIS" in out
    assert "Sentiment vs Forward Return" in out


def test_report_no_crash_without_sentiment():
    df = pd.DataFrame({"close": np.arange(50, dtype=float)})
    out = report(df)
    assert "NEWS-PRICE CORRELATION ANALYSIS" in out


def test_train_all_populates_news_analysis_when_sentiment_present(mock_ccxt_exchange):
    from unittest.mock import patch
    from engine.trainer import train_all

    with patch("ccxt.kraken", return_value=mock_ccxt_exchange):
        bundle = train_all("BTC/USDT", with_news=True, with_micro=True, allow_sample=True)

    assert "sentiment_avg" in bundle["df"].columns
    assert "news_analysis" in bundle
    assert "NEWS-PRICE CORRELATION ANALYSIS" in bundle["news_analysis"]


def test_train_all_no_news_analysis_key_without_news(mock_ccxt_exchange):
    from unittest.mock import patch
    from engine.trainer import train_all

    with patch("ccxt.kraken", return_value=mock_ccxt_exchange):
        bundle = train_all("BTC/USDT", with_news=False, with_micro=True)

    assert bundle.get("news_analysis") is None
