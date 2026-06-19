import pandas as pd
from data.news_features import daily_sentiment, merge_to_price


def _news_df():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    rows = []
    for i, d in enumerate(idx):
        sentiment = float(i)
        rows.append({
            "date": d, "title": f"news {i}", "text": f"text {i}",
            "source": "test", "sentiment": sentiment,
        })
    return pd.DataFrame(rows)


def test_daily_sentiment_shift_default_is_one():
    from utils.config import NEWS
    assert NEWS.sentiment_shift_days >= 1, (
        "Sentiment must be shifted forward by at least 1 day to avoid "
        "using same-day news that would not have been available at "
        "decision time (look-ahead bias)"
    )


def test_daily_sentiment_shift_moves_values_forward_one_day():
    news = _news_df()
    agg_noshift = daily_sentiment(news, shift=0)
    agg_shift = daily_sentiment(news, shift=1)

    d0 = news["date"].min()
    d1 = d0 + pd.Timedelta(days=1)

    val_at_d0_noshift = agg_noshift.loc[d0, "sentiment_avg"]
    val_at_d1_shifted = agg_shift.loc[d1, "sentiment_avg"]

    assert val_at_d1_shifted == val_at_d0_noshift, (
        "shift=1 must move day-d's sentiment to day-(d+1)'s feature row"
    )

    assert pd.isna(agg_shift.loc[d0, "sentiment_avg"]) or agg_shift.loc[d0, "sentiment_avg"] == 0


def test_merge_to_price_no_same_day_leak(synthetic_ohlcv):
    news = _news_df()
    agg_shift = daily_sentiment(news, shift=1)
    merged = merge_to_price(synthetic_ohlcv.copy(), agg_shift, fill=True)

    assert "sentiment_avg" in merged.columns
    assert len(merged) == len(synthetic_ohlcv)

    d0 = news["date"].dt.normalize().min()
    d1 = d0 + pd.Timedelta(days=1)

    if d0 in merged.index:
        assert merged.loc[d0, "sentiment_avg"] == 0, (
            "Day d0's price row must NOT see day d0's same-day news (look-ahead)"
        )
    if d1 in merged.index:
        agg_noshift = daily_sentiment(news, shift=0)
        expected = agg_noshift.loc[d0, "sentiment_avg"]
        assert merged.loc[d1, "sentiment_avg"] == expected, (
            f"Day d1 should carry day d0's sentiment ({expected}), "
            f"got {merged.loc[d1, 'sentiment_avg']}"
        )


def test_merge_to_price_fills_missing_with_zero(synthetic_ohlcv):
    empty_news = pd.DataFrame(columns=["sentiment_avg", "sentiment_std", "news_volume", "sentiment_trend"])
    merged = merge_to_price(synthetic_ohlcv.copy(), empty_news, fill=True)
    assert (merged["sentiment_avg"] == 0).all()
    assert (merged["news_volume"] == 0).all()
