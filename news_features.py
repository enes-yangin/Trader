import pandas as pd
import numpy as np
from data.news_fetcher import fetch_news, generate_sample_news
from data.sentiment import score_batch, init


def score_news_df(news_df):
    if news_df is None or len(news_df) == 0:
        return news_df
    texts = news_df["text"].fillna("").tolist()
    scores = score_batch(texts)
    news_df = news_df.copy()
    news_df["sentiment"] = scores
    return news_df


def daily_sentiment(news_df, shift=1):
    if news_df is None or len(news_df) == 0:
        return pd.DataFrame(columns=["sentiment_avg", "sentiment_std", "news_volume"])
    g = news_df.groupby("date")
    agg = pd.DataFrame({
        "sentiment_avg": g["sentiment"].mean(),
        "sentiment_std": g["sentiment"].std().fillna(0),
        "news_volume": g["sentiment"].count(),
    })
    agg = agg.sort_index()
    agg["sentiment_trend"] = agg["sentiment_avg"].diff().fillna(0)
    if shift > 0:
        full_idx = pd.date_range(agg.index.min(), agg.index.max(), freq="D")
        agg = agg.reindex(full_idx)
        agg["news_volume"] = agg["news_volume"].fillna(0)
        agg["sentiment_avg"] = agg["sentiment_avg"].fillna(0)
        agg["sentiment_std"] = agg["sentiment_std"].fillna(0)
        agg["sentiment_trend"] = agg["sentiment_trend"].fillna(0)
        agg = agg.shift(shift)
    return agg


def merge_to_price(price_df, daily_sent, fill=True):
    attrs = dict(price_df.attrs)
    df = price_df.copy()
    pidx = pd.to_datetime(df.index).normalize()
    df["_d"] = pidx
    ds = daily_sent.copy()
    ds.index = pd.to_datetime(ds.index).normalize()
    df = df.merge(ds, left_on="_d", right_index=True, how="left")
    df.drop(columns=["_d"], inplace=True)
    cols = ["sentiment_avg", "sentiment_std", "news_volume", "sentiment_trend"]
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
    if fill:
        df["news_volume"] = df["news_volume"].fillna(0)
        df["sentiment_avg"] = df["sentiment_avg"].fillna(0)
        df["sentiment_std"] = df["sentiment_std"].fillna(0)
        df["sentiment_trend"] = df["sentiment_trend"].fillna(0)
    df.attrs.update(attrs)
    return df


def add_news_features(price_df, sym, src="auto", days=30, sample=False):
    init()
    span = len(price_df) + 5
    if sample:
        base = sym.split("/")[0]
        news = generate_sample_news(base, n=span)
    else:
        news = fetch_news(sym, src=src, days=days)
        if len(news) == 0:
            base = sym.split("/")[0]
            news = generate_sample_news(base, n=span)
    news = score_news_df(news)
    ds = daily_sentiment(news)
    merged = merge_to_price(price_df, ds)
    cov = (merged["news_volume"] > 0).mean()
    if not sample and cov < 0.3:
        import warnings
        warnings.warn(
            f"News coverage low ({cov:.0%}). RSS only covers recent days; "
            f"set NEWSAPI_KEY for historical coverage in training.",
            stacklevel=2,
        )
    return merged
