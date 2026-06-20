"""
Sosyal Duygu Özellikleri — sosyal medya post'larından bar-başı sinyaller.

Mevcut FinBERT hattını (data/sentiment.py) aynen yeniden kullanır.

Özellikler:
  - social_volume_z: mention patlaması = "fiyattan önce hacim" sinyali
  - social_sentiment_avg: bar başına ortalama duygu skoru
  - social_polarity_shift: duygu yönündeki keskin değişim

shift(≥1) leak-free: bar-d'nin post'ları ancak bar-(d+1)'de görünür.
"""

from __future__ import annotations

from typing import Optional
from datetime import timedelta

import pandas as pd
import numpy as np

from utils.config import NEWS
from utils.logger import get_logger

log = get_logger("social_features")

SOCIAL_FEATURE_COLS = [
    "social_volume_z", "social_sentiment_avg", "social_polarity_shift",
]


def _date_normalize(dt_series: "pd.Series | pd.DatetimeIndex") -> "pd.DatetimeIndex":
    """Seriyi/Index'i tarih seviyesine normalize et (saat bilgisini at)."""
    s = pd.to_datetime(dt_series)
    if isinstance(s, pd.DatetimeIndex):
        return s.normalize()
    return pd.to_datetime(s.dt.normalize())


def social_metrics(
    posts_df: pd.DataFrame,
    price_idx: pd.DatetimeIndex,
    shift_days: int = 1,
    zscore_window: int = 30,
) -> pd.DataFrame:
    """Sosyal medya post'larından günlük duygu metrikleri çıkar.

    Parameters
    ----------
    posts_df : pd.DataFrame
        social_fetcher.fetch_social_posts() çıktısı.
        Columns: datetime, text, source, symbol, engagement
    price_idx : pd.DatetimeIndex
        Fiyat DataFrame'inin index'i (bar tarihleri).
    shift_days : int
        Duygu verisini kaç gün ileri kaydıracağız.
    zscore_window : int
        Volume Z-score pencere boyutu.

    Returns
    -------
    pd.DataFrame
        social_volume_z, social_sentiment_avg, social_polarity_shift
        (price_idx ile aynı index'te)
    """
    result = pd.DataFrame(index=price_idx, data={
        "social_volume_z": 0.0,
        "social_sentiment_avg": 0.0,
        "social_polarity_shift": 0.0,
    })

    if posts_df.empty:
        return result

    posts = posts_df.copy()
    posts["date"] = _date_normalize(posts["datetime"])

    # Her post'a sentiment skoru ata (mevcut FinBERT/VADER pipeline)
    from data.sentiment import score_batch, init
    init()
    texts = posts["text"].fillna("").tolist()
    posts["sentiment"] = score_batch(texts)

    # Günlük agregasyon
    daily = posts.groupby("date").agg(
        post_count=("text", "count"),
        sentiment_avg=("sentiment", "mean"),
        sentiment_std=("sentiment", "std"),
    ).reset_index()
    daily["sentiment_std"] = daily["sentiment_std"].fillna(0)

    # Volume Z-score: mention patlaması = anormal ilgi
    daily["volume_ma"] = daily["post_count"].rolling(zscore_window, min_periods=5).mean()
    daily["volume_std"] = daily["post_count"].rolling(zscore_window, min_periods=5).std()
    daily["volume_z"] = ((daily["post_count"] - daily["volume_ma"]) / daily["volume_std"]).fillna(0)
    daily.loc[daily["post_count"] == 0, "volume_z"] = 0
    daily["volume_z"] = daily["volume_z"].replace([np.inf, -np.inf], 0)

    # Fiyat index'ine merge
    daily["date"] = pd.to_datetime(daily["date"])
    price_dates = _date_normalize(price_idx)

    # Günlük → bar index'ine map
    date_to_vol = dict(zip(daily["date"], daily["volume_z"]))
    date_to_sent = dict(zip(daily["date"], daily["sentiment_avg"]))

    result["social_volume_z"] = [
        date_to_vol.get(d, 0.0) for d in price_dates
    ]
    result["social_sentiment_avg"] = [
        date_to_sent.get(d, 0.0) for d in price_dates
    ]

    # SHIFT: bugünün post'ları ancak yarınki bar'da bilinir
    if shift_days > 0:
        result["social_volume_z"] = result["social_volume_z"].shift(shift_days).fillna(0)
        result["social_sentiment_avg"] = result["social_sentiment_avg"].shift(shift_days).fillna(0)

    # Polarity shift: sentiment yönündeki keskin değişim
    result["social_polarity_shift"] = result["social_sentiment_avg"].diff().fillna(0)

    return result


def add_social_features(
    price_df: pd.DataFrame,
    posts_df: Optional[pd.DataFrame] = None,
    shift_days: int = 1,
    zscore_window: int = 30,
    allow_sample: bool = False,
) -> pd.DataFrame:
    """Fiyat DataFrame'ine sosyal duygu özelliklerini ekle.

    Parameters
    ----------
    price_df : pd.DataFrame
        Günlük OHLCV DataFrame'i.
    posts_df : pd.DataFrame veya None
        Sosyal medya post'ları. None ise otomatik fetch edilir.
    shift_days : int
        Duygu shift gün sayısı (≥1 = leak-free).
    zscore_window : int
        Volume Z-score pencere boyutu.
    allow_sample : bool
        API anahtarı bulunmadığında sentetik/örnek veri kullanılmasına izin verilsin mi?

    Returns
    -------
    pd.DataFrame
        Orijinal df + 3 sosyal özellik eklenmiş.
    """
    df = price_df.copy()
    price_idx = pd.to_datetime(df.index)

    if posts_df is None:
        from data.social_fetcher import fetch_social_posts
        symbol = df.attrs.get("symbol", "BTC/USDT")
        n_days = len(df) + shift_days + 5
        since = price_idx.min() - timedelta(days=shift_days + 1)
        until = price_idx.max()
        posts_df = fetch_social_posts(symbol, since=since, until=until, days=n_days, allow_synthetic=allow_sample)

    metrics = social_metrics(
        posts_df, price_idx, shift_days=shift_days, zscore_window=zscore_window
    )

    for col in SOCIAL_FEATURE_COLS:
        if col in metrics.columns:
            df[col] = metrics[col].values

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df
