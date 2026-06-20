"""
Sosyal Duygu Fetcher — X (Twitter) ve Reddit'ten kripto odaklı post çeker.

Adapter deseni: XAdapter, RedditAdapter — her biri key-gated + sample fallback.
Tüm adapter'lar zaman damgalı post listesi döndürür (datetime, text, source).

Ortak arayüz sayesinde yeni platform eklemek tek bir adapter yazmak kadar kolay.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import hashlib
import os

import pandas as pd
import numpy as np

from utils.config import DATA
from utils.logger import get_logger

log = get_logger("social_fetcher")

POST_COLUMNS = ["datetime", "text", "source", "symbol", "engagement"]


# ------------------------------------------------------------------
# Abstract adapter
# ------------------------------------------------------------------

class SocialAdapter(ABC):
    """Sosyal medya veri kaynağı için soyut arayüz."""

    def __init__(self, api_key: Optional[str] = None, allow_synthetic: bool = False) -> None:
        self.api_key = api_key
        self.allow_synthetic = allow_synthetic

    @abstractmethod
    def fetch(
        self, keyword: str, since: datetime, until: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Post'ları getir. Başarısız olursa boş liste döner, exception throw etmez."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Adapter adı (örn. 'x', 'reddit')."""
        ...

    def _sample_fallback(
        self, keyword: str, since: datetime, until: Optional[datetime] = None,
        limit: int = 100, seed: int = 42,
    ) -> List[Dict[str, Any]]:
        """API yoksa / hatadaysa sentetik post üret (test/sandbox için).

        Only called when self.allow_synthetic is True (B2 fix).
        Uses hashlib.md5 for cross-process deterministic seeding (A6 fix).
        """
        # A6: md5 hash is deterministic across processes
        seed_base = int(hashlib.md5(keyword.encode()).hexdigest(), 16) % (2**31)
        rng = np.random.default_rng(seed + seed_base + int(since.timestamp()) % 10000)
        until = until or datetime.utcnow()
        total_hours = max(1, int((until - since).total_seconds() / 3600))
        n_posts = min(limit, max(5, total_hours // 4))

        posts: List[Dict[str, Any]] = []
        for _ in range(n_posts):
            dt = since + timedelta(
                hours=float(rng.uniform(0, total_hours))
            )
            templates = [
                f"{keyword} looking bullish today, strong momentum building up",
                f"Not sure about {keyword}, chart looks uncertain rn",
                f"Just bought some {keyword}, this dip is a gift",
                f"{keyword} might see a correction soon, taking profits here",
                f"{keyword} fundamentals are solid, long-term holder here",
                f"Bearish divergence on {keyword} daily chart, be careful",
                f"{keyword} accumulation phase seems to be ending, breakout soon?",
                f"Macro headwinds could pressure {keyword} in the short term",
            ]
            text = rng.choice(templates)
            posts.append({
                "datetime": dt,
                "text": text,
                "source": self.name,
                "symbol": keyword,
                "engagement": max(1, int(rng.exponential(10))),
            })
        return sorted(posts, key=lambda p: p["datetime"])


# ------------------------------------------------------------------
# X (Twitter) Adapter
# ------------------------------------------------------------------

class XAdapter(SocialAdapter):
    """X (Twitter) adapter'ı — API key varsa canlı veri, yoksa ve allow_synthetic
    True ise sample fallback; aksi halde boş liste döner (B2 fix)."""

    def __init__(self, api_key: Optional[str] = None, allow_synthetic: bool = False) -> None:
        super().__init__(api_key=api_key or os.environ.get("X_API_KEY", ""),
                         allow_synthetic=allow_synthetic)

    @property
    def name(self) -> str:
        return "x"

    def fetch(
        self, keyword: str, since: datetime, until: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        if not self.api_key:
            if self.allow_synthetic:
                log.info(f"X API key yok, sample fallback: {keyword}")
                return self._sample_fallback(keyword, since, until, limit)
            log.debug(f"X API key yok, returning empty (allow_synthetic=False): {keyword}")
            return []

        try:
            # TODO: Gerçek X API v2 entegrasyonu — ücretsiz tier rate-limit'li.
            if self.allow_synthetic:
                log.info(f"X API henüz entegre edilmedi, sample fallback: {keyword}")
                return self._sample_fallback(keyword, since, until, limit)
            log.debug(f"X API not integrated, returning empty: {keyword}")
            return []
        except Exception as e:
            log.warning(f"X fetch hatası ({keyword}): {e}")
            return []


# ------------------------------------------------------------------
# Reddit Adapter
# ------------------------------------------------------------------

class RedditAdapter(SocialAdapter):
    """Reddit adapter'ı — API key varsa canlı, yoksa ve allow_synthetic True ise
    sample fallback; aksi halde boş liste döner (B2 fix)."""

    def __init__(self, api_key: Optional[str] = None, allow_synthetic: bool = False) -> None:
        super().__init__(api_key=api_key or os.environ.get("REDDIT_API_KEY", ""),
                         allow_synthetic=allow_synthetic)

    @property
    def name(self) -> str:
        return "reddit"

    def fetch(
        self, keyword: str, since: datetime, until: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        if not self.api_key:
            if self.allow_synthetic:
                log.info(f"Reddit API key yok, sample fallback: {keyword}")
                return self._sample_fallback(keyword, since, until, limit)
            log.debug(f"Reddit API key yok, returning empty (allow_synthetic=False): {keyword}")
            return []

        try:
            import requests
            subreddits = "cryptocurrency+bitcoin+ethereum+CryptoMarkets"
            url = f"https://www.reddit.com/r/{subreddits}/search.json"
            headers = {"Authorization": f"Bearer {self.api_key}", "User-Agent": "traderai/1.0"}
            params = {"q": keyword, "sort": "new", "limit": min(limit, 100), "t": "month"}

            resp = requests.get(url, headers=headers, params=params, timeout=10)
            if resp.status_code != 200:
                log.warning(f"Reddit API {resp.status_code}: {resp.text[:100]}")
                if self.allow_synthetic:
                    return self._sample_fallback(keyword, since, until, limit)
                return []

            data = resp.json()
            posts: List[Dict[str, Any]] = []
            for child in data.get("data", {}).get("children", []):
                d = child.get("data", {})
                created = datetime.utcfromtimestamp(d.get("created_utc", 0))
                if created >= since and (until is None or created <= until):
                    posts.append({
                        "datetime": created,
                        "text": f"{d.get('title', '')} {d.get('selftext', '')}"[:500],
                        "source": "reddit",
                        "symbol": keyword,
                        "engagement": d.get("score", 1) + d.get("num_comments", 0),
                    })
            return sorted(posts, key=lambda p: p["datetime"])
        except Exception as e:
            log.warning(f"Reddit fetch hatası ({keyword}): {e}")
            return []


# ------------------------------------------------------------------
# Factory / orchestrator
# ------------------------------------------------------------------

_ADAPTERS: Dict[str, SocialAdapter] = {}


def _get_adapters(allow_synthetic: bool = False) -> Dict[str, SocialAdapter]:
    global _ADAPTERS
    # Recreate if allow_synthetic changed from cached value
    cache_key = f"__synthetic_{allow_synthetic}"
    if cache_key not in _ADAPTERS:
        _ADAPTERS[cache_key] = {
            "x": XAdapter(allow_synthetic=allow_synthetic),
            "reddit": RedditAdapter(allow_synthetic=allow_synthetic),
        }
    return _ADAPTERS[cache_key]


def fetch_social_posts(
    symbol: str,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    days: int = 30,
    limit: int = 100,
    sources: Optional[List[str]] = None,
    allow_synthetic: bool = False,
) -> pd.DataFrame:
    """Tüm sosyal medya kaynaklarından post'ları topla.

    Parameters
    ----------
    symbol : str
        Kripto sembolü (örn. "BTC" veya "BTC/USDT").
    since : datetime or None
        Başlangıç zamanı. None ise `days` gün öncesi.
    until : datetime or None
        Bitiş zamanı. None ise şimdi.
    days : int
        since belirtilmezse kaç gün geriye gidileceği.
    limit : int
        Kaynak başına maksimum post sayısı.
    sources : list or None
        Hangi adapter'lar kullanılacak. None ise hepsi.

    Returns
    -------
    pd.DataFrame
        Columns: datetime, text, source, symbol, engagement
    """
    base = symbol.split("/")[0] if "/" in symbol else symbol
    until = until or datetime.utcnow()
    since = since or (until - timedelta(days=days))

    adapters = _get_adapters(allow_synthetic=allow_synthetic)
    if sources:
        adapters = {s: a for s, a in adapters.items() if s in sources}

    all_posts: List[Dict[str, Any]] = []
    for src_name, adapter in adapters.items():
        try:
            posts = adapter.fetch(base, since=since, until=until, limit=limit)
            all_posts.extend(posts)
        except Exception as e:
            log.warning(f"Adapter {src_name} hata verdi: {e}")

    if not all_posts:
        return pd.DataFrame(columns=POST_COLUMNS)

    df = pd.DataFrame(all_posts)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df
