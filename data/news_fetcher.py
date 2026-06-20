import os
import pandas as pd
import feedparser
import urllib.request
import urllib.parse
import urllib.error
import json
from datetime import datetime, timedelta
from utils.logger import get_logger

log = get_logger("news_fetcher")

RSS_FEEDS = {
    "crypto": [
        "https://cointelegraph.com/rss",
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
    ],
}

NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "")
NEWSAPI_URL = "https://newsapi.org/v2/everything"


def _norm_date(ts):
    try:
        return pd.to_datetime(ts).tz_localize(None).normalize()
    except (ValueError, TypeError):
        try:
            return pd.to_datetime(ts, utc=True).tz_convert(None).normalize()
        except (ValueError, TypeError):
            return pd.NaT


_rss_cache = {}


def fetch_rss(src="crypto", limit=50):
    if src in _rss_cache:
        log.debug(f"Returning cached RSS feed for source: {src}")
        return _rss_cache[src]
    feeds = RSS_FEEDS.get(src, RSS_FEEDS["crypto"])
    rows = []
    for url in feeds:
        try:
            d = feedparser.parse(url)
            if getattr(d, "bozo", 0) and not d.entries:
                log.warning(f"RSS feed unparseable: {url}")
                continue
            for e in d.entries[:limit]:
                title = e.get("title", "")
                summ = e.get("summary", "")
                pub = e.get("published", e.get("updated", ""))
                rows.append({
                    "date": _norm_date(pub),
                    "title": title,
                    "text": f"{title}. {summ}"[:512],
                    "source": "rss",
                })
        except (OSError, ValueError) as e:
            log.warning(f"RSS feed fetch failed for {url}: {type(e).__name__}: {e}")
            continue
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.dropna(subset=["date"])
    _rss_cache[src] = df
    return df


def fetch_newsapi(query, days=30, limit=100):
    if not NEWSAPI_KEY:
        return pd.DataFrame()
    frm = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    params = {
        "q": query,
        "from": frm,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": min(limit, 100),
        "apiKey": NEWSAPI_KEY,
    }
    url = f"{NEWSAPI_URL}?{urllib.parse.urlencode(params)}"
    rows = []
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        for a in data.get("articles", []):
            title = a.get("title", "") or ""
            desc = a.get("description", "") or ""
            rows.append({
                "date": _norm_date(a.get("publishedAt", "")),
                "title": title,
                "text": f"{title}. {desc}"[:512],
                "source": "newsapi",
            })
    except urllib.error.HTTPError as e:
        log.warning(f"NewsAPI HTTP error {e.code}: {e.reason}")
        return pd.DataFrame()
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
        log.warning(f"NewsAPI request failed: {type(e).__name__}: {e}")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.dropna(subset=["date"])
    return df


def _query_for(sym):
    base = sym.split("/")[0]
    m = {
        "BTC": "Bitcoin", "ETH": "Ethereum", "BNB": "Binance Coin",
        "SOL": "Solana", "XRP": "Ripple", "ADA": "Cardano",
        "DOGE": "Dogecoin", "AVAX": "Avalanche",
    }
    return m.get(base, base)


def fetch_news(sym, src="crypto", days=30, use_newsapi=True, use_rss=True):
    frames = []
    if use_rss:
        rss = fetch_rss("crypto")
        if len(rss) > 0:
            frames.append(rss)
    if use_newsapi and NEWSAPI_KEY:
        api = fetch_newsapi(_query_for(sym), days=days)
        if len(api) > 0:
            frames.append(api)
    if not frames:
        return pd.DataFrame(columns=["date", "title", "text", "source"])
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)
    return df


def generate_sample_news(sym="BTC", n=60):
    import numpy as np
    rng = np.random.default_rng(7)
    idx = pd.date_range(end=pd.Timestamp.now().normalize(), periods=n, freq="D")
    pos = ["surges to record high", "sees strong institutional inflow",
           "rallies on positive outlook", "beats market expectations"]
    neg = ["plunges amid sell-off", "faces regulatory pressure",
           "drops on weak demand", "tumbles after warning"]
    neu = ["holds steady", "trades sideways", "shows mixed signals"]
    rows = []
    for d in idx:
        k = rng.choice([0, 1, 2], p=[0.4, 0.3, 0.3])
        pool = [pos, neg, neu][k]
        t = f"{sym} {rng.choice(pool)}"
        rows.append({"date": d, "title": t, "text": t, "source": "sample"})
    return pd.DataFrame(rows)
