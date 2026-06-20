"""External reference time-series as a unified, leak-free feature family.

Seven macro / derivatives / sentiment sources -- DXY, USDT+USDC dominance,
global liquidity, Open Interest, Funding Rate, on-chain activity, Fear & Greed --
are all the same shape: an external daily series merged into the price frame by
date. This module follows two existing project patterns:

  * data/cross_asset.py -- align an external series onto the price index by date
    (daily reindex + forward-fill), with an injectable `ref_data` dict so tests
    never touch the network, and null-safe fillna(0).
  * data/news_fetcher.py -- each real fetch attempts a live endpoint and returns
    None/empty on any failure, so a deterministic synthetic generator can fill
    in. No new dependency: urllib (already used for NewsAPI) + the project's ccxt
    access for funding / open-interest.

Leakage discipline: every series is transformed to a stationary, scale-free
form (change / z-score / normalised), then shifted forward by REFERENCE.shift_days
(>=1) so a value published on day d can only enter the feature row for day
d+shift -- never the same day. This mirrors the news sentiment shift.
"""
import json
import time
import urllib.parse
import urllib.request
import urllib.error
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from utils.config import DATA, FEATURES, REFERENCE
from utils.logger import get_logger

log = get_logger("reference_series")

# Feature column order matches FEATURES.reference_feature_cols.
FEATURE_COLS = list(FEATURES.reference_feature_cols)

# Process-wide memory cache with TTL (B7 fix).
# Maps cache_key -> (series, timestamp).
_series_cache: Dict[str, Tuple[pd.Series, float]] = {}


def clear_cache() -> None:
    """Clear the TTL-tracked memory cache of reference series (B7 fix)."""
    _series_cache.clear()


def _cache_get(key: str, ttl_s: int) -> Optional[pd.Series]:
    """Return cached series if fresh; delete + return None if expired."""
    entry = _series_cache.get(key)
    if entry is None:
        return None
    series, ts = entry
    if time.time() - ts > ttl_s:
        del _series_cache[key]
        return None
    return series


def _cache_set(key: str, series: pd.Series) -> None:
    _series_cache[key] = (series, time.time())


# ------------------------------------------------------------------ #
# Index helpers                                                       #
# ------------------------------------------------------------------ #

def _ext_index(index: pd.Index, buffer: int) -> pd.DatetimeIndex:
    """A daily date range covering `index` plus a leading buffer, so transforms
    that need warm-up (pct_change, rolling z-score, shift) have history."""
    di = pd.to_datetime(index).normalize()
    start = di.min() - pd.Timedelta(days=buffer)
    return pd.date_range(start, di.max(), freq="D")


def _align(series: pd.Series, target_index: pd.Index) -> pd.Series:
    """Daily-normalise a date-indexed series and forward-fill onto target_index
    (identical logic to data/cross_asset.py._align)."""
    s = series.copy()
    s.index = pd.to_datetime(s.index).normalize()
    s = s[~s.index.duplicated(keep="last")]
    ti = pd.to_datetime(target_index).normalize()
    aligned = s.reindex(ti, method="ffill")
    aligned.index = target_index
    return aligned


# ------------------------------------------------------------------ #
# Synthetic generators (deterministic; one seed per series)          #
# ------------------------------------------------------------------ #

def _rw(idx: pd.DatetimeIndex, seed: int, start: float, vol: float,
        floor: Optional[float] = None) -> pd.Series:
    rng = np.random.default_rng(seed)
    vals = start + np.cumsum(rng.standard_normal(len(idx)) * vol)
    if floor is not None:
        vals = np.maximum(vals, floor)
    return pd.Series(vals, index=idx)


def _gen_dxy(idx: pd.DatetimeIndex) -> pd.Series:
    return _rw(idx, 101, start=100.0, vol=0.3, floor=1.0)


def _gen_stable_dom(idx: pd.DatetimeIndex) -> pd.Series:
    """Synthetic stablecoin total market cap (USDT+USDC circulating, DeFiLlama-equivalent).
    The feature name 'stable_dom_chg' reflects the log-return of this series —
    it captures changes in stablecoin supply, a proxy for capital flowing into/out of crypto."""
    return _rw(idx, 102, start=0.08, vol=0.001, floor=1e-4).clip(0.0, 1.0)


def _gen_liquidity(idx: pd.DatetimeIndex) -> pd.Series:
    return _rw(idx, 103, start=100.0, vol=0.4, floor=1.0)


def _gen_oi(idx: pd.DatetimeIndex) -> pd.Series:
    return _rw(idx, 104, start=1.0e9, vol=2.0e7, floor=1.0)


def _gen_funding(idx: pd.DatetimeIndex) -> pd.Series:
    # Mean-reverting small signed rate around 0 (perp funding ~ +/-0.01%).
    rng = np.random.default_rng(105)
    s = np.zeros(len(idx))
    for t in range(1, len(idx)):
        s[t] = 0.85 * s[t - 1] + 0.0003 * rng.standard_normal()
    return pd.Series(s, index=idx)


def _gen_cm_addr(idx: pd.DatetimeIndex) -> pd.Series:
    return _rw(idx, 106, start=1.0e6, vol=1.5e4, floor=1.0)


def _gen_fear_greed(idx: pd.DatetimeIndex) -> pd.Series:
    rng = np.random.default_rng(107)
    vals = 50 + np.cumsum(rng.standard_normal(len(idx)) * 3.0)
    return pd.Series(np.clip(vals, 0, 100), index=idx)


# ------------------------------------------------------------------ #
# Real fetchers (best-effort; return None on any failure)            #
# ------------------------------------------------------------------ #

def _get_json(url: str) -> Optional[dict]:
    try:
        with urllib.request.urlopen(url, timeout=REFERENCE.request_timeout) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError,
            json.JSONDecodeError, ValueError) as e:
        log.warning(f"reference fetch failed for {url}: {type(e).__name__}: {e}")
        return None


def _fetch_fear_greed(sym: str, index: pd.Index) -> Optional[pd.Series]:
    params = {"limit": REFERENCE.fear_greed_limit, "format": "json"}
    url = f"{REFERENCE.fear_greed_url}?{urllib.parse.urlencode(params)}"
    data = _get_json(url)
    if not data or "data" not in data:
        return None
    rows = {}
    for d in data["data"]:
        try:
            ts = pd.to_datetime(int(d["timestamp"]), unit="s").normalize()
            rows[ts] = float(d["value"])
        except (KeyError, ValueError, TypeError):
            continue
    return pd.Series(rows).sort_index() if rows else None


def _fetch_stable_dom(sym: str, index: pd.Index) -> Optional[pd.Series]:
    data = _get_json(REFERENCE.defillama_stables_url)
    if not isinstance(data, list) or not data:
        return None
    rows = {}
    for d in data:
        try:
            ts = pd.to_datetime(int(d["date"]), unit="s").normalize()
            usd = d["totalCirculatingUSD"]
            val = (float(usd["peggedUSD"]) if isinstance(usd, dict)
                   else float(usd))
            rows[ts] = val
        except (KeyError, ValueError, TypeError):
            continue
    return pd.Series(rows).sort_index() if rows else None


def _fetch_ccxt_funding(sym: str, index: pd.Index) -> Optional[pd.Series]:
    return _fetch_ccxt_history(sym, index, kind="funding")


def _fetch_ccxt_oi(sym: str, index: pd.Index) -> Optional[pd.Series]:
    return _fetch_ccxt_history(sym, index, kind="oi")


def _fetch_ccxt_history(sym: str, index: pd.Index, kind: str) -> Optional[pd.Series]:
    try:
        import ccxt
        ex = getattr(ccxt, DATA.exchange_id)()
        if kind == "funding":
            if not getattr(ex, "has", {}).get("fetchFundingRateHistory"):
                return None
            raw = ex.fetchFundingRateHistory(sym, limit=1000)
            pairs = [(r["timestamp"], r.get("fundingRate")) for r in raw]
        else:
            if not getattr(ex, "has", {}).get("fetchOpenInterestHistory"):
                return None
            raw = ex.fetchOpenInterestHistory(sym, timeframe="1d", limit=1000)
            pairs = [(r["timestamp"], r.get("openInterestValue") or r.get("openInterestAmount"))
                     for r in raw]
        rows = {pd.to_datetime(ts, unit="ms").normalize(): float(v)
                for ts, v in pairs if v is not None}
        return pd.Series(rows).sort_index() if rows else None
    except Exception as e:  # ccxt raises a wide variety; degrade gracefully
        log.warning(f"ccxt {kind} history failed for {sym}: {type(e).__name__}: {e}")
        return None


def _fetch_fred(series_id: str, index: pd.Index) -> Optional[pd.Series]:
    if not REFERENCE.fred_api_key:
        return None
    params = {"series_id": series_id, "api_key": REFERENCE.fred_api_key,
              "file_type": "json"}
    data = _get_json(f"{REFERENCE.fred_url}?{urllib.parse.urlencode(params)}")
    if not data or "observations" not in data:
        return None
    rows = {}
    for o in data["observations"]:
        try:
            rows[pd.to_datetime(o["date"]).normalize()] = float(o["value"])
        except (KeyError, ValueError, TypeError):
            continue
    return pd.Series(rows).sort_index() if rows else None


def _fetch_dxy(sym: str, index: pd.Index) -> Optional[pd.Series]:
    return _fetch_fred("DTWEXBGS", index)  # trade-weighted USD index


def _fetch_liquidity(sym: str, index: pd.Index) -> Optional[pd.Series]:
    return _fetch_fred("WALCL", index)  # Fed total assets (liquidity proxy)


def _fetch_coinmetrics(sym: str, index: pd.Index, metric: str) -> Optional[pd.Series]:
    base = sym.split("/")[0].lower()
    params = {"assets": base, "metrics": metric, "frequency": "1d"}
    url = f"{REFERENCE.coinmetrics_url}?{urllib.parse.urlencode(params)}"
    data = _get_json(url)
    if not data or "data" not in data:
        return None
    rows = {}
    for d in data["data"]:
        try:
            ts = pd.to_datetime(d["time"]).tz_localize(None).normalize()
            rows[ts] = float(d[metric])
        except (KeyError, ValueError, TypeError):
            continue
    return pd.Series(rows).sort_index() if rows else None


def _fetch_cm_addr(sym: str, index: pd.Index) -> Optional[pd.Series]:
    return _fetch_coinmetrics(sym, index, "AdrActCnt")


# ------------------------------------------------------------------ #
# Transforms (level series -> stationary, scale-free feature)        #
# ------------------------------------------------------------------ #

def _t_dxy(s: pd.Series) -> pd.Series:
    return s.pct_change().clip(-REFERENCE.dxy_clip, REFERENCE.dxy_clip)


def _t_stable_dom(s: pd.Series) -> pd.Series:
    """Stablecoin market cap log-return (B8: stable_dom_chg = Δ total circulating USDT+USDC)."""
    return s.pct_change().clip(-REFERENCE.dom_clip, REFERENCE.dom_clip)


def _t_liquidity(s: pd.Series) -> pd.Series:
    return s.pct_change().clip(-REFERENCE.liquidity_clip, REFERENCE.liquidity_clip)


def _t_oi(s: pd.Series) -> pd.Series:
    return s.pct_change().clip(-REFERENCE.oi_clip, REFERENCE.oi_clip)


def _t_funding(s: pd.Series) -> pd.Series:
    return s.clip(-REFERENCE.funding_clip, REFERENCE.funding_clip)


def _t_zscore(s: pd.Series) -> pd.Series:
    w = REFERENCE.zscore_window
    mu = s.rolling(w).mean()
    sd = s.rolling(w).std().replace(0, np.nan)
    return ((s - mu) / sd).clip(-REFERENCE.onchain_clip, REFERENCE.onchain_clip)


def _t_cm_addr(s: pd.Series) -> pd.Series:
    return _t_zscore(s)


def _t_fear_greed(s: pd.Series) -> pd.Series:
    return ((s - 50.0) / 50.0).clip(-REFERENCE.fear_greed_clip, REFERENCE.fear_greed_clip)


# name -> (real fetch fn, synthetic gen fn, transform fn)
SERIES: Dict[str, tuple] = {
    "dxy_chg": (_fetch_dxy, _gen_dxy, _t_dxy),
    "stable_dom_chg": (_fetch_stable_dom, _gen_stable_dom, _t_stable_dom),
    "liquidity_chg": (_fetch_liquidity, _gen_liquidity, _t_liquidity),
    "oi_chg": (_fetch_ccxt_oi, _gen_oi, _t_oi),
    "funding_rate": (_fetch_ccxt_funding, _gen_funding, _t_funding),
    "onchain_z": (_fetch_cm_addr, _gen_cm_addr, _t_cm_addr),
    "fear_greed": (_fetch_fear_greed, _gen_fear_greed, _t_fear_greed),
}


# ------------------------------------------------------------------ #
# Public API                                                         #
# ------------------------------------------------------------------ #

def generate_sample_series(name: str, index: pd.Index) -> pd.Series:
    """Deterministic synthetic level series for `name` over a buffered index."""
    gen = SERIES[name][1]
    buffer = REFERENCE.zscore_window + REFERENCE.shift_days + 5
    return gen(_ext_index(index, buffer))


def fetch_reference_series(name: str, sym: str, index: pd.Index,
                           allow_sample: bool = False) -> Optional[pd.Series]:
    """Real fetch for `name`; falls back to the synthetic generator when the
    live fetch is unavailable and allow_sample is set. Returns a date-indexed
    level series (untransformed) or None.

    B7: Cache entries respect REFERENCE.cache_ttl_s — stale entries are
    automatically evicted and re-fetched.
    """
    ttl = REFERENCE.cache_ttl_s

    # Determine cache key: symbol-specific vs global
    is_symbol_specific = name in ("oi_chg", "funding_rate", "onchain_z")
    cache_key = f"{name}:{sym}" if is_symbol_specific else name

    cached = _cache_get(cache_key, ttl)
    if cached is not None:
        log.debug(f"Returning cached reference series for {cache_key}")
        if not cached.empty:
            return cached
        # Endpoint is known-dead this session: skip the network, but still serve
        # a fresh synthetic series (recomputed per index, never cached) so the
        # fallback stays consistent across repeat calls and different symbols.
        return generate_sample_series(name, index) if allow_sample else None

    fetch_fn: Callable = SERIES[name][0]
    raw = fetch_fn(sym, index)

    if raw is not None and len(raw) > 0:
        _cache_set(cache_key, raw)
        return raw

    # Cache an empty marker to avoid re-hitting a dead endpoint; synthetic is
    # served (uncached) above on subsequent calls.
    _cache_set(cache_key, pd.Series(dtype=float))
    if allow_sample:
        return generate_sample_series(name, index)
    return None


def add_reference_features(df: pd.DataFrame, sym: str,
                           ref_data: Optional[Dict[str, pd.Series]] = None,
                           allow_sample: bool = False,
                           shift: int = REFERENCE.shift_days) -> pd.DataFrame:
    """Add the seven external reference features to `df`.

    Each series is transformed to a stationary form, shifted forward by `shift`
    (no same-day leak), aligned onto df's index, and null/inf-filled with 0.
    `ref_data` lets callers (tests) inject raw level series directly, bypassing
    the network -- exactly the cross_asset.py pattern.
    """
    df = df.copy()
    for name in FEATURE_COLS:
        transform = SERIES[name][2]
        if ref_data is not None and name in ref_data:
            raw = ref_data[name]
        else:
            raw = fetch_reference_series(name, sym, df.index, allow_sample=allow_sample)
        if raw is None or len(raw) == 0:
            df[name] = 0.0
            continue
        feat = transform(raw).shift(shift)
        df[name] = _align(feat, df.index)

    df[FEATURE_COLS] = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df
