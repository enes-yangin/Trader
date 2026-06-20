import time
import pandas as pd
import ccxt
from utils.config import DATA, FEATURES


def fetch_crypto(sym, tf=DATA.timeframe, lim=DATA.ohlcv_limit):
    ex = getattr(ccxt, DATA.exchange_id)({"enableRateLimit": True})
    raw = ex.fetch_ohlcv(sym, timeframe=tf, limit=lim)
    df = pd.DataFrame(raw, columns=[FEATURES.date_col, "open", "high", "low", "close", "volume"])
    df[FEATURES.date_col] = pd.to_datetime(df[FEATURES.date_col], unit="ms")
    df.set_index(FEATURES.date_col, inplace=True)
    df = df.astype(float)
    df.attrs["symbol"] = sym
    df.attrs["source"] = "crypto"
    return df


def fetch_crypto_hist(sym, tf=DATA.timeframe, years=DATA.hist_years):
    ex = getattr(ccxt, DATA.exchange_id)({"enableRateLimit": True})
    ms_now = ex.milliseconds()
    since = ms_now - years * 365 * 24 * 60 * 60 * 1000
    all_rows = []
    cursor = since
    while True:
        batch = ex.fetch_ohlcv(sym, timeframe=tf, since=cursor, limit=1000)
        if not batch:
            break
        all_rows += batch
        cursor = batch[-1][0] + 1
        if len(batch) < 1000:
            break
        if cursor >= ms_now:
            break
        time.sleep(ex.rateLimit / 1000.0)
    if not all_rows:
        return fetch_crypto(sym, tf=tf)
    df = pd.DataFrame(all_rows, columns=[FEATURES.date_col, "open", "high", "low", "close", "volume"])
    df[FEATURES.date_col] = pd.to_datetime(df[FEATURES.date_col], unit="ms")
    df = df.drop_duplicates(subset=[FEATURES.date_col])
    df.set_index(FEATURES.date_col, inplace=True)
    df = df.astype(float)
    df.attrs["symbol"] = sym
    df.attrs["source"] = "crypto"
    return df


def fetch_hist(sym, src="crypto", years=DATA.hist_years):
    return fetch_crypto_hist(sym, years=years)


def generate_sample(sym="SAMPLE", n=500):
    import numpy as np
    rng = np.random.default_rng(42)
    idx = pd.date_range(end=pd.Timestamp.now(), periods=n, freq="D")
    p = 100 + np.cumsum(rng.standard_normal(n) * 2)
    p = np.abs(p) + 10
    df = pd.DataFrame({
        "open": p + rng.standard_normal(n) * 0.5,
        "high": p + np.abs(rng.standard_normal(n)) * 2,
        "low": p - np.abs(rng.standard_normal(n)) * 2,
        "close": p,
        "volume": rng.integers(1000, 100000, n).astype(float),
    }, index=idx)
    df.index.name = "date"
    df.attrs["symbol"] = sym
    df.attrs["source"] = "sample"
    return df


def fetch(sym, src="crypto", **kw):
    return fetch_crypto(sym, **kw)


def fetch_live(sym, src="crypto"):
    df = fetch_crypto(sym, tf=DATA.live_timeframe, lim=DATA.live_limit)
    df.attrs["live"] = True
    return df
