import pandas as pd
import ccxt
import yfinance as yf
from utils.config import (
    EXCHANGE_ID, TIMEFRAME, OHLCV_LIMIT,
    STOCK_PERIOD, STOCK_INTERVAL, DATE_COL
)


def fetch_crypto(sym, tf=TIMEFRAME, lim=OHLCV_LIMIT):
    ex = getattr(ccxt, EXCHANGE_ID)({"enableRateLimit": True})
    raw = ex.fetch_ohlcv(sym, timeframe=tf, limit=lim)
    df = pd.DataFrame(raw, columns=[DATE_COL, "open", "high", "low", "close", "volume"])
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], unit="ms")
    df.set_index(DATE_COL, inplace=True)
    df = df.astype(float)
    df.attrs["symbol"] = sym
    df.attrs["source"] = "crypto"
    return df


def fetch_stock(sym, period=STOCK_PERIOD, interval=STOCK_INTERVAL):
    tk = yf.Ticker(sym)
    df = tk.history(period=period, interval=interval)
    df.index.name = DATE_COL
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    keep = ["open", "high", "low", "close", "volume"]
    df = df[[c for c in keep if c in df.columns]]
    df = df.astype(float)
    df.attrs["symbol"] = sym
    df.attrs["source"] = "stock"
    return df


def generate_sample(sym="SAMPLE", n=500):
    import numpy as np
    np.random.seed(42)
    idx = pd.date_range(end=pd.Timestamp.now(), periods=n, freq="D")
    p = 100 + np.cumsum(np.random.randn(n) * 2)
    p = np.abs(p) + 10
    df = pd.DataFrame({
        "open": p + np.random.randn(n) * 0.5,
        "high": p + np.abs(np.random.randn(n)) * 2,
        "low": p - np.abs(np.random.randn(n)) * 2,
        "close": p,
        "volume": np.random.randint(1000, 100000, n).astype(float),
    }, index=idx)
    df.index.name = "date"
    df.attrs["symbol"] = sym
    df.attrs["source"] = "sample"
    return df


def fetch(sym, src="auto", **kw):
    if src == "auto":
        src = "crypto" if "/" in sym else "stock"
    if src == "crypto":
        return fetch_crypto(sym, **kw)
    return fetch_stock(sym, **kw)
