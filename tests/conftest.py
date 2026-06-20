import os
os.environ.setdefault("AI_TRADER_LINEAR_N_TRIALS", "2")
os.environ.setdefault("AI_TRADER_XGB_N_TRIALS", "2")
os.environ.setdefault("AI_TRADER_LSTM_N_TRIALS", "1")
os.environ.setdefault("AI_TRADER_LSTM_OPT_EPOCHS", "1")

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def isolated_cache_dir(tmp_path, monkeypatch):
    """Redirect the dataset cache to a throwaway tmp dir.

    data.store._root() is the single chokepoint every cache read/write goes
    through; it reads DATA.data_dir (default "datasets") dynamically on each
    call. DataConfig is a frozen dataclass, so we can't set DATA.data_dir --
    instead we patch _root() itself. This keeps cache tests from reading or
    writing the developer's real, git-ignored datasets/ folder, so a stale
    parquet from a prior run can't leak across tests.
    """
    from data import store

    cache_dir = tmp_path / "datasets"
    cache_dir.mkdir()
    monkeypatch.setattr(store, "_root", lambda: str(cache_dir))
    return cache_dir


@pytest.fixture
def synthetic_ohlcv():
    np.random.seed(42)
    n = 2000
    idx = pd.date_range("2019-01-01", periods=n, freq="D")
    p = 30000 + np.cumsum(np.random.randn(n) * 200)
    p = np.abs(p) + 1000
    high = p * (1 + np.abs(np.random.randn(n)) * 0.01)
    low = p * (1 - np.abs(np.random.randn(n)) * 0.01)
    vol = 1000 + np.abs(np.random.randn(n)) * 500
    df = pd.DataFrame({
        "open": p, "high": high, "low": low, "close": p, "volume": vol,
    }, index=idx)
    df.index.name = "date"
    return df


@pytest.fixture
def mock_ccxt_exchange():
    from unittest.mock import MagicMock
    np.random.seed(7)
    ms_now = 1718000000000

    def fetch_ohlcv(sym, timeframe=None, since=None, limit=1000):
        if since is not None:
            n = min(limit, 1825)
            rows = []
            p = 30000
            for i in range(n):
                p *= (1 + np.random.randn() * 0.02)
                h = p * (1 + abs(np.random.randn()) * 0.01)
                l = p * (1 - abs(np.random.randn()) * 0.01)
                rows.append([since + i * 86400000, p, h, l, p, 1000 + abs(np.random.randn()) * 500])
            return rows
        base = ms_now - limit * 3600000
        return [[base + i * 3600000, 40000 + i, 40100 + i, 39900 + i, 40050 + i, 50.0]
                for i in range(limit)]

    mob = {"bids": [[29990, 15], [29980, 8]], "asks": [[30010, 5], [30020, 6]], "symbol": "BTC/USDT"}

    mex = MagicMock()
    mex.milliseconds.return_value = ms_now
    mex.fetch_ohlcv.side_effect = fetch_ohlcv
    mex.fetch_order_book.return_value = mob
    mex.rateLimit = 5
    return mex
