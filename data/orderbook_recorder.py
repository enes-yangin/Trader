"""
Order Book Recorder — arka planda orderbook.live_signal()'ı periyodik
kaydeden, append-only Parquet store oluşturan toplayıcı.

Canlı snapshot'ı tarihsel veriye dönüştürür (Feature 1).
"""

from __future__ import annotations

import os
import time
import json
import threading
from typing import Optional, Dict, Any

import pandas as pd
import numpy as np

from utils.config import DATA, ORDERBOOK
from utils.logger import get_logger

log = get_logger("orderbook_recorder")

STORE_SUBDIR = "orderbook_history"
SNAPSHOT_COLS = [
    "timestamp", "mid_price", "microprice", "micro_skew",
    "imbalance", "rel_spread", "bid_walls", "ask_walls", "confidence",
]


class OrderBookRecorder:
    """Arka planda order book snapshot'larını Parquet'e append-only yazan recorder."""

    def __init__(
        self,
        symbol: str,
        interval_s: int = 60,
        depth: int = ORDERBOOK.depth,
        data_dir: Optional[str] = None,
    ) -> None:
        self.symbol = symbol
        self.interval_s = interval_s
        self.depth = depth
        self._root = data_dir or DATA.data_dir
        self._store_dir = os.path.join(self._root, STORE_SUBDIR)
        self._parquet_path = os.path.join(
            self._store_dir, f"{symbol.replace('/', '_')}.parquet"
        )
        self._meta_path = os.path.join(
            self._store_dir, f"{symbol.replace('/', '_')}.meta.json"
        )
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._snapshot_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Daemon thread'te kaydı başlat."""
        if self._thread is not None and self._thread.is_alive():
            log.warning(f"Recorder zaten çalışıyor: {self.symbol}")
            return
        os.makedirs(self._store_dir, exist_ok=True)
        self._stop_event.clear()
        self._snapshot_count = self._count_existing()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name=f"ob-rec-{self.symbol}"
        )
        self._thread.start()
        log.info(f"OrderBookRecorder başladı: {self.symbol} / {self.interval_s}s")

    def stop(self, timeout: float = 10.0) -> None:
        """Kaydı durdur ve thread'i bekle."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            log.info(f"OrderBookRecorder durdu ({self._snapshot_count} snapshot)")

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def snapshot_once(self) -> Optional[Dict[str, Any]]:
        """Tek seferlik senkron snapshot al ve kaydet."""
        snap = self._fetch_snapshot()
        if snap is None:
            return None
        self._append(snap)
        self._snapshot_count += 1
        return snap

    def load_history(self) -> pd.DataFrame:
        """Tüm kayıtlı snapshot'ları DataFrame olarak döndür."""
        if not os.path.exists(self._parquet_path):
            return pd.DataFrame(columns=SNAPSHOT_COLS)
        df = pd.read_parquet(self._parquet_path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)
        return df

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            t0 = time.time()
            snap = self._fetch_snapshot()
            if snap is not None:
                self._append(snap)
                self._snapshot_count += 1
            elapsed = time.time() - t0
            self._stop_event.wait(max(0.0, self.interval_s - elapsed))

    def _fetch_snapshot(self) -> Optional[Dict[str, Any]]:
        try:
            from data.orderbook import live_signal
            sig = live_signal(self.symbol, depth=self.depth)
            return {
                "timestamp": int(time.time() * 1000),
                "mid_price": sig["mid_price"],
                "microprice": sig["microprice"],
                "micro_skew": sig["micro_skew"],
                "imbalance": sig["imbalance"],
                "rel_spread": sig["rel_spread"],
                "bid_walls": sig["bid_walls"],
                "ask_walls": sig["ask_walls"],
                "confidence": sig["confidence"],
            }
        except Exception as e:
            log.warning(f"Snapshot hatası ({self.symbol}): {e}")
            return None

    def _append(self, snap: Dict[str, Any]) -> None:
        row = pd.DataFrame([snap])
        row["timestamp"] = row["timestamp"].astype("int64")
        for c in SNAPSHOT_COLS:
            if c not in row.columns:
                row[c] = np.nan
        write_header = not os.path.exists(self._parquet_path)
        row[SNAPSHOT_COLS].to_parquet(
            self._parquet_path, engine="pyarrow", append=True, index=False,
        )
        if write_header:
            meta = {
                "symbol": self.symbol, "created_at": time.time(),
                "interval_s": self.interval_s, "depth": self.depth,
                "first_snapshot_ms": snap["timestamp"],
            }
            with open(self._meta_path, "w") as f:
                json.dump(meta, f, indent=2)

    def _count_existing(self) -> int:
        if not os.path.exists(self._parquet_path):
            return 0
        try:
            return len(pd.read_parquet(self._parquet_path, columns=["timestamp"]))
        except Exception:
            return 0


def generate_sample_ob_history(
    symbol: str = "BTC/USDT",
    n_snapshots: int = 10000,
    mid_start: float = 30000.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Sandbox testleri için sentetik order book snapshot geçmişi üret."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_snapshots, freq="1min", tz="UTC")
    mid = mid_start * np.exp(np.cumsum(rng.normal(0, 0.0002, n_snapshots)))
    imbalance = np.clip(rng.normal(0, 0.15, n_snapshots), -0.8, 0.8)
    rel_spread = np.abs(rng.normal(0.0005, 0.0002, n_snapshots))
    micro_skew = imbalance * 0.001
    microprice = mid * (1 + micro_skew)
    bid_walls = rng.poisson(0.5, n_snapshots)
    ask_walls = rng.poisson(0.5, n_snapshots)
    confidence = np.clip(np.abs(imbalance) * 100 / ORDERBOOK.imbalance_threshold, 0, 100)
    df = pd.DataFrame({
        "mid_price": mid, "microprice": microprice, "micro_skew": micro_skew,
        "imbalance": imbalance, "rel_spread": rel_spread,
        "bid_walls": bid_walls, "ask_walls": ask_walls, "confidence": confidence,
    }, index=idx)
    df.index.name = "timestamp"
    return df
