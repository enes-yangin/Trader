"""
Thread-safe application state container for TraderAI v2.

Previously, App (the tkinter root) held all state as bare instance attributes:
    self.bundle, self.bundles{}, self.signals{}, self.bt_results,
    self.bt_results_cache{}, self.last_signal, self.live_df, ...

AppState centralizes these behind a re-entrant lock (RLock), providing
atomic get/set operations. The pub/sub bus notifies of state *changes*;
AppState holds the current *truth* — subscribers that connect late can
still read current values without missing events.

Thread safety: every read/write method acquires self._lock.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

import pandas as pd


class AppState:
    """Centralized, thread-safe application state.

    All getters/setters acquire `threading.RLock()` so they are safe to
    call from service executor threads and the tkinter main thread alike.
    """

    def __init__(self):
        self._lock = threading.RLock()

        # Per-symbol caches
        self._bundles: Dict[str, Dict[str, Any]] = {}
        self._signals: Dict[str, Dict[str, Any]] = {}

        # Backtest results (keyed by symbol)
        self._bt_results: Dict[str, Any] = {}
        self._bt_results_cache: Dict[str, Dict[str, Any]] = {}

        # Live data (per symbol)
        self._live_dfs: Dict[str, pd.DataFrame] = {}
        self._ob_trackers: Dict[str, Any] = {}

        # Current UI-selected symbol
        self._current_symbol: str = ""

        # Paper trader state
        self._last_signal: Dict[str, Any] = {}
        self._paper_accuracy: float = 0.0

    # ── Symbol management ──────────────────────────────────────────

    @property
    def current_symbol(self) -> str:
        with self._lock:
            return self._current_symbol

    @current_symbol.setter
    def current_symbol(self, sym: str):
        with self._lock:
            self._current_symbol = sym

    def swap_symbol(self, sym: str) -> Dict[str, Any]:
        """Atomically switch current symbol and return all cached data for it.

        Returns dict with keys: bundle, signal, bt_results.
        Values are None when no cached data exists.
        """
        with self._lock:
            self._current_symbol = sym
            return {
                "bundle": self._bundles.get(sym),
                "signal": self._signals.get(sym),
                "bt_results": self._bt_results_cache.get(sym),
            }

    # ── Bundles ────────────────────────────────────────────────────

    def get_bundle(self, sym: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._bundles.get(sym)

    def set_bundle(self, sym: str, bundle: Dict[str, Any]):
        with self._lock:
            self._bundles[sym] = bundle

    @property
    def bundles(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return dict(self._bundles)  # shallow copy

    # ── Signals ────────────────────────────────────────────────────

    def get_signal(self, sym: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._signals.get(sym)

    def set_signal(self, sym: str, signal: Dict[str, Any]):
        with self._lock:
            self._signals[sym] = signal

    @property
    def signals(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return dict(self._signals)

    # ── Backtest results ───────────────────────────────────────────

    def get_bt_results(self, sym: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._bt_results.get(sym)

    def set_bt_results(self, sym: str, results: Any):
        with self._lock:
            self._bt_results[sym] = results

    def get_bt_cache(self, sym: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._bt_results_cache.get(sym)

    def set_bt_cache(self, sym: str, result: Dict[str, Any]):
        with self._lock:
            self._bt_results_cache[sym] = result

    def clear_bt_cache(self):
        with self._lock:
            self._bt_results_cache.clear()

    # ── Live data ──────────────────────────────────────────────────

    def get_live_df(self, sym: str) -> Optional[pd.DataFrame]:
        with self._lock:
            return self._live_dfs.get(sym)

    def set_live_df(self, sym: str, df: pd.DataFrame):
        with self._lock:
            self._live_dfs[sym] = df

    @property
    def live_df(self) -> Optional[pd.DataFrame]:
        """Convenience: live df for the current symbol."""
        with self._lock:
            return self._live_dfs.get(self._current_symbol)

    def get_ob_tracker(self, sym: str) -> Optional[Any]:
        with self._lock:
            return self._ob_trackers.get(sym)

    def set_ob_tracker(self, sym: str, tracker: Any):
        with self._lock:
            self._ob_trackers[sym] = tracker

    @property
    def ob_tracker(self) -> Optional[Any]:
        """Convenience: ob tracker for the current symbol."""
        with self._lock:
            return self._ob_trackers.get(self._current_symbol)

    # ── Paper trader ───────────────────────────────────────────────

    @property
    def last_signal(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._last_signal)

    @last_signal.setter
    def last_signal(self, sig: Dict[str, Any]):
        with self._lock:
            self._last_signal = sig

    def get_paper_accuracy(self) -> float:
        with self._lock:
            return self._paper_accuracy

    def set_paper_accuracy(self, val: float):
        with self._lock:
            self._paper_accuracy = val
