"""
Data fetching service with ThreadPoolExecutor.

Replaces the raw daemon-thread spawns for:
  - _run_fetch()       → fetch_symbol()
  - _run_fetch_all()   → fetch_all_symbols()
  - _run_preload_all() → preload_all()
  - _fetch_live()      → fetch_live_price()
  - _fetch_ob()        → fetch_live_orderbook()

Every method submits work to a ThreadPoolExecutor and publishes
results / status updates on the MessageBus.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, Optional, Tuple

import ccxt

from messaging.bus import MessageBus
from messaging.topics import DATASET_READY, FETCH_PROGRESS, LIVE_ORDERBOOK, LIVE_PRICE, STATUS
from messaging.messages import DatasetReadyPayload, LivePricePayload, StatusPayload
from utils.config import DATA, SERVICE
from utils.exceptions import AITraderError, DataFetchError, InsufficientDataError
from utils.logger import get_logger

log = get_logger("data_service")


class DataService:
    """Background data-fetching service backed by a fixed-size thread pool.

    All methods are non-blocking — they return a Future immediately and
    publish results on the MessageBus when complete.
    """

    _max_workers: int

    def __init__(self, bus: MessageBus):
        if not isinstance(SERVICE.data_workers, int) or SERVICE.data_workers < 1:
            self._max_workers = 4
        else:
            self._max_workers = SERVICE.data_workers
        self._bus = bus
        self._executor = ThreadPoolExecutor(
            max_workers=self._max_workers,
            thread_name_prefix="datasvc",
        )
        log.info("DataService started with %d workers", self._max_workers)

    # ── Historical data ────────────────────────────────────────────

    def fetch_symbol(self, sym: str, with_news: bool = False,
                     force: bool = False, allow_sample: bool = False) -> Future:
        """Fetch and build dataset for a single symbol → publishes DATASET_READY."""
        return self._executor.submit(
            self._do_fetch_symbol, sym, with_news, force, allow_sample,
        )

    def _do_fetch_symbol(self, sym, with_news, force, allow_sample):
        from data import dataset
        try:
            self._bus.publish_obj(STATUS, StatusPayload(
                text=f"Fetching {DATA.hist_years}y data for {sym}...",
            ))
            df = dataset.build(sym, with_news=with_news, force=force,
                              allow_sample=allow_sample)
            n = len(df)
            rng = f"{df.index.min().date()} → {df.index.max().date()}"
            self._bus.publish_obj(DATASET_READY, DatasetReadyPayload(
                symbol=sym, row_count=n, source="fresh",
            ))
            self._bus.publish_obj(STATUS, StatusPayload(
                text=f"{sym}: {n} rows cached ({rng})",
            ))
        except (DataFetchError, InsufficientDataError) as e:
            log.warning("%s: fetch failed: %s", sym, e)
            self._bus.publish_obj(STATUS, StatusPayload(
                text=f"Fetch error: {e}", level="warning",
            ))
        except AITraderError as e:
            log.error("%s: %s: %s", sym, type(e).__name__, e)
            self._bus.publish_obj(STATUS, StatusPayload(
                text=f"Fetch error: {e}", level="error",
            ))
        except Exception as e:
            log.exception("%s: unexpected fetch error", sym)
            self._bus.publish_obj(STATUS, StatusPayload(
                text=f"Unexpected fetch error: {type(e).__name__}: {e}", level="error",
            ))

    def fetch_all_symbols(self, syms: Tuple[str, ...], with_news: bool = False,
                          allow_sample: bool = False) -> Future:
        """Fetch all symbols in parallel on the worker pool."""
        return self._executor.submit(
            self._do_fetch_all, syms, with_news, allow_sample,
        )

    def _do_fetch_all(self, syms, with_news, allow_sample):
        from data import dataset, store

        total = len(syms)
        done = 0

        def prog(i, tt, s):
            nonlocal done
            done = i
            self._bus.publish_obj(FETCH_PROGRESS, {"done": i, "total": tt, "symbol": s})
            self._bus.publish_obj(STATUS, StatusPayload(
                text=f"[{i}/{tt}] {s}...",
            ))

        try:
            res = dataset.build_many(syms, with_news=with_news, force=True,
                                     progress=prog, allow_sample=allow_sample)
            errs = res.get("_errors", {})
            ok = total - len(errs)
            mb = store.cache_size_mb()
            if errs:
                failed = ", ".join(list(errs.keys())[:3])
                self._bus.publish_obj(STATUS, StatusPayload(
                    text=f"{ok}/{total} cached ({mb:.1f} MB) — failed: {failed}",
                    level="warning",
                ))
            else:
                self._bus.publish_obj(STATUS, StatusPayload(
                    text=f"All {total} symbols cached ({mb:.1f} MB)",
                ))
        except Exception as e:
            log.exception("Fetch all: unexpected error")
            self._bus.publish_obj(STATUS, StatusPayload(
                text=f"Fetch all error: {type(e).__name__}: {e}", level="error",
            ))

    def preload_all(self, syms: Tuple[str, ...], active_sym: str,
                    with_news: bool = False, use_cross_asset: bool = True):
        """Preload data for all symbols (non-blocking, called at startup)."""
        return self._executor.submit(
            self._do_preload_all, syms, active_sym, with_news, use_cross_asset,
        )

    def _do_preload_all(self, syms, active_sym, with_news, use_cross_asset):
        from data import dataset
        try:
            self._bus.publish_obj(STATUS, StatusPayload(
                text="Checking and pre-downloading data for all symbols...",
            ))
            # Active symbol first
            self._bus.publish_obj(STATUS, StatusPayload(
                text=f"Preloading active symbol: {active_sym}...",
            ))
            dataset.ensure(active_sym, with_news=with_news)

            remaining = [s for s in syms if s != active_sym]
            for i, sym in enumerate(remaining):
                self._bus.publish_obj(STATUS, StatusPayload(
                    text=f"Checking/Downloading data [{i+1}/{len(remaining)}]: {sym}...",
                ))
                dataset.ensure(sym, with_news=with_news)

            if use_cross_asset:
                self._bus.publish_obj(STATUS, StatusPayload(
                    text="Downloading reference data...",
                ))
                from data.cross_asset import fetch_reference_data
                fetch_reference_data()

            self._bus.publish_obj(STATUS, StatusPayload(text="All data ready."))
        except Exception as e:
            self._bus.publish_obj(STATUS, StatusPayload(
                text=f"Preload error: {e}", level="error",
            ))

    # ── Live data ──────────────────────────────────────────────────

    def fetch_live_price(self, sym: str) -> Future:
        """Fetch latest OHLCV candles for live display."""
        return self._executor.submit(self._do_fetch_live, sym)

    def _do_fetch_live(self, sym):
        from data.fetcher import fetch_live
        try:
            df = fetch_live(sym)
            last = float(df["close"].iloc[-1])
            self._bus.publish_obj(LIVE_PRICE, LivePricePayload(
                symbol=sym,
                close=last,
                timestamp=str(df.index[-1]),
                df_json=df.to_json(date_format="iso"),
            ))
        except (ccxt.BaseError, ConnectionError, TimeoutError, OSError) as e:
            log.debug("%s: live price fetch failed: %s: %s", sym, type(e).__name__, e)

    def fetch_live_orderbook(self, sym: str) -> Future:
        """Fetch latest orderbook snapshot and publish full OB dict via bus."""
        return self._executor.submit(self._do_fetch_ob, sym)

    def _do_fetch_ob(self, sym):
        from data import orderbook
        try:
            ob = orderbook.fetch_orderbook(sym)
            # Publish the full orderbook dict so the UI's OrderBookTracker
            # can run signal() + EMA smoothing + hysteresis on it
            payload = {
                "symbol": sym,
                "bids": ob.get("bids", []),
                "asks": ob.get("asks", []),
            }
            self._bus.publish(LIVE_ORDERBOOK, payload)
        except (ccxt.BaseError, ConnectionError, TimeoutError, OSError) as e:
            log.debug("%s: order book fetch failed: %s: %s", sym, type(e).__name__, e)

    # ── Shutdown ───────────────────────────────────────────────────

    def shutdown(self):
        """Shutdown the thread pool gracefully (non-blocking)."""
        log.info("DataService shutting down")
        self._executor.shutdown(wait=False)
