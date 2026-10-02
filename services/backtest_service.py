"""
Backtest Service.

Runs backtests on a single-worker ThreadPoolExecutor and publishes
results via the MessageBus.

Replaces: ui/app.py  _on_backtest() + _run_backtest()
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional

from messaging.bus import MessageBus
from messaging.topics import BACKTEST_COMPLETE, STATUS
from messaging.messages import BacktestCompletePayload, StatusPayload
from services.app_state import AppState
from utils.config import SERVICE
from utils.exceptions import AITraderError
from utils.logger import get_logger

log = get_logger("backtest_service")


class BacktestService:
    """Runs backtests on a single background worker."""

    def __init__(self, bus: MessageBus, state: AppState):
        self._bus = bus
        self._state = state
        self._executor = ThreadPoolExecutor(
            max_workers=SERVICE.backtest_workers,
            thread_name_prefix="btsvc",
        )
        log.info("BacktestService started with %d workers", SERVICE.backtest_workers)

    def run_backtest(
        self,
        sym: str,
        capital: float = 1000.0,
        period_days: Optional[int] = None,
    ) -> Future:
        """Submit a backtest for execution.

        Uses state.bundles to decide single-symbol vs portfolio mode.
        Results are stored in state via set_bt_cache() and published
        on BACKTEST_COMPLETE.{sym}.
        """
        return self._executor.submit(self._do_backtest, sym, capital, period_days)

    def _do_backtest(self, sym, capital, period_days):
        bundles = self._state.bundles
        bundle = self._state.get_bundle(sym)

        if bundle is None:
            self._bus.publish_obj(STATUS, StatusPayload(
                text="Train models first", level="warning",
            ))
            return

        if not bundle.get("shield_passed", True):
            self._bus.publish_obj(STATUS, StatusPayload(
                text=f"Backtest blocked: {sym} failed validation shield (NO EDGE)", level="error",
            ))
            return

        try:
            if bundle.get("strategy_type") == "statarb":
                from engine.backtester import run_statarb
                params = bundle["statarb_params"]
                sp_y = bundle["split"]
                sp_x = {
                    "df": params["df_x"],
                    "X_tr": sp_y["X_tr"], "y_tr": sp_y["y_tr"], "idx_tr": sp_y["idx_tr"], "i_tr": sp_y["i_tr"],
                    "X_val": sp_y["X_val"], "y_val": sp_y["y_val"], "idx_val": sp_y["idx_val"], "i_va": sp_y["i_va"],
                    "X_test": sp_y["X_test"], "y_test": sp_y["y_test"], "idx_test": sp_y["idx_test"], "i_te": sp_y["i_te"]
                }
                res = run_statarb(
                    sp_y, sp_x,
                    beta=params["beta"],
                    alpha=params["alpha"],
                    train_mean=params["train_mean"],
                    train_std=params["train_std"],
                    entry=params["entry"],
                    exit=params["exit"],
                    capital=capital,
                    which="test"
                )
                bt_results = {"statarb": res}
                self._state.set_bt_cache(sym, bt_results)
            else:
                # Force portfolio mode path (run_portfolio_all) even for a single symbol
                from engine.backtester import run_portfolio_all
                bt_results = run_portfolio_all(bundles, capital=capital, period_days=period_days)
                for s in bundles.keys():
                    self._state.set_bt_cache(s, bt_results)

            self._state.set_bt_results(sym, bt_results)

            self._bus.publish_obj(
                f"{BACKTEST_COMPLETE}.{sym}",
                BacktestCompletePayload(
                    symbol=sym,
                    portfolio_mode=bundle.get("strategy_type") != "statarb",
                ),
            )
            self._bus.publish_obj(STATUS, StatusPayload(text="Backtest complete"))

        except AITraderError as e:
            log.error("Backtest error: %s: %s", type(e).__name__, e)
            self._bus.publish_obj(STATUS, StatusPayload(
                text=f"Backtest error: {e}", level="error",
            ))
        except Exception as e:
            log.exception("Unexpected backtest error")
            self._bus.publish_obj(STATUS, StatusPayload(
                text=f"Backtest error: {type(e).__name__}: {e}", level="error",
            ))

    def shutdown(self):
        log.info("BacktestService shutting down")
        self._executor.shutdown(wait=False)
