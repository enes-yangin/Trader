"""
ML Training Service.

Encapsulates the full training pipeline that was previously ~110 lines
of inline logic inside App._run_pipeline().  Runs on a single-worker
ThreadPoolExecutor because the pipeline is inherently serial.

Replaces: ui/app.py  _on_run() + _run_pipeline()
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

from messaging.bus import MessageBus
from messaging.topics import TRAINING_STARTED, TRAINING_COMPLETE, STATUS
from messaging.messages import StatusPayload, TrainingCompletePayload
from services.app_state import AppState
from utils.config import BACKTEST, DATA, FEATURES, MODEL, OPTIMIZATION, RISK, SERVICE
from utils.exceptions import AITraderError, DataFetchError, InsufficientDataError
from utils.logger import get_logger

log = get_logger("training_service")


class TrainingService:
    """Runs the ML training pipeline on a single background worker.

    Stores trained bundles and signals in AppState so the UI can
    access them without passing large objects through the message bus.
    """

    def __init__(self, bus: MessageBus, state: AppState):
        self._bus = bus
        self._state = state
        self._executor = ThreadPoolExecutor(
            max_workers=SERVICE.training_workers,
            thread_name_prefix="trainsvc",
        )
        log.info("TrainingService started with %d workers", SERVICE.training_workers)

    def train_pipeline(
        self,
        active_sym: str,
        syms: Tuple[str, ...],
        with_news: bool = False,
        optimize: bool = True,
        weighted: bool = True,
        use_cross_asset: bool = True,
        use_micro: bool = True,
        use_smoothing: bool = True,
        use_reference: bool = True,
        use_orderbook: bool = True,
        use_macro_events: bool = True,
        use_social: bool = True,
        allow_sample: bool = False,
        strategy: str = "ML Directional",
    ) -> Future:
        """Submit the training pipeline for execution.

        Returns a Future that resolves to a dict:
            {bundles: {sym: bundle}, signals: {sym: sig}, failed: {sym: reason}}
        """
        return self._executor.submit(
            self._do_train_pipeline,
            active_sym, syms, with_news, optimize, weighted,
            use_cross_asset, use_micro, use_smoothing, use_reference,
            use_orderbook, use_macro_events, use_social, allow_sample, strategy,
        )

    def _do_train_pipeline(
        self,
        active_sym, syms, with_news, optimize, weighted,
        use_cross_asset, use_micro, use_smoothing, use_reference,
        use_orderbook, use_macro_events, use_social, allow_sample, strategy,
    ):
        from engine.trainer import load_data
        from engine.ensemble import predict_weighted_ensemble
        from engine.predictor import predict_from_bundle
        from utils.types import FeatureSpec, Bundle

        # Plan item D: the regression/classification ("ML Directional") path
        # trains on the curated core_feature_cols set (see utils/types.py
        # FeatureSpec.feature_columns) rather than the full ~22-column union.
        # orderbook/macro_events/social are forced off here because they were
        # measured to be always-zero on this pipeline (N3/N8: no order-book
        # recorder history, static event calendar with no forecast/actual) --
        # keeping them on would cost engineer() time for columns core_feature_cols
        # never reads. Stat-Arb keeps the caller's flags untouched below.
        core_mode = strategy != "Stat-Arb (vs BTC)"
        spec = FeatureSpec(
            news=with_news,
            micro=use_micro or core_mode,
            cross_asset=use_cross_asset and not core_mode,
            smooth=use_smoothing or core_mode,
            reference=use_reference,
            orderbook=use_orderbook and not core_mode,
            macro_events=use_macro_events and not core_mode,
            social=use_social and not core_mode,
            core=core_mode,
        )

        bundles: Dict[str, Any] = {}
        signals: Dict[str, Any] = {}
        failed: Dict[str, str] = {}

        for idx, sym in enumerate(syms):
            try:
                self._bus.publish_obj(STATUS, StatusPayload(
                    text=f"Training {sym} [{idx+1}/{len(syms)}]...",
                ))
                self._bus.publish_obj(TRAINING_STARTED, {
                    "symbol": sym,
                    "spec": {
                        "news": spec.news, "micro": spec.micro,
                        "cross_asset": spec.cross_asset, "smooth": spec.smooth,
                        "reference": spec.reference, "orderbook": spec.orderbook,
                        "macro_events": spec.macro_events, "social": spec.social,
                    },
                })

                if strategy == "Stat-Arb (vs BTC)":
                    self._bus.publish_obj(STATUS, StatusPayload(
                        text=f"[{sym}] Running Stat-Arb Cointegration grid search...",
                    ))
                    # Cointegration partner X:
                    # If sym is BTC/USDT, we use ETH/USDT as X and BTC/USDT as Y.
                    # Else, Y is sym and X is BTC/USDT.
                    if sym == "BTC/USDT":
                        Y_sym = "ETH/USDT"
                        X_sym = "BTC/USDT"
                    else:
                        Y_sym = sym
                        X_sym = "BTC/USDT"

                    df_y = load_data(Y_sym, spec=spec, allow_sample=allow_sample)
                    df_x = load_data(X_sym, spec=spec, allow_sample=allow_sample)
                    
                    common_idx = df_y.index.intersection(df_x.index)
                    if len(common_idx) < 100:
                        raise InsufficientDataError(f"Insufficient overlapping history for {Y_sym} and {X_sym}: {len(common_idx)} rows")
                    
                    y_close = df_y["close"].reindex(common_idx).values
                    x_close = df_x["close"].reindex(common_idx).values
                    
                    # 1. Engle-Granger Cointegration test on all trainval segment (first 80%)
                    tv_end = int(len(y_close) * 0.8)
                    y_tv, x_tv = y_close[:tv_end], x_close[:tv_end]
                    
                    from engine.statarb import engle_granger
                    coint_res = engle_granger(y_tv, x_tv)
                    beta, alpha = coint_res.beta, coint_res.alpha
                    
                    # Estimate spread mean and standard deviation on trainval
                    sp_train = y_tv - (alpha + beta * x_tv)
                    train_mean = float(sp_train.mean())
                    train_std = float(sp_train.std(ddof=0))
                    
                    # 2. Run selection & sealed judgment on holdout (grid search on trainval)
                    # Grid: search entry (1.5, 2.0, 2.5) and exit (0.25, 0.5)
                    grid = [(e, ex) for e in (1.5, 2.0, 2.5) for ex in (0.25, 0.5)]
                    
                    from engine.statarb_signals import select_and_judge
                    # Cost: commission (2 * 0.001) + slippage (2 * 0.0005) = 0.003 round trip,
                    # as a fraction of notional. The spread is in price units (1 unit of Y
                    # against beta units of X), so scale by that gross notional -- from
                    # trainval prices only, no look-ahead.
                    cost_rate = 2 * BACKTEST.commission_pct + 2 * BACKTEST.slippage_pct
                    spread_notional = float(y_tv.mean() + abs(beta) * x_tv.mean())
                    rep = select_and_judge(
                        y_close, x_close, grid,
                        train_frac=0.6, val_frac=0.2,
                        wf_train=120, wf_test=40,
                        cost=cost_rate * spread_notional
                    )
                    
                    # Determine last signal consensus from latest z-score
                    spread_all = y_close - (alpha + beta * x_close)
                    last_spread = spread_all[-1]
                    last_z = (last_spread - train_mean) / (train_std if train_std > 0 else 1.0)
                    
                    entry_th = rep.best_params["entry"]
                    exit_th = rep.best_params["exit"]
                    if last_z > entry_th:
                        consensus = "SHORT SPREAD"
                    elif last_z < -entry_th:
                        consensus = "LONG SPREAD"
                    else:
                        consensus = "HOLD SPREAD"
                        
                    sig = {
                        "symbol": sym,
                        "consensus": consensus,
                        "avg_confidence": float(min(100.0, abs(last_z) / entry_th * 100.0)),
                        "details": [
                            {"model": "ADF Test", "signal": "STATIONARY" if coint_res.cointegrated else "UNIT_ROOT", "confidence": float(abs(coint_res.adf.stat)), "predicted_return": float(coint_res.adf.stat)},
                            {"model": "Hedge Ratio", "signal": f"{beta:.3f}", "confidence": 100.0, "predicted_return": beta}
                        ]
                    }
                    
                    # Construct Bundle
                    import numpy as np
                    from utils.types import Bundle
                    bundle = Bundle(**{
                        "results": {
                            "statarb": {
                                "model": None,
                                "metrics": {
                                    "val": {"rmse": float(-coint_res.adf.stat)},  # Minimize for best model selection
                                    "test": {
                                        "win_rate": rep.holdout_win_rate,
                                        "sharpe": rep.holdout_sharpe,
                                        "total_pnl": rep.holdout_total_pnl,
                                    }
                                }
                            }
                        },
                        "split": {
                            "df": df_y,
                            "X_tr": np.zeros((tv_end, 2)), "y_tr": np.zeros(tv_end), "idx_tr": common_idx[:tv_end], "i_tr": 0,
                            "X_val": np.zeros((0, 2)), "y_val": np.zeros(0), "idx_val": common_idx[tv_end:tv_end], "i_va": tv_end,
                            "X_test": np.zeros((len(y_close)-tv_end, 2)), "y_test": np.zeros(len(y_close)-tv_end), "idx_test": common_idx[tv_end:], "i_te": tv_end
                        },
                        "symbol": sym,
                        "df": df_y,
                        "with_news": False,
                        "with_micro": False,
                        "with_cross_asset": False,
                        "spec": spec,
                        "sample": bool(df_y.attrs.get("sample", False) or df_x.attrs.get("sample", False))
                    })
                    bundle["best_model_name"] = "statarb"
                    bundle["strategy_type"] = "statarb"
                    bundle["pbo"] = 0.0
                    bundle["deflated_p"] = rep.deflated_p_value
                    bundle["cpcv_mean_sharpe"] = rep.holdout_sharpe
                    bundle["cpcv_mean_dir_acc"] = rep.holdout_win_rate
                    bundle["shield_passed"] = bool(rep.deflated_p_value <= 0.05)
                    
                    bundle["statarb_params"] = {
                        "beta": beta,
                        "alpha": alpha,
                        "train_mean": train_mean,
                        "train_std": train_std,
                        "entry": entry_th,
                        "exit": exit_th,
                        "df_x": df_x,
                        "df_y": df_y,
                    }
                    
                    bundles[sym] = bundle
                    signals[sym] = sig
                    
                    self._state.set_bundle(sym, bundle)
                    self._state.set_signal(sym, sig)
                    
                    self._bus.publish_obj(
                        f"{TRAINING_COMPLETE}.{sym}",
                        TrainingCompletePayload(symbol=sym, signal=sig),
                    )
                    continue

                df = load_data(sym, spec=spec, allow_sample=allow_sample,
                               on_first=self._make_on_first())

                # Plan item E: classification (triple-barrier) is now the
                # primary "ML Directional" engine, replacing the MODEL_MAP
                # regression loop. Regression models/MODEL_MAP stay available
                # for direct/low-level use (walkforward, CPCV, scripts) but
                # are no longer what Train&Predict trains by default.
                if optimize:
                    self._bus.publish_obj(STATUS, StatusPayload(
                        text=f"[{sym}] Note: Optuna optimization applies to "
                             f"regression models only; classification engine "
                             f"trains xgb_clf with fixed params.",
                    ))

                from engine.classification_trainer import (
                    build_classifier, split_classification, PRODUCTION_CLASSIFIERS,
                )
                from engine.classifier_adapter import ClassifierSignalAdapter
                from engine.risk import vol_regime_threshold
                import numpy as np

                sp = split_classification(df, spec, labeling="triple_barrier",
                                          pt_mult=MODEL.pt_mult, sl_mult=MODEL.sl_mult)

                # G: freeze the no-trade vol-regime threshold from TRAIN ATR%
                # only (no look-ahead); atr_pct is core_feature_cols[2].
                atr_train_hist = np.array([])
                try:
                    atr_idx = list(FEATURES.core_feature_cols).index("atr_pct")
                    atr_train_hist = sp["X_tr"][:, atr_idx]
                except ValueError:
                    pass
                atr_threshold = (
                    vol_regime_threshold(atr_train_hist, RISK.vol_filter_pctile)
                    if RISK.use_vol_filter else float("inf")
                )

                results: Dict[str, Any] = {}
                for name in PRODUCTION_CLASSIFIERS:
                    clf = build_classifier(name)
                    res = clf.train(sp["X_tr"], sp["y_tr"], sp["X_val"], sp["y_val"])
                    res["test"] = clf.evaluate(sp["X_test"], sp["y_test"])
                    adapter = ClassifierSignalAdapter(
                        clf, pt_mult=MODEL.pt_mult, sl_mult=MODEL.sl_mult,
                        atr_threshold=atr_threshold, horizon=MODEL.pred_horizon)
                    results[name] = {"model": adapter, "metrics": res}

                # Meta-labeling's class-sign semantics don't hold for triple-
                # barrier labels (0/1/2, not signed returns); not wired here.
                # It stays available (and default-off, plan item C) for the
                # regression path via engine.meta_labeling directly.
                meta_model = None

                # Best classifier by validation balanced accuracy (labels are
                # HOLD-heavy under triple-barrier, so plain accuracy favours
                # always-HOLD).
                best_model_name = max(
                    results.keys(),
                    key=lambda k: results[k]["metrics"]["val"]["balanced_accuracy"],
                )

                # Deflated-p: adapter's signed pseudo-return vs. the REAL
                # forward return (sp["fwd_test"]), not the class label.
                best_mdl = results[best_model_name]["model"]
                test_preds = best_mdl.predict(sp["X_test"])
                fwd_test = sp["fwd_test"]
                _n = min(len(test_preds), len(fwd_test))
                test_preds = test_preds[:_n]
                fwd_test = fwd_test[:_n]
                mask = (test_preds != 0) & (fwd_test != 0) & np.isfinite(fwd_test)
                if mask.any():
                    wins = int(np.sum(np.sign(test_preds[mask]) == np.sign(fwd_test[mask])))
                    total_n = int(np.sum(mask))
                else:
                    wins, total_n = 0, 0

                # n_trials = number of classifiers compared (2): Sidak-deflate
                # for having picked the best of that small model-selection set.
                n_trials = len(results)

                from engine.validation_protocol import deflated_p_value
                def_p = deflated_p_value(wins, total_n, n_trials=n_trials)

                # CPCV expects a regression target_col, so the classification
                # shield instead reuses the purged walk-forward distribution
                # from engine.classification_walkforward (same purge/embargo
                # discipline, hit-rate instead of Sharpe).
                try:
                    from engine.classification_walkforward import run_classification
                    # Scale window sizes to the actual data length: the
                    # config defaults (756/189) are sized for years of daily
                    # data and can silently produce zero windows (empty
                    # summary) on a shorter BTC-only cache.
                    n_rows = len(df)
                    wf_train = min(BACKTEST.walkforward_train_size, max(150, n_rows // 3))
                    wf_test = min(BACKTEST.walkforward_test_size, max(30, n_rows // 10))
                    cwf = run_classification(
                        df, best_model_name, spec=spec, labeling="triple_barrier",
                        pt_mult=MODEL.pt_mult, sl_mult=MODEL.sl_mult,
                        purge=MODEL.pred_horizon,
                        train_size=wf_train, test_size=wf_test,
                    )
                    cwf_summary = cwf.get("summary") or {}
                    pbo = 100.0 - cwf_summary.get("pct_windows_above_50", 0.0)
                    cpcv_mean_sharpe = 0.0  # not computed by this walk-forward
                    cpcv_mean_dir_acc = cwf_summary.get("mean_hit_rate", 0.0)
                except Exception as e:
                    log.warning(f"Classification walk-forward shield failed for {sym}: {e}")
                    pbo = 100.0
                    cpcv_mean_sharpe = 0.0
                    cpcv_mean_dir_acc = 0.0

                shield_passed = bool(def_p <= 0.05)

                if not shield_passed:
                    log.warning(f"{sym} failed validation shield: deflated-p={def_p:.4f} > 0.05 (NO EDGE)")
                    self._bus.publish_obj(STATUS, StatusPayload(
                        text=f"{sym} failed validation shield (deflated-p={def_p:.3f} > 0.05). NO EDGE!",
                        level="warning",
                    ))
                else:
                    log.info(f"{sym} passed validation shield: deflated-p={def_p:.4f} <= 0.05")

                bundle = Bundle(**{
                    "results": results, "split": sp, "symbol": sym,
                    "df": df, "with_news": spec.news, "with_micro": spec.micro,
                    "with_cross_asset": spec.cross_asset, "spec": spec,
                    "sample": bool(df.attrs.get("sample", False)),
                })
                bundle["meta_model"] = meta_model
                bundle["best_model_name"] = best_model_name
                bundle["pbo"] = pbo
                bundle["deflated_p"] = def_p
                bundle["cpcv_mean_sharpe"] = cpcv_mean_sharpe
                bundle["cpcv_mean_dir_acc"] = cpcv_mean_dir_acc
                bundle["shield_passed"] = shield_passed

                if spec.news and "sentiment_avg" in df.columns:
                    from engine.news_analysis import report
                    bundle["news_analysis"] = report(df)

                if weighted:
                    sig = predict_weighted_ensemble(bundle)
                else:
                    sig = predict_from_bundle(bundle)

                bundles[sym] = bundle
                signals[sym] = sig

                # Store in AppState so UI can read them without JSON serialization
                self._state.set_bundle(sym, bundle)
                self._state.set_signal(sym, sig)

                self._bus.publish_obj(
                    f"{TRAINING_COMPLETE}.{sym}",
                    TrainingCompletePayload(symbol=sym, signal=sig),
                )

            except (DataFetchError, InsufficientDataError) as e:
                log.warning("%s: data error: %s", sym, e)
                failed[sym] = f"veri: {e}"
                self._bus.publish_obj(STATUS, StatusPayload(
                    text=f"Data error on {sym}: {e}", level="warning",
                ))
            except AITraderError as e:
                log.error("%s: %s: %s", sym, type(e).__name__, e)
                failed[sym] = f"{type(e).__name__}: {e}"
                self._bus.publish_obj(STATUS, StatusPayload(
                    text=f"Model error on {sym}: {e}", level="error",
                ))
            except Exception as e:
                log.exception("%s: unexpected error during training", sym)
                failed[sym] = f"{type(e).__name__}: {e}"
                self._bus.publish_obj(STATUS, StatusPayload(
                    text=f"Unexpected error on {sym}: {e}", level="error",
                ))

        # Summary
        ok_count = len(signals)
        total = len(syms)
        if not failed:
            self._bus.publish_obj(STATUS, StatusPayload(
                text=f"Tüm varlıklar eğitildi ({ok_count}/{total}). Gösterilen: {active_sym}",
            ))
        else:
            detail = "; ".join(f"{s.split('/')[0]}: {r}" for s, r in failed.items())[:300]
            msg = f"{ok_count}/{total} eğitildi. Başarısız → {detail}"
            log.warning("Training summary — failed symbols: %s", failed)
            self._bus.publish_obj(STATUS, StatusPayload(text=msg, level="warning"))

        return {"bundles": bundles, "signals": signals, "failed": failed}

    def _make_on_first(self) -> Optional[Callable]:
        """Create a callback for the first-run data fetch message."""
        bus = self._bus

        def on_first(s: str, years: int):
            bus.publish_obj(STATUS, StatusPayload(
                text=f"First run for {s}: fetching {years}y history (one-time, please wait)...",
            ))

        return on_first

    def shutdown(self):
        log.info("TrainingService shutting down")
        self._executor.shutdown(wait=False)
