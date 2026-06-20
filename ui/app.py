import tkinter as tk
from tkinter import ttk
import threading
import ccxt
from ui.chart import ChartWidget
from ui.portfolio_panel import PortfolioWindow
from ui.panels import (
    SignalCard, ModelDetailPanel, MetricsPanel, OrderBookPanel, StatusBar,
    NewsAnalysisPanel
)
from engine.predictor import predict_from_bundle
from engine.backtester import run_all
from data import dataset, store
from data.fetcher import fetch_live
from data import orderbook
from utils.config import DATA
from utils.exceptions import AITraderError, DataFetchError, InsufficientDataError
from utils.logger import get_logger

log = get_logger("ui")

BG = "#1e1e2e"
BG2 = "#181825"
BG3 = "#313244"
FG = "#cdd6f4"
FG2 = "#a6adc8"
ACCENT = "#89b4fa"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Trader — ML Prediction Engine")
        self.geometry("1280x820")
        self.configure(bg=BG)
        self.minsize(1000, 650)

        self.bundle = None
        self.bundles = {}
        self.signals = {}
        self.bt_results = None
        self.bt_results_cache = {}
        self.last_signal = None
        self.live_on = tk.BooleanVar(value=True)
        self.live_df = None
        self._live_job = None
        self.ob_tracker = orderbook.OrderBookTracker()

        self._style()
        self._toolbar()
        self._layout()

        self.sym_var.trace_add("write", lambda *a: self._on_sym_change())
        self.after(500, self._live_tick)
        self.after(1000, self._startup_preload)
        self.after(1500, self._startup_paper_trader)

    def _startup_preload(self):
        self.status.set("Checking and pre-downloading data for all symbols...")
        wn = self.news_var.get()
        from utils.config import FEATURES, DATA
        threading.Thread(
            target=self._run_preload_all,
            args=(DATA.crypto_symbols, wn, FEATURES.use_cross_asset),
            daemon=True
        ).start()

    def _run_preload_all(self, syms, wn, xasset):
        try:
            from data import dataset
            # Preload the currently selected symbol first so it is ready immediately
            active_sym = self.sym_var.get()
            self.after(0, self.status.set, f"Preloading active symbol: {active_sym}...")
            dataset.ensure(active_sym, with_news=wn)
            
            # Preload the remaining symbols in the background
            remaining_syms = [s for s in syms if s != active_sym]
            for i, sym in enumerate(remaining_syms):
                self.after(0, self.status.set, f"Checking/Downloading data [{i+1}/{len(remaining_syms)}]: {sym}...")
                dataset.ensure(sym, with_news=wn)
                
            if xasset:
                self.after(0, self.status.set, "Downloading reference data...")
                from data.cross_asset import fetch_reference_data
                fetch_reference_data()
            self.after(0, self.status.set, "All data ready.")
        except Exception as e:
            self.after(0, self.status.set, f"Preload error: {e}")

    def _style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background=BG, foreground=FG, fieldbackground=BG3,
                     borderwidth=0)
        s.configure("TCombobox", fieldbackground=BG3, background=BG3,
                     foreground=FG, arrowcolor=FG)
        s.map("TCombobox", fieldbackground=[("readonly", BG3)])
        s.configure("Accent.TButton", background=ACCENT, foreground=BG,
                     font=("Consolas", 10, "bold"), padding=(12, 6))
        s.map("Accent.TButton", background=[("active", "#7ba4e8")])
        s.configure("TButton", background=BG3, foreground=FG,
                     font=("Consolas", 9), padding=(8, 4))
        s.map("TButton", background=[("active", "#45475a")])

    def _toolbar(self):
        bar = tk.Frame(self, bg=BG2, height=48)
        bar.pack(fill="x", side="top")
        bar.pack_propagate(False)

        tk.Label(bar, text="AI TRADER", font=("Consolas", 14, "bold"),
                 bg=BG2, fg=ACCENT).pack(side="left", padx=12)

        syms = DATA.crypto_symbols
        self.sym_var = tk.StringVar(value=syms[0])
        cb = ttk.Combobox(bar, textvariable=self.sym_var, values=syms,
                          state="readonly", width=14, font=("Consolas", 10))
        cb.pack(side="left", padx=(20, 8), pady=10)

        ttk.Button(bar, text="▶ Train & Predict", style="Accent.TButton",
                   command=self._on_run).pack(side="left", padx=4, pady=8)

        ttk.Button(bar, text="Backtest", style="TButton",
                   command=self._on_backtest).pack(side="left", padx=4, pady=8)

        # Capital ($)
        tk.Label(bar, text="Capital:", bg=BG2, fg=FG2, font=("Consolas", 9)).pack(side="left", padx=(8, 2))
        self.capital_var = tk.StringVar(value="1000")
        capital_entry = tk.Entry(bar, textvariable=self.capital_var, bg=BG3, fg=FG,
                                 insertbackground=FG, width=6, font=("Consolas", 9), bd=0,
                                 highlightthickness=1, highlightbackground=BG3, highlightcolor=ACCENT)
        capital_entry.pack(side="left", padx=2, pady=10)

        # Period
        tk.Label(bar, text="Period:", bg=BG2, fg=FG2, font=("Consolas", 9)).pack(side="left", padx=(8, 2))
        self.period_var = tk.StringVar(value="1 Year")
        period_cb = ttk.Combobox(bar, textvariable=self.period_var,
                                 values=["1 Month", "6 Months", "1 Year", "Full Test Set"],
                                 state="readonly", width=13, font=("Consolas", 9))
        period_cb.pack(side="left", padx=2, pady=10)

        ttk.Button(bar, text="⬇ Fetch Data", style="TButton",
                   command=self._on_fetch).pack(side="left", padx=4, pady=8)

        ttk.Button(bar, text="Fetch All", style="TButton",
                   command=self._on_fetch_all).pack(side="left", padx=4, pady=8)

        ttk.Button(bar, text="Portfolio", style="TButton",
                   command=self._on_portfolio).pack(side="left", padx=4, pady=8)

        self.news_var = tk.BooleanVar(value=False)
        chk = tk.Checkbutton(bar, text="News (FinBERT)", variable=self.news_var,
                             bg=BG2, fg=FG, selectcolor=BG3,
                             activebackground=BG2, activeforeground=FG,
                             font=("Consolas", 9))
        chk.pack(side="left", padx=12, pady=8)

        from utils.config import FEATURES, OPTIMIZATION
        self.xasset_var = tk.BooleanVar(value=FEATURES.use_cross_asset)
        chk_x = tk.Checkbutton(bar, text="Cross-Asset", variable=self.xasset_var,
                               bg=BG2, fg=FG, selectcolor=BG3,
                               activebackground=BG2, activeforeground=FG,
                               font=("Consolas", 9))
        chk_x.pack(side="left", padx=4, pady=8)

        self.fg_var = tk.BooleanVar(value=FEATURES.use_reference)
        chk_fg = tk.Checkbutton(bar, text="Fear & Greed", variable=self.fg_var,
                                bg=BG2, fg=FG, selectcolor=BG3,
                                activebackground=BG2, activeforeground=FG,
                                font=("Consolas", 9))
        chk_fg.pack(side="left", padx=4, pady=8)

        self.optimize_var = tk.BooleanVar(value=False)
        chk_o = tk.Checkbutton(bar, text="Optimize", variable=self.optimize_var,
                               bg=BG2, fg=FG, selectcolor=BG3,
                               activebackground=BG2, activeforeground=FG,
                               font=("Consolas", 9))
        chk_o.pack(side="left", padx=4, pady=8)

        self.weighted_var = tk.BooleanVar(value=OPTIMIZATION.ensemble_weighted)
        chk_w = tk.Checkbutton(bar, text="Weighted", variable=self.weighted_var,
                               bg=BG2, fg=FG, selectcolor=BG3,
                               activebackground=BG2, activeforeground=FG,
                               font=("Consolas", 9))
        chk_w.pack(side="left", padx=4, pady=8)

        self.sample_var = tk.BooleanVar(value=False)
        chk_s = tk.Checkbutton(bar, text="Synthetic", variable=self.sample_var,
                               bg=BG2, fg=FG, selectcolor=BG3,
                               activebackground=BG2, activeforeground=FG,
                               font=("Consolas", 9))
        chk_s.pack(side="left", padx=4, pady=8)

        self.live_chk = tk.Checkbutton(bar, text="● Live", bg=BG2, fg="#a6e3a1",
                                       variable=self.live_on,
                                       selectcolor=BG3, activebackground=BG2,
                                       activeforeground="#a6e3a1",
                                       font=("Consolas", 9))
        self.live_chk.pack(side="left", padx=4, pady=8)

    def _layout(self):
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)

        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="both", expand=True)

        self.chart = ChartWidget(left, figsize=(9, 6))

        self.news_panel = NewsAnalysisPanel(left)
        self.news_panel.pack(fill="x", padx=6, pady=(3, 6))

        right = tk.Frame(body, bg=BG2, width=320)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        self.signal_card = SignalCard(right)
        self.signal_card.pack(fill="x", padx=6, pady=(6, 3))

        self.model_panel = ModelDetailPanel(right)
        self.model_panel.pack(fill="x", padx=6, pady=3)

        self.metrics_panel = MetricsPanel(right)
        self.metrics_panel.pack(fill="both", expand=True, padx=6, pady=3)

        div = tk.Frame(right, bg=ACCENT, height=2)
        div.pack(fill="x", padx=6, pady=2)

        self.ob_panel = OrderBookPanel(right)
        self.ob_panel.pack(fill="x", padx=6, pady=(3, 6))

        self.status = StatusBar(self)
        self.status.pack(fill="x", side="bottom")

    def _on_run(self):
        wn = self.news_var.get()
        msg = "Training models (with news sentiment)..." if wn else "Training models..."
        self.status.set(msg)
        threading.Thread(target=self._run_pipeline, daemon=True).start()

    def _run_pipeline(self):
        active_sym = self.sym_var.get()
        syms = [active_sym] + [s for s in DATA.crypto_symbols if s != active_sym]
        wn = self.news_var.get()
        optimize = self.optimize_var.get()
        weighted = self.weighted_var.get()

        def on_first(s, years):
            self.after(0, self.status.set,
                       f"First run for {s}: fetching {years}y history "
                       f"(one-time, please wait)...")

        try:
            from engine.trainer import load_data, split
            from engine.trainer import MODEL_MAP
            from engine.optimizer import build_optimized_model
            from engine.ensemble import predict_weighted_ensemble
            from utils.config import FEATURES, MODEL
            from utils.types import FeatureSpec, Bundle
            spec = FeatureSpec(
                news=wn,
                micro=FEATURES.use_micro,
                cross_asset=self.xasset_var.get(),
                smooth=FEATURES.use_smoothing,
                reference=self.fg_var.get(),
                orderbook=FEATURES.use_orderbook,
                macro_events=FEATURES.use_macro_events,
                social=FEATURES.use_social
            )

            self.bundles = {}
            self.signals = {}
            failed = {}  # sym -> reason (so the summary can name what broke)

            for idx, sym in enumerate(syms):
                try:
                    self.after(0, self.status.set, f"Training {sym} [{idx+1}/{len(syms)}]...")
                    df = load_data(sym, spec=spec, allow_sample=self.sample_var.get(), on_first=on_first)
                    sp = split(df, spec=spec)
                    results = {}
                    for name, cls in MODEL_MAP.items():
                        if optimize and name in ("linear", "xgboost"):
                            self.after(0, self.status.set, f"[{sym}] Optimizing {name} (Optuna)...")
                            mdl, res, best_params, _ = build_optimized_model(name, sp)
                        else:
                            mdl = cls(epochs=MODEL.lstm_params["epochs"]) if name == "lstm" else cls()
                            res = mdl.train(sp["X_tr"], sp["y_tr"], sp["X_val"], sp["y_val"])
                            res["test"] = mdl.evaluate(sp["X_test"], sp["y_test"])
                        results[name] = {"model": mdl, "metrics": res}
                    
                    from engine.meta_labeling import train_meta_model
                    meta_model = train_meta_model(sp, results)
                    
                    bundle = Bundle(**{
                        "results": results, "split": sp, "symbol": sym,
                        "df": df, "with_news": spec.news, "with_micro": spec.micro,
                        "with_cross_asset": spec.cross_asset, "spec": spec,
                        "sample": bool(df.attrs.get("sample", False)),
                    })
                    bundle["meta_model"] = meta_model
                    if spec.news and "sentiment_avg" in df.columns:
                        from engine.news_analysis import report
                        bundle["news_analysis"] = report(df)
                    
                    if weighted:
                        sig = predict_weighted_ensemble(bundle)
                    else:
                        sig = predict_from_bundle(bundle)

                    self.bundles[sym] = bundle
                    self.signals[sym] = sig

                    if sym == self.sym_var.get():
                        self.bundle = bundle
                        self.after(0, self._update_ui, sig)
                except (DataFetchError, InsufficientDataError) as e:
                    log.warning(f"{sym}: data error: {e}")
                    failed[sym] = f"veri: {e}"
                    self.after(0, self.status.set, f"Data error on {sym}: {e}")
                except AITraderError as e:
                    log.error(f"{sym}: {type(e).__name__}: {e}")
                    failed[sym] = f"{type(e).__name__}: {e}"
                    self.after(0, self.status.set, f"Model error on {sym}: {e}")
                except Exception as e:
                    log.exception(f"{sym}: unexpected error during training")
                    failed[sym] = f"{type(e).__name__}: {e}"
                    self.after(0, self.status.set, f"Unexpected error on {sym}: {e}")

            # Training summary — name exactly which symbols failed and why.
            active_sym = self.sym_var.get()
            ok_count = len(self.signals)
            total = len(syms)
            if not failed:
                self.after(0, lambda: self.status.set(
                    f"Tüm varlıklar eğitildi ({ok_count}/{total}). Gösterilen: {active_sym}"))
            else:
                detail = "; ".join(f"{s.split('/')[0]}: {r}" for s, r in failed.items())[:300]
                msg = f"{ok_count}/{total} eğitildi. Başarısız → {detail}"
                log.warning(f"Training summary — failed symbols: {failed}")
                self.after(0, lambda m=msg: self.status.set(m))

        except Exception as e:
            log.exception("Unexpected error in pipeline runner")
            self.after(0, self.status.set, f"Pipeline error: {e}")
            from tkinter import messagebox
            self.after(0, lambda: messagebox.showerror(
                "Beklenmeyen Sistem Hatası",
                f"Eğitim hattı başlatılamadı:\n\n{type(e).__name__}: {e}"
            ))

    def _update_ui(self, sig):
        self.last_signal = sig
        self.signal_card.update(sig)
        self.model_panel.update(sig.get("details", []))

        df = self.bundle["df"]
        self.chart.update(df, symbol=sig.get("symbol", ""))
        self.news_panel.update(self.bundle.get("news_analysis"))

        fng_text = ""
        if "fear_greed" in df.columns:
            last_fng_transformed = float(df["fear_greed"].iloc[-1])
            last_fng_raw = int(round(last_fng_transformed * 50.0 + 50.0))
            if last_fng_raw <= 25:
                fng_cat = "Extreme Fear"
            elif last_fng_raw <= 45:
                fng_cat = "Fear"
            elif last_fng_raw <= 55:
                fng_cat = "Neutral"
            elif last_fng_raw <= 75:
                fng_cat = "Greed"
            else:
                fng_cat = "Extreme Greed"
            fng_text = f" | Fear & Greed: {last_fng_raw} ({fng_cat})"

        warn = "  ⚠ SAMPLE DATA" if self.bundle.get("sample") else ""
        self.status.set(
            f"{sig['symbol']} — {sig['consensus']} "
            f"({sig['avg_confidence']:.1f}%) — {len(df)} rows trained{fng_text}{warn}"
        )

        # Log paper prediction if it is not sample data
        if not self.bundle.get("sample", False):
            try:
                from models.base_model import dynamic_threshold
                from utils.config import SIGNAL, MODEL
                from engine.paper_trader import log_prediction
                
                atr_pct = float(df["atr_pct"].iloc[-1]) if "atr_pct" in df.columns else 0.0
                buy_th, sell_th = dynamic_threshold(atr_pct, SIGNAL.buy_threshold, SIGNAL.sell_threshold)
                
                log_prediction(
                    symbol=sig["symbol"],
                    entry_price=sig["last_close"],
                    consensus=sig["consensus"],
                    confidence=sig["avg_confidence"],
                    buy_threshold=buy_th,
                    sell_threshold=sell_th,
                    horizon_days=MODEL.pred_horizon
                )
                
                # Re-evaluate paper trades in a separate thread
                threading.Thread(target=self._run_paper_trader_evaluation, daemon=True).start()
            except Exception as e:
                log.exception("Error logging paper trade prediction")

    def _startup_paper_trader(self):
        self.status.set("Evaluating past predictions...")
        threading.Thread(target=self._run_paper_trader_evaluation, daemon=True).start()

    def _run_paper_trader_evaluation(self):
        try:
            from engine.paper_trader import evaluate_past_predictions
            stats = evaluate_past_predictions()
            self.after(0, self._update_accuracy_display, stats)
        except Exception as e:
            log.exception("Error during paper trader evaluation")

    def _update_accuracy_display(self, stats):
        if stats["total_resolved"] > 0:
            self.status.set_accuracy(f"Live Acc: {stats['accuracy_pct']:.1f}% ({stats['correct_resolved']}/{stats['total_resolved']})")
        else:
            self.status.set_accuracy("Live Acc: N/A")

    def _on_sym_change(self):
        self.live_df = None
        self.ob_tracker.reset()  # drop smoothed imbalance/state from the old symbol
        sym = self.sym_var.get()
        if hasattr(self, "bundles") and sym in self.bundles:
            self.bundle = self.bundles[sym]
            self.last_signal = self.signals[sym]
            self._update_ui(self.last_signal)
            
            if hasattr(self, "bt_results_cache") and sym in self.bt_results_cache:
                self.bt_results = self.bt_results_cache[sym]
                self._update_backtest()
            else:
                self.bt_results = None
                self.metrics_panel.clear()
        else:
            self.bundle = None
            self.last_signal = None
            self.bt_results = None
            self.signal_card.clear()
            self.model_panel.clear()
            self.metrics_panel.clear()
            self.news_panel.clear()

    def _live_tick(self):
        if self.live_on.get():
            sym = self.sym_var.get()
            threading.Thread(target=self._fetch_live, args=(sym,),
                             daemon=True).start()
            threading.Thread(target=self._fetch_ob, args=(sym,),
                             daemon=True).start()
        if self._live_job is not None:
            try:
                self.after_cancel(self._live_job)
            except Exception:
                pass
        self._live_job = self.after(DATA.live_refresh_s * 1000, self._live_tick)

    def _fetch_live(self, sym):
        try:
            df = fetch_live(sym)
            self.live_df = df
            self.after(0, self._render_live, sym, df)
        except (ccxt.BaseError, ConnectionError, TimeoutError, OSError) as e:
            log.debug(f"{sym}: live price fetch failed: {type(e).__name__}: {e}")

    def _fetch_ob(self, sym):
        try:
            ob = orderbook.fetch_orderbook(sym)
            sig = self.ob_tracker.update(ob, sym=sym)  # EMA smoothing + hysteresis
            self.after(0, self.ob_panel.update, sig)
        except (ccxt.BaseError, ConnectionError, TimeoutError, OSError) as e:
            log.debug(f"{sym}: order book fetch failed: {type(e).__name__}: {e}")

    def _render_live(self, sym, df):
        if self.bundle is not None:
            return
        last = float(df["close"].iloc[-1])
        self.chart.update(df, symbol=f"{sym}  (live)", last_n=min(len(df), 80))
        self.status.set(f"{sym} live: {last:,.2f}")

    def _on_fetch(self):
        sym = self.sym_var.get()
        self.status.set(f"Fetching {DATA.hist_years}y data for {sym}...")
        threading.Thread(target=self._run_fetch, args=(sym,), daemon=True).start()

    def _run_fetch(self, sym):
        wn = self.news_var.get()
        try:
            df = dataset.build(sym, with_news=wn, force=True, allow_sample=self.sample_var.get())
            n = len(df)
            rng = f"{df.index.min().date()} → {df.index.max().date()}"
            self.after(0, self.status.set,
                       f"{sym}: {n} rows cached ({rng})")
        except (DataFetchError, InsufficientDataError) as e:
            log.warning(f"{sym}: fetch failed: {e}")
            self.after(0, self.status.set, f"Fetch error: {e}")
        except AITraderError as e:
            log.error(f"{sym}: {type(e).__name__}: {e}")
            self.after(0, self.status.set, f"Fetch error: {e}")
        except Exception as e:
            log.exception(f"{sym}: unexpected fetch error")
            self.after(0, self.status.set, f"Unexpected fetch error: {type(e).__name__}: {e}")

    def _on_fetch_all(self):
        syms = DATA.crypto_symbols
        self.status.set(f"Fetching {len(syms)} symbols ({DATA.hist_years}y)...")
        threading.Thread(target=self._run_fetch_all, args=(syms,), daemon=True).start()

    def _run_fetch_all(self, syms):
        wn = self.news_var.get()

        def prog(i, total, s):
            self.after(0, self.status.set, f"[{i}/{total}] {s}...")

        try:
            res = dataset.build_many(syms, with_news=wn, force=True, progress=prog, allow_sample=self.sample_var.get())
            errs = res.get("_errors", {})
            ok = len(syms) - len(errs)
            mb = store.cache_size_mb()
            if errs:
                failed = ", ".join(list(errs.keys())[:3])
                self.after(0, self.status.set,
                           f"{ok}/{len(syms)} cached ({mb:.1f} MB) — failed: {failed}")
            else:
                self.after(0, self.status.set,
                           f"All {len(syms)} symbols cached ({mb:.1f} MB)")
        except Exception as e:
            log.exception("Fetch all: unexpected error")
            self.after(0, self.status.set, f"Fetch all error: {type(e).__name__}: {e}")

    def _on_backtest(self):
        if self.bundle is None:
            self.status.set("Train models first")
            return
        self.status.set("Running backtest...")
        threading.Thread(target=self._run_backtest, daemon=True).start()

    def _run_backtest(self):
        try:
            capital_val = 1000.0
            try:
                capital_val = float(self.capital_var.get())
            except ValueError:
                self.after(0, self.status.set, "Invalid capital input, using $1000")

            p_map = {
                "1 Month": 30,
                "6 Months": 180,
                "1 Year": 365,
                "Full Test Set": None
            }
            p_days = p_map.get(self.period_var.get(), None)

            if hasattr(self, "bundles") and len(self.bundles) > 1:
                from engine.backtester import run_portfolio_all
                self.bt_results = run_portfolio_all(self.bundles, capital=capital_val, period_days=p_days)
                for sym in self.bundles.keys():
                    self.bt_results_cache[sym] = self.bt_results
            else:
                self.bt_results = run_all(self.bundle, capital=capital_val, period_days=p_days)
                if self.bundle and "symbol" in self.bundle:
                    self.bt_results_cache[self.bundle["symbol"]] = self.bt_results
            self.after(0, self._update_backtest)
        except AITraderError as e:
            log.error(f"Backtest error: {type(e).__name__}: {e}")
            self.after(0, self.status.set, f"Backtest error: {e}")
        except Exception as e:
            log.exception("Unexpected backtest error")
            self.after(0, self.status.set, f"Backtest error: {type(e).__name__}: {e}")

    def _update_backtest(self):
        self.metrics_panel.update(self.bt_results)

        best = None
        best_eq = None
        best_trades = None
        for name, r in self.bt_results.items():
            if r["equity"] is not None and len(r["equity"]) > 0:
                trades = r["trades"]
                eq = r["equity"]
                if best_eq is None or eq["equity"].iloc[-1] > best_eq["equity"].iloc[-1]:
                    best = name
                    best_eq = eq
                    best_trades = trades

        df = self.bundle["df"]
        sym = self.bundle["symbol"]
        self.chart.update(
            df, trades_df=best_trades if best else None,
            eq_df=best_eq, symbol=f"{sym} [{best}]" if best else sym
        )
        self.status.set("Backtest complete")

    def _on_portfolio(self):
        default_signal = self.last_signal.get("consensus") if self.last_signal else None
        PortfolioWindow(self, default_symbol=self.sym_var.get(), default_signal=default_signal)

    def destroy(self):
        if self._live_job is not None:
            try:
                self.after_cancel(self._live_job)
            except Exception:
                pass
            self._live_job = None
        super().destroy()
