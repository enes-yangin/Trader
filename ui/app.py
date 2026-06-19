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
        self.bt_results = None
        self.last_signal = None
        self.live_on = tk.BooleanVar(value=True)
        self.live_df = None
        self._live_job = None

        self._style()
        self._toolbar()
        self._layout()

        self.sym_var.trace_add("write", lambda *a: self._on_sym_change())
        self.after(500, self._live_tick)

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

        ttk.Button(bar, text="⬇ Fetch Data", style="TButton",
                   command=self._on_fetch).pack(side="left", padx=4, pady=8)

        ttk.Button(bar, text="Fetch All", style="TButton",
                   command=self._on_fetch_all).pack(side="left", padx=4, pady=8)

        ttk.Button(bar, text="Portfolio", style="TButton",
                   command=self._on_portfolio).pack(side="left", padx=4, pady=8)

        self.news_var = tk.BooleanVar(value=True)
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
        sym = self.sym_var.get()
        wn = self.news_var.get()

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
            optimize = self.optimize_var.get()
            weighted = self.weighted_var.get()
            spec = FeatureSpec(news=wn, micro=FEATURES.use_micro, cross_asset=self.xasset_var.get())
            df = load_data(sym, spec=spec, on_first=on_first)
            self.after(0, self.status.set, f"Training on {len(df)} rows...")
            sp = split(df, spec=spec)
            results = {}
            for name, cls in MODEL_MAP.items():
                if optimize and name in ("linear", "xgboost"):
                    self.after(0, self.status.set, f"Optimizing {name} (Optuna)...")
                    mdl, res, best_params, _ = build_optimized_model(name, sp)
                else:
                    mdl = cls(epochs=MODEL.lstm_params["epochs"]) if name == "lstm" else cls()
                    res = mdl.train(sp["X_tr"], sp["y_tr"], sp["X_val"], sp["y_val"])
                    res["test"] = mdl.evaluate(sp["X_test"], sp["y_test"])
                results[name] = {"model": mdl, "metrics": res}
            self.bundle = Bundle(**{
                "results": results, "split": sp, "symbol": sym,
                "df": df, "with_news": spec.news, "with_micro": spec.micro,
                "with_cross_asset": spec.cross_asset, "spec": spec,
                "sample": bool(df.attrs.get("sample", False)),
            })
            if spec.news and "sentiment_avg" in df.columns:
                from engine.news_analysis import report
                self.bundle["news_analysis"] = report(df)
            if weighted:
                sig = predict_weighted_ensemble(self.bundle)
            else:
                sig = predict_from_bundle(self.bundle)
            self.after(0, self._update_ui, sig)
        except (DataFetchError, InsufficientDataError) as e:
            log.warning(f"{sym}: data error: {e}")
            self.after(0, self.status.set, f"Data error: {e}")
        except AITraderError as e:
            log.error(f"{sym}: {type(e).__name__}: {e}")
            self.after(0, self.status.set, f"Error: {e}")
        except Exception as e:
            log.exception(f"{sym}: unexpected error during training")
            self.after(0, self.status.set, f"Unexpected error: {type(e).__name__}: {e}")

    def _update_ui(self, sig):
        self.last_signal = sig
        self.signal_card.update(sig)
        self.model_panel.update(sig.get("details", []))

        df = self.bundle["df"]
        self.chart.update(df, symbol=sig.get("symbol", ""))
        self.news_panel.update(self.bundle.get("news_analysis"))

        warn = "  ⚠ SAMPLE DATA" if self.bundle.get("sample") else ""
        self.status.set(
            f"{sig['symbol']} — {sig['consensus']} "
            f"({sig['avg_confidence']:.1f}%) — {len(df)} rows trained{warn}"
        )

    def _on_sym_change(self):
        self.live_df = None

    def _live_tick(self):
        if self.live_on.get():
            sym = self.sym_var.get()
            threading.Thread(target=self._fetch_live, args=(sym,),
                             daemon=True).start()
            threading.Thread(target=self._fetch_ob, args=(sym,),
                             daemon=True).start()
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
            sig = orderbook.live_signal(sym)
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
            df = dataset.build(sym, with_news=wn, force=True)
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
            res = dataset.build_many(syms, with_news=wn, force=True, progress=prog)
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
            self.bt_results = run_all(self.bundle)
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
