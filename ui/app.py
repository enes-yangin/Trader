"""
TraderAI v2 — Thin UI Controller.

The App class is now a pure presentation + orchestration layer.
Data fetching, training, and backtesting are delegated to services
that communicate via ZeroMQ PUB/SUB (inproc://).

Phase 1 (v2 Kurumsal Mimari): ZeroMQ entegrasyonu, ThreadPoolExecutor,
God Object parçalanması.
"""

import tkinter as tk
from tkinter import ttk
import threading

import pandas as pd

from ui.chart import ChartWidget
from ui.portfolio_panel import PortfolioWindow
from ui.panels import (
    SignalCard, ModelDetailPanel, MetricsPanel, OrderBookPanel, StatusBar,
    NewsAnalysisPanel, ValidationShieldPanel
)
from data import orderbook
from utils.config import DATA, FEATURES, OPTIMIZATION
from utils.logger import get_logger

from messaging.bus import MessageBus
from messaging.topics import (
    LIVE_PRICE, LIVE_ORDERBOOK, TRAINING_COMPLETE, BACKTEST_COMPLETE,
    STATUS, PAPER_ACCURACY, ERROR,
)
from services.app_state import AppState
from services.data_service import DataService
from services.training_service import TrainingService
from services.backtest_service import BacktestService

log = get_logger("ui")

BG = "#1e1e2e"
BG2 = "#181825"
BG3 = "#313244"
FG = "#cdd6f4"
FG2 = "#a6adc8"
ACCENT = "#89b4fa"


class App(tk.Tk):
    """TraderAI main window — thin controller (v2 Phase 1)."""

    def __init__(self):
        super().__init__()
        self.title("AI Trader — ML Prediction Engine")
        self.geometry("1280x820")
        self.configure(bg=BG)
        self.minsize(1000, 650)

        # UI-owned widget state (tkinter variables)
        self.live_on = tk.BooleanVar(value=True)
        self._live_job = None

        # OrderBook tracker — stateful EMA + hysteresis, owned by UI
        self.ob_tracker = orderbook.OrderBookTracker()

        # ── v2: Services ────────────────────────────────────────
        self._init_services()

        # ── UI construction ─────────────────────────────────────
        self._style()
        self._toolbar()
        self._layout()

        # ── Wiring ──────────────────────────────────────────────
        self._subscribe_to_bus()
        self.sym_var.trace_add("write", lambda *a: self._on_sym_change())

        # ── Startup timers ──────────────────────────────────────
        self.after(500, self._live_tick)
        self.after(1000, self._startup_preload)
        self.after(1500, self._startup_paper_trader)

    # ═══════════════════════════════════════════════════════════
    #  v2: Service lifecycle
    # ═══════════════════════════════════════════════════════════

    def _init_services(self):
        """Create the message bus, state manager, and all services."""
        self._bus = MessageBus()
        self._state = AppState()

        self.data_service = DataService(self._bus)
        self.training_service = TrainingService(self._bus, self._state)
        self.backtest_service = BacktestService(self._bus, self._state)

    def _subscribe_to_bus(self):
        """Wire up pub/sub topic handlers with tkinter main-thread dispatch."""
        bus = self._bus
        root = self  # ui_root for after(0, ...) dispatch

        bus.subscribe(LIVE_PRICE, self._on_live_price, ui_root=root)
        bus.subscribe(LIVE_ORDERBOOK, self._on_live_orderbook, ui_root=root)
        bus.subscribe(TRAINING_COMPLETE, self._on_training_complete, ui_root=root)
        bus.subscribe(BACKTEST_COMPLETE, self._on_backtest_complete, ui_root=root)
        bus.subscribe(STATUS, self._on_status_update, ui_root=root)
        bus.subscribe(PAPER_ACCURACY, self._on_accuracy_update, ui_root=root)
        bus.subscribe(ERROR, self._on_error, ui_root=root)

    def _shutdown_services(self):
        """Graceful shutdown: cancel timers, stop services, close bus."""
        if self._live_job is not None:
            try:
                self.after_cancel(self._live_job)
            except Exception:
                pass
            self._live_job = None

        for svc in (self.data_service, self.training_service, self.backtest_service):
            try:
                svc.shutdown()
            except Exception:
                pass

        try:
            self._bus.close()
        except Exception:
            pass

    # ═══════════════════════════════════════════════════════════
    #  Bus → UI callbacks (run on main thread via after(0, ...))
    # ═══════════════════════════════════════════════════════════

    def _on_live_price(self, msg: dict):
        """Handle data.live.price — update chart with live data."""
        sym = msg.get("symbol", "")
        df_json = msg.get("df_json", "")
        if df_json:
            try:
                df = pd.read_json(df_json)
            except Exception:
                return
        else:
            return

        self._state.set_live_df(sym, df)

        # Only render live if no trained bundle exists for this symbol
        if self._state.get_bundle(sym) is not None:
            return
        last = float(df["close"].iloc[-1])
        self.chart.update(df, symbol=f"{sym}  (live)", last_n=min(len(df), 80))
        self.status.set(f"{sym} live: {last:,.2f}")

    def _on_live_orderbook(self, msg: dict):
        """Handle data.live.orderbook — update OB tracker and panel."""
        sym = msg.get("symbol", "")
        # msg is the full raw orderbook dict (bids/asks lists)
        # Reconstruct: remove symbol key, pass the rest as orderbook
        ob_dict = {
            "bids": msg.get("bids", []),
            "asks": msg.get("asks", []),
            "symbol": sym,
        }
        try:
            sig = self.ob_tracker.update(ob_dict, sym=sym)
            self.ob_panel.update(sig)
        except Exception:
            log.debug("OB tracker update failed for %s", sym, exc_info=True)

    def _on_training_complete(self, msg: dict):
        """Handle ml.training.complete.{sym} — update UI for completed symbol."""
        sym = msg.get("symbol", "")
        sig = msg.get("signal", {})
        if not sym or not sig:
            return

        self._state.set_signal(sym, sig)
        self._state.last_signal = sig

        # Only update UI if the completed symbol is currently selected
        if sym == self.sym_var.get():
            self._update_ui(sig)

    def _on_backtest_complete(self, msg: dict):
        """Handle backtest.complete.{sym} — push backtest results to UI."""
        sym = msg.get("symbol", "")
        if sym == self.sym_var.get():
            self._update_backtest()

    def _on_status_update(self, msg: dict):
        """Handle ui.status — update the status bar."""
        text = msg.get("text", "")
        level = msg.get("level", "info")
        self.status.set(text)

    def _on_accuracy_update(self, msg: dict):
        """Handle paper.accuracy — update paper trader display."""
        stats = msg
        self._update_accuracy_display(stats)

    def _on_error(self, msg: dict):
        """Handle system.error — show error in status bar."""
        text = msg.get("message", msg.get("text", "Unknown error"))
        self.status.set(f"Error: {text}")

    # ═══════════════════════════════════════════════════════════
    #  UI Construction (unchanged)
    # ═══════════════════════════════════════════════════════════

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

        syms = list(DATA.crypto_symbols)
        self.sym_var = tk.StringVar(value=syms[0])
        cb = ttk.Combobox(bar, textvariable=self.sym_var, values=syms,
                          state="readonly", width=14, font=("Consolas", 10))
        cb.pack(side="left", padx=(20, 8), pady=10)

        # Strategy Selection
        tk.Label(bar, text="Strategy:", bg=BG2, fg=FG2, font=("Consolas", 9)).pack(side="left", padx=(8, 2))
        self.strategy_var = tk.StringVar(value="ML Directional")
        self.strategy_cb = ttk.Combobox(bar, textvariable=self.strategy_var,
                                         values=["ML Directional", "Stat-Arb (vs BTC)"],
                                         state="readonly", width=18, font=("Consolas", 9))
        self.strategy_cb.pack(side="left", padx=2, pady=10)

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

        self.shield_panel = ValidationShieldPanel(right)
        self.shield_panel.pack(fill="x", padx=6, pady=3)

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

    # ═══════════════════════════════════════════════════════════
    #  Startup tasks
    # ═══════════════════════════════════════════════════════════

    def _startup_preload(self):
        """Kick off background preload of all symbol data (v2: via DataService)."""
        self.status.set("Checking and pre-downloading data for all symbols...")
        self.data_service.preload_all(
            syms=DATA.crypto_symbols,
            active_sym=self.sym_var.get(),
            with_news=self.news_var.get(),
            use_cross_asset=FEATURES.use_cross_asset,
        )

    # ═══════════════════════════════════════════════════════════
    #  Button handlers → thin delegators to services
    # ═══════════════════════════════════════════════════════════

    def _on_run(self):
        """Train & Predict button → TrainingService."""
        active_sym = self.sym_var.get()
        syms = tuple([active_sym] + [s for s in DATA.crypto_symbols if s != active_sym])

        wn = self.news_var.get()
        msg = "Training models (with news sentiment)..." if wn else "Training models..."
        self.status.set(msg)

        self.training_service.train_pipeline(
            active_sym=active_sym,
            syms=syms,
            with_news=wn,
            optimize=self.optimize_var.get(),
            weighted=self.weighted_var.get(),
            use_cross_asset=self.xasset_var.get(),
            use_micro=FEATURES.use_micro,
            use_smoothing=FEATURES.use_smoothing,
            use_reference=self.fg_var.get(),
            use_orderbook=FEATURES.use_orderbook,
            use_macro_events=FEATURES.use_macro_events,
            use_social=FEATURES.use_social,
            allow_sample=self.sample_var.get(),
            strategy=self.strategy_var.get(),
        )

    def _on_fetch(self):
        """Fetch Data button → DataService."""
        sym = self.sym_var.get()
        self.status.set(f"Fetching {DATA.hist_years}y data for {sym}...")
        self.data_service.fetch_symbol(
            sym, with_news=self.news_var.get(), force=True,
            allow_sample=self.sample_var.get(),
        )

    def _on_fetch_all(self):
        """Fetch All button → DataService."""
        syms = DATA.crypto_symbols
        self.status.set(f"Fetching {len(syms)} symbols ({DATA.hist_years}y)...")
        self.data_service.fetch_all_symbols(
            syms, with_news=self.news_var.get(),
            allow_sample=self.sample_var.get(),
        )

    def _on_backtest(self):
        """Backtest button → BacktestService."""
        sym = self.sym_var.get()
        if self._state.get_bundle(sym) is None:
            self.status.set("Train models first")
            return

        self.status.set("Running backtest...")

        try:
            capital_val = float(self.capital_var.get())
        except ValueError:
            capital_val = 1000.0
            self.status.set("Invalid capital input, using $1000")

        p_map = {"1 Month": 30, "6 Months": 180, "1 Year": 365, "Full Test Set": None}
        p_days = p_map.get(self.period_var.get(), None)

        self.backtest_service.run_backtest(sym, capital=capital_val, period_days=p_days)

    def _on_portfolio(self):
        """Portfolio button → open PortfolioWindow."""
        last_sig = self._state.last_signal
        default_signal = last_sig.get("consensus") if last_sig else None
        PortfolioWindow(self, default_symbol=self.sym_var.get(), default_signal=default_signal)

    # ═══════════════════════════════════════════════════════════
    #  Live data (timer-driven)
    # ═══════════════════════════════════════════════════════════

    def _live_tick(self):
        """Periodic live data refresh — uses DataService's ThreadPoolExecutor."""
        if self.live_on.get():
            sym = self.sym_var.get()
            self.data_service.fetch_live_price(sym)
            self.data_service.fetch_live_orderbook(sym)

        if self._live_job is not None:
            try:
                self.after_cancel(self._live_job)
            except Exception:
                pass
        self._live_job = self.after(DATA.live_refresh_s * 1000, self._live_tick)

    # ═══════════════════════════════════════════════════════════
    #  UI update helpers
    # ═══════════════════════════════════════════════════════════

    def _update_ui(self, sig: dict):
        """Push a signal + bundle data to all UI panels."""
        sym = sig.get("symbol", self.sym_var.get())
        bundle = self._state.get_bundle(sym)
        if bundle is None:
            return

        self._state.last_signal = sig
        self.signal_card.update(sig)
        self.shield_panel.update(bundle)
        self.model_panel.update(sig.get("details", []))

        df = bundle["df"]
        self.chart.update(df, symbol=sig.get("symbol", ""))
        self.news_panel.update(bundle.get("news_analysis"))

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

        warn = "  ⚠ SAMPLE DATA" if bundle.get("sample") else ""
        self.status.set(
            f"{sig['symbol']} — {sig['consensus']} "
            f"({sig['avg_confidence']:.1f}%) — {len(df)} rows trained{fng_text}{warn}"
        )

        # Log paper prediction if not sample data
        if not bundle.get("sample", False):
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
                    horizon_days=MODEL.pred_horizon,
                )

                # Re-evaluate paper trades in background
                threading.Thread(target=self._run_paper_trader_evaluation, daemon=True).start()
            except Exception:
                log.exception("Error logging paper trade prediction")

    def _update_backtest(self):
        """Push backtest results to metrics panel and chart."""
        sym = self.sym_var.get()
        bt_results = self._state.get_bt_cache(sym)
        if bt_results is None:
            return
        bundle = self._state.get_bundle(sym)
        if bundle is None:
            return

        self.metrics_panel.update(bt_results)

        best = None
        best_eq = None
        best_trades = None
        for name, r in bt_results.items():
            if r["equity"] is not None and len(r["equity"]) > 0:
                trades = r["trades"]
                eq = r["equity"]
                if best_eq is None or eq["equity"].iloc[-1] > best_eq["equity"].iloc[-1]:
                    best = name
                    best_eq = eq
                    best_trades = trades

        df = bundle["df"]
        chart_sym = bundle["symbol"]
        self.chart.update(
            df, trades_df=best_trades if best else None,
            eq_df=best_eq, symbol=f"{chart_sym} [{best}]" if best else chart_sym,
        )
        self.status.set("Backtest complete")

    # ═══════════════════════════════════════════════════════════
    #  Symbol switching
    # ═══════════════════════════════════════════════════════════

    def _on_sym_change(self):
        """Symbol combobox changed — swap cached data via AppState."""
        self.ob_tracker.reset()  # drop smoothed state from old symbol
        sym = self.sym_var.get()
        self._state.current_symbol = sym

        data = self._state.swap_symbol(sym)
        sig = data["signal"]
        bt = data["bt_results"]

        if sig is not None:
            self._update_ui(sig)
            if bt is not None:
                self._state.set_bt_results(sym, bt)
                self._update_backtest()
            else:
                self.metrics_panel.clear()
        else:
            self.signal_card.clear()
            self.shield_panel.clear()
            self.model_panel.clear()
            self.metrics_panel.clear()
            self.news_panel.clear()

    # ═══════════════════════════════════════════════════════════
    #  Paper trader (kept inline — small enough)
    # ═══════════════════════════════════════════════════════════

    def _startup_paper_trader(self):
        self.status.set("Evaluating past predictions...")
        threading.Thread(target=self._run_paper_trader_evaluation, daemon=True).start()

    def _run_paper_trader_evaluation(self):
        try:
            from engine.paper_trader import evaluate_past_predictions
            stats = evaluate_past_predictions()
            self._bus.publish_obj(PAPER_ACCURACY, stats)
        except Exception:
            log.exception("Error during paper trader evaluation")

    def _update_accuracy_display(self, stats: dict):
        if stats.get("total_resolved", 0) > 0:
            self.status.set_accuracy(
                f"Live Acc: {stats['accuracy_pct']:.1f}% "
                f"({stats['correct_resolved']}/{stats['total_resolved']})"
            )
        else:
            self.status.set_accuracy("Live Acc: N/A")

    # ═══════════════════════════════════════════════════════════
    #  Cleanup
    # ═══════════════════════════════════════════════════════════

    def destroy(self):
        self._shutdown_services()
        super().destroy()
