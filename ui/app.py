import tkinter as tk
from tkinter import ttk
import threading
from ui.chart import ChartWidget
from ui.panels import SignalCard, ModelDetailPanel, MetricsPanel, StatusBar
from engine.trainer import train_all, compare
from engine.predictor import predict_from_bundle
from engine.backtester import run_all
from utils.config import CRYPTO_SYMBOLS, STOCK_SYMBOLS

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

        self._style()
        self._toolbar()
        self._layout()

        self.bundle = None
        self.bt_results = None

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

        syms = CRYPTO_SYMBOLS + STOCK_SYMBOLS
        self.sym_var = tk.StringVar(value=syms[0])
        cb = ttk.Combobox(bar, textvariable=self.sym_var, values=syms,
                          state="readonly", width=14, font=("Consolas", 10))
        cb.pack(side="left", padx=(20, 8), pady=10)

        ttk.Button(bar, text="▶ Train & Predict", style="Accent.TButton",
                   command=self._on_run).pack(side="left", padx=4, pady=8)

        ttk.Button(bar, text="Backtest", style="TButton",
                   command=self._on_backtest).pack(side="left", padx=4, pady=8)

        self.news_var = tk.BooleanVar(value=True)
        chk = tk.Checkbutton(bar, text="News (FinBERT)", variable=self.news_var,
                             bg=BG2, fg=FG, selectcolor=BG3,
                             activebackground=BG2, activeforeground=FG,
                             font=("Consolas", 9))
        chk.pack(side="left", padx=12, pady=8)

    def _layout(self):
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)

        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="both", expand=True)

        self.chart = ChartWidget(left, figsize=(9, 6))

        right = tk.Frame(body, bg=BG2, width=320)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        self.signal_card = SignalCard(right)
        self.signal_card.pack(fill="x", padx=6, pady=(6, 3))

        self.model_panel = ModelDetailPanel(right)
        self.model_panel.pack(fill="x", padx=6, pady=3)

        self.metrics_panel = MetricsPanel(right)
        self.metrics_panel.pack(fill="both", expand=True, padx=6, pady=(3, 6))

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
        try:
            self.bundle = train_all(sym, with_news=wn)
            sig = predict_from_bundle(self.bundle)
            self.after(0, self._update_ui, sig)
        except Exception as e:
            self.after(0, self.status.set, f"Error: {e}")

    def _update_ui(self, sig):
        self.signal_card.update(sig)
        self.model_panel.update(sig.get("details", []))

        df = self.bundle["df"]
        self.chart.update(df, symbol=sig.get("symbol", ""))

        self.status.set(
            f"{sig['symbol']} — {sig['consensus']} "
            f"({sig['avg_confidence']:.1f}%) — Models trained"
        )

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
        except Exception as e:
            self.after(0, self.status.set, f"Backtest error: {e}")

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
