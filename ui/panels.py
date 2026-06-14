import tkinter as tk
from tkinter import ttk

BG = "#1e1e2e"
BG2 = "#181825"
BG3 = "#313244"
FG = "#cdd6f4"
FG2 = "#a6adc8"
GREEN = "#a6e3a1"
RED = "#f38ba8"
YELLOW = "#f9e2af"
BLUE = "#89b4fa"
FONT = ("Consolas", 10)
FONT_B = ("Consolas", 11, "bold")
FONT_L = ("Consolas", 20, "bold")
FONT_S = ("Consolas", 8)


class SignalCard(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG2, **kw)
        self.columnconfigure(0, weight=1)

        self.lbl_sym = tk.Label(self, text="---", font=FONT_B, bg=BG2, fg=FG)
        self.lbl_sym.grid(row=0, column=0, sticky="w", padx=12, pady=(10, 0))

        self.lbl_signal = tk.Label(self, text="---", font=FONT_L, bg=BG2, fg=FG2)
        self.lbl_signal.grid(row=1, column=0, sticky="w", padx=12)

        self.lbl_conf = tk.Label(self, text="", font=FONT, bg=BG2, fg=FG2)
        self.lbl_conf.grid(row=2, column=0, sticky="w", padx=12)

        self.lbl_ret = tk.Label(self, text="", font=FONT, bg=BG2, fg=FG2)
        self.lbl_ret.grid(row=3, column=0, sticky="w", padx=12)

        self.lbl_price = tk.Label(self, text="", font=FONT_S, bg=BG2, fg=FG2)
        self.lbl_price.grid(row=4, column=0, sticky="w", padx=12, pady=(0, 10))

        self.frm_votes = tk.Frame(self, bg=BG2)
        self.frm_votes.grid(row=5, column=0, sticky="ew", padx=12, pady=(0, 10))

    def update(self, sig):
        sym = sig.get("symbol", "---")
        consensus = sig.get("consensus", "---")
        conf = sig.get("avg_confidence", 0)
        ret = sig.get("avg_predicted_return", 0)
        price = sig.get("last_close", 0)
        votes = sig.get("votes", {})

        color = GREEN if consensus == "BUY" else RED if consensus == "SELL" else YELLOW
        self.lbl_sym.config(text=sym)
        self.lbl_signal.config(text=consensus, fg=color)
        self.lbl_conf.config(text=f"Confidence: {conf:.1f}%")
        self.lbl_ret.config(text=f"Pred Return: {ret:+.4%}")
        self.lbl_price.config(text=f"Last Close: {price:,.2f}")

        for w in self.frm_votes.winfo_children():
            w.destroy()
        for i, (k, v) in enumerate(votes.items()):
            c = GREEN if k == "BUY" else RED if k == "SELL" else YELLOW
            tk.Label(self.frm_votes, text=f"{k}:{v}", font=FONT_S,
                     bg=BG2, fg=c).pack(side="left", padx=(0, 10))


class ModelDetailPanel(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG2, **kw)
        self.columnconfigure(0, weight=1)
        tk.Label(self, text="Model Details", font=FONT_B, bg=BG2,
                 fg=FG).pack(anchor="w", padx=10, pady=(8, 4))
        self.container = tk.Frame(self, bg=BG2)
        self.container.pack(fill="both", expand=True, padx=10, pady=(0, 8))

    def update(self, details):
        for w in self.container.winfo_children():
            w.destroy()
        if not details:
            return
        for d in details:
            f = tk.Frame(self.container, bg=BG3, pady=4, padx=8)
            f.pack(fill="x", pady=2)
            c = GREEN if d["signal"] == "BUY" else RED if d["signal"] == "SELL" else YELLOW
            tk.Label(f, text=d["model"], font=FONT, bg=BG3,
                     fg=FG, width=18, anchor="w").pack(side="left")
            tk.Label(f, text=d["signal"], font=FONT_B, bg=BG3,
                     fg=c, width=6).pack(side="left")
            tk.Label(f, text=f"{d['confidence']:.1f}%", font=FONT, bg=BG3,
                     fg=FG2, width=8).pack(side="left")
            tk.Label(f, text=f"{d['predicted_return']:+.4%}", font=FONT, bg=BG3,
                     fg=FG2).pack(side="left", padx=(4, 0))


class MetricsPanel(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG2, **kw)
        tk.Label(self, text="Backtest Results", font=FONT_B, bg=BG2,
                 fg=FG).pack(anchor="w", padx=10, pady=(8, 4))
        self.container = tk.Frame(self, bg=BG2)
        self.container.pack(fill="both", expand=True, padx=10, pady=(0, 8))

    def update(self, bt_results):
        for w in self.container.winfo_children():
            w.destroy()
        if not bt_results:
            return

        cols = ["Return", "Equity", "Trades", "WinRate", "Sharpe", "MaxDD"]
        hdr = tk.Frame(self.container, bg=BG3)
        hdr.pack(fill="x", pady=(0, 2))
        tk.Label(hdr, text="Model", font=FONT_B, bg=BG3, fg=FG,
                 width=12, anchor="w").pack(side="left", padx=(8, 0))
        for c in cols:
            tk.Label(hdr, text=c, font=FONT_S, bg=BG3, fg=FG2,
                     width=10, anchor="e").pack(side="left")

        for name, r in bt_results.items():
            m = r["metrics"]
            row = tk.Frame(self.container, bg=BG2)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=name, font=FONT, bg=BG2, fg=FG,
                     width=12, anchor="w").pack(side="left", padx=(8, 0))

            ret_c = GREEN if m["total_return"] > 0 else RED if m["total_return"] < 0 else FG2
            vals = [
                (f"{m['total_return']:+.2f}%", ret_c),
                (f"${m['final_equity']:,.0f}", FG2),
                (f"{m['n_trades']}", FG2),
                (f"{m['win_rate']:.0f}%", GREEN if m["win_rate"] > 50 else FG2),
                (f"{m['sharpe']:.2f}", BLUE),
                (f"{m['max_drawdown']:.1f}%", RED if m["max_drawdown"] < -10 else FG2),
            ]
            for txt, clr in vals:
                tk.Label(row, text=txt, font=FONT, bg=BG2, fg=clr,
                         width=10, anchor="e").pack(side="left")


class StatusBar(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG3, height=24, **kw)
        self.lbl = tk.Label(self, text="Ready", font=FONT_S, bg=BG3, fg=FG2)
        self.lbl.pack(side="left", padx=8)
        self.pack_propagate(False)

    def set(self, msg):
        self.lbl.config(text=msg)
        self.update_idletasks()
