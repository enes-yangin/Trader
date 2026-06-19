import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
from portfolio.ledger import Ledger
from portfolio.position import Trade
from portfolio.pnl import open_positions, with_live_prices, monthly_summary, signal_alignment
from utils.exceptions import PortfolioError

BG = "#1e1e2e"
BG2 = "#181825"
BG3 = "#313244"
FG = "#cdd6f4"
FG2 = "#a6adc8"
FONT = ("Consolas", 10)
FONT_B = ("Consolas", 11, "bold")


class PortfolioWindow(tk.Toplevel):
    def __init__(self, parent, default_symbol="BTC/USDT", default_signal=None):
        super().__init__(parent)
        self.title("Portfolio — Manual Trade Ledger")
        self.geometry("760x640")
        self.configure(bg=BG)
        self.minsize(640, 480)

        self.ledger = Ledger()
        self.default_symbol = default_symbol
        self.default_signal = default_signal

        self._form()
        self._trades_table()
        self._summary_tables()
        self._refresh_all()

    def _form(self):
        frm = tk.Frame(self, bg=BG2)
        frm.pack(fill="x", padx=8, pady=8)

        tk.Label(frm, text="Add Trade", font=FONT_B, bg=BG2, fg=FG).grid(
            row=0, column=0, columnspan=6, sticky="w", padx=4, pady=(4, 8))

        tk.Label(frm, text="Symbol", bg=BG2, fg=FG2, font=FONT).grid(row=1, column=0, sticky="w", padx=4)
        self.sym_var = tk.StringVar(value=self.default_symbol)
        tk.Entry(frm, textvariable=self.sym_var, bg=BG3, fg=FG, font=FONT,
                 insertbackground=FG, width=12).grid(row=2, column=0, padx=4)

        tk.Label(frm, text="Side", bg=BG2, fg=FG2, font=FONT).grid(row=1, column=1, sticky="w", padx=4)
        self.side_var = tk.StringVar(value="BUY")
        ttk.Combobox(frm, textvariable=self.side_var, values=["BUY", "SELL"],
                     state="readonly", width=6, font=FONT).grid(row=2, column=1, padx=4)

        tk.Label(frm, text="Quantity", bg=BG2, fg=FG2, font=FONT).grid(row=1, column=2, sticky="w", padx=4)
        self.qty_var = tk.StringVar()
        tk.Entry(frm, textvariable=self.qty_var, bg=BG3, fg=FG, font=FONT,
                 insertbackground=FG, width=10).grid(row=2, column=2, padx=4)

        tk.Label(frm, text="Price", bg=BG2, fg=FG2, font=FONT).grid(row=1, column=3, sticky="w", padx=4)
        self.price_var = tk.StringVar()
        tk.Entry(frm, textvariable=self.price_var, bg=BG3, fg=FG, font=FONT,
                 insertbackground=FG, width=10).grid(row=2, column=3, padx=4)

        tk.Label(frm, text="Fee", bg=BG2, fg=FG2, font=FONT).grid(row=1, column=4, sticky="w", padx=4)
        self.fee_var = tk.StringVar(value="0")
        tk.Entry(frm, textvariable=self.fee_var, bg=BG3, fg=FG, font=FONT,
                 insertbackground=FG, width=8).grid(row=2, column=4, padx=4)

        tk.Label(frm, text="Date (YYYY-MM-DD)", bg=BG2, fg=FG2, font=FONT).grid(row=1, column=5, sticky="w", padx=4)
        self.date_var = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
        tk.Entry(frm, textvariable=self.date_var, bg=BG3, fg=FG, font=FONT,
                 insertbackground=FG, width=12).grid(row=2, column=5, padx=4)

        sig_label = f"AI signal at entry: {self.default_signal}" if self.default_signal else \
            "AI signal at entry: (none -- run Train & Predict first)"
        self.sig_lbl = tk.Label(frm, text=sig_label, bg=BG2, fg=FG2, font=FONT)
        self.sig_lbl.grid(row=3, column=0, columnspan=4, sticky="w", padx=4, pady=(8, 4))

        ttk.Button(frm, text="Add Trade", style="Accent.TButton",
                   command=self._on_add).grid(row=3, column=4, columnspan=2, padx=4, pady=(8, 4), sticky="e")

    def _trades_table(self):
        frm = tk.Frame(self, bg=BG2)
        frm.pack(fill="both", expand=False, padx=8, pady=(0, 8))

        tk.Label(frm, text="Trades", font=FONT_B, bg=BG2, fg=FG).pack(anchor="w", padx=4, pady=(4, 2))

        cols = ("date", "symbol", "side", "qty", "price", "fee", "signal")
        self.tree = ttk.Treeview(frm, columns=cols, show="headings", height=6)
        for c in cols:
            self.tree.heading(c, text=c.capitalize())
            self.tree.column(c, width=90, anchor="center")
        self.tree.pack(fill="x", padx=4, pady=2)

        ttk.Button(frm, text="Delete Selected", command=self._on_delete).pack(anchor="e", padx=4, pady=4)

    def _summary_tables(self):
        frm = tk.Frame(self, bg=BG2)
        frm.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        tk.Label(frm, text="Open Positions", font=FONT_B, bg=BG2, fg=FG).pack(anchor="w", padx=4, pady=(4, 2))
        self.pos_text = tk.Text(frm, height=5, bg=BG3, fg=FG, font=FONT, relief="flat")
        self.pos_text.pack(fill="x", padx=4, pady=2)

        tk.Label(frm, text="Monthly Realized PnL", font=FONT_B, bg=BG2, fg=FG).pack(anchor="w", padx=4, pady=(8, 2))
        self.month_text = tk.Text(frm, height=5, bg=BG3, fg=FG, font=FONT, relief="flat")
        self.month_text.pack(fill="x", padx=4, pady=2)

        tk.Label(frm, text="Signal Alignment (entry signal vs outcome)", font=FONT_B, bg=BG2, fg=FG).pack(
            anchor="w", padx=4, pady=(8, 2))
        self.sig_text = tk.Text(frm, height=4, bg=BG3, fg=FG, font=FONT, relief="flat")
        self.sig_text.pack(fill="x", padx=4, pady=2)

        ttk.Button(frm, text="Refresh (fetch live prices)", command=self._refresh_all).pack(anchor="e", padx=4, pady=4)

    def _on_add(self):
        try:
            qty = float(self.qty_var.get())
            price = float(self.price_var.get())
            fee = float(self.fee_var.get() or 0)
            ts = datetime.strptime(self.date_var.get().strip(), "%Y-%m-%d")
        except ValueError:
            messagebox.showerror("Invalid input", "Quantity, price, fee must be numbers and date must be YYYY-MM-DD")
            return

        trade = Trade(
            symbol=self.sym_var.get().strip().upper(),
            side=self.side_var.get(),
            qty=qty, price=price, timestamp=ts, fee=fee,
            signal=self.default_signal,
        )
        try:
            self.ledger.add_trade(trade)
        except PortfolioError as e:
            messagebox.showerror("Trade rejected", str(e))
            return

        self.qty_var.set("")
        self.price_var.set("")
        self.fee_var.set("0")
        self._refresh_all()

    def _on_delete(self):
        sel = self.tree.selection()
        if not sel:
            return
        idx = self.tree.index(sel[0])
        try:
            self.ledger.remove_trade(idx)
        except PortfolioError as e:
            messagebox.showerror("Delete failed", str(e))
            return
        self._refresh_all()

    def _refresh_all(self):
        self._refresh_trades()
        self._refresh_positions()
        self._refresh_monthly()
        self._refresh_signal_alignment()

    def _refresh_trades(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        for t in self.ledger.list_trades():
            self.tree.insert("", "end", values=(
                t.timestamp.strftime("%Y-%m-%d"), t.symbol, t.side,
                f"{t.qty:g}", f"{t.price:g}", f"{t.fee:g}", t.signal or "-",
            ))

    def _refresh_positions(self):
        trades = self.ledger.list_trades()
        positions = with_live_prices(open_positions(trades))
        self.pos_text.config(state="normal")
        self.pos_text.delete("1.0", "end")
        if not positions:
            self.pos_text.insert("1.0", "  (no open positions)")
        for p in positions:
            if p.unrealized_pnl is not None:
                line = (f"  {p.symbol:10s} qty={p.qty:g}  avg={p.avg_price:.4f}  "
                        f"cur={p.current_price:.4f}  uPnL={p.unrealized_pnl:+.2f} "
                        f"({p.unrealized_pnl_pct:+.2f}%)\n")
            else:
                line = f"  {p.symbol:10s} qty={p.qty:g}  avg={p.avg_price:.4f}  cur=N/A\n"
            self.pos_text.insert("end", line)
        self.pos_text.config(state="disabled")

    def _refresh_monthly(self):
        trades = self.ledger.list_trades()
        df = monthly_summary(trades)
        self.month_text.config(state="normal")
        self.month_text.delete("1.0", "end")
        if df.empty:
            self.month_text.insert("1.0", "  (no closed trades yet)")
        for _, r in df.iterrows():
            line = (f"  {r['month']}  realized={r['realized_pnl']:+.2f}  "
                    f"n={int(r['n_closed'])}  win_rate={r['win_rate']:.1f}%  "
                    f"avg={r['avg_pnl_pct']:+.2f}%\n")
            self.month_text.insert("end", line)
        self.month_text.config(state="disabled")

    def _refresh_signal_alignment(self):
        trades = self.ledger.list_trades()
        df = signal_alignment(trades)
        self.sig_text.config(state="normal")
        self.sig_text.delete("1.0", "end")
        if df.empty:
            self.sig_text.insert("1.0", "  (no closed trades with recorded AI signal)")
        for _, r in df.iterrows():
            line = (f"  entry={r['buy_signal']:5s}  n={int(r['n_closed'])}  "
                    f"win_rate={r['win_rate']:.1f}%  total_pnl={r['total_pnl']:+.2f}\n")
            self.sig_text.insert("end", line)
        self.sig_text.config(state="disabled")
