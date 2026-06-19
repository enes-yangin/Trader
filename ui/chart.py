import numpy as np
import ta
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

BG = "#1e1e2e"
FG = "#cdd6f4"
GREEN = "#a6e3a1"
RED = "#f38ba8"
BLUE = "#89b4fa"
YELLOW = "#f9e2af"
GRID = "#313244"
CANDLE_UP = "#a6e3a1"
CANDLE_DN = "#f38ba8"


def apply_style(fig, axes):
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG, labelsize=8)
        ax.xaxis.label.set_color(FG)
        ax.yaxis.label.set_color(FG)
        ax.title.set_color(FG)
        ax.grid(True, color=GRID, alpha=0.5, linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_color(GRID)


def draw_candlestick(ax, df, last_n=60):
    d = df.tail(last_n).copy()
    d = d.reset_index()
    if "date" in d.columns:
        d["x"] = np.arange(len(d))
    else:
        d["x"] = np.arange(len(d))

    up = d[d["close"] >= d["open"]]
    dn = d[d["close"] < d["open"]]

    ax.bar(up["x"], up["close"] - up["open"], bottom=up["open"],
           color=CANDLE_UP, width=0.6, alpha=0.9)
    ax.bar(up["x"], up["high"] - up["close"], bottom=up["close"],
           color=CANDLE_UP, width=0.15)
    ax.bar(up["x"], up["open"] - up["low"], bottom=up["low"],
           color=CANDLE_UP, width=0.15)

    ax.bar(dn["x"], dn["close"] - dn["open"], bottom=dn["open"],
           color=CANDLE_DN, width=0.6, alpha=0.9)
    ax.bar(dn["x"], dn["high"] - dn["open"], bottom=dn["open"],
           color=CANDLE_DN, width=0.15)
    ax.bar(dn["x"], dn["close"] - dn["low"], bottom=dn["low"],
           color=CANDLE_DN, width=0.15)

    if len(d) >= 20:
        close = df["close"]
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        ema20 = ta.trend.EMAIndicator(close, window=20).ema_indicator()
        t_bb_high = bb.bollinger_hband().tail(last_n).values
        t_bb_low = bb.bollinger_lband().tail(last_n).values
        t_ema20 = ema20.tail(last_n).values
        xs = np.arange(len(d))
        ax.plot(xs, t_bb_high, color=BLUE, alpha=0.3, linewidth=0.8)
        ax.plot(xs, t_bb_low, color=BLUE, alpha=0.3, linewidth=0.8)
        ax.fill_between(xs, t_bb_high, t_bb_low, color=BLUE, alpha=0.05)
        ax.plot(xs, t_ema20, color=YELLOW, linewidth=1, label="EMA20")

    ax.set_ylabel("Price", fontsize=9)
    ax.legend(loc="upper left", fontsize=7, facecolor=BG, edgecolor=GRID,
              labelcolor=FG)
    return d


def draw_volume(ax, df, last_n=60):
    d = df.tail(last_n).copy().reset_index()
    xs = np.arange(len(d))
    colors = [CANDLE_UP if d["close"].iloc[i] >= d["open"].iloc[i]
              else CANDLE_DN for i in range(len(d))]
    ax.bar(xs, d["volume"], color=colors, alpha=0.5, width=0.6)
    ax.set_ylabel("Vol", fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))


def draw_rsi(ax, df, last_n=60):
    if "rsi" not in df.columns:
        return
    t = df.tail(last_n)
    xs = np.arange(len(t))
    rsi_vals = t["rsi"].values
    if np.nanmax(rsi_vals) <= 1.5:
        rsi_vals = rsi_vals * 100
    ax.plot(xs, rsi_vals, color=BLUE, linewidth=1.2)
    ax.axhline(70, color=RED, linestyle="--", linewidth=0.7, alpha=0.7)
    ax.axhline(30, color=GREEN, linestyle="--", linewidth=0.7, alpha=0.7)
    ax.fill_between(xs, 30, 70, color=GRID, alpha=0.15)
    ax.set_ylabel("RSI", fontsize=8)
    ax.set_ylim(0, 100)


def draw_signals(ax, trades_df, df, last_n=60):
    if trades_df is None or len(trades_df) == 0:
        return
    d = df.tail(last_n).reset_index()
    start_date = df.index[-last_n] if len(df) >= last_n else df.index[0]

    for _, tr in trades_df.iterrows():
        if "idx" not in tr or tr["idx"] < start_date:
            continue
        mask = d["date"] == tr["idx"] if "date" in d.columns else None
        if mask is not None and mask.any():
            xi = d[mask].index[0]
            if tr["action"] == "BUY":
                ax.scatter(xi, tr["price"], marker="^", color=GREEN,
                          s=100, zorder=5, edgecolors="white", linewidths=0.5)
            else:
                ax.scatter(xi, tr["price"], marker="v", color=RED,
                          s=100, zorder=5, edgecolors="white", linewidths=0.5)


def draw_equity(ax, eq_df):
    if eq_df is None or len(eq_df) == 0:
        return
    xs = np.arange(len(eq_df))
    vals = eq_df["equity"].values
    base = vals[0]
    ax.plot(xs, vals, color=BLUE, linewidth=1.5)
    ax.fill_between(xs, base, vals,
                    where=vals >= base, color=GREEN, alpha=0.15)
    ax.fill_between(xs, base, vals,
                    where=vals < base, color=RED, alpha=0.15)
    ax.axhline(base, color=FG, linestyle=":", linewidth=0.7, alpha=0.5)
    ax.set_ylabel("Equity", fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))


class ChartWidget:
    def __init__(self, parent, figsize=(10, 7)):
        self.fig, self.axes = plt.subplots(
            4, 1, figsize=figsize,
            gridspec_kw={"height_ratios": [3, 1, 1, 2]},
            sharex=False
        )
        self.fig.subplots_adjust(hspace=0.35, left=0.08, right=0.96,
                                  top=0.95, bottom=0.05)
        apply_style(self.fig, self.axes)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def update(self, df, trades_df=None, eq_df=None, symbol="", last_n=60):
        for ax in self.axes:
            ax.clear()
        apply_style(self.fig, self.axes)

        draw_candlestick(self.axes[0], df, last_n)
        if trades_df is not None:
            draw_signals(self.axes[0], trades_df, df, last_n)
        self.axes[0].set_title(f"  {symbol}", fontsize=11, fontweight="bold",
                                loc="left")

        draw_volume(self.axes[1], df, last_n)
        draw_rsi(self.axes[2], df, last_n)
        draw_equity(self.axes[3], eq_df)

        self.canvas.draw_idle()

    def destroy(self):
        plt.close(self.fig)
