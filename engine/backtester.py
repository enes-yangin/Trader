import numpy as np
import pandas as pd
from data.indicators import get_features
from utils.config import TARGET_COL, SIGNAL_THRESH_BUY, SIGNAL_THRESH_SELL


def run(mdl, sp, buy_th=SIGNAL_THRESH_BUY, sell_th=SIGNAL_THRESH_SELL, capital=10000.0):
    X_val = sp["X_val"]
    y_val = sp["y_val"]
    idx = sp["idx_val"]
    df = sp["df"]
    close = df["close"].values
    sp_idx = sp["split_idx"]

    if hasattr(mdl, "predict_last"):
        preds = _lstm_preds(mdl, sp)
    else:
        preds = mdl.predict(X_val)

    n = min(len(preds), len(y_val))
    preds = preds[:n]
    y_val = y_val[:n]
    idx = idx[:n]

    trades = []
    cash = capital
    pos = 0.0
    entry_px = 0.0
    equity_curve = []

    for i in range(n):
        ci = sp_idx + i
        if ci >= len(close):
            break
        px = close[ci]
        p = preds[i]

        if p > buy_th and pos == 0:
            pos = cash / px
            entry_px = px
            cash = 0.0
            trades.append({"idx": idx[i], "action": "BUY", "price": px, "pred": p})
        elif p < sell_th and pos > 0:
            cash = pos * px
            pnl = (px - entry_px) / entry_px
            trades.append({"idx": idx[i], "action": "SELL", "price": px, "pred": p, "pnl": pnl})
            pos = 0.0
            entry_px = 0.0

        eq = cash + pos * px
        equity_curve.append({"date": idx[i], "equity": eq})

    if pos > 0:
        last_px = close[min(sp_idx + n - 1, len(close) - 1)]
        cash = pos * last_px
        pos = 0.0

    final_eq = cash if cash > 0 else capital
    eq_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    metrics = _calc_metrics(eq_df, trades_df, capital, final_eq)
    return {
        "metrics": metrics,
        "trades": trades_df,
        "equity": eq_df,
        "model": mdl.name,
    }


def _lstm_preds(mdl, sp):
    X_full = np.vstack([sp["X_tr"], sp["X_val"]])
    y_full = np.concatenate([sp["y_tr"], sp["y_val"]])
    all_preds = mdl.predict(X_full)
    offset = len(all_preds) - len(sp["y_val"])
    if offset < 0:
        offset = 0
    return all_preds[offset:]


def _calc_metrics(eq_df, trades_df, capital, final_eq):
    total_ret = (final_eq - capital) / capital
    n_trades = len(trades_df)

    sells = trades_df[trades_df["action"] == "SELL"] if n_trades > 0 else pd.DataFrame()
    n_closed = len(sells)
    wins = len(sells[sells["pnl"] > 0]) if n_closed > 0 else 0
    win_rate = wins / n_closed if n_closed > 0 else 0.0

    avg_pnl = float(sells["pnl"].mean()) if n_closed > 0 else 0.0
    max_win = float(sells["pnl"].max()) if n_closed > 0 else 0.0
    max_loss = float(sells["pnl"].min()) if n_closed > 0 else 0.0

    sharpe = 0.0
    max_dd = 0.0
    if len(eq_df) > 1:
        rets = eq_df["equity"].pct_change().dropna()
        if rets.std() > 0:
            sharpe = float(rets.mean() / rets.std() * np.sqrt(252))
        peak = eq_df["equity"].cummax()
        dd = (eq_df["equity"] - peak) / peak
        max_dd = float(dd.min())

    return {
        "total_return": round(total_ret * 100, 2),
        "final_equity": round(final_eq, 2),
        "n_trades": n_trades,
        "n_closed": n_closed,
        "win_rate": round(win_rate * 100, 1),
        "avg_pnl": round(avg_pnl * 100, 2),
        "max_win": round(max_win * 100, 2),
        "max_loss": round(max_loss * 100, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_dd * 100, 2),
    }


def run_all(bundle, **kw):
    sp = bundle["split"]
    results = {}
    for name, r in bundle["results"].items():
        results[name] = run(r["model"], sp, **kw)
    return results


def summary_table(bt_results):
    rows = []
    for name, r in bt_results.items():
        row = {"model": name, **r["metrics"]}
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def format_report(bt_results):
    lines = ["=" * 60, "  BACKTEST REPORT", "=" * 60]
    for name, r in bt_results.items():
        m = r["metrics"]
        lines.append(f"\n  [{name.upper()}]")
        lines.append(f"    Return:     {m['total_return']:+.2f}%")
        lines.append(f"    Equity:     ${m['final_equity']:,.2f}")
        lines.append(f"    Trades:     {m['n_trades']} ({m['n_closed']} closed)")
        lines.append(f"    Win Rate:   {m['win_rate']:.1f}%")
        lines.append(f"    Avg PnL:    {m['avg_pnl']:+.2f}%")
        lines.append(f"    Best/Worst: {m['max_win']:+.2f}% / {m['max_loss']:+.2f}%")
        lines.append(f"    Sharpe:     {m['sharpe']:.2f}")
        lines.append(f"    Max DD:     {m['max_drawdown']:.2f}%")
    lines.append("\n" + "=" * 60)
    return "\n".join(lines)
