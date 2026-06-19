from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from models.base_model import BaseModel, dynamic_threshold
from engine.risk import position_size, stop_loss_hit
from utils.config import BACKTEST, RISK, SIGNAL
from utils.types import SplitDict, SplitName, Bundle, BacktestResult, BacktestMetrics


def _select_set(sp: SplitDict, which: SplitName) -> Tuple[np.ndarray, np.ndarray, pd.Index, int]:
    """Type-safe split-set selection (replaces dynamic sp[f"X_{which}"] access)."""
    if which == "train":
        return sp["X_tr"], sp["y_tr"], sp["idx_tr"], 0
    elif which == "val":
        return sp["X_val"], sp["y_val"], sp["idx_val"], sp["i_tr"]
    elif which == "test":
        return sp["X_test"], sp["y_test"], sp["idx_test"], sp["i_va"]
    raise ValueError(f"Unknown split name: {which!r} (expected train/val/test)")


def run(mdl: BaseModel, sp: SplitDict, buy_th: float = SIGNAL.buy_threshold,
        sell_th: float = SIGNAL.sell_threshold,
        capital: float = BACKTEST.initial_capital, which: SplitName = "test",
        commission_pct: float = BACKTEST.commission_pct,
        slippage_pct: float = BACKTEST.slippage_pct,
        stop_loss_pct: float = RISK.stop_loss_pct,
        use_kelly: bool = RISK.use_kelly_sizing,
        kelly_fraction: float = RISK.kelly_fraction,
        max_position_pct: float = RISK.max_position_pct,
        min_position_pct: float = RISK.min_position_pct,
        kelly_min_trades: int = RISK.kelly_min_trades) -> BacktestResult:
    X_set, y_set, idx, start_idx = _select_set(sp, which)
    df = sp["df"]
    close = df["close"].values

    if hasattr(mdl, "predict_last"):
        preds = _lstm_preds(mdl, sp, which)
    else:
        preds = mdl.predict(X_set)

    n = min(len(preds), len(y_set))
    preds = preds[:n]
    y_set = y_set[:n]
    idx = idx[:n]

    if "atr_pct" in df.columns:
        atr_vals = df["atr_pct"].reindex(idx).fillna(0.0).values
    else:
        atr_vals = np.zeros(n)

    trades = []
    cash = capital
    pos = 0.0
    entry_px = 0.0
    entry_capital = 0.0
    equity_curve = []
    total_costs = 0.0
    trade_pnls: List[float] = []
    n_stops = 0

    def _close_position(px: float, action: str, pred: float) -> None:
        nonlocal cash, pos, entry_px, entry_capital, total_costs, n_stops
        fill_px = px * (1 - slippage_pct)
        gross = pos * fill_px
        commission = gross * commission_pct
        proceeds = gross - commission
        total_costs += commission
        pnl = (proceeds - entry_capital) / entry_capital if entry_capital > 0 else 0.0
        trades.append({
            "idx": idx_i, "action": action, "price": px,
            "fill_price": fill_px, "pred": pred, "pnl": pnl,
            "commission": commission,
        })
        trade_pnls.append(pnl)
        cash += proceeds
        if action == "STOP":
            n_stops += 1
        pos = 0.0
        entry_px = 0.0
        entry_capital = 0.0

    for i in range(n):
        ci = start_idx + i
        if ci >= len(close):
            break
        px = close[ci]
        p = preds[i]
        idx_i = idx[i]
        bt, st = dynamic_threshold(atr_vals[i], buy_th, sell_th)

        if pos > 0 and stop_loss_hit(entry_px, px, stop_loss_pct=stop_loss_pct):
            _close_position(px, "STOP", p)
        elif p > bt and pos == 0:
            frac = position_size(
                trade_pnls, use_kelly=use_kelly, kelly_frac=kelly_fraction,
                max_pos=max_position_pct, min_pos=min_position_pct,
                min_trades=kelly_min_trades,
            )
            alloc = cash * frac
            fill_px = px * (1 + slippage_pct)
            commission = alloc * commission_pct
            net_alloc = alloc - commission
            pos = net_alloc / fill_px
            entry_px = fill_px
            entry_capital = alloc
            total_costs += commission
            cash -= alloc
            trades.append({
                "idx": idx_i, "action": "BUY", "price": px,
                "fill_price": fill_px, "pred": p, "commission": commission,
                "size_pct": round(frac * 100, 1),
            })
        elif p < st and pos > 0:
            _close_position(px, "SELL", p)

        eq = cash + pos * px
        equity_curve.append({"date": idx_i, "equity": eq})

    if pos > 0:
        last_px = close[min(start_idx + n - 1, len(close) - 1)]
        idx_i = idx[n - 1] if n > 0 else None
        _close_position(last_px, "CLOSE", 0.0)
        trade_pnls.pop()

    final_eq = cash
    eq_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    metrics = _calc_metrics(eq_df, trades_df, capital, final_eq)
    metrics["total_costs"] = round(total_costs, 2)
    metrics["costs_pct_of_capital"] = round(total_costs / capital * 100, 3)
    metrics["n_stop_losses"] = n_stops
    return {
        "metrics": metrics,
        "trades": trades_df,
        "equity": eq_df,
        "model": mdl.name,
        "which": which,
    }


def _lstm_preds(mdl: BaseModel, sp: SplitDict, which: SplitName = "test") -> np.ndarray:
    if which == "test":
        X_full = np.vstack([sp["X_tr"], sp["X_val"], sp["X_test"]])
        target_len = len(sp["y_test"])
    elif which == "val":
        X_full = np.vstack([sp["X_tr"], sp["X_val"]])
        target_len = len(sp["y_val"])
    else:
        X_full = sp["X_tr"]
        target_len = len(sp["y_tr"])
    all_preds = mdl.predict(X_full)
    offset = len(all_preds) - target_len
    if offset < 0:
        offset = 0
    return all_preds[offset:]


def _calc_metrics(eq_df: pd.DataFrame, trades_df: pd.DataFrame,
                  capital: float, final_eq: float) -> BacktestMetrics:
    total_ret = (final_eq - capital) / capital
    n_trades = len(trades_df)

    exit_actions = {"SELL", "STOP", "CLOSE"}
    sells = trades_df[trades_df["action"].isin(exit_actions)] if n_trades > 0 else pd.DataFrame()
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
        "total_costs": 0.0,
        "costs_pct_of_capital": 0.0,
        "n_stop_losses": 0,
    }


def run_all(bundle: Bundle, **kw: Any) -> Dict[str, BacktestResult]:
    sp = bundle["split"]
    results = {}
    for name, r in bundle["results"].items():
        results[name] = run(r["model"], sp, **kw)
    return results


def summary_table(bt_results: Dict[str, BacktestResult]) -> pd.DataFrame:
    rows = []
    for name, r in bt_results.items():
        row = {"model": name, **r["metrics"]}
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def format_report(bt_results: Dict[str, BacktestResult]) -> str:
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
        lines.append(f"    Costs:      ${m['total_costs']:,.2f} ({m['costs_pct_of_capital']:.3f}% of capital)")
        lines.append(f"    Stop-Outs:  {m['n_stop_losses']}")
    lines.append("\n" + "=" * 60)
    return "\n".join(lines)
