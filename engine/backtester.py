from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from models.base_model import BaseModel, dynamic_threshold
from engine.risk import (
    position_size, stop_loss_hit, vol_adjusted_max_leverage,
    funding_cost, check_liquidation, liquidation_buffer_price
)
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


def _enter_position(cash: float, px: float, frac: float,
                    commission_pct: float, slippage_pct: float):
    """Common position-sizing math used by run() and run_portfolio().

    Returns (alloc, fill_px, commission, net_alloc, pos)."""
    alloc = cash * frac
    fill_px = px * (1 + slippage_pct)
    commission = alloc * commission_pct
    net_alloc = alloc - commission
    pos = net_alloc / fill_px
    return alloc, fill_px, commission, net_alloc, pos


def _exit_position_calc(pos: float, px: float, entry_capital: float,
                        entry_leverage: float, slippage_pct: float,
                        commission_pct: float):
    """Common exit PnL math used by run() and run_portfolio().

    Returns (fill_px, gross, commission, proceeds, pnl).
    Handles both leveraged (entry_leverage > 1.0) and spot positions."""
    fill_px = px * (1 - slippage_pct)
    gross = pos * fill_px
    commission = gross * commission_pct
    if entry_leverage > 1.0:
        margin = entry_capital / entry_leverage
        pnl = (gross - entry_capital - commission) / margin
        proceeds = margin + (gross - entry_capital) - commission
    else:
        proceeds = gross - commission
        pnl = (proceeds - entry_capital) / entry_capital if entry_capital > 0 else 0.0
    return fill_px, gross, commission, proceeds, pnl


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
        kelly_min_trades: int = RISK.kelly_min_trades,
        period_days: Optional[int] = None) -> BacktestResult:
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

    if period_days is not None:
        limit = min(period_days, n)
        preds = preds[-limit:]
        y_set = y_set[-limit:]
        idx = idx[-limit:]
        start_idx = start_idx + (n - limit)
        n = limit

    if "atr_pct" in df.columns:
        atr_vals = df["atr_pct"].reindex(idx).fillna(0.0).values
    else:
        atr_vals = np.zeros(n)

    trades = []
    cash = capital
    pos = 0.0
    entry_px = 0.0
    entry_capital = 0.0
    entry_leverage = 1.0
    equity_curve = []
    total_costs = 0.0
    trade_pnls: List[float] = []
    n_stops = 0

    def _close_position(px: float, action: str, pred: float) -> None:
        nonlocal cash, pos, entry_px, entry_capital, total_costs, n_stops, entry_leverage
        fill_px, gross, commission, proceeds, pnl = _exit_position_calc(
            pos, px, entry_capital, entry_leverage, slippage_pct, commission_pct,
        )
        total_costs += commission
        if entry_leverage > 1.0 and action == "LIQUIDATION":
            pnl = -1.0
            proceeds = 0.0
        trades.append({
            "idx": idx_i, "action": action, "price": px,
            "fill_price": fill_px, "pred": pred, "pnl": pnl,
            "commission": commission,
        })
        trade_pnls.append(pnl)
        cash += proceeds
        if action in ("STOP", "LIQ_PREVENT_STOP"):
            n_stops += 1
        pos = 0.0
        entry_px = 0.0
        entry_capital = 0.0
        entry_leverage = 1.0

    bars_held = 0
    from utils.config import MODEL
    pred_horizon = MODEL.pred_horizon

    for i in range(n):
        ci = start_idx + i
        if ci >= len(close):
            break
        px = close[ci]
        p = preds[i]
        idx_i = idx[i]
        bt, st = dynamic_threshold(atr_vals[i], buy_th, sell_th)

        if pos > 0:
            bars_held += 1
            if entry_leverage > 1.0:
                # Annualized funding cost charged per bar (assuming daily bars)
                fc = funding_cost(pos * px, funding_rate_annual=RISK.funding_rate_annual, holding_days=1.0)
                cash -= fc
                total_costs += fc

                # Check for liquidation
                is_liq, liq_px = check_liquidation(entry_px, px, entry_leverage, side="long", maintenance_margin_pct=RISK.maintenance_margin)
                if is_liq:
                    _close_position(px, "LIQUIDATION", p)
                    bars_held = 0
                    eq = cash
                    equity_curve.append({"date": idx_i, "equity": eq})
                    continue

                # Liquidation prevention buffer stop
                buffer_px = liquidation_buffer_price(entry_px, entry_leverage, side="long", buffer_pct=RISK.liquidation_buffer)
                if px <= buffer_px:
                    _close_position(buffer_px, "LIQ_PREVENT_STOP", p)
                    bars_held = 0
                    eq = cash
                    equity_curve.append({"date": idx_i, "equity": eq})
                    continue

        atr_pct_val = atr_vals[i] if i < len(atr_vals) else 0.0
        stop_loss_pct_dyn = max(0.01, min(0.15, atr_pct_val * 3.0)) if atr_pct_val > 0 else stop_loss_pct
        if pos > 0 and stop_loss_hit(entry_px, px, stop_loss_pct=stop_loss_pct_dyn):
            _close_position(px, "STOP", p)
            bars_held = 0
        elif pos > 0 and bars_held >= pred_horizon:
            _close_position(px, "TIME_EXIT", p)
            bars_held = 0
        elif p < st and pos > 0:
            _close_position(px, "SELL", p)
            bars_held = 0
        elif p > bt and pos == 0:
            frac = position_size(
                trade_pnls, use_kelly=use_kelly, kelly_frac=kelly_fraction,
                max_pos=max_position_pct, min_pos=min_position_pct,
                min_trades=kelly_min_trades,
            )
            margin = cash * frac

            # Volatility-adjusted leverage calculation
            if RISK.use_leverage:
                leverage = vol_adjusted_max_leverage(atr_vals[i], max_lev=RISK.max_leverage) if atr_vals[i] > 0 else RISK.max_leverage
            else:
                leverage = 1.0

            entry_capital, fill_px, commission, net_alloc, pos = _enter_position(
                cash, px, frac * leverage, commission_pct, slippage_pct,
            )
            entry_px = fill_px
            entry_leverage = leverage
            total_costs += commission
            cash -= (entry_capital / leverage)
            trades.append({
                "idx": idx_i, "action": "BUY", "price": px,
                "fill_price": fill_px, "pred": p, "commission": commission,
                "size_pct": round(frac * 100, 1),
                "leverage": round(leverage, 2)
            })
            bars_held = 0

        eq = cash + pos * px
        equity_curve.append({"date": idx_i, "equity": eq})

    if pos > 0:
        last_px = close[min(start_idx + n - 1, len(close) - 1)]
        idx_i = idx[n - 1] if n > 0 else None
        _close_position(last_px, "CLOSE", 0.0)
        trade_pnls.pop()
        trades.pop()

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

    sells = (
        trades_df[trades_df["action"] != "BUY"]
        if n_trades > 0 and "action" in trades_df.columns
        else pd.DataFrame()
    )
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
            sharpe = float(rets.mean() / rets.std() * np.sqrt(365))
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


def run_portfolio(bundles: Dict[str, Bundle], model_name: str,
                  buy_th: float = SIGNAL.buy_threshold,
                  sell_th: float = SIGNAL.sell_threshold,
                  capital: float = BACKTEST.initial_capital,
                  which: SplitName = "test",
                  commission_pct: float = BACKTEST.commission_pct,
                  slippage_pct: float = BACKTEST.slippage_pct,
                  stop_loss_pct: float = RISK.stop_loss_pct,
                  use_kelly: bool = RISK.use_kelly_sizing,
                  kelly_fraction: float = RISK.kelly_fraction,
                  max_position_pct: float = RISK.max_position_pct,
                  min_position_pct: float = RISK.min_position_pct,
                  kelly_min_trades: int = RISK.kelly_min_trades,
                  period_days: Optional[int] = None) -> BacktestResult:
    import numpy as np
    import pandas as pd
    from models.base_model import dynamic_threshold
    from engine.risk import position_size, stop_loss_hit
    from utils.config import MODEL
    
    pred_horizon = MODEL.pred_horizon
    
    # 1. Gather index-aligned prediction and price data for each symbol
    symbol_data = {}
    all_dates = pd.Index([])
    
    for sym, bundle in bundles.items():
        sp = bundle["split"]
        X_set, y_set, idx, start_idx = _select_set(sp, which)
        df = sp["df"]
        close_vals = df["close"].values
        
        if model_name not in bundle["results"]:
            continue
        mdl = bundle["results"][model_name]["model"]
        
        if hasattr(mdl, "predict_last"):
            preds = _lstm_preds(mdl, sp, which)
        else:
            preds = mdl.predict(X_set)
            
        n = min(len(preds), len(y_set))
        preds = preds[:n]
        idx = idx[:n]
        
        if "atr_pct" in df.columns:
            atr_vals = df["atr_pct"].reindex(idx).fillna(0.0).values
        else:
            atr_vals = np.zeros(n)
            
        symbol_data[sym] = {
            "dates": list(idx),
            "preds": preds,
            "atr": atr_vals,
            "start_idx": start_idx,
            "close": close_vals,
        }
        all_dates = all_dates.union(idx)
        
    all_dates = sorted(all_dates)
    if not all_dates:
        return {
            "metrics": _calc_metrics(pd.DataFrame(columns=["equity"]), pd.DataFrame(), capital, capital),
            "trades": pd.DataFrame(),
            "equity": pd.DataFrame(columns=["date", "equity"]),
            "model": model_name,
            "which": which,
        }
        
    if period_days is not None:
        all_dates = all_dates[-min(period_days, len(all_dates)):]
        
    trades = []
    cash = capital
    pos = 0.0
    current_symbol = None
    entry_px = 0.0
    entry_capital = 0.0
    equity_curve = []
    total_costs = 0.0
    trade_pnls = []
    n_stops = 0
    bars_held = 0
    
    def _close_pos(px: float, action: str, pred: float, date_val: Any) -> None:
        nonlocal cash, pos, entry_px, entry_capital, total_costs, n_stops, current_symbol
        fill_px, gross, commission, proceeds, pnl = _exit_position_calc(
            pos, px, entry_capital, 1.0, slippage_pct, commission_pct,
        )
        total_costs += commission
        trades.append({
            "idx": date_val, "action": action, "price": px,
            "fill_price": fill_px, "pred": pred, "pnl": pnl,
            "commission": commission,
            "symbol": current_symbol,
        })
        trade_pnls.append(pnl)
        cash += proceeds
        if action == "STOP":
            n_stops += 1
        pos = 0.0
        entry_px = 0.0
        entry_capital = 0.0
        current_symbol = None

    for d in all_dates:
        candidates = {}
        for sym, sdata in symbol_data.items():
            if d in sdata["dates"]:
                idx_i = sdata["dates"].index(d)
                p = sdata["preds"][idx_i]
                c_idx = sdata["start_idx"] + idx_i
                px = sdata["close"][c_idx] if c_idx < len(sdata["close"]) else sdata["close"][-1]
                bt, st = dynamic_threshold(sdata["atr"][idx_i], buy_th, sell_th)
                candidates[sym] = {
                    "pred": p,
                    "price": px,
                    "buy_th": bt,
                    "sell_th": st,
                }
                
        # 1. Manage existing position
        if current_symbol is not None:
            bars_held += 1
            if current_symbol in candidates:
                cdata = candidates[current_symbol]
                cur_px = cdata["price"]
                cur_pred = cdata["pred"]
                cur_st = cdata["sell_th"]
                
                sdata = symbol_data[current_symbol]
                idx_i = sdata["dates"].index(d)
                cur_atr = sdata["atr"][idx_i] if idx_i < len(sdata["atr"]) else 0.0
                stop_loss_pct_dyn = max(0.01, min(0.15, cur_atr * 3.0)) if cur_atr > 0 else stop_loss_pct
                is_stop = stop_loss_hit(entry_px, cur_px, stop_loss_pct=stop_loss_pct_dyn)
                is_time = bars_held >= pred_horizon
                is_sell = cur_pred < cur_st
                
                # Check rotation trigger
                better_sym = None
                better_pred = cur_pred
                for sym, cand in candidates.items():
                    if sym != current_symbol and cand["pred"] > cand["buy_th"]:
                        if cand["pred"] > better_pred + 0.005:  # 0.5% higher expected return
                            better_sym = sym
                            better_pred = cand["pred"]
                            
                if is_stop:
                    _close_pos(cur_px, "STOP", cur_pred, d)
                    bars_held = 0
                elif is_time:
                    _close_pos(cur_px, "TIME_EXIT", cur_pred, d)
                    bars_held = 0
                elif is_sell:
                    _close_pos(cur_px, "SELL", cur_pred, d)
                    bars_held = 0
                elif better_sym is not None:
                    _close_pos(cur_px, "ROTATE_EXIT", cur_pred, d)
                    bars_held = 0
                    
                    bcand = candidates[better_sym]
                    b_px = bcand["price"]
                    b_pred = bcand["pred"]
                    
                    frac = position_size(
                        trade_pnls, use_kelly=use_kelly, kelly_frac=kelly_fraction,
                        max_pos=max_position_pct, min_pos=min_position_pct,
                        min_trades=kelly_min_trades,
                    )
                    entry_capital, fill_px, commission, net_alloc, pos = _enter_position(
                        cash, b_px, frac, commission_pct, slippage_pct,
                    )
                    entry_px = fill_px
                    total_costs += commission
                    cash -= entry_capital
                    current_symbol = better_sym
                    trades.append({
                        "idx": d, "action": "BUY", "price": b_px,
                        "fill_price": fill_px, "pred": b_pred, "commission": commission,
                        "size_pct": round(frac * 100, 1),
                        "symbol": better_sym,
                    })
            else:
                last_px = entry_px
                _close_pos(last_px, "CLOSE", 0.0, d)
                bars_held = 0
                
        # 2. Enter new position if flat
        if current_symbol is None:
            best_sym = None
            best_pred = -999.0
            for sym, cand in candidates.items():
                if cand["pred"] > cand["buy_th"]:
                    if cand["pred"] > best_pred:
                        best_sym = sym
                        best_pred = cand["pred"]
                        
            if best_sym is not None:
                bcand = candidates[best_sym]
                b_px = bcand["price"]
                b_pred = bcand["pred"]
                
                frac = position_size(
                    trade_pnls, use_kelly=use_kelly, kelly_frac=kelly_fraction,
                    max_pos=max_position_pct, min_pos=min_position_pct,
                    min_trades=kelly_min_trades,
                )
                entry_capital, fill_px, commission, net_alloc, pos = _enter_position(
                    cash, b_px, frac, commission_pct, slippage_pct,
                )
                entry_px = fill_px
                total_costs += commission
                cash -= entry_capital
                current_symbol = best_sym
                trades.append({
                    "idx": d, "action": "BUY", "price": b_px,
                    "fill_price": fill_px, "pred": b_pred, "commission": commission,
                    "size_pct": round(frac * 100, 1),
                    "symbol": best_sym,
                })
                bars_held = 0
                
        # 3. Calculate equity
        eq = cash
        if current_symbol is not None and current_symbol in candidates:
            eq += pos * candidates[current_symbol]["price"]
        equity_curve.append({"date": d, "equity": eq})
        
    if current_symbol is not None:
        sdata = symbol_data[current_symbol]
        last_px = sdata["close"][-1]
        _close_pos(last_px, "CLOSE", 0.0, all_dates[-1])
        if len(trade_pnls) > 0:
            trade_pnls.pop()
        if len(trades) > 0:
            trades.pop()
        
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
        "model": model_name,
        "which": which,
    }


def run_portfolio_all(bundles: Dict[str, Bundle], **kw: Any) -> Dict[str, BacktestResult]:
    model_names = ["linear", "xgboost", "lstm"]
    results = {}
    for name in model_names:
        results[name] = run_portfolio(bundles, name, **kw)
    return results
