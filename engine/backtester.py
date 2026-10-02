from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from models.base_model import BaseModel, dynamic_threshold
from engine.risk import (
    position_size, stop_loss_hit, vol_adjusted_max_leverage,
    funding_cost, check_liquidation, liquidation_buffer_price
)
from utils.config import BACKTEST, MODEL, RISK, SIGNAL
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
        period_days: Optional[int] = None,
        pt_mult: float = MODEL.pt_mult, sl_mult: float = MODEL.sl_mult,
        seed: int = 42) -> BacktestResult:
    X_set, y_set, idx, start_idx = _select_set(sp, which)
    df = sp["df"]
    close = df["close"].values
    rng = np.random.default_rng(seed)

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
    entry_atr = 0.0
    entry_pred = 0.0
    entry_capital = 0.0
    entry_leverage = 1.0
    equity_curve = []
    total_costs = 0.0
    trade_pnls: List[float] = []
    unleveraged_pnls: List[float] = []
    n_stops = 0

    bars_held = 0
    from utils.config import MODEL
    pred_horizon = MODEL.pred_horizon

    # Event Queue based HFT Simulation
    import queue
    events = queue.Queue()

    # Queue the first market tick as TICK event
    if n > 0:
        events.put(('TICK', {
            'i': 0,
            'ci': start_idx,
            'px': close[start_idx],
            'p': preds[0],
            'idx_i': idx[0],
            'atr_val': atr_vals[0]
        }))

    next_tick_idx = 1

    while not events.empty():
        event_type, event_data = events.get()
        if event_type == 'TICK':
            i = event_data['i']
            ci = event_data['ci']
            px = event_data['px']
            p = event_data['p']
            idx_i = event_data['idx_i']
            atr_val = event_data['atr_val']
            bt, st = dynamic_threshold(atr_val, buy_th, sell_th)

            if pos > 0:
                bars_held += 1
                if entry_leverage > 1.0:
                    fc = funding_cost(pos * px, funding_rate_annual=RISK.funding_rate_annual, holding_days=1.0)
                    cash -= fc
                    total_costs += fc

                    is_liq, liq_px = check_liquidation(entry_px, px, entry_leverage, side="long", maintenance_margin_pct=RISK.maintenance_margin)
                    if is_liq:
                        events.put(('ORDER_CLOSE', {'px': px, 'action': 'LIQUIDATION', 'pred': p, 'idx_i': idx_i, 'atr': atr_val}))
                        if next_tick_idx < n:
                            events.put(('TICK', {
                                'i': next_tick_idx,
                                'ci': start_idx + next_tick_idx,
                                'px': close[start_idx + next_tick_idx],
                                'p': preds[next_tick_idx],
                                'idx_i': idx[next_tick_idx],
                                'atr_val': atr_vals[next_tick_idx]
                            }))
                            next_tick_idx += 1
                        continue

                    buffer_px = liquidation_buffer_price(entry_px, entry_leverage, side="long", buffer_pct=RISK.liquidation_buffer)
                    if px <= buffer_px:
                        events.put(('ORDER_CLOSE', {'px': buffer_px, 'action': 'LIQ_PREVENT_STOP', 'pred': p, 'idx_i': idx_i, 'atr': atr_val}))
                        if next_tick_idx < n:
                            events.put(('TICK', {
                                'i': next_tick_idx,
                                'ci': start_idx + next_tick_idx,
                                'px': close[start_idx + next_tick_idx],
                                'p': preds[next_tick_idx],
                                'idx_i': idx[next_tick_idx],
                                'atr_val': atr_vals[next_tick_idx]
                            }))
                            next_tick_idx += 1
                        continue

            # User-specified FIXED-percentage exit (SignalConfig): hold a long
            # until it either takes profit (+take_profit_pct, or +take_profit_high_pct
            # when the entry signal's expected move >= tp_high_trigger) or hits the
            # stop (-stop_loss_pct). No time barrier and no exit at a small loss:
            # "kesin kar yoksa satma, %5'ten az zararda satma".
            if pos > 0:
                tp_frac = (SIGNAL.take_profit_high_pct
                           if abs(entry_pred) >= SIGNAL.tp_high_trigger
                           else SIGNAL.take_profit_pct)
                tp_px = entry_px * (1.0 + tp_frac)
                sl_px = entry_px * (1.0 - SIGNAL.stop_loss_pct)
                if px >= tp_px:
                    events.put(('ORDER_CLOSE', {'px': px, 'action': 'TAKE_PROFIT', 'pred': p, 'idx_i': idx_i, 'atr': atr_val}))
                elif px <= sl_px:
                    events.put(('ORDER_CLOSE', {'px': px, 'action': 'STOP', 'pred': p, 'idx_i': idx_i, 'atr': atr_val}))
            elif p > bt and pos == 0:
                events.put(('ORDER_OPEN', {'px': px, 'pred': p, 'idx_i': idx_i, 'atr': atr_val}))

            eq = cash + pos * px
            equity_curve.append({"date": idx_i, "equity": eq})

            # Queue the next tick only after processing this tick's main evaluation
            if next_tick_idx < n:
                events.put(('TICK', {
                    'i': next_tick_idx,
                    'ci': start_idx + next_tick_idx,
                    'px': close[start_idx + next_tick_idx],
                    'p': preds[next_tick_idx],
                    'idx_i': idx[next_tick_idx],
                    'atr_val': atr_vals[next_tick_idx]
                }))
                next_tick_idx += 1

        elif event_type == 'ORDER_CLOSE':
            px = event_data['px']
            action = event_data['action']
            p = event_data['pred']
            idx_i = event_data['idx_i']
            atr_val = event_data['atr']
            
            # Dynamic exit slippage including latency and market impact
            dynamic_slip = slippage_pct + 0.1 * atr_val
            market_impact = ((pos * px / entry_capital) ** 2 * 0.005) if entry_capital > 0 else 0.0
            latency_slip = abs(rng.normal(0, max(1e-5, atr_val * 0.05)))
            total_slip = dynamic_slip + market_impact + latency_slip
            
            fill_px, gross, commission, proceeds, pnl = _exit_position_calc(
                pos, px, entry_capital, entry_leverage, total_slip, commission_pct,
            )
            total_costs += commission
            if entry_leverage > 1.0 and action == "LIQUIDATION":
                pnl = -1.0
                proceeds = 0.0
            trades.append({
                "idx": idx_i, "action": action, "price": px,
                "fill_price": fill_px, "pred": p, "pnl": pnl,
                "commission": commission,
            })
            trade_pnls.append(pnl)
            unleveraged_pnls.append(pnl / entry_leverage)
            cash += proceeds
            if action in ("STOP", "LIQ_PREVENT_STOP"):
                n_stops += 1
            pos = 0.0
            entry_px = 0.0
            entry_capital = 0.0
            entry_leverage = 1.0
            bars_held = 0

        elif event_type == 'ORDER_OPEN':
            px = event_data['px']
            p = event_data['pred']
            idx_i = event_data['idx_i']
            atr_val = event_data['atr']
            
            frac = position_size(
                unleveraged_pnls, use_kelly=use_kelly, kelly_frac=kelly_fraction,
                max_pos=max_position_pct, min_pos=min_position_pct,
                min_trades=kelly_min_trades,
            )
            
            if RISK.use_leverage:
                leverage = vol_adjusted_max_leverage(atr_val, max_lev=RISK.max_leverage) if atr_val > 0 else RISK.max_leverage
            else:
                leverage = 1.0
                
            # Dynamic entry slippage including latency and market impact
            dynamic_slip = slippage_pct + 0.1 * atr_val
            market_impact = (frac ** 2) * 0.005
            latency_slip = abs(rng.normal(0, max(1e-5, atr_val * 0.05)))
            total_slip = dynamic_slip + market_impact + latency_slip
            
            entry_capital, fill_px, commission, net_alloc, pos = _enter_position(
                cash, px, frac * leverage, commission_pct, total_slip,
            )
            entry_px = fill_px
            entry_atr = atr_val
            entry_pred = p
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

    if pos > 0:
        last_px = close[min(start_idx + n - 1, len(close) - 1)]
        idx_i = idx[n - 1] if n > 0 else None
        
        # Last close
        dynamic_slip = slippage_pct + 0.1 * atr_vals[n - 1]
        market_impact = ((pos * last_px / entry_capital) ** 2 * 0.005) if entry_capital > 0 else 0.0
        latency_slip = abs(rng.normal(0, max(1e-5, atr_vals[n - 1] * 0.05)))
        total_slip = dynamic_slip + market_impact + latency_slip
        
        fill_px, gross, commission, proceeds, pnl = _exit_position_calc(
            pos, last_px, entry_capital, entry_leverage, total_slip, commission_pct,
        )
        total_costs += commission
        trades.append({
            "idx": idx_i, "action": "CLOSE", "price": last_px,
            "fill_price": fill_px, "pred": 0.0, "pnl": pnl,
            "commission": commission,
        })
        trade_pnls.append(pnl)
        cash += proceeds
        
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
                  period_days: Optional[int] = None,
                  pt_mult: float = MODEL.pt_mult, sl_mult: float = MODEL.sl_mult,
                  seed: int = 42) -> BacktestResult:
    import numpy as np
    import pandas as pd
    from models.base_model import dynamic_threshold
    from engine.risk import position_size, stop_loss_hit
    from utils.config import MODEL
    
    rng = np.random.default_rng(seed)
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
    entry_atr = 0.0
    entry_pred = 0.0
    entry_capital = 0.0
    equity_curve = []
    total_costs = 0.0
    trade_pnls = []
    n_stops = 0
    bars_held = 0

    def _close_pos(px: float, action: str, pred: float, date_val: Any) -> None:
        nonlocal cash, pos, entry_px, entry_capital, total_costs, n_stops, current_symbol
        
        # Dynamic slippage/impact/latency
        cur_atr = 0.005
        if current_symbol in symbol_data:
            sdata = symbol_data[current_symbol]
            if date_val in sdata["dates"]:
                idx_i = sdata["dates"].index(date_val)
                cur_atr = sdata["atr"][idx_i] if idx_i < len(sdata["atr"]) else 0.005
                
        dynamic_slip = slippage_pct + 0.1 * cur_atr
        market_impact = ((pos * px / entry_capital) ** 2 * 0.005) if entry_capital > 0 else 0.0
        latency_slip = abs(rng.normal(0, max(1e-5, cur_atr * 0.05)))
        total_slip = dynamic_slip + market_impact + latency_slip
        
        fill_px, gross, commission, proceeds, pnl = _exit_position_calc(
            pos, px, entry_capital, 1.0, total_slip, commission_pct,
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

    # Event Queue based HFT Simulation for Portfolio
    import queue
    events = queue.Queue()
    for d in all_dates:
        events.put(('TICK', d))

    while not events.empty():
        event_type, d = events.get()
        if event_type == 'TICK':
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
                    
            closed_this_tick = False
            
            # 1. Manage existing position
            if current_symbol is not None:
                bars_held += 1
                if current_symbol in candidates:
                    cdata = candidates[current_symbol]
                    cur_px = cdata["price"]
                    cur_pred = cdata["pred"]
                    cur_st = cdata["sell_th"]
                    
                    # User-specified FIXED-percentage exit (see run() / SignalConfig):
                    # hold until take-profit (+take_profit_pct, or +take_profit_high_pct
                    # when entry signal expected move >= tp_high_trigger) or stop
                    # (-stop_loss_pct). No time barrier, no small-loss exit, no
                    # mid-trade rotation. Rotation happens only from flat.
                    tp_frac = (SIGNAL.take_profit_high_pct
                               if abs(entry_pred) >= SIGNAL.tp_high_trigger
                               else SIGNAL.take_profit_pct)
                    tp_px = entry_px * (1.0 + tp_frac)
                    sl_px = entry_px * (1.0 - SIGNAL.stop_loss_pct)
                    if cur_px >= tp_px:
                        _close_pos(cur_px, "TAKE_PROFIT", cur_pred, d)
                        bars_held = 0
                        closed_this_tick = True
                    elif cur_px <= sl_px:
                        _close_pos(cur_px, "STOP", cur_pred, d)
                        bars_held = 0
                        closed_this_tick = True
                else:
                    last_px = entry_px
                    _close_pos(last_px, "CLOSE", 0.0, d)
                    bars_held = 0
                    closed_this_tick = True
                    
            # 2. Enter new position if flat
            if current_symbol is None and not closed_this_tick:
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
                    
                    # Get entry ATR for dynamic slippage
                    b_atr = 0.005
                    if best_sym in symbol_data:
                        sdata_entry = symbol_data[best_sym]
                        if d in sdata_entry["dates"]:
                            idx_entry = sdata_entry["dates"].index(d)
                            b_atr = sdata_entry["atr"][idx_entry] if idx_entry < len(sdata_entry["atr"]) else 0.005
                            
                    dynamic_slip = slippage_pct + 0.1 * b_atr
                    market_impact = (frac ** 2) * 0.005
                    latency_slip = abs(rng.normal(0, max(1e-5, b_atr * 0.05)))
                    total_slip = dynamic_slip + market_impact + latency_slip
                    
                    entry_capital, fill_px, commission, net_alloc, pos = _enter_position(
                        cash, b_px, frac, commission_pct, total_slip,
                    )
                    entry_px = fill_px
                    entry_atr = b_atr
                    entry_pred = b_pred
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
        
        # Last close
        cur_atr = 0.005
        if current_symbol in symbol_data:
            sdata_entry = symbol_data[current_symbol]
            cur_atr = sdata_entry["atr"][-1] if len(sdata_entry["atr"]) > 0 else 0.005
            
        dynamic_slip = slippage_pct + 0.1 * cur_atr
        market_impact = ((pos * last_px / entry_capital) ** 2 * 0.005) if entry_capital > 0 else 0.0
        latency_slip = abs(rng.normal(0, max(1e-5, cur_atr * 0.05)))
        total_slip = dynamic_slip + market_impact + latency_slip
        
        fill_px, gross, commission, proceeds, pnl = _exit_position_calc(
            pos, last_px, entry_capital, 1.0, total_slip, commission_pct,
        )
        total_costs += commission
        trades.append({
            "idx": all_dates[-1], "action": "CLOSE", "price": last_px,
            "fill_price": fill_px, "pred": 0.0, "pnl": pnl,
            "commission": commission,
            "symbol": current_symbol,
        })
        trade_pnls.append(pnl)
        cash += proceeds
        
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
    # Dynamic: read the trained model set from whichever bundle trained anything,
    # rather than hardcoding ["linear","xgboost","lstm"] (A7/A: lstm no longer
    # trained by default, and the set should track MODEL_MAP, not duplicate it).
    model_names: List[str] = []
    for b in bundles.values():
        model_names = list(b["results"].keys())
        break
    results = {}
    for name in model_names:
        results[name] = run_portfolio(bundles, name, **kw)
    return results


def run_statarb(sp_y: SplitDict, sp_x: SplitDict, beta: float, alpha: float,
                train_mean: float, train_std: float, entry: float = 2.0,
                exit: float = 0.5, capital: float = BACKTEST.initial_capital,
                commission_pct: float = BACKTEST.commission_pct,
                slippage_pct: float = BACKTEST.slippage_pct,
                funding_rate_annual: float = RISK.funding_rate_annual,
                which: SplitName = "test", seed: int = 42) -> BacktestResult:
    """Run a realistic dual-legged statistical arbitrage spread backtest.
    
    Includes entry/exit commission and slippage on both legs, and perp funding carry on the short leg.
    """
    X_y, y_y, idx_y, start_y = _select_set(sp_y, which)
    X_x, y_x, idx_x, start_x = _select_set(sp_x, which)
    
    df_y = sp_y["df"]
    df_x = sp_x["df"]
    
    # Align dates
    common_idx = idx_y.intersection(idx_x)
    n = len(common_idx)
    
    if n == 0:
        return {
            "metrics": _calc_metrics(pd.DataFrame(columns=["equity"]), pd.DataFrame(), capital, capital),
            "trades": pd.DataFrame(),
            "equity": pd.DataFrame(columns=["date", "equity"]),
            "model": "statarb",
            "which": which,
        }
        
    y_close = df_y["close"].reindex(common_idx).values
    x_close = df_x["close"].reindex(common_idx).values
    
    # Calculate spread
    spread_vals = y_close - (alpha + beta * x_close)
    
    # Calculate z-score
    z = (spread_vals - train_mean) / (train_std if train_std > 0 else 1.0)
    
    # Generate positions
    from engine.statarb_signals import generate_positions
    pos = generate_positions(z, entry=entry, exit=exit)
    
    # Force close at the last bar
    if pos.size > 0:
        pos[-1] = 0.0
        
    trades = []
    cash = capital
    equity_curve = []
    total_costs = 0.0
    
    qty_y = 0.0
    qty_x = 0.0
    entry_equity = capital
    active_pos = 0.0  # current position state: 1.0 (long spread), -1.0 (short spread)
    
    for t in range(n):
        eq = cash
        if active_pos == 1.0:
            # We are LONG spread: long Y, short X
            eq += qty_y * y_close[t] - qty_x * x_close[t]
        elif active_pos == -1.0:
            # We are SHORT spread: short Y, long X
            eq += -qty_y * y_close[t] + qty_x * x_close[t]
            
        equity_curve.append({"date": common_idx[t], "equity": eq})
        
        # Check transitions
        target_pos = pos[t]
        if target_pos != active_pos:
            # Close existing position if open
            if active_pos != 0.0:
                # Exit slippage & commission
                exit_y_px = y_close[t] * (1 - slippage_pct if active_pos == 1.0 else 1 + slippage_pct)
                exit_x_px = x_close[t] * (1 + slippage_pct if active_pos == 1.0 else 1 - slippage_pct)
                
                comm_y = qty_y * exit_y_px * commission_pct
                comm_x = qty_x * exit_x_px * commission_pct
                total_costs += comm_y + comm_x
                
                # Realised PnL
                if active_pos == 1.0:
                    gross_pnl = (qty_y * exit_y_px - qty_x * exit_x_px) - entry_equity
                else:
                    gross_pnl = (-qty_y * exit_y_px + qty_x * exit_x_px) - entry_equity
                    
                net_pnl = gross_pnl - (comm_y + comm_x)
                pnl_pct = net_pnl / entry_equity if entry_equity > 0 else 0.0
                
                trades.append({
                    "idx": common_idx[t], "action": "CLOSE", "price": spread_vals[t],
                    "fill_price": exit_y_px - beta * exit_x_px, "pred": 0.0, "pnl": pnl_pct,
                    "commission": comm_y + comm_x,
                })
                
                cash += entry_equity + net_pnl
                qty_y = 0.0
                qty_x = 0.0
                active_pos = 0.0
                
            # Open new position if target is not 0
            if target_pos != 0.0:
                frac = 0.5  # Allocate 50% of capital to this spread trade
                entry_equity = cash * frac
                
                # Entry slippage & commission
                entry_y_px = y_close[t] * (1 + slippage_pct if target_pos == 1.0 else 1 - slippage_pct)
                entry_x_px = x_close[t] * (1 - slippage_pct if target_pos == 1.0 else 1 + slippage_pct)
                
                # Calculate quantities
                qty_y = entry_equity / entry_y_px
                qty_x = qty_y * beta
                
                comm_y = qty_y * entry_y_px * commission_pct
                comm_x = qty_x * entry_x_px * commission_pct
                total_costs += comm_y + comm_x
                
                # Subtract margin / entry equity from cash
                cash -= entry_equity
                active_pos = target_pos
                
                trades.append({
                    "idx": common_idx[t], "action": "BUY" if target_pos == 1.0 else "SELL", "price": spread_vals[t],
                    "fill_price": entry_y_px - beta * entry_x_px, "pred": z[t], "commission": comm_y + comm_x,
                    "size_pct": round(frac * 100, 1),
                })
                
        # Carry funding cost
        if active_pos != 0.0:
            # The short leg pays perp funding
            if active_pos == 1.0:
                # Short X
                fc = qty_x * x_close[t] * (funding_rate_annual / 365.0)
            else:
                # Short Y
                fc = qty_y * y_close[t] * (funding_rate_annual / 365.0)
            cash -= fc
            total_costs += fc
            
    final_eq = cash
    
    eq_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    metrics = _calc_metrics(eq_df, trades_df, capital, final_eq)
    metrics["total_costs"] = round(total_costs, 2)
    metrics["costs_pct_of_capital"] = round(total_costs / capital * 100, 3)
    
    return {
        "metrics": metrics,
        "trades": trades_df,
        "equity": eq_df,
        "model": "statarb",
        "which": which,
    }
