from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Optional
import pandas as pd
from data.fetcher import fetch
from portfolio.position import ClosedLot, OpenLot, Trade, match_fifo
from utils.logger import get_logger

log = get_logger("pnl")


@dataclass
class PositionSummary:
    symbol: str
    qty: float
    avg_price: float
    cost_basis: float
    current_price: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    unrealized_pnl_pct: Optional[float] = None


def realized_trades(trades: List[Trade]) -> List[ClosedLot]:
    closed, _ = match_fifo(trades)
    return closed


def open_positions(trades: List[Trade]) -> List[PositionSummary]:
    _, open_lots = match_fifo(trades)
    by_symbol: Dict[str, List[OpenLot]] = defaultdict(list)
    for lot in open_lots:
        by_symbol[lot.symbol].append(lot)

    out = []
    for sym, lots in by_symbol.items():
        qty = sum(lot.qty for lot in lots)
        cost = sum(lot.qty * lot.price + lot.qty * lot.fee_per_unit for lot in lots)
        avg_price = cost / qty if qty > 1e-12 else 0.0
        out.append(PositionSummary(symbol=sym, qty=qty, avg_price=avg_price, cost_basis=cost))
    return sorted(out, key=lambda p: p.symbol)


def with_live_prices(positions: List[PositionSummary], src: str = "crypto") -> List[PositionSummary]:
    for p in positions:
        try:
            df = fetch(p.symbol, src=src)
            px = float(df["close"].iloc[-1])
            p.current_price = px
            p.unrealized_pnl = (px - p.avg_price) * p.qty
            p.unrealized_pnl_pct = (px / p.avg_price - 1) * 100 if p.avg_price > 0 else 0.0
        except Exception as e:
            log.warning(f"{p.symbol}: live price fetch failed ({type(e).__name__}: {e})")
    return positions


def total_realized_pnl(trades: List[Trade]) -> float:
    return sum(c.pnl for c in realized_trades(trades))


def monthly_summary(trades: List[Trade]) -> pd.DataFrame:
    closed = realized_trades(trades)
    cols = ["month", "realized_pnl", "n_closed", "win_rate", "avg_pnl_pct", "best", "worst"]
    if not closed:
        return pd.DataFrame(columns=cols)

    rows = []
    for c in closed:
        rows.append({
            "month": c.sell_time.strftime("%Y-%m"),
            "pnl": c.pnl,
            "pnl_pct": c.pnl_pct,
        })
    df = pd.DataFrame(rows)
    grouped = df.groupby("month")

    summary = grouped.agg(
        realized_pnl=("pnl", "sum"),
        n_closed=("pnl", "count"),
        avg_pnl_pct=("pnl_pct", "mean"),
        best=("pnl", "max"),
        worst=("pnl", "min"),
    )
    summary["win_rate"] = grouped["pnl"].apply(lambda s: round((s > 0).mean() * 100, 1))
    summary = summary.round(4).reset_index()
    return summary[cols]


def signal_alignment(trades: List[Trade]) -> pd.DataFrame:
    """For closed positions, compare the AI signal recorded at entry time
    against the realized outcome. Helps answer: "when the AI said BUY and
    the user acted on it, how often was that profitable?"
    """
    closed = realized_trades(trades)
    cols = ["buy_signal", "n_closed", "win_rate", "avg_pnl_pct", "total_pnl"]
    with_signal = [c for c in closed if c.buy_signal is not None]
    if not with_signal:
        return pd.DataFrame(columns=cols)

    rows = [{"buy_signal": c.buy_signal, "pnl": c.pnl, "pnl_pct": c.pnl_pct} for c in with_signal]
    df = pd.DataFrame(rows)
    grouped = df.groupby("buy_signal")
    summary = grouped.agg(
        n_closed=("pnl", "count"),
        avg_pnl_pct=("pnl_pct", "mean"),
        total_pnl=("pnl", "sum"),
    )
    summary["win_rate"] = grouped["pnl"].apply(lambda s: round((s > 0).mean() * 100, 1))
    summary = summary.round(4).reset_index()
    return summary[cols]
