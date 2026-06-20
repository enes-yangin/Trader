from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Dict, List, Literal, Optional, Tuple
from utils.exceptions import PortfolioError

Side = Literal["BUY", "SELL"]
EPS = 1e-12


@dataclass
class Trade:
    symbol: str
    side: Side
    qty: float
    price: float
    timestamp: datetime
    fee: float = 0.0
    note: str = ""
    signal: Optional[str] = None


@dataclass
class OpenLot:
    symbol: str
    qty: float
    price: float
    timestamp: datetime
    fee_per_unit: float
    signal: Optional[str] = None


@dataclass
class ClosedLot:
    symbol: str
    qty: float
    buy_price: float
    sell_price: float
    buy_time: datetime
    sell_time: datetime
    buy_fee: float = 0.0
    sell_fee: float = 0.0
    buy_signal: Optional[str] = None
    sell_signal: Optional[str] = None

    @property
    def cost_basis(self) -> float:
        return self.buy_price * self.qty + self.buy_fee

    @property
    def proceeds(self) -> float:
        return self.sell_price * self.qty - self.sell_fee

    @property
    def pnl(self) -> float:
        return self.proceeds - self.cost_basis

    @property
    def pnl_pct(self) -> float:
        if self.cost_basis == 0:
            return 0.0
        return self.pnl / abs(self.cost_basis) * 100

    @property
    def hold_days(self) -> float:
        return (self.sell_time - self.buy_time).total_seconds() / 86400


def _validate_trade(t: Trade) -> None:
    if t.side not in ("BUY", "SELL"):
        raise PortfolioError(f"Invalid side '{t.side}', must be BUY or SELL")
    if t.qty <= 0:
        raise PortfolioError(f"Trade qty must be positive, got {t.qty}")
    if t.price <= 0:
        raise PortfolioError(f"Trade price must be positive, got {t.price}")
    if t.fee < 0:
        raise PortfolioError(f"Trade fee cannot be negative, got {t.fee}")


def match_fifo(trades: List[Trade]) -> Tuple[List[ClosedLot], List[OpenLot]]:
    for t in trades:
        _validate_trade(t)

    ordered = sorted(trades, key=lambda t: t.timestamp)
    open_lots: Dict[str, Deque[OpenLot]] = defaultdict(deque)
    closed: List[ClosedLot] = []

    for t in ordered:
        if t.side == "BUY":
            fee_per_unit = t.fee / t.qty if t.qty else 0.0
            open_lots[t.symbol].append(OpenLot(
                symbol=t.symbol, qty=t.qty, price=t.price,
                timestamp=t.timestamp, fee_per_unit=fee_per_unit, signal=t.signal,
            ))
            continue

        remaining = t.qty
        sell_fee_per_unit = t.fee / t.qty if t.qty else 0.0
        lots = open_lots[t.symbol]

        while remaining > EPS:
            if not lots:
                raise PortfolioError(
                    f"Cannot sell {t.qty} {t.symbol} on {t.timestamp}: "
                    f"no matching open position (oversell). "
                    f"Check that all prior BUY trades for {t.symbol} are recorded."
                )
            lot = lots[0]
            matched = min(lot.qty, remaining)
            closed.append(ClosedLot(
                symbol=t.symbol, qty=matched,
                buy_price=lot.price, sell_price=t.price,
                buy_time=lot.timestamp, sell_time=t.timestamp,
                buy_fee=lot.fee_per_unit * matched,
                sell_fee=sell_fee_per_unit * matched,
                buy_signal=lot.signal, sell_signal=t.signal,
            ))
            lot.qty -= matched
            remaining -= matched
            if lot.qty <= EPS:
                lots.popleft()

    open_out = [lot for lots in open_lots.values() for lot in lots if lot.qty > EPS]
    return closed, open_out
