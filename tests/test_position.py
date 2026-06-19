from datetime import datetime, timedelta
import pytest
from portfolio.position import Trade, match_fifo
from utils.exceptions import PortfolioError

T0 = datetime(2026, 1, 1)


def trade(symbol, side, qty, price, days=0, fee=0.0, signal=None):
    return Trade(symbol, side, qty, price, T0 + timedelta(days=days), fee=fee, signal=signal)


def test_simple_round_trip():
    trades = [
        trade("BTC/USDT", "BUY", 1.0, 100.0, days=0),
        trade("BTC/USDT", "SELL", 1.0, 110.0, days=5),
    ]
    closed, open_lots = match_fifo(trades)
    assert len(closed) == 1
    assert open_lots == []
    c = closed[0]
    assert c.qty == 1.0
    assert c.buy_price == 100.0
    assert c.sell_price == 110.0
    assert c.pnl == pytest.approx(10.0)
    assert c.hold_days == pytest.approx(5.0)


def test_partial_sell_leaves_open_lot():
    trades = [
        trade("BTC/USDT", "BUY", 2.0, 100.0, days=0),
        trade("BTC/USDT", "SELL", 0.5, 110.0, days=1),
    ]
    closed, open_lots = match_fifo(trades)
    assert len(closed) == 1
    assert closed[0].qty == 0.5
    assert len(open_lots) == 1
    assert open_lots[0].qty == pytest.approx(1.5)
    assert open_lots[0].price == 100.0


def test_sell_spans_multiple_buy_lots_fifo_order():
    trades = [
        trade("BTC/USDT", "BUY", 1.0, 100.0, days=0),
        trade("BTC/USDT", "BUY", 1.0, 200.0, days=1),
        trade("BTC/USDT", "SELL", 1.5, 300.0, days=2),
    ]
    closed, open_lots = match_fifo(trades)
    assert len(closed) == 2
    assert closed[0].buy_price == 100.0
    assert closed[0].qty == pytest.approx(1.0)
    assert closed[1].buy_price == 200.0
    assert closed[1].qty == pytest.approx(0.5)
    assert len(open_lots) == 1
    assert open_lots[0].price == 200.0
    assert open_lots[0].qty == pytest.approx(0.5)


def test_fees_allocated_proportionally():
    trades = [
        trade("BTC/USDT", "BUY", 2.0, 100.0, days=0, fee=2.0),
        trade("BTC/USDT", "SELL", 1.0, 110.0, days=1, fee=1.0),
    ]
    closed, _ = match_fifo(trades)
    c = closed[0]
    assert c.buy_fee == pytest.approx(1.0)
    assert c.sell_fee == pytest.approx(1.0)
    assert c.cost_basis == pytest.approx(101.0)
    assert c.proceeds == pytest.approx(109.0)
    assert c.pnl == pytest.approx(8.0)


def test_oversell_raises():
    trades = [trade("BTC/USDT", "SELL", 1.0, 100.0, days=0)]
    with pytest.raises(PortfolioError, match="oversell"):
        match_fifo(trades)


def test_oversell_after_partial_position_raises():
    trades = [
        trade("BTC/USDT", "BUY", 1.0, 100.0, days=0),
        trade("BTC/USDT", "SELL", 2.0, 110.0, days=1),
    ]
    with pytest.raises(PortfolioError, match="oversell"):
        match_fifo(trades)


@pytest.mark.parametrize("qty,price,fee", [
    (0.0, 100.0, 0.0),
    (-1.0, 100.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, -5.0, 0.0),
    (1.0, 100.0, -1.0),
])
def test_invalid_trade_values_raise(qty, price, fee):
    with pytest.raises(PortfolioError):
        match_fifo([trade("BTC/USDT", "BUY", qty, price, fee=fee)])


def test_invalid_side_raises():
    bad = Trade("BTC/USDT", "HOLD", 1.0, 100.0, T0)  # type: ignore[arg-type]
    with pytest.raises(PortfolioError, match="Invalid side"):
        match_fifo([bad])


def test_multi_symbol_independent_matching():
    trades = [
        trade("BTC/USDT", "BUY", 1.0, 100.0, days=0),
        trade("ETH/USDT", "BUY", 1.0, 2000.0, days=0),
        trade("BTC/USDT", "SELL", 1.0, 120.0, days=1),
    ]
    closed, open_lots = match_fifo(trades)
    assert len(closed) == 1
    assert closed[0].symbol == "BTC/USDT"
    assert len(open_lots) == 1
    assert open_lots[0].symbol == "ETH/USDT"


def test_signal_carried_through_to_closed_lot():
    trades = [
        trade("BTC/USDT", "BUY", 1.0, 100.0, days=0, signal="BUY"),
        trade("BTC/USDT", "SELL", 1.0, 110.0, days=1, signal="SELL"),
    ]
    closed, _ = match_fifo(trades)
    assert closed[0].buy_signal == "BUY"
    assert closed[0].sell_signal == "SELL"


def test_empty_trades_returns_empty():
    closed, open_lots = match_fifo([])
    assert closed == []
    assert open_lots == []


def test_unordered_input_sorted_by_timestamp():
    trades = [
        trade("BTC/USDT", "SELL", 1.0, 110.0, days=1),
        trade("BTC/USDT", "BUY", 1.0, 100.0, days=0),
    ]
    closed, open_lots = match_fifo(trades)
    assert len(closed) == 1
    assert closed[0].buy_price == 100.0
    assert closed[0].sell_price == 110.0
