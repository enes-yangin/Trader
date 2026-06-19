from datetime import datetime, timedelta
from unittest.mock import patch
import pandas as pd
import pytest
from portfolio.position import Trade
from portfolio.pnl import (
    realized_trades, open_positions, with_live_prices,
    total_realized_pnl, monthly_summary, signal_alignment,
)

T0 = datetime(2026, 1, 1)


def trade(symbol, side, qty, price, days=0, **kw):
    return Trade(symbol, side, qty, price, T0 + timedelta(days=days), **kw)


def test_realized_trades_empty():
    assert realized_trades([]) == []
    assert total_realized_pnl([]) == 0.0


def test_total_realized_pnl_sums_closed_lots():
    trades = [
        trade("BTC/USDT", "BUY", 1.0, 100.0, days=0),
        trade("BTC/USDT", "SELL", 1.0, 120.0, days=5),
    ]
    assert total_realized_pnl(trades) == pytest.approx(20.0)


def test_open_positions_aggregates_across_lots():
    trades = [
        trade("BTC/USDT", "BUY", 1.0, 100.0, days=0),
        trade("BTC/USDT", "BUY", 1.0, 200.0, days=1),
    ]
    positions = open_positions(trades)
    assert len(positions) == 1
    p = positions[0]
    assert p.symbol == "BTC/USDT"
    assert p.qty == pytest.approx(2.0)
    assert p.avg_price == pytest.approx(150.0)
    assert p.cost_basis == pytest.approx(300.0)


def test_open_positions_empty_when_fully_closed():
    trades = [
        trade("BTC/USDT", "BUY", 1.0, 100.0, days=0),
        trade("BTC/USDT", "SELL", 1.0, 110.0, days=1),
    ]
    assert open_positions(trades) == []


def test_with_live_prices_success():
    trades = [trade("BTC/USDT", "BUY", 1.0, 100.0, days=0)]
    positions = open_positions(trades)
    fake_df = pd.DataFrame({"close": [150.0]})
    with patch("portfolio.pnl.fetch", return_value=fake_df):
        out = with_live_prices(positions)
    p = out[0]
    assert p.current_price == 150.0
    assert p.unrealized_pnl == pytest.approx(50.0)
    assert p.unrealized_pnl_pct == pytest.approx(50.0)


def test_with_live_prices_failure_leaves_fields_none():
    trades = [trade("BTC/USDT", "BUY", 1.0, 100.0, days=0)]
    positions = open_positions(trades)
    with patch("portfolio.pnl.fetch", side_effect=ConnectionError("offline")):
        out = with_live_prices(positions)
    p = out[0]
    assert p.current_price is None
    assert p.unrealized_pnl is None


def test_monthly_summary_empty():
    df = monthly_summary([])
    assert df.empty
    assert list(df.columns) == ["month", "realized_pnl", "n_closed", "win_rate", "avg_pnl_pct", "best", "worst"]


def test_monthly_summary_groups_by_month():
    trades = [
        trade("BTC/USDT", "BUY", 1.0, 100.0, days=0),
        trade("BTC/USDT", "SELL", 1.0, 110.0, days=2),
        trade("BTC/USDT", "BUY", 1.0, 100.0, days=35),
        trade("BTC/USDT", "SELL", 1.0, 90.0, days=37),
    ]
    df = monthly_summary(trades)
    assert len(df) == 2
    jan = df[df["month"] == "2026-01"].iloc[0]
    assert jan["n_closed"] == 1
    assert jan["realized_pnl"] == pytest.approx(10.0)
    assert jan["win_rate"] == pytest.approx(100.0)

    feb = df[df["month"] == "2026-02"].iloc[0]
    assert feb["n_closed"] == 1
    assert feb["realized_pnl"] == pytest.approx(-10.0)
    assert feb["win_rate"] == pytest.approx(0.0)


def test_signal_alignment_empty_when_no_signals():
    trades = [
        trade("BTC/USDT", "BUY", 1.0, 100.0, days=0),
        trade("BTC/USDT", "SELL", 1.0, 110.0, days=1),
    ]
    df = signal_alignment(trades)
    assert df.empty


def test_signal_alignment_groups_by_buy_signal():
    trades = [
        trade("BTC/USDT", "BUY", 1.0, 100.0, days=0, signal="BUY"),
        trade("BTC/USDT", "SELL", 1.0, 120.0, days=1, signal="SELL"),
        trade("BTC/USDT", "BUY", 1.0, 100.0, days=2, signal="HOLD"),
        trade("BTC/USDT", "SELL", 1.0, 90.0, days=3, signal="SELL"),
    ]
    df = signal_alignment(trades)
    assert set(df["buy_signal"]) == {"BUY", "HOLD"}
    buy_row = df[df["buy_signal"] == "BUY"].iloc[0]
    assert buy_row["n_closed"] == 1
    assert buy_row["total_pnl"] == pytest.approx(20.0)
    assert buy_row["win_rate"] == pytest.approx(100.0)

    hold_row = df[df["buy_signal"] == "HOLD"].iloc[0]
    assert hold_row["total_pnl"] == pytest.approx(-10.0)
    assert hold_row["win_rate"] == pytest.approx(0.0)
