from datetime import datetime, timedelta
import pytest
from portfolio.ledger import Ledger
from portfolio.position import Trade
from utils.exceptions import PortfolioError

T0 = datetime(2026, 1, 1)


def trade(symbol, side, qty, price, days=0, **kw):
    return Trade(symbol, side, qty, price, T0 + timedelta(days=days), **kw)


@pytest.fixture
def ledger_path(tmp_path):
    return tmp_path / "ledger.json"


def test_new_ledger_is_empty(ledger_path):
    led = Ledger(path=ledger_path)
    assert led.list_trades() == []
    assert led.symbols() == []


def test_add_and_list_trades(ledger_path):
    led = Ledger(path=ledger_path)
    led.add_trade(trade("BTC/USDT", "BUY", 1.0, 100.0, days=0))
    led.add_trade(trade("ETH/USDT", "BUY", 2.0, 2000.0, days=1))
    assert len(led.list_trades()) == 2
    assert led.symbols() == ["BTC/USDT", "ETH/USDT"]
    assert len(led.list_trades(symbol="BTC/USDT")) == 1


def test_persistence_across_instances(ledger_path):
    led = Ledger(path=ledger_path)
    led.add_trade(trade("BTC/USDT", "BUY", 1.0, 100.0, days=0, fee=1.0, signal="BUY"))
    assert ledger_path.exists()

    reloaded = Ledger(path=ledger_path)
    trades = reloaded.list_trades()
    assert len(trades) == 1
    t = trades[0]
    assert t.symbol == "BTC/USDT"
    assert t.side == "BUY"
    assert t.qty == 1.0
    assert t.price == 100.0
    assert t.fee == 1.0
    assert t.signal == "BUY"
    assert t.timestamp == T0


def test_remove_trade(ledger_path):
    led = Ledger(path=ledger_path)
    led.add_trade(trade("BTC/USDT", "BUY", 1.0, 100.0, days=0))
    led.add_trade(trade("BTC/USDT", "BUY", 1.0, 110.0, days=1))
    removed = led.remove_trade(0)
    assert removed.price == 100.0
    assert len(led.list_trades()) == 1
    assert led.list_trades()[0].price == 110.0

    reloaded = Ledger(path=ledger_path)
    assert len(reloaded.list_trades()) == 1


def test_remove_invalid_index_raises(ledger_path):
    led = Ledger(path=ledger_path)
    led.add_trade(trade("BTC/USDT", "BUY", 1.0, 100.0, days=0))
    with pytest.raises(PortfolioError):
        led.remove_trade(5)
    with pytest.raises(PortfolioError):
        led.remove_trade(-1)


def test_oversell_rejected_and_not_persisted(ledger_path):
    led = Ledger(path=ledger_path)
    led.add_trade(trade("BTC/USDT", "BUY", 1.0, 100.0, days=0))
    with pytest.raises(PortfolioError, match="oversell"):
        led.add_trade(trade("BTC/USDT", "SELL", 5.0, 110.0, days=1))
    assert len(led.list_trades()) == 1

    reloaded = Ledger(path=ledger_path)
    assert len(reloaded.list_trades()) == 1


def test_validate_false_skips_fifo_check(ledger_path):
    led = Ledger(path=ledger_path)
    led.add_trade(trade("BTC/USDT", "SELL", 5.0, 110.0, days=0), validate=False)
    assert len(led.list_trades()) == 1


def test_clear(ledger_path):
    led = Ledger(path=ledger_path)
    led.add_trade(trade("BTC/USDT", "BUY", 1.0, 100.0, days=0))
    led.clear()
    assert led.list_trades() == []

    reloaded = Ledger(path=ledger_path)
    assert reloaded.list_trades() == []


def test_load_missing_file_is_empty(tmp_path):
    led = Ledger(path=tmp_path / "does_not_exist.json")
    assert led.list_trades() == []
