from data.orderbook import (
    best_prices, spread, microprice, signal, imbalance, generate_sample_ob
)


def test_best_prices_both_sides():
    ob = {"bids": [[100, 5]], "asks": [[102, 3]], "symbol": "X"}
    bb, ba, mid = best_prices(ob)
    assert bb == 100
    assert ba == 102
    assert mid == 101


def test_best_prices_bid_only():
    ob = {"bids": [[100, 5], [99, 3]], "asks": [], "symbol": "X"}
    bb, ba, mid = best_prices(ob)
    assert bb == 100
    assert ba is None
    assert mid == 100, "mid should equal best_bid, not best_bid/2"


def test_best_prices_ask_only():
    ob = {"bids": [], "asks": [[200, 5]], "symbol": "X"}
    bb, ba, mid = best_prices(ob)
    assert bb is None
    assert ba == 200
    assert mid == 200, "mid should equal best_ask, not best_ask/2"


def test_best_prices_empty_book():
    ob = {"bids": [], "asks": [], "symbol": "X"}
    bb, ba, mid = best_prices(ob)
    assert bb is None
    assert ba is None
    assert mid == 0.0


def test_spread_one_sided_returns_zero():
    ob = {"bids": [[100, 5]], "asks": [], "symbol": "X"}
    abs_sp, rel_sp = spread(ob)
    assert abs_sp == 0.0
    assert rel_sp == 0.0


def test_microprice_one_sided_falls_back_to_mid():
    ob = {"bids": [[100, 5]], "asks": [], "symbol": "X"}
    mp = microprice(ob)
    assert mp == 100


def test_signal_empty_book_no_crash():
    ob = {"bids": [], "asks": [], "symbol": "X"}
    sig = signal(ob)
    assert sig["state"] == "BALANCED"
    assert sig["mid_price"] == 0.0
    assert sig["confidence"] == 0.0


def test_signal_buy_pressure_detected():
    ob = generate_sample_ob("BTC/USDT", mid=30000, skew=0.5)
    sig = signal(ob)
    assert sig["imbalance"] > 0
    assert sig["state"] == "BUY PRESSURE"


def test_signal_sell_pressure_detected():
    ob = generate_sample_ob("BTC/USDT", mid=30000, skew=-0.5)
    sig = signal(ob)
    assert sig["imbalance"] < 0
    assert sig["state"] == "SELL PRESSURE"


def test_imbalance_zero_volume_returns_zero():
    ob = {"bids": [[100, 0]], "asks": [[101, 0]], "symbol": "X"}
    assert imbalance(ob) == 0.0
