from data.orderbook import (
    best_prices, spread, microprice, signal, imbalance, generate_sample_ob,
    book_reliability, _resolve_state, OrderBookTracker,
)


def _book(bv, av, n=10, mid=100.0):
    """Order book with `n` levels per side, each bid level holding `bv` and each
    ask level `av`, so imbalance = (bv - av) / (bv + av) exactly."""
    bids = [[mid * (1 - 0.001 * (i + 1)), bv] for i in range(n)]
    asks = [[mid * (1 + 0.001 * (i + 1)), av] for i in range(n)]
    return {"bids": bids, "asks": asks, "symbol": "X"}


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


# ------------------------------------------------------------------ #
# Honest confidence (no clamp-to-100 the moment threshold is crossed)  #
# ------------------------------------------------------------------ #

def test_confidence_is_honest_not_clamped():
    # imbalance 0.4 must read ~40% confidence, NOT 100%.
    sig = signal(_book(7, 3))  # (7-3)/10 = 0.4
    assert abs(sig["imbalance"] - 0.4) < 1e-9
    assert 35 <= sig["confidence"] <= 45


def test_confidence_full_only_when_one_sided():
    sig = signal(_book(10, 0))  # imbalance +1.0
    assert sig["confidence"] == 100.0


# ------------------------------------------------------------------ #
# Thin-book guard                                                     #
# ------------------------------------------------------------------ #

def test_book_reliability_accepts_balanced_deep_book():
    ok, bv, av = book_reliability(_book(10, 10))
    assert ok


def test_book_reliability_rejects_one_sided():
    ok, _, _ = book_reliability(_book(10, 0.001))  # ask side ~empty
    assert not ok


def test_book_reliability_rejects_too_few_levels():
    ok, _, _ = book_reliability(_book(10, 10, n=3))  # < min_side_levels
    assert not ok


# ------------------------------------------------------------------ #
# Hysteresis state machine                                            #
# ------------------------------------------------------------------ #

def test_hysteresis_enters_and_holds_and_exits():
    assert _resolve_state(0.4, "BALANCED") == "BUY PRESSURE"      # enter
    assert _resolve_state(0.2, "BUY PRESSURE") == "BUY PRESSURE"  # hold in dead-band
    assert _resolve_state(0.1, "BUY PRESSURE") == "BALANCED"      # exit below 0.15


def test_hysteresis_no_direct_flip_buy_to_sell():
    # From BUY, a mild negative must go to BALANCED first, not straight to SELL.
    assert _resolve_state(-0.2, "BUY PRESSURE") == "BALANCED"
    assert _resolve_state(-0.4, "BUY PRESSURE") == "SELL PRESSURE"


# ------------------------------------------------------------------ #
# Tracker: smoothing + hysteresis kill the whipsaw                    #
# ------------------------------------------------------------------ #

def test_tracker_smoothing_suppresses_whipsaw():
    # Raw imbalance alternating +-0.5 every refresh WOULD flip BUY/SELL each
    # tick. With EMA smoothing + hysteresis the state must never reach SELL.
    tr = OrderBookTracker()
    states = []
    for raw in [0.5, -0.5, 0.5, -0.5, 0.5, -0.5]:
        bv, av = (7.5, 2.5) if raw > 0 else (2.5, 7.5)
        states.append(tr.update(_book(bv, av), sym="X")["state"])
    assert "SELL PRESSURE" not in states, f"whipsaw not suppressed: {states}"


def test_tracker_reports_thin_book_without_corrupting_state():
    tr = OrderBookTracker()
    tr.update(_book(8, 2), sym="X")            # reliable warm-up -> BUY-ish
    sig = tr.update(_book(10, 0.001), sym="X")  # thin snapshot
    assert sig["state"] == "THIN BOOK"
    assert sig["confidence"] == 0.0
    assert sig["reliable"] is False


def test_tracker_reset_clears_state():
    tr = OrderBookTracker()
    tr.update(_book(9, 1), sym="X")
    tr.reset()
    assert tr._ema == {} and tr._state == {}
