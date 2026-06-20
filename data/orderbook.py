import numpy as np
import ccxt
from utils.config import DATA, ORDERBOOK


def fetch_orderbook(sym, depth=ORDERBOOK.depth):
    ex = getattr(ccxt, DATA.exchange_id)({"enableRateLimit": True})
    ob = ex.fetch_order_book(sym, limit=depth)
    return ob


def imbalance(ob, levels=ORDERBOOK.imbalance_levels):
    bids = ob["bids"][:levels]
    asks = ob["asks"][:levels]
    bid_vol = sum(b[1] for b in bids)
    ask_vol = sum(a[1] for a in asks)
    tot = bid_vol + ask_vol
    if tot == 0:
        return 0.0
    return (bid_vol - ask_vol) / tot


def best_prices(ob):
    bids = ob.get("bids") or []
    asks = ob.get("asks") or []
    best_bid = bids[0][0] if bids else None
    best_ask = asks[0][0] if asks else None
    if best_bid is not None and best_ask is not None:
        mid = (best_bid + best_ask) / 2
    elif best_bid is not None:
        mid = best_bid
    elif best_ask is not None:
        mid = best_ask
    else:
        mid = 0.0
    return best_bid, best_ask, mid


def spread(ob):
    best_bid, best_ask, mid = best_prices(ob)
    if best_bid is None or best_ask is None:
        return 0.0, 0.0
    abs_sp = best_ask - best_bid
    rel_sp = abs_sp / mid if mid > 0 else 0.0
    return abs_sp, rel_sp


def detect_walls(ob, levels=ORDERBOOK.wall_levels, factor=ORDERBOOK.wall_factor):
    bids = ob["bids"][:levels]
    asks = ob["asks"][:levels]
    walls: dict = {"bid": [], "ask": []}
    # Order-book rows are [price, volume] on some venues (Binance) but
    # [price, volume, timestamp] on others (Kraken) -- index, don't unpack.
    if bids:
        bvols = [b[1] for b in bids]
        bmean = np.mean(bvols)
        for lvl in bids:
            px, vol = lvl[0], lvl[1]
            if vol > bmean * factor:
                walls["bid"].append({"price": px, "size": vol})
    if asks:
        avols = [a[1] for a in asks]
        amean = np.mean(avols)
        for lvl in asks:
            px, vol = lvl[0], lvl[1]
            if vol > amean * factor:
                walls["ask"].append({"price": px, "size": vol})
    return walls


def microprice(ob):
    best_bid, best_ask, mid = best_prices(ob)
    if best_bid is None or best_ask is None:
        return mid
    bv = ob["bids"][0][1]
    av = ob["asks"][0][1]
    tot = bv + av
    if tot == 0:
        return mid
    return (best_bid * av + best_ask * bv) / tot


def signal(ob, imb_th=ORDERBOOK.imbalance_threshold):
    imb = imbalance(ob)
    abs_sp, rel_sp = spread(ob)
    walls = detect_walls(ob)
    mp = microprice(ob)
    best_bid, best_ask, mid = best_prices(ob)

    if imb > imb_th:
        state = "BUY PRESSURE"
    elif imb < -imb_th:
        state = "SELL PRESSURE"
    else:
        state = "BALANCED"

    # Honest confidence: imbalance already lives in [-1, 1], so report its
    # magnitude directly instead of clamping to 100% the moment it passes the
    # (low) 0.3 threshold -- imb 0.31 is NOT "100% sure".
    conf = min(abs(imb) * 100, 100)
    return {
        "symbol": ob.get("symbol", "N/A"),
        "state": state,
        "imbalance": round(imb, 4),
        "confidence": round(conf, 1),
        "rel_spread": round(rel_sp, 6),
        "mid_price": round(mid, 2),
        "microprice": round(mp, 2),
        "micro_skew": round((mp - mid) / mid, 6) if mid > 0 else 0.0,
        "bid_walls": len(walls["bid"]),
        "ask_walls": len(walls["ask"]),
        "walls": walls,
    }


def book_reliability(ob, levels=ORDERBOOK.imbalance_levels,
                     min_levels=ORDERBOOK.min_side_levels,
                     thin_ratio=ORDERBOOK.thin_book_ratio):
    """Whether the top-`levels` book is deep enough on both sides to trust the
    imbalance. A near-one-sided book makes imbalance saturate to +/-1, which is
    what drives the 100%-buy <-> 100%-sell whipsaw. Returns (reliable, bid_vol,
    ask_vol)."""
    bids = ob.get("bids") or []
    asks = ob.get("asks") or []
    bid_vol = sum(b[1] for b in bids[:levels])
    ask_vol = sum(a[1] for a in asks[:levels])
    tot = bid_vol + ask_vol
    if len(bids) < min_levels or len(asks) < min_levels or tot <= 0:
        return False, bid_vol, ask_vol
    weak_share = min(bid_vol, ask_vol) / tot
    return weak_share >= thin_ratio, bid_vol, ask_vol


def _resolve_state(imb, prev_state, enter_th=ORDERBOOK.imbalance_threshold,
                   exit_th=ORDERBOOK.imbalance_exit_threshold):
    """Hysteresis state machine: enter a directional state only past `enter_th`,
    but hold it until imbalance retreats inside `exit_th`. This stops the state
    from flipping every time noisy imbalance jitters across the 0.3 line."""
    if prev_state == "BUY PRESSURE" and imb > exit_th:
        return "BUY PRESSURE"
    if prev_state == "SELL PRESSURE" and imb < -exit_th:
        return "SELL PRESSURE"
    if imb > enter_th:
        return "BUY PRESSURE"
    if imb < -enter_th:
        return "SELL PRESSURE"
    return "BALANCED"


class OrderBookTracker:
    """Stateful wrapper around `signal()` that adds EMA smoothing + hysteresis +
    a thin-book guard, keyed per symbol. The UI holds one instance and calls
    `update(ob)` each refresh; this is what tames the live-panel whipsaw.

    A thin/one-sided snapshot is reported as state 'THIN BOOK' (confidence 0)
    and does NOT update the smoothed imbalance, so garbage books cannot corrupt
    the running estimate or flip the state.
    """

    def __init__(self, alpha=ORDERBOOK.imbalance_smooth_alpha,
                 enter_th=ORDERBOOK.imbalance_threshold,
                 exit_th=ORDERBOOK.imbalance_exit_threshold):
        self.alpha = alpha
        self.enter_th = enter_th
        self.exit_th = exit_th
        self._ema: dict = {}
        self._state: dict = {}

    def reset(self, sym=None):
        if sym is None:
            self._ema.clear()
            self._state.clear()
        else:
            self._ema.pop(sym, None)
            self._state.pop(sym, None)

    def update(self, ob, sym=None):
        sym = sym or ob.get("symbol", "N/A")
        sig = signal(ob)
        sig["symbol"] = sym
        sig["raw_imbalance"] = sig["imbalance"]

        reliable, _bv, _av = book_reliability(ob)
        if not reliable:
            # Hold the smoothed value; surface unreliability instead of a fake 100%.
            sig["state"] = "THIN BOOK"
            sig["confidence"] = 0.0
            sig["reliable"] = False
            sig["imbalance"] = round(self._ema.get(sym, 0.0), 4)
            return sig

        raw = sig["raw_imbalance"]
        prev_ema = self._ema.get(sym)
        ema = raw if prev_ema is None else self.alpha * raw + (1 - self.alpha) * prev_ema
        self._ema[sym] = ema

        prev_state = self._state.get(sym, "BALANCED")
        state = _resolve_state(ema, prev_state, self.enter_th, self.exit_th)
        self._state[sym] = state

        sig["state"] = state
        sig["imbalance"] = round(ema, 4)
        sig["confidence"] = round(min(abs(ema) * 100, 100), 1)
        sig["reliable"] = True
        return sig


def live_signal(sym, depth=ORDERBOOK.depth):
    ob = fetch_orderbook(sym, depth=depth)
    sig = signal(ob)
    sig["symbol"] = sym
    return sig


def format_signal(sig):
    lines = [
        "=" * 42,
        f"  LIVE ORDER BOOK — {sig['symbol']}",
        "=" * 42,
        f"  State:       {sig['state']} ({sig['confidence']:.0f}%)",
        f"  Imbalance:   {sig['imbalance']:+.4f}",
        f"  Mid Price:   {sig['mid_price']:,.2f}",
        f"  Microprice:  {sig['microprice']:,.2f} (skew {sig['micro_skew']:+.5f})",
        f"  Rel Spread:  {sig['rel_spread']:.4%}",
        f"  Walls:       {sig['bid_walls']} bid / {sig['ask_walls']} ask",
        "=" * 42,
    ]
    return "\n".join(lines)


def generate_sample_ob(sym="BTC/USDT", mid=30000.0, skew=0.0):
    bids, asks = [], []
    for i in range(50):
        bp = mid * (1 - 0.0001 * (i + 1))
        ap = mid * (1 + 0.0001 * (i + 1))
        bv = abs(np.random.randn()) * 10 * max(1 + skew, 0.0)
        av = abs(np.random.randn()) * 10 * max(1 - skew, 0.0)
        bids.append([bp, bv])
        asks.append([ap, av])
    return {"bids": bids, "asks": asks, "symbol": sym}
