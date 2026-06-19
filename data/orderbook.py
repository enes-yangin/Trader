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
    if bids:
        bvols = [b[1] for b in bids]
        bmean = np.mean(bvols)
        for px, vol in bids:
            if vol > bmean * factor:
                walls["bid"].append({"price": px, "size": vol})
    if asks:
        avols = [a[1] for a in asks]
        amean = np.mean(avols)
        for px, vol in asks:
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

    conf = min(abs(imb) / imb_th * 100, 100) if imb_th > 0 else 0.0
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
        bv = abs(np.random.randn()) * 10 * (1 + skew)
        av = abs(np.random.randn()) * 10 * (1 - skew)
        bids.append([bp, bv])
        asks.append([ap, av])
    return {"bids": bids, "asks": asks, "symbol": sym}
