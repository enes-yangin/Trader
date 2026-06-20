import numpy as np
import pandas as pd
from utils.config import MODEL, FEATURES

LABEL_SELL = 0
LABEL_HOLD = 1
LABEL_BUY = 2
LABEL_NAMES = {LABEL_SELL: "SELL", LABEL_HOLD: "HOLD", LABEL_BUY: "BUY"}
CLASS_TARGET_COL = "target_cls"


def make_labels(df, h=None, threshold=0.0, atr_normalize=True):
    """Convert forward h-period return into a 3-class directional label.

    The continuous return target is dominated by noise; turning it into
    SELL/HOLD/BUY lets a classifier optimize directly for direction instead
    of fighting to predict an exact (mostly unpredictable) magnitude.

    threshold: dead-zone half-width. Returns whose magnitude falls inside the
        band become HOLD. With atr_normalize=True the band scales with each
        row's ATR%, so a flat 1% move in a calm regime and a 5% move in a wild
        regime are judged relative to prevailing volatility rather than by a
        single fixed cutoff.
    """
    h = h if h is not None else MODEL.pred_horizon
    fwd = df[FEATURES.close_col].pct_change(periods=h).shift(-h)

    if atr_normalize and "atr_pct" in df.columns:
        band = threshold * df["atr_pct"] if threshold > 0 else df["atr_pct"] * 0.0
        excess = fwd
        up = (excess > band).fillna(False)
        down = (excess < -band).fillna(False)
    else:
        up = (fwd > threshold).fillna(False)
        down = (fwd < -threshold).fillna(False)

    labels = np.full(len(df), LABEL_HOLD, dtype=float)
    labels[up.values] = LABEL_BUY
    labels[down.values] = LABEL_SELL
    labels[fwd.isna().values] = np.nan
    return pd.Series(labels, index=df.index, name=CLASS_TARGET_COL)


def add_labels(df, h=None, threshold=0.0, atr_normalize=True):
    df = df.copy()
    df[CLASS_TARGET_COL] = make_labels(df, h=h, threshold=threshold, atr_normalize=atr_normalize)
    return df


def triple_barrier_labels(df, h=None, pt_mult=1.5, sl_mult=1.0,
                           vol_col="atr_pct", min_vol=1e-4):
    """Label each bar by which of three barriers the path touches first.

    For every bar t, set an upper barrier at +pt_mult*vol[t], a lower barrier
    at -sl_mult*vol[t], and a vertical (time) barrier h bars ahead. Walking
    forward from t+1 to t+h:
      - if the high crosses the upper barrier first  -> BUY  (profit-taking hit)
      - if the low crosses the lower barrier first   -> SELL (stop-loss hit)
      - if neither is touched by the vertical barrier -> HOLD (timed out)

    This beats a fixed "return h bars later" label because it respects the
    *path*: a position that hits its profit target on day 2 and reverses by
    day 5 is correctly a BUY win, not whatever the day-5 snapshot happens to
    show. Barriers scale with per-bar volatility (ATR%), so the same label
    logic adapts across calm and turbulent regimes.

    Barriers are evaluated using future highs/lows, so labels (like any
    forward-looking target) must never be used as model inputs -- only as the
    prediction target on the training portion of a chronological split.
    """
    h = h if h is not None else MODEL.pred_horizon
    close = df[FEATURES.close_col].values
    high = df["high"].values if "high" in df.columns else close
    low = df["low"].values if "low" in df.columns else close

    if vol_col in df.columns:
        vol = df[vol_col].values.astype(float)
    else:
        vol = np.full(len(df), min_vol)
    vol = np.where(np.isnan(vol) | (vol < min_vol), min_vol, vol)

    n = len(df)
    labels = np.full(n, np.nan)
    for t in range(n):
        end = t + h
        if end >= n:
            break
        entry = close[t]
        up = entry * (1 + pt_mult * vol[t])
        dn = entry * (1 - sl_mult * vol[t])
        lbl = LABEL_HOLD
        for j in range(t + 1, end + 1):
            if high[j] >= up:
                lbl = LABEL_BUY
                break
            if low[j] <= dn:
                lbl = LABEL_SELL
                break
        labels[t] = lbl
    return pd.Series(labels, index=df.index, name=CLASS_TARGET_COL)


def add_triple_barrier_labels(df, h=None, pt_mult=1.5, sl_mult=1.0):
    df = df.copy()
    df[CLASS_TARGET_COL] = triple_barrier_labels(df, h=h, pt_mult=pt_mult, sl_mult=sl_mult)
    return df


def label_distribution(labels):
    s = pd.Series(labels).dropna()
    if len(s) == 0:
        return {"SELL": 0.0, "HOLD": 0.0, "BUY": 0.0}
    vc = s.value_counts(normalize=True)
    return {
        "SELL": float(vc.get(LABEL_SELL, 0.0)),
        "HOLD": float(vc.get(LABEL_HOLD, 0.0)),
        "BUY": float(vc.get(LABEL_BUY, 0.0)),
    }
