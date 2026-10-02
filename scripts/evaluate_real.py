"""Real-data evaluation: the numbers quoted in README.md come from this script.

Fetches daily OHLCV straight from the exchange (no dataset cache, no synthetic
fallback -- a failed or stale fetch aborts that symbol), then runs:

  1. Directional classifier (xgb_clf, triple-barrier labels) through the
     purged walk-forward, per symbol.
  2. Stat-arb (each alt vs BTC, log prices): Engle-Granger on the first 80%,
     then the app's grid -> sealed-holdout -> deflated-p protocol.

Usage:
    python scripts/evaluate_real.py                       # exchange from .env / config
    AI_TRADER_EXCHANGE=binance python scripts/evaluate_real.py
Writes docs/results/real_data_eval_<exchange>.json and prints a markdown summary.
(Kraken's public API serves only ~720 daily bars; Binance paginates the full 5 years.)
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.fetcher import fetch_hist
from data.indicators import engineer
from engine.classification_walkforward import run_classification
from engine.statarb import engle_granger
from engine.statarb_signals import select_and_judge
from engine.validation_protocol import binomial_p_value
from utils.config import BACKTEST, DATA, MODEL
from utils.types import FeatureSpec

MAX_STALE_DAYS = 3
SPEC = FeatureSpec(micro=True, smooth=True)
STATARB_GRID = [(e, x) for e in (1.5, 2.0, 2.5) for x in (0.25, 0.5)]
STATARB_COST = 2 * BACKTEST.commission_pct + 2 * BACKTEST.slippage_pct


def fetch_real(sym):
    df = fetch_hist(sym)
    if df.attrs.get("source") != "crypto":
        raise RuntimeError(f"{sym}: non-exchange source {df.attrs.get('source')!r}")
    age = (pd.Timestamp.now("UTC").tz_localize(None) - df.index[-1]).days
    if age > MAX_STALE_DAYS:
        raise RuntimeError(f"{sym}: last bar {df.index[-1].date()} is {age}d old")
    return df


def eval_directional(df):
    feats = engineer(df, spec=SPEC)
    n = len(feats)
    train = min(BACKTEST.walkforward_train_size, max(150, n // 3))
    test = min(BACKTEST.walkforward_test_size, max(30, n // 10))
    res = run_classification(feats, "xgb_clf", spec=SPEC, labeling="triple_barrier",
                             purge=MODEL.pred_horizon, train_size=train, test_size=test)
    folds, s = res["folds"], res["summary"]
    if not s:
        return {"n_windows": 0}
    f = folds.dropna(subset=["hit_rate"])
    n_sig = int(f["n_signals"].sum())
    wins = int(round((f["hit_rate"] * f["n_signals"]).sum()))
    hit = wins / n_sig if n_sig else float("nan")
    # Baseline: share of up-moves over the same test span. A constant
    # always-BUY / always-SELL "model" scores max(up, 1 - up) with zero skill.
    h = MODEL.pred_horizon
    fwd = feats["close"].pct_change(h).shift(-h)
    span = fwd.loc[str(folds["test_start"].iloc[0]):str(folds["test_end"].iloc[-1])].dropna()
    up = float((span > 0).mean())
    # Forward returns overlap h bars, so signals are not independent trials:
    # the honest sample size is roughly signals / h.
    n_eff = n_sig // h
    return {
        "n_windows": s["n_windows"], "train_size": train, "test_size": test,
        "test_from": str(folds["test_start"].iloc[0]), "test_to": str(folds["test_end"].iloc[-1]),
        "test_bars": int(s["n_windows"] * test), "signals": n_sig, "wins": wins,
        "pooled_hit_rate": round(hit, 4) if n_sig else None,
        "base_rate_up": round(up, 4),
        "best_constant_hit_rate": round(max(up, 1 - up), 4),
        "pct_windows_above_50": float(s["pct_windows_above_50"]),
        "naive_p": round(binomial_p_value(wins, n_sig), 4) if n_sig else None,
        "n_eff": n_eff,
        "p_eff": round(binomial_p_value(int(round(hit * n_eff)), n_eff), 4) if n_eff else None,
    }


def eval_statarb(df_y, df_x):
    idx = df_y.index.intersection(df_x.index)
    # Log prices: the spread is then scale-free and the round-trip cost is a
    # fraction of notional. (The app's UI path uses raw prices and scales the
    # cost by the spread's notional instead -- see README "Known limitations".)
    y, x = np.log(df_y["close"].reindex(idx).values), np.log(df_x["close"].reindex(idx).values)
    eg = engle_granger(y[:int(len(y) * 0.8)], x[:int(len(x) * 0.8)])
    rep = select_and_judge(y, x, STATARB_GRID, train_frac=0.6, val_frac=0.2,
                           wf_train=120, wf_test=40, cost=STATARB_COST)
    return {
        "rows": len(idx), "cointegrated_5pct": bool(eg.cointegrated),
        "eg_adf_stat": round(float(eg.adf.stat), 3), "best_params": rep.best_params,
        "n_trials": rep.n_trials, "holdout_trades": rep.holdout_trades,
        "holdout_wins": rep.holdout_wins,
        "holdout_win_rate": round(rep.holdout_win_rate, 4),
        "holdout_total_pnl": round(rep.holdout_total_pnl, 4),
        "raw_p": round(rep.raw_p_value, 4), "deflated_p": round(rep.deflated_p_value, 4),
    }


def main():
    out = {
        "run_at_utc": pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M"),
        "exchange": DATA.exchange_id, "timeframe": DATA.timeframe,
        "horizon_bars": MODEL.pred_horizon, "pt_mult": MODEL.pt_mult, "sl_mult": MODEL.sl_mult,
        "features": "technical + microstructure + smoothing",
        "statarb_round_trip_cost": STATARB_COST,
        "data": {}, "directional": {}, "statarb": {}, "errors": {},
    }
    prices = {}
    for sym in DATA.crypto_symbols:
        try:
            df = fetch_real(sym)
        except Exception as e:
            out["errors"][sym] = f"{type(e).__name__}: {str(e)[:200]}"
            print(f"[FAIL] {sym}: {out['errors'][sym]}", flush=True)
            continue
        prices[sym] = df
        out["data"][sym] = {"rows": len(df), "from": str(df.index[0].date()),
                            "to": str(df.index[-1].date()),
                            "last_close": float(df["close"].iloc[-1])}
        print(f"[data] {sym}: {out['data'][sym]}", flush=True)

    for sym, df in prices.items():
        try:
            out["directional"][sym] = eval_directional(df)
        except Exception as e:
            out["errors"][f"directional:{sym}"] = f"{type(e).__name__}: {str(e)[:200]}"
        print(f"[directional] {sym}: {out['directional'].get(sym)}", flush=True)

    if "BTC/USDT" in prices:
        for sym, df in prices.items():
            if sym == "BTC/USDT":
                continue
            try:
                out["statarb"][sym] = eval_statarb(df, prices["BTC/USDT"])
            except Exception as e:
                out["errors"][f"statarb:{sym}"] = f"{type(e).__name__}: {str(e)[:200]}"
            print(f"[statarb] {sym} vs BTC: {out['statarb'].get(sym)}", flush=True)

    d = [v for v in out["directional"].values() if v.get("signals")]
    sig, wins = sum(v["signals"] for v in d), sum(v["wins"] for v in d)
    out["directional_pooled"] = {
        "symbols": len(d), "signals": sig, "wins": wins,
        "hit_rate": round(wins / sig, 4) if sig else None,
        "symbols_beating_constant_baseline": [
            s for s, v in out["directional"].items()
            if v.get("signals") and v["pooled_hit_rate"] > v["best_constant_hit_rate"]],
        "symbols_significant_p_eff": [
            s for s, v in out["directional"].items()
            if v.get("p_eff") is not None and v["p_eff"] < 0.05],
    }
    out["statarb_passed_shield"] = [s for s, v in out["statarb"].items() if v["deflated_p"] < 0.05]

    res_dir = os.path.join(ROOT, "docs", "results")
    os.makedirs(res_dir, exist_ok=True)
    path = os.path.join(res_dir, f"real_data_eval_{DATA.exchange_id}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print("\n| Symbol | Rows | Windows | Signals | Hit rate | Best constant | Windows >50% | n_eff | p_eff |")
    print("|---|---|---|---|---|---|---|---|---|")
    for sym, v in out["directional"].items():
        if v.get("n_windows"):
            print(f"| {sym} | {out['data'][sym]['rows']} | {v['n_windows']} | {v['signals']} | "
                  f"{v['pooled_hit_rate']} | {v['best_constant_hit_rate']} | "
                  f"{v['pct_windows_above_50']}% | {v['n_eff']} | {v['p_eff']} |")
    print(f"\nPooled: {out['directional_pooled']}")
    print("\n| Pair | Cointegrated | Holdout trades | Win rate | Total PnL | Deflated p |")
    print("|---|---|---|---|---|---|")
    for sym, v in out["statarb"].items():
        print(f"| {sym} vs BTC | {v['cointegrated_5pct']} | {v['holdout_trades']} | "
              f"{v['holdout_win_rate']} | {v['holdout_total_pnl']} | {v['deflated_p']} |")
    print(f"\nWrote {path}")
    if out["errors"]:
        print(f"Errors: {out['errors']}")


if __name__ == "__main__":
    main()
