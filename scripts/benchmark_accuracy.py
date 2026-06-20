"""
Real-data model accuracy benchmark across all 8 crypto pairs.

Loads cached OHLCV from datasets/, engineers features across 6 combo
levels (baseline -> full), trains linear / xgboost / lstm on each
symbol, and reports R2 + RMSE on the chronological train/val/test split.

Usage:
    python scripts/benchmark_accuracy.py                    # all combos, all pairs
    python scripts/benchmark_accuracy.py --top 3 --no-lstm  # quick run
    python scripts/benchmark_accuracy.py --combo full       # full features only
    python scripts/benchmark_accuracy.py --no-winsorize     # disable outlier capping
    python scripts/benchmark_accuracy.py --json             # machine-readable
"""

import argparse
import json
import os
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import cache_policy, store
from data.indicators import engineer, get_features
from engine.trainer import split, build_model
from utils.config import FEATURES, MODEL, SPLIT
from utils.types import FeatureSpec

# -- symbols -----------------------------------------------------------
SYMBOLS: List[str] = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT",
    "XRP/USDT", "ADA/USDT", "DOGE/USDT", "AVAX/USDT",
]

# -- feature combos ----------------------------------------------------
FEATURE_COMBOS: List[Dict[str, Any]] = [
    {
        "id": "baseline",
        "label": "technical only",
        "spec": FeatureSpec(
            news=False, micro=False, cross_asset=False,
            smooth=False, reference=False,
        ),
    },
    {
        "id": "+micro",
        "label": "+microstructure",
        "spec": FeatureSpec(
            news=False, micro=True, cross_asset=False,
            smooth=False, reference=False,
        ),
    },
    {
        "id": "+smooth",
        "label": "+micro+smoothing",
        "spec": FeatureSpec(
            news=False, micro=True, cross_asset=False,
            smooth=True, reference=False,
        ),
    },
    {
        "id": "+news",
        "label": "+micro+smooth+news",
        "spec": FeatureSpec(
            news=True, micro=True, cross_asset=False,
            smooth=True, reference=False,
        ),
    },
    {
        "id": "+cross",
        "label": "+micro+smooth+news+cross_asset",
        "spec": FeatureSpec(
            news=True, micro=True, cross_asset=True,
            smooth=True, reference=False,
        ),
    },
    {
        "id": "full",
        "label": "full (all families)",
        "spec": FeatureSpec(
            news=True, micro=True, cross_asset=True,
            smooth=True, reference=True,
        ),
    },
]


# -- helpers ------------------------------------------------------------

def winsorize_features(df: pd.DataFrame, feature_cols: List[str],
                       limits: Tuple[float, float] = (0.005, 0.995)
                       ) -> pd.DataFrame:
    """Clip feature columns at quantile limits to reduce outlier influence.

    Target column is never winsorized; only input features are capped.
    Returns the same DataFrame (mutated in-place).
    """
    for col in feature_cols:
        if col not in df.columns:
            continue
        if df[col].dtype not in ("float64", "float32", "float16"):
            continue
        lo = df[col].quantile(limits[0])
        hi = df[col].quantile(limits[1])
        if lo < hi:
            df[col] = df[col].clip(lo, hi)
    return df


def load_cached_data(sym: str, spec: FeatureSpec,
                     verbose: bool = True) -> pd.DataFrame:
    """Load cached dataset respecting the feature spec (price vs news cache).

    Falls back to ``dataset.build(force=True)`` when cache is missing.
    """
    with_news = spec.news
    df = cache_policy.load_cached_dataset(sym, with_news=with_news)
    if df is not None:
        kind = "dataset_news" if with_news else "dataset"
        meta = store.meta(sym, kind) or {}
        if verbose:
            start = str(meta.get("start", "?"))[:10]
            end = str(meta.get("end", "?"))[:10]
            print(f"  {sym:15s}  loaded {len(df):>5d} rows  "
                  f"({start} -> {end})")
        return df

    if verbose:
        print(f"  {sym:15s}  cache miss -- building (with_news={with_news})...")
    from data import dataset
    return dataset.build(sym, src="crypto", with_news=with_news,
                         force=True, allow_sample=False)


def engineer_data(raw: pd.DataFrame, sym: str,
                  spec: FeatureSpec) -> pd.DataFrame:
    """Add indicators + target to raw OHLCV using the given FeatureSpec."""
    return engineer(raw, sym=sym, spec=spec, precomputed_news=spec.news)


def train_one_model(df: pd.DataFrame, model_name: str,
                    spec: FeatureSpec) -> Tuple[Dict[str, float], float]:
    """Train a single model, return (metrics_dict, elapsed_seconds)."""
    sp = split(df, spec=spec)
    t0 = time.perf_counter()
    mdl = build_model(model_name)
    res = mdl.train(sp["X_tr"], sp["y_tr"], sp["X_val"], sp["y_val"])
    res["test"] = mdl.evaluate(sp["X_test"], sp["y_test"])
    elapsed = time.perf_counter() - t0

    return {
        "train_r2": float(res["train"]["r2"]),
        "val_r2":   float(res["val"]["r2"]),
        "test_r2":  float(res["test"]["r2"]),
        "train_rmse": float(res["train"]["rmse"]),
        "val_rmse":   float(res["val"]["rmse"]),
        "test_rmse":  float(res["test"]["rmse"]),
    }, elapsed


# -- main loop ---------------------------------------------------------

def run_benchmark(
    symbols: List[str],
    skip_lstm: bool = False,
    combos: Optional[List[Dict[str, Any]]] = None,
    do_winsorize: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run benchmark across symbols x combos. Returns nested results dict."""
    if combos is None:
        combos = FEATURE_COMBOS

    models_to_run = ["linear", "xgboost"]
    if not skip_lstm:
        models_to_run.append("lstm")

    all_results: Dict[str, Any] = {}
    total_start = time.perf_counter()

    for combo in combos:
        combo_id = combo["id"]
        label = combo["label"]
        spec = combo["spec"]
        n_feat = spec.n_features

        if verbose:
            print(f"\n{'='*70}")
            print(f"  COMBO: {label}  ({n_feat} features)")
            print(f"{'='*70}")

        combo_results: Dict[str, Any] = {
            "_meta": {
                "label": label,
                "n_features": n_feat,
                "news": spec.news,
                "micro": spec.micro,
                "cross_asset": spec.cross_asset,
                "smooth": spec.smooth,
                "reference": spec.reference,
            },
        }
        for sym in symbols:
            if verbose:
                print(f"\n  --- {sym} ---")

            try:
                raw = load_cached_data(sym, spec, verbose=verbose)
            except Exception as e:
                print(f"  !! load failed: {type(e).__name__}: {e}")
                combo_results[sym] = {"_error": f"load: {e}"}
                continue

            try:
                df = engineer_data(raw, sym, spec)
            except Exception as e:
                print(f"  !! engineer failed: {type(e).__name__}: {e}")
                combo_results[sym] = {"_error": f"engineer: {e}"}
                continue

            # Winsorize features (never target)
            if do_winsorize:
                feat_cols = list(get_features(df, spec=spec).columns)
                df = winsorize_features(df, feat_cols)

            sym_results: Dict[str, Any] = {
                "rows": len(df),
                "n_features": len(get_features(df, spec=spec).columns),
            }
            for mdl_name in models_to_run:
                try:
                    metrics, elapsed = train_one_model(df, mdl_name, spec)
                    sym_results[mdl_name] = {**metrics, "elapsed_s": round(elapsed, 1)}
                    if verbose:
                        print(
                            f"  {mdl_name:10s}  "
                            f"train R2={metrics['train_r2']:+.4f}  "
                            f"val R2={metrics['val_r2']:+.4f}  "
                            f"test R2={metrics['test_r2']:+.4f}  "
                            f"({elapsed:.1f}s)"
                        )
                except Exception as e:
                    print(f"  {mdl_name:10s}  !! FAILED: {type(e).__name__}: {e}")
                    sym_results[mdl_name] = {"_error": str(e)}

            combo_results[sym] = sym_results

        all_results[combo_id] = combo_results

    total_elapsed = time.perf_counter() - total_start
    all_results["_meta"] = {
        "total_elapsed_s": round(total_elapsed, 1),
        "n_combos": len(combos),
        "symbols": symbols,
        "models": models_to_run,
        "pred_horizon": MODEL.pred_horizon,
        "winsorize": do_winsorize,
        "split": {
            "train": SPLIT.train_ratio,
            "val": SPLIT.val_ratio,
            "test": SPLIT.test_ratio,
        },
    }
    if verbose:
        print(f"\n{'='*70}")
        print(f"  Total time: {total_elapsed:.0f}s")
    return all_results


# -- reporting ----------------------------------------------------------

def _successful_symbols(combo_results: Dict) -> List[str]:
    return [s for s in SYMBOLS
            if s in combo_results
            and "_error" not in combo_results[s]]


def print_combo_summary(all_results: Dict[str, Any]) -> None:
    """Print what level of R2 each feature family provides."""
    combos = [k for k in all_results if k not in ("_meta",) and not k.startswith("_")]
    models = all_results["_meta"]["models"]
    combo_labels = {c["id"]: c["label"] for c in FEATURE_COMBOS}

    print(f"\n{'='*90}")
    print(f"  FEATURE COMBO SUMMARY  --  AVERAGE TEST R2 ACROSS ALL SYMBOLS")
    print(f"  (target: {FEATURES.target_col}  |  horizon: {MODEL.pred_horizon}d  |  "
          f"split: {int(SPLIT.train_ratio*100)}/"
          f"{int(SPLIT.val_ratio*100)}/"
          f"{int(SPLIT.test_ratio*100)}  |  "
          f"winsorize={'yes' if all_results['_meta'].get('winsorize') else 'no'})")
    print(f"{'='*90}")

    # Header
    header = f"  {'combo':28s} {'n_feat':>6s}"
    for m in models:
        header += f"  {m:>10s}"
    print(header)
    print(f"  {'-'*28}{'------'}{'----------' * len(models)}")

    for combo_id in combos:
        combo_data = all_results[combo_id]
        label = combo_labels.get(combo_id, combo_id)
        n_feat = combo_data.get("_meta", {}).get("n_features", 0)
        success = _successful_symbols(combo_data)

        line = f"  {label:28s} {n_feat:>6d}"
        for m in models:
            vals = []
            for s in success:
                v = combo_data[s].get(m, {}).get("test_r2")
                if v is not None:
                    vals.append(v)
            if vals:
                line += f"  {np.mean(vals):>+10.4f}"
            else:
                line += f"  {'--':>10s}"
        print(line)


def print_incremental_value(all_results: Dict[str, Any]) -> None:
    """Show the marginal R2 gain/loss when each feature family is added."""
    models = all_results["_meta"]["models"]
    # We need pairs: (before, after)
    transitions = [
        ("baseline -> +micro",    "baseline", "+micro",    "microstructure"),
        ("+micro -> +smooth",     "+micro",   "+smooth",   "smoothing"),
        ("+smooth -> +news",      "+smooth",  "+news",     "news sentiment"),
        ("+news -> +cross",       "+news",    "+cross",    "cross-asset"),
        ("+cross -> full",        "+cross",   "full",      "reference series"),
    ]

    print(f"\n{'='*90}")
    print(f"  FEATURE FAMILY INCREMENTAL VALUE  (delta of avg test R2)")
    print(f"{'='*90}")
    header = f"  {'transition':30s} {'from_feat':>7s} {'to_feat':>7s}"
    for m in models:
        header += f"  {m:>10s}"
    print(header)
    print(f"  {'-'*30}{'-------'*2}{'----------' * len(models)}")

    for label, before_id, after_id, family_name in transitions:
        if before_id not in all_results or after_id not in all_results:
            continue
        before = all_results[before_id]
        after = all_results[after_id]
        b_feat = before.get("_meta", {}).get("n_features", 0)
        a_feat = after.get("_meta", {}).get("n_features", 0)

        line = f"  {label:30s} {b_feat:>7d} {a_feat:>7d}"
        for m in models:
            b_vals, a_vals = [], []
            for s in SYMBOLS:
                bv = before.get(s, {}).get(m, {}).get("test_r2")
                av = after.get(s, {}).get(m, {}).get("test_r2")
                if bv is not None and av is not None:
                    b_vals.append(bv)
                    a_vals.append(av)
            if b_vals and a_vals:
                delta = np.mean(a_vals) - np.mean(b_vals)
                line += f"  {delta:>+10.4f}"
            else:
                line += f"  {'--':>10s}"
        print(line)


def print_best_per_symbol(all_results: Dict[str, Any]) -> None:
    """Show each symbol's best combo and model."""
    models = all_results["_meta"]["models"]
    combo_labels = {c["id"]: c["label"] for c in FEATURE_COMBOS}
    combos = [k for k in all_results if k not in ("_meta",) and not k.startswith("_")]

    print(f"\n{'='*90}")
    print(f"  BEST CONFIG PER SYMBOL  (highest test R2 across all combos)")
    print(f"{'='*90}")
    print(f"  {'symbol':15s} {'best_combo':30s} {'model':10s} {'test_r2':>8s} {'train_r2':>8s}")

    for sym in SYMBOLS:
        best_r2 = -999.0
        best_combo = "?"
        best_model = "?"
        best_train = 0.0
        for combo_id in combos:
            sd = all_results[combo_id].get(sym, {})
            if "_error" in sd:
                continue
            for m in models:
                v = sd.get(m, {}).get("test_r2")
                if v is not None and v > best_r2:
                    best_r2 = v
                    best_combo = combo_labels.get(combo_id, combo_id)
                    best_model = m
                    best_train = sd.get(m, {}).get("train_r2", 0)
        label = combo_labels.get(best_combo, best_combo)
        print(f"  {sym:15s} {label:30s} {best_model:10s} {best_r2:>+8.4f} {best_train:>+8.4f}")


def print_full_detail(all_results: Dict[str, Any]) -> None:
    """Print the per-symbol x per-model table for the 'full' combo only."""
    full_data = all_results.get("full")
    if full_data is None:
        return

    models = all_results["_meta"]["models"]
    success = _successful_symbols(full_data)

    print(f"\n{'='*90}")
    print(f"  FULL CONFIG: TEST R2 BY SYMBOL x MODEL")
    print(f"{'='*90}")

    header = f"  {'symbol':15s}"
    for m in models:
        header += f"  {m:>10s}"
    header += "  |  best"
    print(header)
    print(f"  {'-'*15}{'----------'*len(models)}--+---------")

    rows_for_avg = {m: [] for m in models}
    for sym in success:
        r = full_data[sym]
        line = f"  {sym:15s}"
        best_name, best_val = None, -999.0
        for m in models:
            v = r.get(m, {}).get("test_r2")
            if v is None:
                line += f"  {'FAIL':>10s}"
            else:
                line += f"  {v:>+10.4f}"
                rows_for_avg[m].append(v)
                if v > best_val:
                    best_val, best_name = v, m
        line += f"  |  {best_name}"
        print(line)

    print(f"  {'-'*15}{'----------'*len(models)}--+---------")
    avg_line = f"  {'AVERAGE':15s}"
    for m in models:
        vals = rows_for_avg[m]
        if vals:
            avg_line += f"  {np.mean(vals):>+10.4f}"
        else:
            avg_line += f"  {'--':>10s}"
    print(avg_line)


def to_json(all_results: Dict[str, Any]) -> str:
    """Serialise results to JSON (errors become string keys)."""
    out: Dict[str, Any] = {}
    for key, val in all_results.items():
        if key == "_meta":
            out["_meta"] = val
            continue
        out[key] = {}
        for sub_key, sub_val in val.items():
            if isinstance(sub_val, dict):
                out[key][sub_key] = {}
                for k, v in sub_val.items():
                    if isinstance(v, dict):
                        out[key][sub_key][k] = {
                            kk: (str(vv) if isinstance(vv, Exception) else vv)
                            for kk, vv in v.items()
                        }
                    elif isinstance(v, Exception):
                        out[key][sub_key][k] = str(v)
                    else:
                        out[key][sub_key][k] = v
            else:
                out[key][sub_key] = sub_val
    return json.dumps(out, indent=2, default=str)


# -- entry point -------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark model accuracy across feature combos")
    parser.add_argument("--top", type=int, default=0,
                        help="Only benchmark the first N symbols")
    parser.add_argument("--no-lstm", action="store_true",
                        help="Skip LSTM (much faster)")
    parser.add_argument("--json", action="store_true",
                        help="Print results as JSON")
    parser.add_argument("--symbols", nargs="+",
                        help="Specific symbols to benchmark")
    parser.add_argument("--combo", choices=[c["id"] for c in FEATURE_COMBOS] + ["all"],
                        default="all",
                        help="Which feature combo(s) to run (default: all)")
    parser.add_argument("--winsorize", action="store_true", default=True,
                        help="Winsorize features at 0.5/99.5%% (default: yes)")
    parser.add_argument("--no-winsorize", dest="winsorize", action="store_false",
                        help="Disable winsorizing")
    args = parser.parse_args()

    symbols = SYMBOLS
    if args.symbols:
        symbols = args.symbols
    if args.top and args.top > 0:
        symbols = symbols[:args.top]

    combos = FEATURE_COMBOS if args.combo == "all" else [
        c for c in FEATURE_COMBOS if c["id"] == args.combo
    ]

    print("=" * 70)
    print("  REAL-DATA MODEL ACCURACY BENCHMARK")
    print("=" * 70)
    print(f"  Symbols:    {len(symbols)} pair(s)")
    print(f"  Models:     linear, xgboost" + (", lstm" if not args.no_lstm else ""))
    print(f"  Target:     {FEATURES.target_col}  (horizon = {MODEL.pred_horizon}d)")
    print(f"  Split:      {int(SPLIT.train_ratio*100)}/"
          f"{int(SPLIT.val_ratio*100)}/"
          f"{int(SPLIT.test_ratio*100)}")
    print(f"  Combos:     {len(combos)} ({', '.join(c['id'] for c in combos)})")
    print(f"  Winsorize:  {'yes' if args.winsorize else 'no'}")

    results = run_benchmark(
        symbols,
        skip_lstm=args.no_lstm,
        combos=combos,
        do_winsorize=args.winsorize,
    )

    if args.json:
        print(to_json(results))
    else:
        print_combo_summary(results)
        print_incremental_value(results)
        print_best_per_symbol(results)
        print_full_detail(results)


if __name__ == "__main__":
    main()
