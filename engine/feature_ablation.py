from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from engine.classification_walkforward import run_classification
from utils.types import FeatureSpec
from utils.config import BACKTEST


def _spec_variants() -> Dict[str, FeatureSpec]:
    """Feature-group on/off combinations to compare.

    'cross_asset' is excluded here because it needs reference-symbol data that
    is not present on a single engineered frame; ablation focuses on the
    families derivable from one asset's OHLCV.
    """
    return {
        "technical_only": FeatureSpec(news=False, micro=False, cross_asset=False),
        "technical+micro": FeatureSpec(news=False, micro=True, cross_asset=False),
    }


def run_ablation(df: pd.DataFrame, model_name: str = "xgb_clf",
                  threshold: float = 0.5, labeling: str = "fixed",
                  train_size: int = BACKTEST.walkforward_train_size,
                  test_size: int = BACKTEST.walkforward_test_size,
                  variants: Optional[Dict[str, FeatureSpec]] = None) -> pd.DataFrame:
    """For each feature-group variant, run a classification walk-forward and
    collect the mean hit-rate and its stability. A feature family earns its
    place only if it lifts mean hit rate without inflating variance; on
    near-random data most families add noise, not signal, which is exactly
    what an honest ablation should reveal."""
    variants = variants or _spec_variants()
    rows: List[Dict[str, Any]] = []
    for label, spec in variants.items():
        res = run_classification(df, model_name, spec=spec, threshold=threshold,
                                  labeling=labeling,
                                  train_size=train_size, test_size=test_size)
        s = res["summary"]
        if not s:
            rows.append({"variant": label, "n_features": spec.n_features,
                         "n_windows": 0, "mean_hit_rate": np.nan,
                         "std_hit_rate": np.nan, "pct_above_50": np.nan,
                         "mean_f1": np.nan})
            continue
        rows.append({
            "variant": label,
            "n_features": spec.n_features,
            "n_windows": s["n_windows"],
            "mean_hit_rate": s["mean_hit_rate"],
            "std_hit_rate": s["std_hit_rate"],
            "pct_above_50": s["pct_windows_above_50"],
            "mean_f1": s["mean_f1"],
        })
    return pd.DataFrame(rows)


def summarize_ablation(table: pd.DataFrame) -> str:
    lines = ["=== Feature Ablation (walk-forward mean hit rate) ==="]
    for _, r in table.iterrows():
        hr = "n/a" if pd.isna(r["mean_hit_rate"]) else f"{r['mean_hit_rate']:.3f}"
        sd = "n/a" if pd.isna(r["std_hit_rate"]) else f"{r['std_hit_rate']:.3f}"
        pa = "n/a" if pd.isna(r["pct_above_50"]) else f"{r['pct_above_50']:.0f}%"
        lines.append(
            f"  {r['variant']:18s} feats={int(r['n_features']):2d}  "
            f"hit={hr} (std {sd})  windows>50%={pa}"
        )
    valid = table.dropna(subset=["mean_hit_rate"])
    if len(valid) >= 2:
        best = valid.loc[valid["mean_hit_rate"].idxmax()]
        lines.append(f"  -> best mean hit rate: {best['variant']} ({best['mean_hit_rate']:.3f})")
        delta = valid["mean_hit_rate"].max() - valid["mean_hit_rate"].min()
        lines.append(f"  -> spread across variants: {delta:.3f} "
                     f"({'meaningful' if delta > 0.03 else 'within noise'})")
    return "\n".join(lines)
