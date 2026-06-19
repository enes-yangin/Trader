from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
from data.indicators import get_features
from data.labeling import add_labels, CLASS_TARGET_COL, LABEL_BUY, LABEL_SELL
from engine.classification_trainer import build_classifier
from engine.purging import purge_window
from engine.walkforward import make_windows
from utils.config import FEATURES, MODEL, BACKTEST
from utils.types import FeatureSpec


def _hit_rate(preds: np.ndarray, fwd: np.ndarray) -> Dict[str, float]:
    mask = (preds == LABEL_BUY) | (preds == LABEL_SELL)
    n_sig = int(mask.sum())
    if n_sig == 0:
        return {"n_signals": 0, "hit_rate": float("nan")}
    correct = 0
    for p, f in zip(preds[mask], fwd[mask]):
        if (p == LABEL_BUY and f > 0) or (p == LABEL_SELL and f < 0):
            correct += 1
    return {"n_signals": n_sig, "hit_rate": correct / n_sig}


def run_classification(df: pd.DataFrame, model_name: str,
                        spec: Optional[FeatureSpec] = None,
                        threshold: float = 0.5,
                        labeling: str = "fixed",
                        pt_mult: float = 1.5, sl_mult: float = 1.0,
                        purge: Optional[int] = None, embargo: int = 0,
                        train_size: int = BACKTEST.walkforward_train_size,
                        test_size: int = BACKTEST.walkforward_test_size,
                        step: Optional[int] = None,
                        model_kw: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    spec = spec or FeatureSpec.default()
    model_kw = model_kw or {}
    h = MODEL.pred_horizon

    if labeling == "triple_barrier":
        from data.labeling import triple_barrier_labels
        df = df.copy()
        df[CLASS_TARGET_COL] = triple_barrier_labels(df, h=h, pt_mult=pt_mult, sl_mult=sl_mult)
    else:
        df = add_labels(df, h=h, threshold=threshold, atr_normalize=True)
    fwd = df[FEATURES.close_col].pct_change(h).shift(-h)
    df = df.assign(_fwd=fwd).dropna(subset=[CLASS_TARGET_COL, "_fwd"])

    X = get_features(df, spec=spec).values
    y = df[CLASS_TARGET_COL].values.astype(int)
    fwd_arr = df["_fwd"].values
    idx = df.index

    wins = make_windows(len(X), train_size, test_size, step)
    if not wins:
        return {"folds": pd.DataFrame(), "summary": {}, "n_windows": 0}

    label_horizon = h if purge is None else purge

    rows: List[Dict[str, Any]] = []
    for k, w in enumerate(wins):
        a, b = purge_window(w["train"], w["test"], label_horizon, embargo)
        c, d = w["test"]
        if b <= a:
            continue
        mdl = build_classifier(model_name, **model_kw)
        mdl.train(X[a:b], y[a:b])
        preds = mdl.predict(X[c:d])
        me = mdl.evaluate(X[c:d], y[c:d])
        hr = _hit_rate(preds, fwd_arr[c:d])
        ts = idx[c].date() if hasattr(idx[c], "date") else int(idx[c])
        te = idx[d - 1].date() if hasattr(idx[d - 1], "date") else int(idx[d - 1])
        rows.append({
            "window": k + 1,
            "test_start": ts,
            "test_end": te,
            "accuracy": round(me["accuracy"], 4),
            "f1_macro": round(me["f1_macro"], 4),
            "n_signals": hr["n_signals"],
            "hit_rate": round(hr["hit_rate"], 4) if not np.isnan(hr["hit_rate"]) else np.nan,
        })

    folds = pd.DataFrame(rows)
    if folds.empty:
        return {"folds": folds, "summary": {}, "n_windows": 0}
    valid_hr = folds["hit_rate"].dropna()
    summary = {
        "n_windows": len(rows),
        "mean_accuracy": round(folds["accuracy"].mean(), 4),
        "mean_f1": round(folds["f1_macro"].mean(), 4),
        "mean_hit_rate": round(valid_hr.mean(), 4) if len(valid_hr) else float("nan"),
        "std_hit_rate": round(valid_hr.std(), 4) if len(valid_hr) > 1 else float("nan"),
        "pct_windows_above_50": round((valid_hr > 0.5).mean() * 100, 1) if len(valid_hr) else 0.0,
        "total_signals": int(folds["n_signals"].sum()),
    }
    return {"folds": folds, "summary": summary, "n_windows": len(rows)}


def format_classification_report(name: str, res: Dict[str, Any]) -> str:
    s = res["summary"]
    if not s:
        return f"{name}: no walk-forward windows (insufficient data)"
    lines = [
        f"=== {name} walk-forward ({s['n_windows']} windows) ===",
        f"  mean accuracy:    {s['mean_accuracy']:.3f}",
        f"  mean F1 (macro):  {s['mean_f1']:.3f}",
        f"  mean hit rate:    {s['mean_hit_rate']:.3f}  (std {s['std_hit_rate']:.3f})",
        f"  windows > 50%:    {s['pct_windows_above_50']:.0f}%",
        f"  total signals:    {s['total_signals']}",
    ]
    return "\n".join(lines)
