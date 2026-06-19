import numpy as np
import pandas as pd
from typing import Optional, Type, Dict, Any, List, Tuple
from data.indicators import get_features
from engine.purging import purge_window
from models.base_model import BaseModel
from utils.exceptions import InsufficientDataError
from utils.config import BACKTEST, FEATURES, MODEL
from utils.types import FeatureSpec, WalkForwardResult, WalkForwardSummary


def make_windows(n: int, train_size: int, test_size: int,
                 step: Optional[int] = None) -> List[Dict[str, Tuple[int, int]]]:
    if step is None:
        step = test_size
    wins: List[Dict[str, Tuple[int, int]]] = []
    start = 0
    while start + train_size + test_size <= n:
        tr = (start, start + train_size)
        te = (start + train_size, start + train_size + test_size)
        wins.append({"train": tr, "test": te})
        start += step
    return wins


def _aligned_preds(mdl: BaseModel, X_seg: np.ndarray,
                    y_seg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    seq_len = getattr(mdl, "seq_len", None)
    preds = mdl.predict(X_seg)
    if seq_len and len(preds) == len(X_seg) - seq_len:
        yt = y_seg[seq_len:]
    else:
        n = min(len(preds), len(y_seg))
        preds = preds[:n]
        yt = y_seg[:n]
    return preds, yt


def run(df: pd.DataFrame, model_cls: Type[BaseModel],
        train_size: int = BACKTEST.walkforward_train_size,
        test_size: int = BACKTEST.walkforward_test_size, step: Optional[int] = None,
        spec: Optional[FeatureSpec] = None,
        with_news: bool = FEATURES.use_news, with_micro: bool = FEATURES.use_micro,
        with_cross_asset: bool = FEATURES.use_cross_asset,
        purge: Optional[int] = None, embargo: int = 0,
        model_kw: Optional[Dict[str, Any]] = None) -> WalkForwardResult:
    spec = spec or FeatureSpec.from_bools(with_news, with_micro, with_cross_asset)
    model_kw = model_kw or {}
    if FEATURES.target_col not in df.columns:
        raise InsufficientDataError(
            f"walkforward.run expects an engineered DataFrame containing the "
            f"'{FEATURES.target_col}' column and feature columns. Got raw OHLCV instead. "
            f"Call data.indicators.engineer(df, ...) first."
        )
    X = get_features(df, spec=spec).values
    y = df[FEATURES.target_col].values
    idx = df.index
    n = len(X)
    wins = make_windows(n, train_size, test_size, step)
    if not wins:
        return {"folds": pd.DataFrame(), "summary": WalkForwardSummary(), "n_windows": 0}

    label_horizon = MODEL.pred_horizon if purge is None else purge

    rows = []
    for k, w in enumerate(wins):
        a, b = purge_window(w["train"], w["test"], label_horizon, embargo)
        c, d = w["test"]
        if b <= a:
            continue
        mdl = model_cls(**model_kw)
        mdl.train(X[a:b], y[a:b])
        m = mdl.evaluate(X[c:d], y[c:d])
        preds, yt = _aligned_preds(mdl, X[c:d], y[c:d])
        dir_acc = float(np.mean(np.sign(preds) == np.sign(yt))) if len(preds) else 0.0
        rows.append({
            "window": k + 1,
            "train_start": idx[a].date(),
            "test_start": idx[c].date(),
            "test_end": idx[d - 1].date(),
            "r2": round(m["r2"], 4),
            "rmse": round(m["rmse"], 5),
            "dir_acc": round(dir_acc, 4),
        })

    folds = pd.DataFrame(rows)
    if folds.empty:
        return {"folds": folds, "summary": WalkForwardSummary(), "n_windows": 0}
    summary: WalkForwardSummary = {
        "n_windows": len(rows),
        "mean_r2": round(folds["r2"].mean(), 4),
        "std_r2": round(folds["r2"].std(), 4),
        "mean_dir_acc": round(folds["dir_acc"].mean(), 4),
        "std_dir_acc": round(folds["dir_acc"].std(), 4),
        "pct_r2_positive": round((folds["r2"] > 0).mean() * 100, 1),
        "pct_dir_above_50": round((folds["dir_acc"] > 0.5).mean() * 100, 1),
    }
    return {"folds": folds, "summary": summary, "n_windows": len(rows)}


def run_model(df: pd.DataFrame, model_name: str, **kw: Any) -> WalkForwardResult:
    from models.linear_model import LinearModel
    from models.xgb_model import XGBModel
    from models.lstm_model import LSTMModel
    mp: Dict[str, Type[BaseModel]] = {"linear": LinearModel, "xgboost": XGBModel, "lstm": LSTMModel}
    cls = mp[model_name.lower()]
    mk = {"epochs": MODEL.lstm_params["epochs"]} if model_name == "lstm" else {}
    return run(df, cls, model_kw=mk, **kw)


def format_report(name: str, res: WalkForwardResult) -> str:
    s = res["summary"]
    if not s:
        return f"  {name}: insufficient data for walk-forward"
    lines = [
        f"  [{name.upper()}] — {s['n_windows']} windows",
        f"    Mean R2:      {s['mean_r2']:+.4f} (std {s['std_r2']:.4f})",
        f"    Mean Dir Acc: {s['mean_dir_acc']:.1%} (std {s['std_dir_acc']:.1%})",
        f"    R2 > 0:       {s['pct_r2_positive']:.0f}% of windows",
        f"    Dir > 50%:    {s['pct_dir_above_50']:.0f}% of windows",
    ]
    return "\n".join(lines)
