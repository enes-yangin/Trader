from typing import Optional, Any, Dict
import numpy as np
import pandas as pd
from data.indicators import get_features
from data.labeling import add_labels, CLASS_TARGET_COL
from models.base_classifier import BaseClassifier
from models.classifier_models import LogisticModel, XGBClassifierModel
from utils.config import FEATURES, MODEL, SPLIT
from utils.types import FeatureSpec
from engine.trainer import load_data, _resolve_spec

CLASSIFIER_MAP = {
    "logistic": LogisticModel,
    "xgb_clf": XGBClassifierModel,
}


def build_classifier(name: str, **kw: Any) -> BaseClassifier:
    cls = CLASSIFIER_MAP.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown classifier: {name}. Options: {list(CLASSIFIER_MAP.keys())}")
    return cls(**kw)


def split_classification(df: pd.DataFrame, spec: FeatureSpec,
                          threshold: float = 0.5, atr_normalize: bool = True,
                          tr: float = SPLIT.train_ratio,
                          va: float = SPLIT.val_ratio) -> Dict[str, Any]:
    df = add_labels(df, threshold=threshold, atr_normalize=atr_normalize)
    df = df.dropna(subset=[CLASS_TARGET_COL])
    X = get_features(df, spec=spec).values
    y = df[CLASS_TARGET_COL].values.astype(int)
    fwd = df[FEATURES.close_col].pct_change(MODEL.pred_horizon).shift(-MODEL.pred_horizon)
    fwd_arr = fwd.values
    idx = df.index
    n = len(X)
    i_tr, i_va = int(n * tr), int(n * (tr + va))
    return {
        "X_tr": X[:i_tr], "y_tr": y[:i_tr],
        "X_val": X[i_tr:i_va], "y_val": y[i_tr:i_va],
        "X_test": X[i_va:], "y_test": y[i_va:],
        "fwd_test": fwd_arr[i_va:],
        "idx_test": idx[i_va:],
        "spec": spec,
    }


def train_classifier(sym: str, model_name: str, src: str = "crypto",
                      horizon: int = MODEL.pred_horizon, spec: Optional[FeatureSpec] = None,
                      with_news: bool = FEATURES.use_news, with_micro: bool = FEATURES.use_micro,
                      with_cross_asset: bool = FEATURES.use_cross_asset,
                      threshold: float = 0.5, allow_sample: bool = False,
                      **kw: Any) -> Dict[str, Any]:
    spec = _resolve_spec(spec, with_news, with_micro, with_cross_asset)
    df = load_data(sym, src=src, horizon=horizon, spec=spec, allow_sample=allow_sample)
    sp = split_classification(df, spec, threshold=threshold)
    mdl = build_classifier(model_name, **kw)
    res = mdl.train(sp["X_tr"], sp["y_tr"], sp["X_val"], sp["y_val"])
    res["test"] = mdl.evaluate(sp["X_test"], sp["y_test"])
    return {"model": mdl, "metrics": res, "split": sp, "symbol": sym}


def directional_hit_rate(preds: np.ndarray, fwd_returns: np.ndarray) -> Dict[str, float]:
    from data.labeling import LABEL_BUY, LABEL_SELL
    n = min(len(preds), len(fwd_returns))
    preds, fwd_returns = preds[:n], fwd_returns[:n]
    mask = (preds == LABEL_BUY) | (preds == LABEL_SELL)
    n_sig = int(mask.sum())
    if n_sig == 0:
        return {"n_signals": 0, "hit_rate": float("nan")}
    correct = 0
    for p, f in zip(preds[mask], fwd_returns[mask]):
        if (p == LABEL_BUY and f > 0) or (p == LABEL_SELL and f < 0):
            correct += 1
    return {"n_signals": n_sig, "hit_rate": correct / n_sig}
