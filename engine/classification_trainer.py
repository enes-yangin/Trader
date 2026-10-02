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

# Classifiers actually trained in production. Both stay registered above (usable
# by ev_optimization, walk-forward, tests), but only these drive live signals /
# backtests. Measured on real BTC+alt data: with the min_proba confidence gate,
# xgb_clf's well-calibrated probabilities let it abstain when unsure (portfolio
# ~breakeven), whereas LogisticRegression's flatter probabilities cleared the
# gate too often, over-traded (121 trades) and lost ~-12%. This is a calibration
# argument, not a fit to one test window -- see backtest-exit-alignment memory.
PRODUCTION_CLASSIFIERS = ("xgb_clf",)


def build_classifier(name: str, **kw: Any) -> BaseClassifier:
    cls = CLASSIFIER_MAP.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown classifier: {name}. Options: {list(CLASSIFIER_MAP.keys())}")
    return cls(**kw)


from engine.purging import purged_train_end


def split_classification(df: pd.DataFrame, spec: FeatureSpec,
                          threshold: float = 0.5, atr_normalize: bool = True,
                          labeling: str = "fixed",
                          pt_mult: float = MODEL.pt_mult, sl_mult: float = MODEL.sl_mult,
                          tr: float = SPLIT.train_ratio,
                          va: float = SPLIT.val_ratio) -> Dict[str, Any]:
    if labeling == "triple_barrier":
        from data.labeling import triple_barrier_labels
        df = df.copy()
        df[CLASS_TARGET_COL] = triple_barrier_labels(
            df, h=MODEL.pred_horizon, pt_mult=pt_mult, sl_mult=sl_mult)
    else:
        df = add_labels(df, threshold=threshold, atr_normalize=atr_normalize)
    df = df.dropna(subset=[CLASS_TARGET_COL])
    X = get_features(df, spec=spec).values
    y = df[CLASS_TARGET_COL].values.astype(int)
    fwd = df[FEATURES.close_col].pct_change(MODEL.pred_horizon).shift(-MODEL.pred_horizon)
    fwd_arr = fwd.values
    idx = df.index
    n = len(X)
    i_tr, i_va = int(n * tr), int(n * (tr + va))

    # A1 & B6: Purge train and val endpoints to prevent forward-label leakage in classification splits
    label_horizon = MODEL.pred_horizon
    purged_i_tr = purged_train_end(0, i_tr, i_tr, label_horizon)
    purged_i_va = purged_train_end(i_tr, i_va, i_va, label_horizon)

    return {
        "X_tr": X[:purged_i_tr], "y_tr": y[:purged_i_tr],
        "X_val": X[purged_i_tr:purged_i_va], "y_val": y[purged_i_tr:purged_i_va],
        "X_test": X[purged_i_va:], "y_test": y[purged_i_va:],
        "fwd_test": fwd_arr[purged_i_va:],
        # Plan item I: filled in so engine.backtester's _select_set/run() can
        # score a classifier bundle exactly like a regression SplitDict --
        # idx_tr/idx_val/i_tr/i_va/df/split_idx mirror engine.trainer.split()'s
        # SplitDict shape (i_tr/i_va are the PURGED boundaries, since those are
        # the true start row of X_val/X_test after purging).
        "idx_tr": idx[:purged_i_tr], "idx_val": idx[purged_i_tr:purged_i_va],
        "idx_test": idx[purged_i_va:],
        "i_tr": purged_i_tr, "i_va": purged_i_va, "split_idx": purged_i_tr,
        "df": df,
        "spec": spec,
    }


def train_classifier(sym: str, model_name: str, src: str = "crypto",
                      horizon: int = MODEL.pred_horizon, spec: Optional[FeatureSpec] = None,
                      with_news: bool = FEATURES.use_news, with_micro: bool = FEATURES.use_micro,
                      with_cross_asset: bool = FEATURES.use_cross_asset,
                      with_smoothing: bool = FEATURES.use_smoothing,
                      with_reference: bool = FEATURES.use_reference,
                      with_orderbook: bool = FEATURES.use_orderbook,
                      with_macro_events: bool = FEATURES.use_macro_events,
                      with_social: bool = FEATURES.use_social,
                      threshold: float = 0.5, allow_sample: bool = False,
                      **kw: Any) -> Dict[str, Any]:
    spec = _resolve_spec(spec, with_news, with_micro, with_cross_asset,
                         with_smoothing, with_reference, with_orderbook,
                         with_macro_events, with_social)
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
