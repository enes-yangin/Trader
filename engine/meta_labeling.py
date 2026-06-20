import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.ensemble import RandomForestClassifier
from utils.types import Bundle, SplitDict, EnsembleSignal
from engine.backtester import _lstm_preds
from utils.logger import get_logger

log = get_logger("meta_labeling")


def train_meta_model(sp: SplitDict, primary_results: Dict[str, Any]) -> Any:
    """
    Trains a secondary RandomForestClassifier on the validation split.
    The targets are meta-labels (1 if primary ensemble is directionally correct, 0 otherwise).
    Uses out-of-sample predictions on validation set to prevent future data leakage.
    """
    y_val = sp["y_val"]
    X_val = sp["X_val"]
    
    if len(y_val) == 0 or len(X_val) == 0:
        log.warning("Validation split is empty. Cannot train meta-model.")
        return None

    # 1. Gather out-of-sample validation predictions for all primary models
    all_preds = []
    for name, r in primary_results.items():
        mdl = r["model"]
        try:
            if hasattr(mdl, "predict_last"):
                preds = _lstm_preds(mdl, sp, "val")
            else:
                preds = mdl.predict(X_val)
            all_preds.append(preds)
        except Exception as e:
            log.warning(f"Failed to get validation predictions for {name}: {e}")
            
    if not all_preds:
        log.warning("No primary model predictions available. Cannot train meta-model.")
        return None
        
    # Truncate to minimum length to align
    min_len = min(len(y_val), *(len(p) for p in all_preds))
    if min_len < 5:
        log.warning(f"Validation set too small ({min_len} rows) for meta-labeling.")
        return None
        
    all_preds_aligned = [p[:min_len] for p in all_preds]
    ens_preds_val = np.mean(all_preds_aligned, axis=0)
    y_val_aligned = y_val[:min_len]
    X_val_aligned = X_val[:min_len]
    
    # 2. Construct meta-labels: 1 if direction matches, 0 otherwise
    y_meta = (np.sign(ens_preds_val) == np.sign(y_val_aligned)).astype(int)
    
    # Safeguard if only one class exists
    if len(np.unique(y_meta)) <= 1:
        log.warning("Meta-labels contain only one class. Using dummy classifier fallback.")
        class DummyMetaClassifier:
            def predict_proba(self, X):
                # Always predict 100% confidence for class 1 (correct prediction)
                return np.column_stack([np.zeros(len(X)), np.ones(len(X))])
        return DummyMetaClassifier()
        
    # 3. Fit RandomForestClassifier
    try:
        clf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
        clf.fit(X_val_aligned, y_meta)
        log.info(f"Meta-model trained successfully on {min_len} validation rows.")
        return clf
    except Exception as e:
        log.exception(f"Failed to fit meta-model: {e}")
        return None


def filter_signal_with_meta(sig: EnsembleSignal, df: pd.DataFrame, spec: Any,
                             meta_model: Any, selected_features: List[str] = None,
                             threshold: float = 0.50) -> None:
    """
    Evaluates the meta-model on the last feature row of the DataFrame.
    If the probability of correctness is below the threshold, overrides the consensus signal to HOLD.
    """
    if meta_model is None:
        return
        
    try:
        from data.indicators import get_features
        feat_df = get_features(df, spec=spec)
        if selected_features is not None:
            feat_df = feat_df[selected_features]
            
        X_last = feat_df.values[-1:]
        
        # Predict probability of correctness (class 1)
        prob_correct = float(meta_model.predict_proba(X_last)[0, 1])
        sig["meta_confidence"] = round(prob_correct * 100, 1)
        
        # If probability is below threshold, override direction to HOLD to filter out false alarms
        if prob_correct < threshold and sig["consensus"] in ("BUY", "SELL"):
            log.info(f"Meta-model filtered out signal {sig['consensus']} (confidence: {prob_correct:.2f} < {threshold})")
            sig["consensus"] = "HOLD"
    except Exception as e:
        log.warning(f"Error applying meta-labeling filter: {e}")
