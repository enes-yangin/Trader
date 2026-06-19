from typing import List, Tuple, Dict, Any
import numpy as np
from models.xgb_model import XGBModel
from utils.config import OPTIMIZATION
from utils.types import SplitDict
from utils.logger import get_logger

log = get_logger("feature_selection")


def rank_features(sp: SplitDict, feature_names: List[str]) -> List[Tuple[str, float]]:
    if len(feature_names) != sp["X_tr"].shape[1]:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) does not match "
            f"X_tr feature count ({sp['X_tr'].shape[1]})"
        )
    mdl = XGBModel()
    mdl.train(sp["X_tr"], sp["y_tr"])
    importances = mdl.model.feature_importances_
    order = np.argsort(importances)[::-1]
    return [(feature_names[i], float(importances[i])) for i in order]


def select_top_k(sp: SplitDict, feature_names: List[str],
                  k: int = OPTIMIZATION.feature_selection_top_k) -> Dict[str, Any]:
    k = min(k, len(feature_names))
    ranked = rank_features(sp, feature_names)
    selected_names = [name for name, _ in ranked[:k]]
    selected_idx = [feature_names.index(n) for n in selected_names]

    new_sp = dict(sp)
    for key in ("X_tr", "X_val", "X_test"):
        new_sp[key] = sp[key][:, selected_idx]
    new_sp["selected_features"] = selected_names
    new_sp["feature_importance"] = ranked

    log.info(f"Selected {len(selected_names)}/{len(feature_names)} features: {selected_names}")
    return new_sp


def format_importance(ranked: List[Tuple[str, float]], top_n: int = 10) -> str:
    lines = ["  Feature Importance (XGBoost)", "  " + "-" * 30]
    for name, imp in ranked[:top_n]:
        bar = "#" * max(1, int(imp * 50))
        lines.append(f"  {name:16s} {imp:.4f}  {bar}")
    return "\n".join(lines)
