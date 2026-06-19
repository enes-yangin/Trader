from typing import Optional, Any, Dict
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from models.base_classifier import BaseClassifier
from utils.config import MODEL
from utils.types import ClassTrainResult
from data.labeling import LABEL_SELL, LABEL_HOLD, LABEL_BUY

ALL_LABELS = [LABEL_SELL, LABEL_HOLD, LABEL_BUY]


class LogisticModel(BaseClassifier):
    def __init__(self, C: float = 1.0, **kw: Any):
        super().__init__(name="LogisticRegression")
        self.scaler = StandardScaler()
        self.model: LogisticRegression = LogisticRegression(
            C=C, max_iter=1000, class_weight="balanced", **kw,
        )

    def train(self, X_tr: np.ndarray, y_tr: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> ClassTrainResult:
        Xs = self.scaler.fit_transform(X_tr)
        self.model.fit(Xs, y_tr.astype(int))
        self.trained = True
        self.n_features_ = X_tr.shape[1]
        out: ClassTrainResult = {"train": self.evaluate(X_tr, y_tr)}
        if X_val is not None and y_val is not None:
            out["val"] = self.evaluate(X_val, y_val)
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_ready(X)
        Xs = self.scaler.transform(X)
        return self.model.predict(Xs).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_ready(X)
        Xs = self.scaler.transform(X)
        return _align_proba(self.model.predict_proba(Xs), self.model.classes_)


class XGBClassifierModel(BaseClassifier):
    def __init__(self, **kw: Any):
        super().__init__(name="XGBClassifier")
        early = kw.pop("early_stopping_rounds", None)
        params = {**MODEL.xgb_params, **kw}
        self.early_stopping_rounds: int = (
            MODEL.xgb_early_stopping_rounds if early is None else early
        )
        self.model: xgb.XGBClassifier = xgb.XGBClassifier(
            **params, objective="multi:softprob", num_class=3,
            verbosity=0, random_state=42,
        )

    def train(self, X_tr: np.ndarray, y_tr: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> ClassTrainResult:
        fit_kw: Dict[str, Any] = {}
        if X_val is not None and y_val is not None and self.early_stopping_rounds > 0:
            self.model.set_params(early_stopping_rounds=self.early_stopping_rounds)
            fit_kw["eval_set"] = [(X_val, y_val.astype(int))]
            fit_kw["verbose"] = False
        else:
            self.model.set_params(early_stopping_rounds=None)
        self.model.fit(X_tr, y_tr.astype(int), **fit_kw)
        self.trained = True
        self.n_features_ = X_tr.shape[1]
        out: ClassTrainResult = {"train": self.evaluate(X_tr, y_tr)}
        if X_val is not None and y_val is not None:
            out["val"] = self.evaluate(X_val, y_val)
        best_it = getattr(self.model, "best_iteration", None)
        if best_it is not None:
            out["best_iteration"] = int(best_it)
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_ready(X)
        return self.model.predict(X).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_ready(X)
        return _align_proba(self.model.predict_proba(X), self.model.classes_)


def _align_proba(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Reorder/pad probability columns to always be [SELL, HOLD, BUY] even if
    some class was absent from the training split."""
    out = np.zeros((proba.shape[0], 3))
    for col, cls in enumerate(classes):
        if int(cls) in ALL_LABELS:
            out[:, int(cls)] = proba[:, col]
    return out
