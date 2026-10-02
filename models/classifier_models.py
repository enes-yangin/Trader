from typing import Optional, Any, Dict
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
        self._params = {**MODEL.xgb_params, **kw}
        self.early_stopping_rounds: int = (
            MODEL.xgb_early_stopping_rounds if early is None else early
        )
        # num_class is set dynamically in train() (not fixed at 3 here): xgboost's
        # sklearn wrapper requires the label values passed to fit() to be dense
        # 0..n_classes-1. Wide triple-barrier labels (e.g. a long horizon where
        # the barrier is touched almost every time) can leave a whole class --
        # usually HOLD -- entirely absent from a training split, giving e.g.
        # {SELL=0, BUY=2} with no 1s. Encoding through LabelEncoder first (and
        # decoding predictions back) makes this class-absence case safe instead
        # of a hard ValueError ("Expected: [0 1], got [0 2]").
        self.model: xgb.XGBClassifier = xgb.XGBClassifier(
            **self._params, objective="multi:softprob",
            verbosity=0, random_state=42,
        )
        self._label_encoder: Optional[LabelEncoder] = None

    def train(self, X_tr: np.ndarray, y_tr: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> ClassTrainResult:
        self._label_encoder = LabelEncoder()
        y_tr_enc = self._label_encoder.fit_transform(y_tr.astype(int))
        self.model.set_params(num_class=len(self._label_encoder.classes_))

        fit_kw: Dict[str, Any] = {}
        if X_val is not None and y_val is not None and self.early_stopping_rounds > 0:
            try:
                y_val_enc = self._label_encoder.transform(y_val.astype(int))
                self.model.set_params(early_stopping_rounds=self.early_stopping_rounds)
                fit_kw["eval_set"] = [(X_val, y_val_enc)]
                fit_kw["verbose"] = False
            except ValueError:
                # Validation split contains a class absent from training
                # (can't be encoded) -- skip early stopping rather than crash.
                self.model.set_params(early_stopping_rounds=None)
        else:
            self.model.set_params(early_stopping_rounds=None)
        self.model.fit(X_tr, y_tr_enc, **fit_kw)
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
        assert self._label_encoder is not None
        raw = self.model.predict(X)
        if raw.ndim > 1:
            # Edge case: with only 2 classes present (num_class=2),
            # "multi:softprob" can make predict() return the (n, 2)
            # probability matrix itself instead of hard labels -- reduce
            # via argmax rather than crash on inverse_transform.
            enc_preds = raw.argmax(axis=1).astype(int)
        else:
            enc_preds = raw.astype(int)
        return self._label_encoder.inverse_transform(enc_preds).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_ready(X)
        assert self._label_encoder is not None
        proba = self.model.predict_proba(X)
        return _align_proba(proba, self._label_encoder.classes_)


def _align_proba(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Reorder/pad probability columns to always be [SELL, HOLD, BUY] even if
    some class was absent from the training split."""
    label_to_col = {label: i for i, label in enumerate(ALL_LABELS)}
    out = np.zeros((proba.shape[0], len(ALL_LABELS)))
    for col, cls in enumerate(classes):
        cls_int = int(cls)
        if cls_int in label_to_col:
            out[:, label_to_col[cls_int]] = proba[:, col]
    return out
