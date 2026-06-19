from abc import ABC, abstractmethod
from typing import Optional, Any
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score
from utils.exceptions import ModelNotTrainedError, FeatureMismatchError
from utils.types import ClassMetrics, ClassTrainResult, ModelSignal
from data.labeling import LABEL_SELL, LABEL_HOLD, LABEL_BUY


class BaseClassifier(ABC):
    def __init__(self, name: str = "base_clf"):
        self.name = name
        self.model: Optional[Any] = None
        self.trained: bool = False
        self.n_features_: Optional[int] = None

    @abstractmethod
    def train(self, X_tr: np.ndarray, y_tr: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> ClassTrainResult:
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ...

    def _check_ready(self, X: np.ndarray) -> None:
        if not self.trained:
            raise ModelNotTrainedError(f"{self.name}: predict() called before train()")
        if self.n_features_ is not None:
            n = X.shape[1] if X.ndim > 1 else X.shape[-1]
            if n != self.n_features_:
                raise FeatureMismatchError(
                    f"{self.name}: expected {self.n_features_} features, got {n}."
                )

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> ClassMetrics:
        preds = self.predict(X)
        n = min(len(preds), len(y))
        preds, y = preds[:n], y[:n]
        labels = [LABEL_SELL, LABEL_HOLD, LABEL_BUY]
        prec = precision_score(y, preds, labels=labels, average=None,
                                zero_division=0)
        prec_map = {lbl: float(prec[i]) for i, lbl in enumerate(labels)}
        return {
            "accuracy": float(accuracy_score(y, preds)),
            "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
            "f1_macro": float(f1_score(y, preds, labels=labels, average="macro",
                                        zero_division=0)),
            "precision_buy": prec_map[LABEL_BUY],
            "precision_sell": prec_map[LABEL_SELL],
            "n_buy_pred": int(np.sum(preds == LABEL_BUY)),
            "n_sell_pred": int(np.sum(preds == LABEL_SELL)),
            "n_hold_pred": int(np.sum(preds == LABEL_HOLD)),
        }

    def signal(self, X_last: np.ndarray, min_proba: float = 0.5) -> ModelSignal:
        proba = self.predict_proba(X_last.reshape(1, -1))[0]
        cls_idx = int(np.argmax(proba))
        confidence = float(proba[cls_idx]) * 100
        label_to_cls = {LABEL_SELL: "SELL", LABEL_HOLD: "HOLD", LABEL_BUY: "BUY"}
        sig_name = label_to_cls.get(cls_idx, "HOLD")
        if cls_idx != LABEL_HOLD and proba[cls_idx] < min_proba:
            sig_name = "HOLD"
        return {
            "signal": sig_name,  # type: ignore[typeddict-item]
            "confidence": round(confidence, 1),
            "predicted_return": 0.0,
        }

    def __repr__(self) -> str:
        st = "trained" if self.trained else "untrained"
        return f"<{self.name} [{st}]>"
