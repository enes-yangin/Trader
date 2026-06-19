from abc import ABC, abstractmethod
from typing import Optional, Any, Literal, Tuple
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils.exceptions import ModelNotTrainedError, FeatureMismatchError
from utils.config import SIGNAL
from utils.types import MetricsDict, TrainResult, ModelSignal


def dynamic_threshold(atr_pct: float, buy_th: float = SIGNAL.buy_threshold,
                       sell_th: float = SIGNAL.sell_threshold) -> Tuple[float, float]:
    """Scale BUY/SELL thresholds with recent volatility (ATR%).

    In low-volatility regimes the fixed +-2% threshold is rarely reached
    (model stays HOLD); in high-volatility regimes it is reached on noise
    alone. Scaling the threshold by atr_pct keeps signal sensitivity roughly
    proportional to the asset's current volatility regime. Falls back to the
    fixed thresholds if dynamic thresholding is disabled or atr_pct is
    non-positive/unavailable.
    """
    if not SIGNAL.use_dynamic_threshold or atr_pct <= 0 or np.isnan(atr_pct):
        return buy_th, sell_th
    th = SIGNAL.atr_mult * atr_pct
    return th, -th


class BaseModel(ABC):
    def __init__(self, name: str = "base"):
        self.name = name
        self.model: Optional[Any] = None
        self.trained: bool = False
        self.n_features_: Optional[int] = None

    @abstractmethod
    def train(self, X_tr: np.ndarray, y_tr: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> TrainResult:
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def _check_ready(self, X: np.ndarray) -> None:
        if not self.trained:
            raise ModelNotTrainedError(
                f"{self.name}: predict() called before train()"
            )
        if self.n_features_ is not None:
            n = X.shape[1] if X.ndim > 1 else X.shape[-1]
            if n != self.n_features_:
                raise FeatureMismatchError(
                    f"{self.name}: expected {self.n_features_} features, "
                    f"got {n}. Ensure with_news/with_micro flags match "
                    f"between training and inference."
                )

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> MetricsDict:
        preds = self.predict(X)
        mse = mean_squared_error(y, preds)
        mae = mean_absolute_error(y, preds)
        r2 = r2_score(y, preds)
        rmse = np.sqrt(mse)
        return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

    def signal(self, pred: float, buy_th: float = SIGNAL.buy_threshold,
               sell_th: float = SIGNAL.sell_threshold) -> ModelSignal:
        scale = SIGNAL.confidence_scale
        cmax = SIGNAL.confidence_max
        sig: Literal["BUY", "HOLD", "SELL"]
        if pred > buy_th:
            conf = min((pred - buy_th) / buy_th * scale, cmax)
            sig = "BUY"
        elif pred < sell_th:
            conf = min(abs(pred - sell_th) / abs(sell_th) * scale, cmax)
            sig = "SELL"
        else:
            conf = max(0, (1 - abs(pred) / buy_th) * scale)
            sig = "HOLD"
        return {"signal": sig, "confidence": round(float(conf), 1), "predicted_return": round(float(pred), 6)}

    def __repr__(self) -> str:
        st = "trained" if self.trained else "untrained"
        return f"<{self.name} [{st}]>"
