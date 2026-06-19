from typing import Optional
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from models.base_model import BaseModel
from utils.types import TrainResult


class LinearModel(BaseModel):
    def __init__(self, alpha: float = 1.0):
        super().__init__(name="LinearRegression")
        self.scaler = StandardScaler()
        self.model: Ridge = Ridge(alpha=alpha)

    def train(self, X_tr: np.ndarray, y_tr: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> TrainResult:
        Xs = self.scaler.fit_transform(X_tr)
        self.model.fit(Xs, y_tr)
        self.trained = True
        self.n_features_ = X_tr.shape[1]
        out: TrainResult = {"train": self.evaluate(X_tr, y_tr)}
        if X_val is not None and y_val is not None:
            out["val"] = self.evaluate(X_val, y_val)
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_ready(X)
        Xs = self.scaler.transform(X)
        return self.model.predict(Xs).flatten()
