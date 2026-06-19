from typing import Optional, Sequence, Dict, Union, Any
import numpy as np
import xgboost as xgb
from models.base_model import BaseModel
from utils.config import MODEL
from utils.types import TrainResult


class XGBModel(BaseModel):
    def __init__(self, **kw: Any):
        super().__init__(name="XGBoost")
        early_stopping_rounds = kw.pop("early_stopping_rounds", None)
        params = {**MODEL.xgb_params, **kw}
        self.early_stopping_rounds: int = (
            MODEL.xgb_early_stopping_rounds if early_stopping_rounds is None
            else early_stopping_rounds
        )
        self.model: xgb.XGBRegressor = xgb.XGBRegressor(
            **params,
            objective="reg:squarederror",
            verbosity=0,
            random_state=42,
        )

    def train(self, X_tr: np.ndarray, y_tr: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> TrainResult:
        fit_kw: Dict[str, Any] = {}
        if X_val is not None and y_val is not None and self.early_stopping_rounds > 0:
            self.model.set_params(early_stopping_rounds=self.early_stopping_rounds)
            fit_kw["eval_set"] = [(X_val, y_val)]
            fit_kw["verbose"] = False
        else:
            self.model.set_params(early_stopping_rounds=None)
        self.model.fit(X_tr, y_tr, **fit_kw)
        self.trained = True
        self.n_features_ = X_tr.shape[1]
        out: TrainResult = {"train": self.evaluate(X_tr, y_tr)}
        if X_val is not None and y_val is not None:
            out["val"] = self.evaluate(X_val, y_val)
        best_it = getattr(self.model, "best_iteration", None)
        if best_it is not None:
            out["best_iteration"] = int(best_it)
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_ready(X)
        return self.model.predict(X).flatten()

    def feature_importance(self, cols: Optional[Sequence[str]] = None) -> Union[Dict[str, float], np.ndarray]:
        imp = self.model.feature_importances_
        if cols is not None:
            return dict(sorted(zip(cols, imp), key=lambda x: -x[1]))
        return imp
