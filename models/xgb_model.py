import numpy as np
import xgboost as xgb
from models.base_model import BaseModel
from utils.config import XGB_PARAMS


class XGBModel(BaseModel):
    def __init__(self, **kw):
        super().__init__(name="XGBoost")
        params = {**XGB_PARAMS, **kw}
        self.model = xgb.XGBRegressor(
            **params,
            objective="reg:squarederror",
            verbosity=0,
            random_state=42,
        )

    def train(self, X_tr, y_tr, X_val=None, y_val=None):
        fit_kw = {}
        if X_val is not None and y_val is not None:
            fit_kw["eval_set"] = [(X_val, y_val)]
            fit_kw["verbose"] = False
        self.model.fit(X_tr, y_tr, **fit_kw)
        self.trained = True
        out = {"train": self.evaluate(X_tr, y_tr)}
        if X_val is not None and y_val is not None:
            out["val"] = self.evaluate(X_val, y_val)
        return out

    def predict(self, X):
        return self.model.predict(X).flatten()

    def feature_importance(self, cols=None):
        imp = self.model.feature_importances_
        if cols is not None:
            return dict(sorted(zip(cols, imp), key=lambda x: -x[1]))
        return imp
