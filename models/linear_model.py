import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from models.base_model import BaseModel


class LinearModel(BaseModel):
    def __init__(self, alpha=1.0):
        super().__init__(name="LinearRegression")
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)

    def train(self, X_tr, y_tr, X_val=None, y_val=None):
        Xs = self.scaler.fit_transform(X_tr)
        self.model.fit(Xs, y_tr)
        self.trained = True
        out = {"train": self.evaluate(X_tr, y_tr)}
        if X_val is not None and y_val is not None:
            out["val"] = self.evaluate(X_val, y_val)
        return out

    def predict(self, X):
        Xs = self.scaler.transform(X)
        return self.model.predict(Xs).flatten()
