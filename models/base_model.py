from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class BaseModel(ABC):
    def __init__(self, name="base"):
        self.name = name
        self.model = None
        self.trained = False

    @abstractmethod
    def train(self, X_tr, y_tr, X_val=None, y_val=None):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def evaluate(self, X, y):
        preds = self.predict(X)
        mse = mean_squared_error(y, preds)
        mae = mean_absolute_error(y, preds)
        r2 = r2_score(y, preds)
        rmse = np.sqrt(mse)
        return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

    def signal(self, pred, buy_th=0.02, sell_th=-0.02):
        if pred > buy_th:
            conf = min((pred - buy_th) / buy_th * 50, 99)
            sig = "BUY"
        elif pred < sell_th:
            conf = min(abs(pred - sell_th) / abs(sell_th) * 50, 99)
            sig = "SELL"
        else:
            conf = max(0, (1 - abs(pred) / buy_th) * 50)
            sig = "HOLD"
        return {"signal": sig, "confidence": round(float(conf), 1), "predicted_return": round(float(pred), 6)}

    def __repr__(self):
        st = "trained" if self.trained else "untrained"
        return f"<{self.name} [{st}]>"
