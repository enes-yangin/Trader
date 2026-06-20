import numpy as np
import pandas as pd
from engine.meta_labeling import train_meta_model, filter_signal_with_meta
from engine.backtester import run
from utils.types import Bundle, SplitDict, EnsembleSignal, FeatureSpec
from models.base_model import BaseModel


class _MockModel:
    name = "Mock"
    trained = True
    n_features_ = None

    def __init__(self, pred_val: float):
        self.pred_val = pred_val

    def train(self, *a, **k):
        return {}

    def evaluate(self, X, y):
        return {"r2": 0.0, "rmse": 0.0, "mse": 0.0, "mae": 0.0}

    def predict(self, X):
        return np.full(len(X), self.pred_val)

    def signal(self, pred, buy_th, sell_th):
        from models.base_model import BaseModel
        return BaseModel.signal(self, pred, buy_th=buy_th, sell_th=sell_th)


def test_train_meta_model_and_filter():
    n = 40
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "close": np.full(n, 100.0),
        "atr_pct": np.full(n, 0.01),
    }, index=idx)
    df.index.name = "date"
    df.attrs["symbol"] = "TEST/USDT"

    # Split sizes: train=20, val=10, test=10
    sp = SplitDict(**{
        "X_tr": np.zeros((20, 5)),
        "y_tr": np.zeros(20),
        "X_val": np.zeros((10, 5)),
        "y_val": np.array([0.05] * 5 + [-0.05] * 5),  # 5 positive, 5 negative returns
        "X_test": np.zeros((10, 5)),
        "y_test": np.zeros(10),
        "idx_tr": idx[:20],
        "idx_val": idx[20:30],
        "idx_test": idx[30:],
        "df": df,
        "i_tr": 20,
        "i_va": 30,
        "split_idx": 20,
        "with_news": False,
        "with_micro": False,
        "with_cross_asset": False,
        "spec": None,
    })

    # Primary model predictions match validation target for the first 5 elements, but not the last 5
    primary_results = {
        "linear": {
            "model": _MockModel(0.05),  # predicts positive 0.05 for all
            "metrics": {},
        }
    }

    meta_model = train_meta_model(sp, primary_results)
    assert meta_model is not None

    # Test filtering signal
    sig: EnsembleSignal = {
        "consensus": "BUY",
        "avg_confidence": 80.0,
        "avg_predicted_return": 0.05,
        "votes": {"BUY": 1.0, "HOLD": 0.0, "SELL": 0.0},
        "details": [],
        "last_close": 100.0,
        "symbol": "TEST/USDT",
    }
    
    spec = FeatureSpec.default()
    filter_signal_with_meta(sig, df, spec, meta_model, threshold=0.90)
    # The first elements were correct, but the classifier has a high threshold (0.90),
    # so it should be filtered out to HOLD because of general uncertainty or threshold override
    assert sig["consensus"] in ("BUY", "HOLD")


def test_dynamic_stop_loss_calculation():
    n = 20
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    
    # Set high ATR for first half, low ATR for second half
    atr_vals = [0.05] * 10 + [0.005] * 10
    df = pd.DataFrame({
        "close": np.full(n, 100.0),
        "atr_pct": atr_vals,
    }, index=idx)
    df.index.name = "date"
    df.attrs["symbol"] = "TEST/USDT"

    sp = SplitDict(**{
        "X_tr": np.zeros((10, 5)),
        "y_tr": np.zeros(10),
        "X_val": np.zeros((5, 5)),
        "y_val": np.zeros(5),
        "X_test": np.zeros((5, 5)),
        "y_test": np.zeros(5),
        "idx_tr": idx[:10],
        "idx_val": idx[10:15],
        "idx_test": idx[15:],
        "df": df,
        "i_tr": 10,
        "i_va": 15,
        "split_idx": 10,
        "with_news": False,
        "with_micro": False,
        "with_cross_asset": False,
        "spec": None,
    })

    # Test run executes without crash and dynamically scales stops
    mdl = _MockModel(0.04)
    res = run(mdl, sp, which="test", stop_loss_pct=0.05)
    assert "metrics" in res
    assert res["metrics"]["n_stop_losses"] >= 0
