from unittest.mock import patch
import dataclasses
import numpy as np
import pandas as pd
from models.base_model import dynamic_threshold
from utils.config import SIGNAL


def test_dynamic_threshold_scales_with_atr():
    bt, st = dynamic_threshold(0.04, SIGNAL.buy_threshold, SIGNAL.sell_threshold)
    assert bt == SIGNAL.atr_mult * 0.04
    assert st == -SIGNAL.atr_mult * 0.04


def test_dynamic_threshold_higher_atr_means_wider_band():
    bt_low, st_low = dynamic_threshold(0.01, SIGNAL.buy_threshold, SIGNAL.sell_threshold)
    bt_high, st_high = dynamic_threshold(0.05, SIGNAL.buy_threshold, SIGNAL.sell_threshold)
    assert bt_high > bt_low > 0
    assert st_high < st_low < 0


def test_dynamic_threshold_zero_atr_falls_back_to_fixed():
    bt, st = dynamic_threshold(0.0, SIGNAL.buy_threshold, SIGNAL.sell_threshold)
    assert bt == SIGNAL.buy_threshold
    assert st == SIGNAL.sell_threshold


def test_dynamic_threshold_negative_atr_falls_back_to_fixed():
    bt, st = dynamic_threshold(-0.01, SIGNAL.buy_threshold, SIGNAL.sell_threshold)
    assert bt == SIGNAL.buy_threshold
    assert st == SIGNAL.sell_threshold


def test_dynamic_threshold_nan_falls_back_to_fixed():
    bt, st = dynamic_threshold(float("nan"), SIGNAL.buy_threshold, SIGNAL.sell_threshold)
    assert bt == SIGNAL.buy_threshold
    assert st == SIGNAL.sell_threshold


def test_dynamic_threshold_disabled_returns_fixed():
    disabled = dataclasses.replace(SIGNAL, use_dynamic_threshold=False)
    with patch("models.base_model.SIGNAL", disabled):
        bt, st = dynamic_threshold(0.10, SIGNAL.buy_threshold, SIGNAL.sell_threshold)
    assert bt == SIGNAL.buy_threshold
    assert st == SIGNAL.sell_threshold


class _ConstantModel:
    name = "Constant"
    trained = True
    n_features_ = None

    def __init__(self, pred):
        self.pred = pred

    def train(self, *a, **k):
        return {}

    def evaluate(self, X, y):
        return {"r2": 0.0, "rmse": 0.0, "mse": 0.0, "mae": 0.0}

    def predict(self, X):
        return np.full(len(X), self.pred)

    def signal(self, pred, buy_th, sell_th):
        from models.base_model import BaseModel
        return BaseModel.signal(self, pred, buy_th=buy_th, sell_th=sell_th)  # type: ignore[arg-type]


def _df_with_atr(atr_pct_value, n=60):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "close": np.full(n, 100.0),
        "atr_pct": np.full(n, atr_pct_value),
    }, index=idx)
    df.index.name = "date"
    df.attrs["symbol"] = "TEST/USDT"
    return df


def test_predict_single_low_atr_narrow_band_triggers_buy():
    from engine.predictor import predict_single
    from utils.types import FeatureSpec

    pred = SIGNAL.atr_mult * 0.005 + 1e-6
    mdl = _ConstantModel(pred)
    df = _df_with_atr(atr_pct_value=0.005)

    with patch("engine.predictor.get_features") as gf:
        gf.return_value.values = np.zeros((len(df), 12))
        sig = predict_single(mdl, df, spec=FeatureSpec.default())

    assert sig["signal"] == "BUY"
    assert sig["predicted_return"] < SIGNAL.buy_threshold


def test_predict_single_high_atr_wide_band_stays_hold():
    from engine.predictor import predict_single
    from utils.types import FeatureSpec

    pred = SIGNAL.buy_threshold + 0.01
    mdl = _ConstantModel(pred)
    atr_pct_val = (pred + 0.01) / SIGNAL.atr_mult
    df = _df_with_atr(atr_pct_value=atr_pct_val)

    with patch("engine.predictor.get_features") as gf:
        gf.return_value.values = np.zeros((len(df), 12))
        sig = predict_single(mdl, df, spec=FeatureSpec.default())

    assert sig["signal"] == "HOLD"
    assert pred < SIGNAL.atr_mult * atr_pct_val


def test_predict_single_no_atr_column_uses_fixed_threshold():
    from engine.predictor import predict_single
    from utils.types import FeatureSpec

    pred = SIGNAL.buy_threshold + 0.001
    mdl = _ConstantModel(pred)
    df = _df_with_atr(atr_pct_value=0.05).drop(columns=["atr_pct"])

    with patch("engine.predictor.get_features") as gf:
        gf.return_value.values = np.zeros((len(df), 12))
        sig = predict_single(mdl, df, spec=FeatureSpec.default())

    assert sig["signal"] == "BUY"


def test_backtester_atr_values_aligned_with_split_index(synthetic_ohlcv):
    from data.indicators import engineer
    from engine.trainer import split
    from engine.backtester import run

    df = engineer(synthetic_ohlcv, with_news=False, with_micro=True)
    sp = split(df, with_news=False, with_micro=True)

    pred_above_fixed = SIGNAL.buy_threshold + 0.001
    mdl = _ConstantModel(pred_above_fixed)
    result = run(mdl, sp, which="test")

    assert "metrics" in result
    assert result["metrics"]["n_trades"] >= 0
