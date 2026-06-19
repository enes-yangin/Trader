import numpy as np
import pytest
from models.linear_model import LinearModel
from models.xgb_model import XGBModel
from utils.exceptions import ModelNotTrainedError, FeatureMismatchError


@pytest.mark.parametrize("cls", [LinearModel, XGBModel])
def test_predict_before_train_raises(cls):
    mdl = cls()
    X = np.random.randn(10, 5)
    with pytest.raises(ModelNotTrainedError):
        mdl.predict(X)


@pytest.mark.parametrize("cls", [LinearModel, XGBModel])
def test_feature_mismatch_after_train_raises(cls):
    mdl = cls()
    X_tr = np.random.randn(100, 5)
    y_tr = np.random.randn(100)
    mdl.train(X_tr, y_tr)

    X_wrong = np.random.randn(10, 3)
    with pytest.raises(FeatureMismatchError):
        mdl.predict(X_wrong)


@pytest.mark.parametrize("cls", [LinearModel, XGBModel])
def test_predict_after_train_succeeds(cls):
    mdl = cls()
    X_tr = np.random.randn(100, 5)
    y_tr = np.random.randn(100)
    mdl.train(X_tr, y_tr)

    preds = mdl.predict(np.random.randn(10, 5))
    assert len(preds) == 10
    assert mdl.n_features_ == 5
    assert mdl.trained is True


def test_signal_thresholds_produce_expected_labels():
    mdl = LinearModel()
    buy = mdl.signal(0.05, buy_th=0.02, sell_th=-0.02)
    sell = mdl.signal(-0.05, buy_th=0.02, sell_th=-0.02)
    hold = mdl.signal(0.0, buy_th=0.02, sell_th=-0.02)

    assert buy["signal"] == "BUY"
    assert sell["signal"] == "SELL"
    assert hold["signal"] == "HOLD"
    assert 0 <= buy["confidence"] <= 100
    assert 0 <= sell["confidence"] <= 100
    assert 0 <= hold["confidence"] <= 100
