import warnings
import numpy as np
import pytest
from tests.realistic_data import make_realistic_ohlcv
from data.indicators import engineer
from engine.trainer import split
from models.linear_model import LinearModel
from models.xgb_model import XGBModel
from models.lstm_model import LSTMModel

warnings.filterwarnings("ignore")


def _prep(seed=0, n=900):
    df = engineer(make_realistic_ohlcv(n=n, seed=seed), with_news=False, with_micro=True)
    sp = split(df, with_news=False, with_micro=True)
    return sp


@pytest.fixture(scope="module")
def split_data():
    return _prep(seed=0)


def test_target_has_meaningful_variance(split_data):
    sp = split_data
    assert sp["y_test"].std() > 1e-4
    assert np.isfinite(sp["y_test"]).all()


def test_linear_out_of_sample_r2_in_plausible_range(split_data):
    sp = split_data
    m = LinearModel()
    m.train(sp["X_tr"], sp["y_tr"], sp["X_val"], sp["y_val"])
    r2 = m.evaluate(sp["X_test"], sp["y_test"])["r2"]
    assert -2.0 < r2 < 0.30, f"Linear test R2={r2:.4f} outside plausible band for near-random returns"


def test_xgboost_overfit_gap_is_small(split_data):
    """A small train-vs-test R2 gap is the success criterion for
    regularization: the model should fit training data only marginally
    better than test data, not dramatically (which would indicate it
    memorized noise instead of learning generalizable structure).
    """
    sp = split_data
    m = XGBModel()
    m.train(sp["X_tr"], sp["y_tr"], sp["X_val"], sp["y_val"])
    tr = m.evaluate(sp["X_tr"], sp["y_tr"])["r2"]
    te = m.evaluate(sp["X_test"], sp["y_test"])["r2"]
    gap = tr - te
    assert gap < 0.3, f"train R2 ({tr:.4f}) - test R2 ({te:.4f}) = {gap:.4f} too large, overfitting"


def test_xgboost_train_r2_not_overfit(split_data):
    """With shallow trees, min_child_weight, L1/L2 penalties, and early
    stopping, XGBoost should not memorize training noise: train R2 should
    stay low and close to the near-zero test R2, not near 1.0.
    """
    sp = split_data
    m = XGBModel()
    res = m.train(sp["X_tr"], sp["y_tr"], sp["X_val"], sp["y_val"])
    tr = res["train"]["r2"]
    assert m.trained is True
    assert -0.5 < tr < 0.3, f"train R2={tr:.4f} suggests under- or over-fitting"


def test_model_not_worse_than_naive_zero_baseline(split_data):
    sp = split_data
    y = sp["y_test"]
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    naive_pred = np.zeros_like(y)
    ss_res_naive = float(np.sum((y - naive_pred) ** 2))
    naive_r2 = 1 - ss_res_naive / ss_tot

    m = LinearModel()
    m.train(sp["X_tr"], sp["y_tr"], sp["X_val"], sp["y_val"])
    model_r2 = m.evaluate(sp["X_test"], sp["y_test"])["r2"]

    assert model_r2 > naive_r2 - 0.5, (
        f"Linear model R2={model_r2:.4f} catastrophically below "
        f"zero-prediction baseline R2={naive_r2:.4f}"
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_r2_stable_across_seeds(seed):
    sp = _prep(seed=seed)
    m = LinearModel()
    m.train(sp["X_tr"], sp["y_tr"], sp["X_val"], sp["y_val"])
    r2 = m.evaluate(sp["X_test"], sp["y_test"])["r2"]
    assert -3.0 < r2 < 0.5, f"seed={seed} test R2={r2:.4f} wildly out of range"


def test_directional_accuracy_near_coinflip(split_data):
    sp = split_data
    m = LinearModel()
    m.train(sp["X_tr"], sp["y_tr"], sp["X_val"], sp["y_val"])
    preds = m.predict(sp["X_test"])
    dir_acc = float(np.mean(np.sign(preds) == np.sign(sp["y_test"])))
    assert 0.30 < dir_acc < 0.70, (
        f"Directional accuracy={dir_acc:.3f} implausibly far from coin-flip "
        f"for near-random returns (likely look-ahead leak if very high)"
    )


def test_lstm_early_stopping_limits_overfit(split_data):
    """LSTM early stopping (restoring the best-val-loss checkpoint) should
    keep train R2 from running away to near 1.0 the way an unstopped, fully
    trained network would on noisy returns.
    """
    sp = split_data
    m = LSTMModel()
    res = m.train(sp["X_tr"], sp["y_tr"], sp["X_val"], sp["y_val"])
    assert "best_epoch" in res
    assert len(res["history"]["train_loss"]) <= m.p["epochs"]
    tr = res["train"]["r2"]
    assert tr < 0.5, f"train R2={tr:.4f} suggests early stopping did not limit overfitting"
