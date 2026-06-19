import warnings
import numpy as np
import pytest
from tests.realistic_data import make_realistic_ohlcv
from data.indicators import engineer, get_features
from data.labeling import (
    make_labels, add_labels, label_distribution,
    CLASS_TARGET_COL, LABEL_BUY, LABEL_SELL, LABEL_HOLD,
)
from models.classifier_models import LogisticModel, XGBClassifierModel
from models.base_classifier import BaseClassifier

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def labeled_split():
    df = engineer(make_realistic_ohlcv(n=1200, seed=0), with_news=False, with_micro=True)
    df = add_labels(df, threshold=0.5, atr_normalize=True)
    df = df.dropna(subset=[CLASS_TARGET_COL])
    X = get_features(df, with_micro=True).values
    y = df[CLASS_TARGET_COL].values.astype(int)
    n = len(X)
    i_tr, i_va = int(n * 0.7), int(n * 0.85)
    return {
        "X_tr": X[:i_tr], "y_tr": y[:i_tr],
        "X_val": X[i_tr:i_va], "y_val": y[i_tr:i_va],
        "X_test": X[i_va:], "y_test": y[i_va:],
    }


def test_labels_are_three_classes():
    df = engineer(make_realistic_ohlcv(n=600, seed=1), with_news=False, with_micro=True)
    labels = make_labels(df, threshold=0.5, atr_normalize=True).dropna()
    assert set(labels.unique()).issubset({LABEL_SELL, LABEL_HOLD, LABEL_BUY})


def test_last_h_rows_are_nan():
    df = engineer(make_realistic_ohlcv(n=600, seed=1), with_news=False, with_micro=True)
    from utils.config import MODEL
    labels = make_labels(df)
    assert labels.iloc[-MODEL.pred_horizon:].isna().all()


def test_zero_threshold_has_no_hold():
    df = engineer(make_realistic_ohlcv(n=600, seed=1), with_news=False, with_micro=True)
    dist = label_distribution(make_labels(df, threshold=0.0, atr_normalize=False))
    assert dist["HOLD"] < 0.05


def test_higher_threshold_more_hold():
    df = engineer(make_realistic_ohlcv(n=600, seed=1), with_news=False, with_micro=True)
    low = label_distribution(make_labels(df, threshold=0.3, atr_normalize=True))
    high = label_distribution(make_labels(df, threshold=1.5, atr_normalize=True))
    assert high["HOLD"] > low["HOLD"]


def test_add_labels_does_not_mutate_input():
    df = engineer(make_realistic_ohlcv(n=400, seed=2), with_news=False, with_micro=True)
    cols_before = set(df.columns)
    add_labels(df)
    assert CLASS_TARGET_COL not in cols_before
    assert CLASS_TARGET_COL not in df.columns


@pytest.mark.parametrize("Model", [LogisticModel, XGBClassifierModel])
def test_classifier_trains_and_predicts(labeled_split, Model):
    sp = labeled_split
    m = Model()
    res = m.train(sp["X_tr"], sp["y_tr"], sp["X_val"], sp["y_val"])
    assert m.trained
    assert "train" in res and "val" in res
    preds = m.predict(sp["X_test"])
    assert len(preds) == len(sp["y_test"])
    assert set(np.unique(preds)).issubset({LABEL_SELL, LABEL_HOLD, LABEL_BUY})


@pytest.mark.parametrize("Model", [LogisticModel, XGBClassifierModel])
def test_predict_proba_shape_and_normalization(labeled_split, Model):
    sp = labeled_split
    m = Model()
    m.train(sp["X_tr"], sp["y_tr"], sp["X_val"], sp["y_val"])
    proba = m.predict_proba(sp["X_test"])
    assert proba.shape == (len(sp["X_test"]), 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert (proba >= 0).all()


@pytest.mark.parametrize("Model", [LogisticModel, XGBClassifierModel])
def test_classifier_metrics_in_valid_ranges(labeled_split, Model):
    sp = labeled_split
    m = Model()
    m.train(sp["X_tr"], sp["y_tr"], sp["X_val"], sp["y_val"])
    me = m.evaluate(sp["X_test"], sp["y_test"])
    assert 0.0 <= me["accuracy"] <= 1.0
    assert 0.0 <= me["balanced_accuracy"] <= 1.0
    assert 0.0 <= me["f1_macro"] <= 1.0
    total = me["n_buy_pred"] + me["n_sell_pred"] + me["n_hold_pred"]
    assert total == len(sp["y_test"])


def test_signal_returns_valid_structure(labeled_split):
    sp = labeled_split
    m = LogisticModel()
    m.train(sp["X_tr"], sp["y_tr"], sp["X_val"], sp["y_val"])
    sig = m.signal(sp["X_test"][-1])
    assert sig["signal"] in ("BUY", "HOLD", "SELL")
    assert 0.0 <= sig["confidence"] <= 100.0


def test_min_proba_forces_hold_when_unconfident(labeled_split):
    sp = labeled_split
    m = LogisticModel()
    m.train(sp["X_tr"], sp["y_tr"], sp["X_val"], sp["y_val"])
    sig = m.signal(sp["X_test"][-1], min_proba=0.99)
    assert sig["signal"] == "HOLD"


def test_classifier_directional_hit_rate_plausible(labeled_split):
    """Sanity bound: directional hit rate on BUY/SELL signals should sit in a
    plausible band around coin-flip. A value far above 0.65 on test data would
    signal a look-ahead leak rather than skill."""
    sp = labeled_split
    m = XGBClassifierModel()
    m.train(sp["X_tr"], sp["y_tr"], sp["X_val"], sp["y_val"])
    preds = m.predict(sp["X_test"])
    mask = (preds == LABEL_BUY) | (preds == LABEL_SELL)
    if mask.sum() > 5:
        hit = float(np.mean(preds[mask] == sp["y_test"][mask]))
        assert 0.0 < hit < 0.80, f"hit rate {hit:.3f} implausible"


def test_base_classifier_is_abstract():
    with pytest.raises(TypeError):
        BaseClassifier()  # type: ignore[abstract]
