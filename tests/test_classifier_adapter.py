"""Plan item I: ClassifierSignalAdapter wraps a BaseClassifier as a BaseModel
so engine.backtester's regression-oriented scoring works unmodified with
triple-barrier classifiers."""
import numpy as np
import pytest

from data.labeling import LABEL_BUY, LABEL_HOLD, LABEL_SELL
from engine.classifier_adapter import ClassifierSignalAdapter
from utils.config import FEATURES


class _FakeClassifier:
    """Returns a fixed, caller-supplied label sequence -- no real training.
    `confidence` is the winning-class probability predict_proba reports for each
    label (default 1.0 so directional calls clear any min_proba gate)."""

    def __init__(self, labels, confidence=1.0):
        self.name = "fake_clf"
        self.trained = True
        self.n_features_ = len(FEATURES.core_feature_cols)
        self._labels = np.asarray(labels)
        self._confidence = confidence

    def predict(self, X):
        return self._labels[: len(X)]

    def predict_proba(self, X):
        # Aligned [SELL, HOLD, BUY]; confidence on the predicted label, the rest
        # split evenly across the other two classes.
        labels = self._labels[: len(X)]
        out = np.zeros((len(labels), 3))
        rest = (1.0 - self._confidence) / 2.0
        for i, lbl in enumerate(labels):
            out[i, :] = rest
            out[i, int(lbl)] = self._confidence
        return out


def _X(atr_values):
    """Build a core-feature-shaped matrix with the given atr_pct column."""
    n = len(atr_values)
    idx = list(FEATURES.core_feature_cols).index("atr_pct")
    X = np.zeros((n, len(FEATURES.core_feature_cols)))
    X[:, idx] = atr_values
    return X


def test_buy_label_produces_positive_pseudo_return():
    # atr=0.25, pt_mult=1.5 -> move=0.375, comfortably above the 20% profit
    # target (user request) as well as the plain cost floor.
    clf = _FakeClassifier([LABEL_BUY])
    adapter = ClassifierSignalAdapter(clf, pt_mult=1.5, sl_mult=1.0, atr_threshold=float("inf"))
    out = adapter.predict(_X([0.25]))
    assert out[0] == pytest.approx(1.5 * 0.25)


def test_sell_label_produces_negative_pseudo_return():
    clf = _FakeClassifier([LABEL_SELL])
    adapter = ClassifierSignalAdapter(clf, pt_mult=1.5, sl_mult=1.0, atr_threshold=float("inf"))
    out = adapter.predict(_X([0.25]))
    assert out[0] == pytest.approx(-1.0 * 0.25)


def test_hold_label_is_zero():
    clf = _FakeClassifier([LABEL_HOLD])
    adapter = ClassifierSignalAdapter(clf, atr_threshold=float("inf"))
    out = adapter.predict(_X([0.25]))
    assert out[0] == 0.0


def test_low_confidence_directional_is_forced_to_hold():
    # A BUY the model is only 0.40 sure of is below the 0.45 gate -> HOLD (0.0),
    # even though the ATR-based move would otherwise clear costs.
    clf = _FakeClassifier([LABEL_BUY], confidence=0.40)
    adapter = ClassifierSignalAdapter(clf, pt_mult=1.5, sl_mult=1.0,
                                      atr_threshold=float("inf"), min_proba=0.45)
    out = adapter.predict(_X([0.25]))
    assert out[0] == 0.0


def test_high_confidence_directional_fires():
    clf = _FakeClassifier([LABEL_BUY], confidence=0.60)
    adapter = ClassifierSignalAdapter(clf, pt_mult=1.5, sl_mult=1.0,
                                      atr_threshold=float("inf"), min_proba=0.45)
    out = adapter.predict(_X([0.25]))
    assert out[0] == pytest.approx(1.5 * 0.25)


def test_buy_below_cost_floor_is_forced_to_zero():
    # F: a tiny ATR means pt_mult*atr_pct can't clear round-trip costs.
    clf = _FakeClassifier([LABEL_BUY])
    adapter = ClassifierSignalAdapter(clf, pt_mult=1.5, sl_mult=1.0, atr_threshold=float("inf"))
    out = adapter.predict(_X([0.0001]))
    assert out[0] == 0.0


def test_buy_below_profit_target_is_forced_to_zero():
    # User request: only trades expected to yield >5% pass. atr=0.025,
    # pt_mult=1.5 -> move=0.0375 (3.75%) clears round-trip costs comfortably
    # but not the 5% profit target -- must still be rejected.
    clf = _FakeClassifier([LABEL_BUY])
    adapter = ClassifierSignalAdapter(clf, pt_mult=1.5, sl_mult=1.0, atr_threshold=float("inf"))
    out = adapter.predict(_X([0.025]))
    assert out[0] == 0.0


def test_buy_in_blocked_vol_regime_is_forced_to_zero():
    from utils.config import RISK
    assert RISK.use_vol_filter, "test assumes the G default (use_vol_filter=True)"
    clf = _FakeClassifier([LABEL_BUY])
    # atr=0.25 clears both the cost floor and the 20% profit target (F), so
    # only the frozen vol-regime threshold (G) is responsible for the block.
    adapter = ClassifierSignalAdapter(clf, pt_mult=1.5, sl_mult=1.0, atr_threshold=0.01)
    out = adapter.predict(_X([0.25]))
    assert out[0] == 0.0


def test_mixed_batch_maps_each_row_independently():
    clf = _FakeClassifier([LABEL_BUY, LABEL_HOLD, LABEL_SELL])
    adapter = ClassifierSignalAdapter(clf, pt_mult=1.5, sl_mult=1.0, atr_threshold=float("inf"))
    out = adapter.predict(_X([0.25, 0.25, 0.25]))
    assert out[0] > 0.0
    assert out[1] == 0.0
    assert out[2] < 0.0


def test_horizon_scales_expected_move_by_sqrt_horizon():
    # atr=0.03 (typical BTC daily ATR%), pt_mult=1.5 -> 1-day move = 4.5%,
    # far below the 20% target. Over a 20-day horizon the sqrt-time-scaled
    # move is 4.5% * sqrt(20) ~= 20.1%, just clearing it -- this is the
    # mechanism that makes the 20% profit target achievable at all once the
    # horizon is extended from 1 day to 20.
    clf = _FakeClassifier([LABEL_BUY])
    adapter_1d = ClassifierSignalAdapter(clf, pt_mult=1.5, atr_threshold=float("inf"), horizon=1)
    adapter_20d = ClassifierSignalAdapter(clf, pt_mult=1.5, atr_threshold=float("inf"), horizon=20)
    out_1d = adapter_1d.predict(_X([0.03]))
    out_20d = adapter_20d.predict(_X([0.03]))
    assert out_1d[0] == 0.0, "1-day move (4.5%) must not clear the 20% target"
    assert out_20d[0] == pytest.approx(1.5 * 0.03 * np.sqrt(20))
    assert out_20d[0] > 0.20, "20-day sqrt-scaled move should clear the 20% target here"


def test_horizon_rejects_less_than_one():
    clf = _FakeClassifier([LABEL_HOLD])
    with pytest.raises(ValueError):
        ClassifierSignalAdapter(clf, horizon=0)


def test_train_raises_not_implemented():
    clf = _FakeClassifier([LABEL_HOLD])
    adapter = ClassifierSignalAdapter(clf)
    with pytest.raises(NotImplementedError):
        adapter.train(np.zeros((1, 1)), np.zeros(1))
