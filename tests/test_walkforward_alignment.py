import numpy as np
from data.indicators import engineer
from engine.walkforward import _aligned_preds, make_windows, run


class _ConstantModel:
    seq_len = 30

    def __init__(self, **kw):
        pass

    def train(self, X, y, *a, **kw):
        return {"train": {"r2": 0, "rmse": 0, "mse": 0, "mae": 0}}

    def predict(self, X):
        return np.zeros(max(0, len(X) - self.seq_len))

    def evaluate(self, X, y):
        return {"r2": 0.0, "rmse": 0.0, "mse": 0.0, "mae": 0.0}


class _PassthroughModel:
    def __init__(self, **kw):
        pass

    def train(self, X, y, *a, **kw):
        return {"train": {"r2": 0, "rmse": 0, "mse": 0, "mae": 0}}

    def predict(self, X):
        return np.zeros(len(X))

    def evaluate(self, X, y):
        return {"r2": 0.0, "rmse": 0.0, "mse": 0.0, "mae": 0.0}


def test_aligned_preds_lstm_offset():
    X = np.random.randn(100, 5)
    y = np.arange(100, dtype=float)
    mdl = _ConstantModel()

    preds, yt = _aligned_preds(mdl, X, y)

    assert len(preds) == len(yt) == 100 - mdl.seq_len
    assert yt[0] == y[mdl.seq_len], (
        "yt must start at index seq_len: LSTM's first prediction "
        "corresponds to the target after the first full sequence"
    )
    assert yt[-1] == y[-1]


def test_aligned_preds_non_sequential_model():
    X = np.random.randn(50, 5)
    y = np.arange(50, dtype=float)
    mdl = _PassthroughModel()

    preds, yt = _aligned_preds(mdl, X, y)

    assert len(preds) == len(yt) == 50
    assert np.array_equal(yt, y)


def test_make_windows_chronological():
    wins = make_windows(1000, train_size=400, test_size=100)
    assert len(wins) > 0
    for w in wins:
        a, b = w["train"]
        c, d = w["test"]
        assert a < b == c < d
        assert b - a == 400
        assert d - c == 100


def test_walkforward_run_no_crash(synthetic_ohlcv):
    df = engineer(synthetic_ohlcv, with_news=False, with_micro=True)
    res = run(df, _PassthroughModel, train_size=400, test_size=100,
              with_news=False, with_micro=True)
    assert res["n_windows"] > 0
    assert "mean_r2" in res["summary"]
    assert len(res["folds"]) == res["n_windows"]


def test_lstm_predict_shorter_than_seqlen_returns_empty():
    from models.lstm_model import LSTMModel
    from data.indicators import engineer
    from engine.trainer import split
    import pandas as pd

    rng = np.random.RandomState(11)
    n = 100
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    p = 30000 + np.cumsum(rng.randn(n) * 200)
    p = np.abs(p) + 1000
    df0 = pd.DataFrame({"open": p, "high": p * 1.01, "low": p * 0.99,
                        "close": p, "volume": 1000 + np.abs(rng.randn(n)) * 500}, index=idx)
    df0.index.name = "date"
    df = engineer(df0, with_news=False, with_micro=True)
    sp = split(df, with_news=False, with_micro=True)

    m = LSTMModel(epochs=2)
    res = m.train(sp["X_tr"], sp["y_tr"], sp["X_val"], sp["y_val"])
    assert "val" in res
    preds = m.predict(sp["X_test"])
    assert isinstance(preds, np.ndarray)


def test_walkforward_raw_df_raises_clear_error():
    import pandas as pd
    from engine.walkforward import run_model
    from utils.exceptions import InsufficientDataError

    idx = pd.date_range("2020-01-01", periods=600, freq="D")
    raw = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0,
                        "close": 1.0, "volume": 1.0}, index=idx)
    try:
        run_model(raw, "linear", train_size=400, test_size=100, with_news=False)
        assert False, "should raise InsufficientDataError on raw df"
    except InsufficientDataError:
        pass
