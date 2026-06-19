import warnings
import numpy as np
import pytest
from tests.realistic_data import make_realistic_ohlcv
from data.indicators import engineer
from engine.classification_walkforward import (
    run_classification, format_classification_report,
)
from utils.types import FeatureSpec

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def engineered():
    return engineer(make_realistic_ohlcv(n=1200, seed=0), with_news=False, with_micro=True)


def test_walkforward_produces_multiple_windows(engineered):
    spec = FeatureSpec.from_bools(False, True, False)
    res = run_classification(engineered, "logistic", spec=spec,
                              train_size=400, test_size=100)
    assert res["n_windows"] >= 3
    assert len(res["folds"]) == res["n_windows"]


def test_walkforward_summary_keys(engineered):
    spec = FeatureSpec.from_bools(False, True, False)
    res = run_classification(engineered, "xgb_clf", spec=spec,
                              train_size=400, test_size=100)
    s = res["summary"]
    for k in ("mean_accuracy", "mean_f1", "mean_hit_rate", "std_hit_rate",
              "pct_windows_above_50", "total_signals"):
        assert k in s


def test_hit_rate_in_valid_range(engineered):
    spec = FeatureSpec.from_bools(False, True, False)
    res = run_classification(engineered, "logistic", spec=spec,
                              train_size=400, test_size=100)
    folds = res["folds"]
    valid = folds["hit_rate"].dropna()
    assert ((valid >= 0) & (valid <= 1)).all()


def test_walkforward_hit_rate_not_implausibly_high(engineered):
    """On near-random synthetic returns, mean walk-forward hit rate must stay
    near coin-flip. A mean above 0.65 across many windows would indicate a
    look-ahead leak entering through the labeling or feature pipeline."""
    spec = FeatureSpec.from_bools(False, True, False)
    res = run_classification(engineered, "xgb_clf", spec=spec,
                              train_size=400, test_size=100)
    mhr = res["summary"]["mean_hit_rate"]
    if not np.isnan(mhr):
        assert mhr < 0.65, f"mean hit rate {mhr:.3f} implausibly high"


def test_insufficient_data_returns_empty():
    spec = FeatureSpec.from_bools(False, True, False)
    df = engineer(make_realistic_ohlcv(n=200, seed=0), with_news=False, with_micro=True)
    res = run_classification(df, "logistic", spec=spec,
                              train_size=400, test_size=100)
    assert res["n_windows"] == 0
    assert res["folds"].empty


def test_format_report_runs(engineered):
    spec = FeatureSpec.from_bools(False, True, False)
    res = run_classification(engineered, "logistic", spec=spec,
                              train_size=400, test_size=100)
    txt = format_classification_report("logistic", res)
    assert "walk-forward" in txt
    assert "hit rate" in txt
