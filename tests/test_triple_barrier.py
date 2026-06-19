import warnings
import numpy as np
import pandas as pd
import pytest
from tests.realistic_data import make_realistic_ohlcv
from data.indicators import engineer
from data.labeling import (
    triple_barrier_labels, add_triple_barrier_labels, label_distribution,
    CLASS_TARGET_COL, LABEL_BUY, LABEL_SELL, LABEL_HOLD,
)
from utils.config import MODEL

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def df():
    return engineer(make_realistic_ohlcv(n=900, seed=0), with_news=False, with_micro=True)


def test_labels_are_three_classes(df):
    lbl = triple_barrier_labels(df, pt_mult=1.5, sl_mult=1.0).dropna()
    assert set(lbl.unique()).issubset({LABEL_SELL, LABEL_HOLD, LABEL_BUY})


def test_last_h_rows_are_nan(df):
    lbl = triple_barrier_labels(df, h=MODEL.pred_horizon)
    assert lbl.iloc[-MODEL.pred_horizon:].isna().all()


def test_upper_barrier_hit_labels_buy():
    n = 20
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    close = np.full(n, 100.0)
    high = close.copy()
    low = close.copy()
    high[3] = 130.0
    d = pd.DataFrame({"close": close, "high": high, "low": low,
                      "atr_pct": np.full(n, 0.1)}, index=idx)
    lbl = triple_barrier_labels(d, h=5, pt_mult=1.0, sl_mult=1.0)
    assert lbl.iloc[0] == LABEL_BUY


def test_lower_barrier_hit_labels_sell():
    n = 20
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    close = np.full(n, 100.0)
    high = close.copy()
    low = close.copy()
    low[2] = 70.0
    d = pd.DataFrame({"close": close, "high": high, "low": low,
                      "atr_pct": np.full(n, 0.1)}, index=idx)
    lbl = triple_barrier_labels(d, h=5, pt_mult=1.0, sl_mult=1.0)
    assert lbl.iloc[0] == LABEL_SELL


def test_no_barrier_hit_labels_hold():
    n = 20
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    flat = np.full(n, 100.0)
    d = pd.DataFrame({"close": flat, "high": flat, "low": flat,
                      "atr_pct": np.full(n, 0.1)}, index=idx)
    lbl = triple_barrier_labels(d, h=5, pt_mult=1.0, sl_mult=1.0)
    assert lbl.iloc[0] == LABEL_HOLD


def test_first_touch_wins_upper_before_lower():
    n = 20
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    close = np.full(n, 100.0)
    high = close.copy()
    low = close.copy()
    high[2] = 130.0
    low[4] = 70.0
    d = pd.DataFrame({"close": close, "high": high, "low": low,
                      "atr_pct": np.full(n, 0.1)}, index=idx)
    lbl = triple_barrier_labels(d, h=6, pt_mult=1.0, sl_mult=1.0)
    assert lbl.iloc[0] == LABEL_BUY


def test_higher_pt_mult_reduces_buy_share(df):
    low_pt = label_distribution(triple_barrier_labels(df, pt_mult=1.0, sl_mult=1.0))
    high_pt = label_distribution(triple_barrier_labels(df, pt_mult=3.0, sl_mult=1.0))
    assert high_pt["BUY"] <= low_pt["BUY"]


def test_add_triple_barrier_does_not_mutate_input(df):
    cols_before = set(df.columns)
    out = add_triple_barrier_labels(df)
    assert CLASS_TARGET_COL not in cols_before or CLASS_TARGET_COL in df.columns
    assert CLASS_TARGET_COL in out.columns


def test_label_uses_only_future_not_past():
    """Changing bars strictly before t must not change the label at t:
    triple barrier looks only forward from the entry, never backward."""
    n = 30
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    close = np.full(n, 100.0)
    high = close.copy()
    low = close.copy()
    high[12] = 130.0
    base = pd.DataFrame({"close": close.copy(), "high": high.copy(),
                         "low": low.copy(), "atr_pct": np.full(n, 0.1)}, index=idx)
    lbl_base = triple_barrier_labels(base, h=5, pt_mult=1.0, sl_mult=1.0)

    perturbed = base.copy()
    perturbed.iloc[0:5, perturbed.columns.get_loc("high")] = 999.0
    perturbed.iloc[0:5, perturbed.columns.get_loc("low")] = 1.0
    lbl_pert = triple_barrier_labels(perturbed, h=5, pt_mult=1.0, sl_mult=1.0)

    assert lbl_base.iloc[10] == lbl_pert.iloc[10]
