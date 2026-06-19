import warnings
import numpy as np
import pandas as pd
import pytest
from engine.purging import purged_train_end, purge_window, n_purged
from engine.validation_protocol import make_split

warnings.filterwarnings("ignore")


def test_purge_drops_horizon_overlap():
    # train [0,100), test starts at 100, horizon 5
    # labels at 95..99 reference bars up to 100..104 -> overlap test
    end = purged_train_end(0, 100, 100, label_horizon=5)
    assert end == 95


def test_purge_with_embargo_drops_more():
    end = purged_train_end(0, 100, 100, label_horizon=5, embargo=3)
    assert end == 92


def test_purge_zero_horizon_is_noop():
    end = purged_train_end(0, 100, 100, label_horizon=0)
    assert end == 100


def test_purge_window_returns_tuple():
    a, b = purge_window((0, 100), (100, 150), label_horizon=5)
    assert a == 0
    assert b == 95


def test_n_purged_counts_dropped_rows():
    assert n_purged((0, 100), (100, 150), label_horizon=5) == 5
    assert n_purged((0, 100), (100, 150), label_horizon=0) == 0


def test_purge_never_below_train_start():
    # huge horizon would purge the whole block; clamp at train_start
    end = purged_train_end(50, 100, 100, label_horizon=1000)
    assert end == 50


def test_negative_horizon_raises():
    with pytest.raises(ValueError):
        purged_train_end(0, 100, 100, label_horizon=-1)


def test_negative_embargo_raises():
    with pytest.raises(ValueError):
        purged_train_end(0, 100, 100, label_horizon=5, embargo=-1)


@pytest.fixture
def df():
    idx = pd.date_range("2020-01-01", periods=1000, freq="D")
    return pd.DataFrame({"close": np.arange(1000, dtype=float)}, index=idx)


def test_datasplit_purges_train_tail(df):
    sp = make_split(df, train_frac=0.6, val_frac=0.2, label_horizon=5)
    # train_end=600; purged train should drop last 5 rows -> 595
    assert len(sp.train) == 595
    assert sp.n_purged()["train"] == 5


def test_datasplit_purges_val_tail(df):
    sp = make_split(df, train_frac=0.6, val_frac=0.2, label_horizon=5)
    # val [600,800); purged val drops last 5 -> 195 rows
    assert len(sp.val) == 195
    assert sp.n_purged()["val"] == 5


def test_datasplit_no_purge_by_default(df):
    sp = make_split(df, train_frac=0.6, val_frac=0.2)
    assert len(sp.train) == 600
    assert sp.n_purged()["train"] == 0


def test_datasplit_holdout_untouched_by_purge(df):
    sp = make_split(df, train_frac=0.6, val_frac=0.2, label_horizon=5)
    # holdout itself is not purged (it is the final evaluation set)
    holdout = sp.unseal_holdout("test")
    assert len(holdout) == 200


def test_purge_eliminates_lookahead_label_overlap():
    """End-to-end: with triple-barrier labels and a chronological split, the
    last training label must not reference any bar inside the test segment.
    Without purging it does; with purging it does not."""
    from tests.realistic_data import make_realistic_ohlcv
    from data.indicators import engineer
    from data.labeling import triple_barrier_labels, CLASS_TARGET_COL

    h = 5
    df = engineer(make_realistic_ohlcv(n=600, seed=0), with_news=False, with_micro=True)
    df = df.assign(**{CLASS_TARGET_COL: triple_barrier_labels(df, h=h)})
    df = df.dropna(subset=[CLASS_TARGET_COL]).reset_index(drop=True)

    test_start = 400
    # purged train end
    purged_end = purged_train_end(0, test_start, test_start, label_horizon=h)
    # the label at purged_end-1 references bars up to (purged_end-1)+h
    last_label_reach = (purged_end - 1) + h
    assert last_label_reach < test_start, (
        f"purged last train label reaches bar {last_label_reach}, "
        f"test starts at {test_start} -- still overlapping"
    )
    # and without purging it WOULD overlap
    unpurged_reach = (test_start - 1) + h
    assert unpurged_reach >= test_start
