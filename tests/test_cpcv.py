import warnings
import math
import numpy as np
import pytest

warnings.filterwarnings("ignore")

from tests.realistic_data import make_realistic_ohlcv
from data.indicators import engineer
from models.linear_model import LinearModel
from engine.cpcv import (
    group_bounds,
    n_paths,
    cpcv_combinations,
    train_indices,
    run_cpcv,
    CPCVResult,
    PathMetrics,
)


@pytest.fixture(scope="module")
def eng_df():
    return engineer(make_realistic_ohlcv(n=900, seed=0), with_news=False,
                    with_micro=True)


# ------------------------------------------------------------------ #
# Geometry                                                            #
# ------------------------------------------------------------------ #

def test_group_bounds_partition_is_contiguous_and_complete():
    bounds = group_bounds(100, 5)
    assert bounds[0][0] == 0
    assert bounds[-1][1] == 100
    for i in range(len(bounds) - 1):
        assert bounds[i][1] == bounds[i + 1][0]  # no gaps/overlaps


def test_group_bounds_rejects_too_few_groups():
    with pytest.raises(ValueError):
        group_bounds(100, 1)


def test_n_paths_formula():
    # Lopez de Prado worked example: N=6, k=2 -> phi = C(5,1) = 5.
    assert n_paths(6, 2) == 5
    assert n_paths(10, 2) == 9
    assert n_paths(6, 3) == math.comb(5, 2)


def test_cpcv_combinations_count():
    combos = cpcv_combinations(6, 2)
    assert len(combos) == math.comb(6, 2)
    assert all(len(c) == 2 for c in combos)


def test_each_group_tested_phi_times():
    # The key identity underpinning path reconstruction: every group appears in
    # exactly phi = C(N-1, k-1) combinations.
    n_groups, k = 6, 2
    combos = cpcv_combinations(n_groups, k)
    counts = {g: 0 for g in range(n_groups)}
    for c in combos:
        for g in c:
            counts[g] += 1
    assert set(counts.values()) == {n_paths(n_groups, k)}


# ------------------------------------------------------------------ #
# Purging / embargo                                                   #
# ------------------------------------------------------------------ #

def test_train_indices_exclude_test_bars():
    n = 100
    tr = train_indices(n, [(40, 60)], label_horizon=0)
    assert not any(40 <= t < 60 for t in tr)


def test_train_indices_purge_left_label_overlap():
    # With horizon h, a train bar t in [start-h, start) has label reaching into
    # the test group and must be dropped.
    n = 100
    h = 5
    tr = train_indices(n, [(40, 60)], label_horizon=h)
    # bars 35..39 reach into [40,60) via their h-step label -> purged
    assert not any(35 <= t < 40 for t in tr)
    # bar 34's label [34,39] does not reach 40 -> kept
    assert 34 in tr


def test_train_indices_embargo_right():
    n = 100
    tr = train_indices(n, [(40, 60)], label_horizon=0, embargo=3)
    # bars 60,61,62 embargoed
    assert not any(60 <= t < 63 for t in tr)
    assert 63 in tr


def test_train_indices_interior_test_purges_both_sides():
    # An interior test group purges left (label overlap) and embargoes right.
    n = 120
    tr = set(train_indices(n, [(50, 70)], label_horizon=4, embargo=2).tolist())
    assert all(t not in tr for t in range(46, 70))   # purge-left + test
    assert all(t not in tr for t in range(70, 72))   # embargo-right
    assert 45 in tr and 72 in tr


def test_train_indices_no_lookahead_against_horizon():
    # End-to-end leakage check: for every retained train bar, its label window
    # [t, t+h] must not intersect any test interval.
    n = 200
    h = 5
    test_intervals = [(40, 60), (120, 140)]
    tr = train_indices(n, test_intervals, label_horizon=h, embargo=0)
    test_bars = set()
    for s, e in test_intervals:
        test_bars.update(range(s, e))
    for t in tr:
        label_window = set(range(t, min(n, t + h + 1)))
        assert not (label_window & test_bars), (
            f"train bar {t} label window overlaps a test interval"
        )


# ------------------------------------------------------------------ #
# run_cpcv                                                            #
# ------------------------------------------------------------------ #

def test_run_cpcv_produces_phi_paths(eng_df):
    res = run_cpcv(eng_df, LinearModel, n_groups=6, k_test=2, purge=5)
    assert isinstance(res, CPCVResult)
    assert res.n_paths == n_paths(6, 2)
    assert res.n_combinations == math.comb(6, 2)


def test_run_cpcv_path_metrics_in_range(eng_df):
    res = run_cpcv(eng_df, LinearModel, n_groups=6, k_test=2, purge=5)
    for p in res.paths:
        assert isinstance(p, PathMetrics)
        assert 0.0 <= p.dir_acc <= 1.0
        assert p.n > 0


def test_run_cpcv_summary_has_distribution(eng_df):
    res = run_cpcv(eng_df, LinearModel, n_groups=6, k_test=2, purge=5)
    s = res.summary()
    assert s["n_paths"] == n_paths(6, 2)
    assert "std_dir_acc" in s
    assert "pct_sharpe_positive" in s
    # A distribution, not a point estimate: more than one path.
    assert s["n_paths"] > 1


def test_run_cpcv_k3_more_paths_than_k2(eng_df):
    r2 = run_cpcv(eng_df, LinearModel, n_groups=6, k_test=2, purge=5)
    r3 = run_cpcv(eng_df, LinearModel, n_groups=6, k_test=3, purge=5)
    assert r3.n_paths > r2.n_paths  # C(5,2)=10 > C(5,1)=5


def test_run_cpcv_raises_on_raw_ohlcv():
    raw = make_realistic_ohlcv(n=300, seed=1)
    from utils.exceptions import InsufficientDataError
    with pytest.raises(InsufficientDataError):
        run_cpcv(raw, LinearModel, n_groups=4, k_test=2)


def test_run_cpcv_raises_when_train_too_small(eng_df):
    from utils.exceptions import InsufficientDataError
    # Huge min_train can never be satisfied -> informative error.
    with pytest.raises(InsufficientDataError):
        run_cpcv(eng_df, LinearModel, n_groups=6, k_test=2, min_train=100000)


def test_run_cpcv_report_formats(eng_df):
    res = run_cpcv(eng_df, LinearModel, n_groups=6, k_test=2, purge=5)
    report = res.format_report("linear")
    assert "paths" in report
    assert "Sharpe" in report
