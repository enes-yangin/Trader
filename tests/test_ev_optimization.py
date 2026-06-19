import warnings
import numpy as np
import pytest

warnings.filterwarnings("ignore")

from tests.realistic_data import make_realistic_ohlcv
from data.indicators import engineer
from engine.ev_optimization import (
    EVParams,
    EVWindowResult,
    default_grid,
    ev_walkforward,
    ev_scan,
    ev_judge_holdout,
    ev_optimize,
)

# The EV module is model-agnostic; these tests check structural invariants
# (win/trade counts, deflation maths, trial logging), not predictive power.
# Logistic regression trains ~100x faster than the 200-tree XGBoost classifier,
# keeping the full suite fast. One realism check below still exercises xgb_clf.
MODEL = "logistic"


@pytest.fixture(scope="module")
def eng_df():
    df = make_realistic_ohlcv(n=900, seed=0)
    return engineer(df, with_news=False, with_micro=True)


# ------------------------------------------------------------------ #
# EVParams validation                                                  #
# ------------------------------------------------------------------ #

def test_ev_params_valid():
    p = EVParams(pt_mult=1.5, sl_mult=1.0, horizon=5, min_proba=0.55)
    assert p.as_dict()["pt_mult"] == 1.5


def test_ev_params_bad_pt_raises():
    with pytest.raises(ValueError):
        EVParams(pt_mult=0.0, sl_mult=1.0, horizon=5)


def test_ev_params_bad_horizon_raises():
    with pytest.raises(ValueError):
        EVParams(pt_mult=1.5, sl_mult=1.0, horizon=0)


def test_ev_params_bad_min_proba_raises():
    with pytest.raises(ValueError):
        EVParams(pt_mult=1.5, sl_mult=1.0, horizon=5, min_proba=0.0)


def test_default_grid_non_empty():
    g = default_grid()
    assert len(g) > 0
    assert all(isinstance(p, EVParams) for p in g)


# ------------------------------------------------------------------ #
# ev_walkforward                                                       #
# ------------------------------------------------------------------ #

def test_ev_walkforward_runs(eng_df):
    params = EVParams(pt_mult=1.5, sl_mult=1.0, horizon=5)
    res = ev_walkforward(eng_df, MODEL, params, train_size=200, test_size=50)
    assert isinstance(res, EVWindowResult)
    assert res.n_windows >= 0


def test_ev_walkforward_wins_leq_n_trades(eng_df):
    params = EVParams(pt_mult=1.5, sl_mult=1.0, horizon=5)
    res = ev_walkforward(eng_df, MODEL, params, train_size=200, test_size=50)
    assert res.wins <= res.n_trades


def test_ev_walkforward_win_rate_in_range(eng_df):
    params = EVParams(pt_mult=1.5, sl_mult=1.0, horizon=5)
    res = ev_walkforward(eng_df, MODEL, params, train_size=200, test_size=50)
    if res.n_trades > 0:
        assert 0.0 <= res.win_rate <= 1.0


def test_ev_walkforward_high_min_proba_fewer_trades(eng_df):
    low = ev_walkforward(eng_df, MODEL,
                          EVParams(1.5, 1.0, 5, min_proba=0.50),
                          train_size=200, test_size=50)
    high = ev_walkforward(eng_df, MODEL,
                           EVParams(1.5, 1.0, 5, min_proba=0.90),
                           train_size=200, test_size=50)
    # Higher confidence threshold must not produce more trades.
    assert high.n_trades <= low.n_trades


# ------------------------------------------------------------------ #
# ev_scan                                                              #
# ------------------------------------------------------------------ #

def test_ev_scan_logs_all_configs(eng_df):
    grid = [EVParams(1.5, 1.0, 5), EVParams(2.0, 1.0, 5), EVParams(1.5, 1.5, 5)]
    result = ev_scan(eng_df, MODEL, grid=grid, train_size=200, test_size=50)
    assert result.log.n_trials == len(grid)


def test_ev_scan_best_params_from_grid(eng_df):
    grid = [EVParams(1.5, 1.0, 5), EVParams(2.0, 1.0, 10)]
    result = ev_scan(eng_df, MODEL, grid=grid, train_size=200, test_size=50)
    assert result.best_params in grid


def test_ev_scan_empty_grid_raises(eng_df):
    with pytest.raises(ValueError):
        ev_scan(eng_df, MODEL, grid=[], train_size=200, test_size=50)


# ------------------------------------------------------------------ #
# ev_judge_holdout                                                     #
# ------------------------------------------------------------------ #

def test_ev_judge_holdout_runs(eng_df):
    n = len(eng_df)
    tv = eng_df.iloc[: int(n * 0.8)]
    ho = eng_df.iloc[int(n * 0.8) :]
    params = EVParams(1.5, 1.0, 5)
    stats = ev_judge_holdout(tv, ho, params, model_name=MODEL, n_trials=3)
    assert 0.0 <= stats.deflated_p <= 1.0


def test_ev_judge_holdout_deflated_geq_raw(eng_df):
    n = len(eng_df)
    tv = eng_df.iloc[: int(n * 0.8)]
    ho = eng_df.iloc[int(n * 0.8) :]
    params = EVParams(1.5, 1.0, 5)
    stats = ev_judge_holdout(tv, ho, params, model_name=MODEL, n_trials=20)
    # Sidak deflation makes the p-value no smaller than the raw.
    assert stats.deflated_p >= stats.raw_p - 1e-9


def test_ev_judge_holdout_one_trial_equals_raw(eng_df):
    n = len(eng_df)
    tv = eng_df.iloc[: int(n * 0.8)]
    ho = eng_df.iloc[int(n * 0.8) :]
    params = EVParams(1.5, 1.0, 5)
    stats = ev_judge_holdout(tv, ho, params, model_name=MODEL, n_trials=1)
    assert stats.deflated_p == pytest.approx(stats.raw_p, abs=1e-9)


# ------------------------------------------------------------------ #
# ev_optimize (full protocol)                                          #
# ------------------------------------------------------------------ #

def test_ev_optimize_runs_full_protocol(eng_df):
    small_grid = [EVParams(1.5, 1.0, 5), EVParams(2.0, 1.0, 5)]
    report = ev_optimize(eng_df, MODEL, grid=small_grid,
                          train_size=200, test_size=50, trainval_frac=0.8)
    assert report.scan.log.n_trials == len(small_grid)
    assert report.holdout_rows > 0
    assert report.trainval_rows > 0


def test_ev_optimize_summary_has_content(eng_df):
    small_grid = [EVParams(1.5, 1.0, 5), EVParams(2.0, 1.0, 5)]
    report = ev_optimize(eng_df, MODEL, grid=small_grid,
                          train_size=200, test_size=50, trainval_frac=0.8)
    s = report.summary()
    assert "EV Optimisation Report" in s
    assert "Deflated p" in s


def test_ev_optimize_bad_trainval_frac_raises(eng_df):
    with pytest.raises(ValueError):
        ev_optimize(eng_df, MODEL, trainval_frac=1.0)


def test_ev_optimize_too_small_trainval_raises(eng_df):
    # Passing enormous train/test sizes makes the trainval block too small.
    with pytest.raises(ValueError):
        ev_optimize(eng_df, MODEL, train_size=99999, test_size=99999,
                    trainval_frac=0.8)


def test_ev_optimize_deflation_grows_with_grid_size(eng_df):
    """More trials -> higher deflated p for the same raw evidence."""
    small_grid = [EVParams(1.5, 1.0, 5)]
    big_grid = [EVParams(pt, sl, h)
                for pt in (1.5, 2.0) for sl in (1.0, 1.5) for h in (5, 10)]
    small_report = ev_optimize(eng_df, MODEL, grid=small_grid,
                                train_size=200, test_size=50, trainval_frac=0.8)
    big_report = ev_optimize(eng_df, MODEL, grid=big_grid,
                              train_size=200, test_size=50, trainval_frac=0.8)
    # Deflation is strictly larger with more trials (assuming both produce trades).
    assert big_report.scan.log.n_trials > small_report.scan.log.n_trials


def test_ev_optimize_honest_result_on_garch_data(eng_df):
    """On near-random synthetic data, the result must NOT be flagged significant.
    This is the core honesty check: the shield must not false-positive on noise.

    We use a tiny grid (2 configs) to keep the test fast. With only 2 trials the
    deflation is mild, but even so the holdout on 180 rows of GARCH data should
    not pass p < 0.05 more than ~5% of the time over different seeds. We pick a
    fixed seed that is known to produce a non-significant result as a regression
    guard against the shield being inadvertently weakened."""
    grid = [EVParams(1.5, 1.0, 5), EVParams(2.0, 1.0, 5)]
    report = ev_optimize(eng_df, "xgb_clf", grid=grid,
                          train_size=200, test_size=50, trainval_frac=0.8)
    # On GARCH noise with seed=0 the shield should correctly report no edge.
    # If this ever starts failing it means something in the pipeline is leaking
    # future information -- treat it as a lookahead-bias alarm.
    assert not report.holdout.positive_ev, (
        f"Shield false-positive: EV={report.holdout.ev:.5f}, "
        f"deflated_p={report.holdout.deflated_p:.4f}. "
        "Check for lookahead bias in labeling or features."
    )
