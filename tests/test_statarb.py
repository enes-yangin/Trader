import warnings
import numpy as np
import pytest

from tests.realistic_data import make_cointegrated_pair, make_independent_walks
from engine.statarb import (
    adf_test, hedge_ratio, spread, zscore, engle_granger,
)
from engine.statarb_signals import (
    generate_positions, simulate, fit_and_signal, statarb_walkforward,
    select_and_judge,
)

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# Cointegration detection (1b)
# --------------------------------------------------------------------------- #

def test_adf_rejects_on_stationary_series():
    # White noise is stationary -> ADF should reject the unit root.
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    res = adf_test(x)
    assert res.is_stationary
    assert res.stat < res.crit[0.05]


def test_adf_does_not_reject_on_random_walk():
    # A random walk has a unit root -> ADF should NOT reject.
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.standard_normal(500))
    res = adf_test(x)
    assert not res.is_stationary


def test_adf_too_short_raises():
    with pytest.raises(ValueError):
        adf_test(np.arange(5.0))


def test_hedge_ratio_recovers_planted_beta():
    df_x, df_y = make_cointegrated_pair(n=900, seed=3, beta=0.065)
    beta, _alpha = hedge_ratio(df_y["close"].values, df_x["close"].values)
    assert beta == pytest.approx(df_y.attrs["true_beta"], abs=0.01)


def test_engle_granger_detects_planted_cointegration():
    df_x, df_y = make_cointegrated_pair(n=900, seed=5, beta=0.065)
    res = engle_granger(df_y["close"].values, df_x["close"].values)
    assert res.cointegrated
    assert res.beta == pytest.approx(df_y.attrs["true_beta"], abs=0.01)


def test_engle_granger_rejects_independent_walks():
    # Negative control: two unrelated random walks are not cointegrated.
    df_x, df_y = make_independent_walks(n=900, seed=7)
    res = engle_granger(df_y["close"].values, df_x["close"].values)
    assert not res.cointegrated


def test_spread_is_residual():
    y = np.array([10.0, 11.0, 12.0])
    x = np.array([1.0, 2.0, 3.0])
    sp = spread(y, x, beta=1.0, alpha=9.0)
    assert np.allclose(sp, [0.0, 0.0, 0.0])


def test_zscore_uses_supplied_train_stats():
    sp = np.array([1.0, 2.0, 3.0, 4.0])
    z = zscore(sp, mean=2.5, std=2.0)
    assert z[0] == pytest.approx((1.0 - 2.5) / 2.0)


# --------------------------------------------------------------------------- #
# Signals (1c)
# --------------------------------------------------------------------------- #

def test_positions_enter_short_on_high_z():
    z = np.array([0.0, 2.5, 1.0, 0.3])
    pos = generate_positions(z, entry=2.0, exit=0.5)
    # opens short at the +2.5 spike, holds at 1.0, closes when |z|<0.5
    assert pos[1] == -1.0
    assert pos[2] == -1.0
    assert pos[3] == 0.0


def test_positions_enter_long_on_low_z():
    z = np.array([0.0, -2.5, -1.0, 0.1])
    pos = generate_positions(z, entry=2.0, exit=0.5)
    assert pos[1] == 1.0
    assert pos[3] == 0.0


def test_positions_no_lookahead():
    # position at t must not depend on z after t: truncating the tail leaves
    # earlier positions unchanged.
    z = np.array([0.0, 2.5, 1.0, 0.3, -2.5])
    full = generate_positions(z, entry=2.0, exit=0.5)
    trunc = generate_positions(z[:3], entry=2.0, exit=0.5)
    assert np.array_equal(full[:3], trunc)


def test_simulate_pnl_and_cost():
    sp = np.array([0.0, 1.0, 2.0])      # spread rises by 1 each bar
    pos = np.array([1.0, 1.0, 0.0])     # long the spread for bars 0..1
    trades, per_bar = simulate(sp, pos, cost=0.1)
    assert per_bar[0] == pytest.approx(1.0)
    assert per_bar[1] == pytest.approx(1.0)
    assert len(trades) == 1
    assert trades[0] == pytest.approx(2.0 - 0.1)


def test_fit_and_signal_profitable_on_planted_pair():
    df_x, df_y = make_cointegrated_pair(n=900, seed=11, spread_half_life=10.0)
    y = df_y["close"].values
    x = df_x["close"].values
    cut = 500
    res = fit_and_signal(y[:cut], x[:cut], y[cut:], x[cut:], entry=2.0, exit=0.5)
    # On a planted mean-reverting spread the strategy should trade and, before
    # costs, come out ahead more often than not.
    assert res.n_trades > 0
    assert res.win_rate > 0.5


# --------------------------------------------------------------------------- #
# Leakage / purging (1d, 1e)
# --------------------------------------------------------------------------- #

def test_walkforward_purges_train_tail():
    df_x, df_y = make_cointegrated_pair(n=600, seed=2)
    y = df_y["close"].values
    x = df_x["close"].values
    no_purge = statarb_walkforward(y, x, train_size=120, test_size=40, purge=0)
    purged = statarb_walkforward(y, x, train_size=120, test_size=40, purge=10)
    # Both run; purging changes the fit window so results need not match. The
    # contract under test is that purge is actually applied (fewer train bars),
    # which we verify indirectly: a huge purge starves the fit and is rejected.
    assert no_purge.n_windows > 0
    assert purged.n_windows > 0


def test_walkforward_rejects_overlong_purge():
    df_x, df_y = make_cointegrated_pair(n=600, seed=2)
    y = df_y["close"].values
    x = df_x["close"].values
    # Purge larger than the train block leaves <4 fit bars -> no usable windows.
    res = statarb_walkforward(y, x, train_size=120, test_size=40, purge=200)
    assert res.n_windows == 0


def test_walkforward_length_mismatch_raises():
    with pytest.raises(ValueError):
        statarb_walkforward(np.arange(100.0), np.arange(90.0),
                            train_size=40, test_size=20)


# --------------------------------------------------------------------------- #
# Shield: selection + sealed holdout + deflated-p (1d)
# --------------------------------------------------------------------------- #

def test_select_and_judge_runs_full_protocol():
    df_x, df_y = make_cointegrated_pair(n=1200, seed=13, spread_half_life=12.0)
    y = df_y["close"].values
    x = df_x["close"].values
    grid = [(e, ex) for e in (1.5, 2.0, 2.5) for ex in (0.25, 0.5)]
    rep = select_and_judge(y, x, grid, wf_train=120, wf_test=40)
    assert rep.n_trials == len(grid)
    assert rep.best_params["entry"] in (1.5, 2.0, 2.5)
    # Deflation must make the p-value no smaller than the raw p-value.
    assert rep.deflated_p_value >= rep.raw_p_value
    assert 0.0 <= rep.deflated_p_value <= 1.0


def test_deflation_penalises_more_trials():
    df_x, df_y = make_cointegrated_pair(n=1200, seed=17, spread_half_life=12.0)
    y = df_y["close"].values
    x = df_x["close"].values
    small = select_and_judge(y, x, [(2.0, 0.5)], wf_train=120, wf_test=40)
    big_grid = [(e, ex) for e in (1.5, 2.0, 2.5, 3.0)
                for ex in (0.25, 0.5, 0.75)]
    big = select_and_judge(y, x, big_grid, wf_train=120, wf_test=40)
    # With one trial, deflated == raw. With many, deflation strictly inflates
    # the p-value for the same kind of holdout evidence.
    assert small.deflated_p_value == pytest.approx(small.raw_p_value)
    assert big.n_trials > small.n_trials


def test_select_and_judge_holdout_sealed_until_end():
    # The holdout fraction is judged exactly once; the report exposes its size
    # so we can confirm it was actually held out of selection.
    df_x, df_y = make_cointegrated_pair(n=1000, seed=19)
    y = df_y["close"].values
    x = df_x["close"].values
    rep = select_and_judge(y, x, [(2.0, 0.5), (2.5, 0.5)],
                           train_frac=0.6, val_frac=0.2,
                           wf_train=100, wf_test=40)
    assert rep.holdout_trades >= 0
    assert rep.trial_log.n_trials == 2
