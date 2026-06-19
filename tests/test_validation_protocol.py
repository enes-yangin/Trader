import warnings
import numpy as np
import pandas as pd
import pytest
from engine.validation_protocol import (
    make_split, TrialLog,
    binomial_p_value, deflated_p_value, breakeven_winrate, expected_value,
)
from utils.exceptions import AITraderError

warnings.filterwarnings("ignore")


@pytest.fixture
def df():
    idx = pd.date_range("2020-01-01", periods=1000, freq="D")
    return pd.DataFrame({"close": np.arange(1000, dtype=float)}, index=idx)


def test_split_segments_are_contiguous_and_chronological(df):
    sp = make_split(df, train_frac=0.6, val_frac=0.2)
    assert sp.train.index[-1] < sp.val.index[0]
    val_end = sp.val.index[-1]
    holdout = sp.unseal_holdout("test")
    assert val_end < holdout.index[0]


def test_split_sizes_sum_to_total(df):
    sp = make_split(df, train_frac=0.6, val_frac=0.2)
    s = sp.sizes()
    assert s["train"] + s["val"] + s["holdout"] == len(df)
    assert s["train"] == 600
    assert s["val"] == 200
    assert s["holdout"] == 200


def test_trainval_covers_train_plus_val(df):
    sp = make_split(df, train_frac=0.6, val_frac=0.2)
    assert len(sp.trainval) == sp.train_end == 600 or len(sp.trainval) == sp.val_end
    assert len(sp.trainval) == 800


def test_holdout_starts_sealed(df):
    sp = make_split(df)
    assert sp.holdout_was_touched is False
    assert sp.unseal_count == 0


def test_unsealing_holdout_is_recorded(df):
    sp = make_split(df)
    _ = sp.unseal_holdout("final eval")
    assert sp.holdout_was_touched is True
    assert sp.unseal_count == 1
    _ = sp.unseal_holdout("again")
    assert sp.unseal_count == 2


def test_split_rejects_no_room_for_holdout(df):
    with pytest.raises(AITraderError):
        make_split(df, train_frac=0.7, val_frac=0.3)


def test_split_rejects_invalid_fracs(df):
    with pytest.raises(AITraderError):
        make_split(df, train_frac=1.5, val_frac=0.2)


def test_trial_log_counts_and_finds_best():
    log = TrialLog()
    log.record({"pt": 1.0}, 0.48)
    log.record({"pt": 1.5}, 0.53)
    log.record({"pt": 2.0}, 0.50)
    assert log.n_trials == 3
    assert log.best()["params"]["pt"] == 1.5
    assert log.best()["score"] == 0.53


def test_trial_log_empty_best_is_none():
    assert TrialLog().best() is None


def test_binomial_p_value_fair_coin():
    p = binomial_p_value(50, 100, 0.5)
    assert 0.4 < p < 0.6


def test_binomial_p_value_strong_signal_is_small():
    p = binomial_p_value(70, 100, 0.5)
    assert p < 0.001


def test_deflation_increases_p_value():
    raw = binomial_p_value(62, 100, 0.5)
    deflated = deflated_p_value(62, 100, n_trials=20)
    assert deflated > raw


def test_deflation_single_trial_equals_raw():
    raw = binomial_p_value(60, 100, 0.5)
    deflated = deflated_p_value(60, 100, n_trials=1)
    assert abs(raw - deflated) < 1e-12


def test_deflation_can_kill_marginal_significance():
    """A result that looks significant raw (p<0.05) can become
    non-significant after accounting for many trials."""
    raw = binomial_p_value(59, 100, 0.5)
    deflated = deflated_p_value(59, 100, n_trials=30)
    assert raw < 0.05
    assert deflated > 0.05


def test_breakeven_winrate():
    assert breakeven_winrate(1.0) == pytest.approx(0.5)
    assert breakeven_winrate(1.5) == pytest.approx(0.4)
    assert breakeven_winrate(3.0) == pytest.approx(0.25)


def test_expected_value_sign():
    assert expected_value(0.5, 0.02, 0.02) == pytest.approx(0.0)
    assert expected_value(0.6, 0.02, 0.02) > 0
    assert expected_value(0.4, 0.02, 0.02) < 0
    assert expected_value(0.5, 0.02, 0.02, cost=0.001) < 0
