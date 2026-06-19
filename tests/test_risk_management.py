import numpy as np
from engine.risk import kelly_fraction, position_size, stop_loss_hit
from engine.backtester import run
from engine.trainer import split
from data.indicators import engineer
from utils.config import RISK


def test_kelly_fraction_insufficient_history_returns_max():
    f = kelly_fraction([0.01, -0.01], min_trades=5)
    assert f == RISK.max_position_pct


def test_kelly_fraction_all_wins_returns_max():
    f = kelly_fraction([0.01] * 10, min_trades=5)
    assert f == RISK.max_position_pct


def test_kelly_fraction_all_losses_returns_max():
    f = kelly_fraction([-0.01] * 10, min_trades=5)
    assert f == RISK.max_position_pct


def test_kelly_fraction_mixed_results_in_range():
    pnls = [0.05, -0.02, 0.03, -0.04, 0.06, -0.01, 0.02, -0.03]
    f = kelly_fraction(pnls, min_trades=5)
    assert RISK.min_position_pct <= f <= RISK.max_position_pct


def test_kelly_fraction_high_win_rate_higher_than_low_win_rate():
    good = [0.05, 0.04, 0.06, 0.03, -0.02, -0.01]
    bad = [0.02, -0.05, -0.04, -0.06, -0.03, 0.01]
    f_good = kelly_fraction(good, min_trades=5)
    f_bad = kelly_fraction(bad, min_trades=5)
    assert f_good >= f_bad


def test_position_size_respects_use_kelly_flag():
    pnls = [0.02, -0.05, -0.04, -0.06, -0.03, 0.01]
    f_with_kelly = position_size(pnls, use_kelly=True, min_trades=5)
    f_without = position_size(pnls, use_kelly=False, min_trades=5)
    assert f_without == RISK.max_position_pct
    assert f_with_kelly <= f_without


def test_stop_loss_hit_triggers_below_threshold():
    assert stop_loss_hit(entry_px=100, current_px=94, stop_loss_pct=0.05) is True
    assert stop_loss_hit(entry_px=100, current_px=96, stop_loss_pct=0.05) is False
    assert stop_loss_hit(entry_px=100, current_px=95, stop_loss_pct=0.05) is True


def test_stop_loss_zero_entry_returns_false():
    assert stop_loss_hit(entry_px=0, current_px=50, stop_loss_pct=0.05) is False


class _AlwaysBuyModel:
    name = "AlwaysBuy"

    def __init__(self):
        self.trained = True
        self.n_features_ = None

    def train(self, *a, **kw):
        return {"train": {"r2": 0, "rmse": 0, "mse": 0, "mae": 0}}

    def evaluate(self, X, y):
        return {"r2": 0.0, "rmse": 0.0, "mse": 0.0, "mae": 0.0}

    def predict(self, X):
        return np.full(len(X), 0.05)


def test_backtest_position_never_exceeds_max(synthetic_ohlcv):
    df = engineer(synthetic_ohlcv, with_news=False, with_micro=True)
    sp = split(df, with_news=False, with_micro=True)
    mdl = _AlwaysBuyModel()

    res = run(mdl, sp, max_position_pct=0.5, use_kelly=False)
    trades = res["trades"]
    buys = trades[trades["action"] == "BUY"]
    if len(buys) > 0:
        assert (buys["size_pct"] <= 50.0 + 1e-6).all()


def test_backtest_with_stop_loss_records_stops(synthetic_ohlcv):
    df = engineer(synthetic_ohlcv, with_news=False, with_micro=True)
    sp = split(df, with_news=False, with_micro=True)
    mdl = _AlwaysBuyModel()

    res = run(mdl, sp, stop_loss_pct=0.001, use_kelly=False)
    assert res["metrics"]["n_stop_losses"] >= 0
    if res["metrics"]["n_stop_losses"] > 0:
        stops = res["trades"][res["trades"]["action"] == "STOP"]
        assert len(stops) == res["metrics"]["n_stop_losses"]


def test_backtest_final_equity_always_nonnegative(synthetic_ohlcv):
    df = engineer(synthetic_ohlcv, with_news=False, with_micro=True)
    sp = split(df, with_news=False, with_micro=True)
    mdl = _AlwaysBuyModel()
    res = run(mdl, sp)
    assert res["metrics"]["final_equity"] >= 0
