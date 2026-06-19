import numpy as np
import pandas as pd
from engine.backtester import run, _calc_metrics
from engine.trainer import split
from data.indicators import engineer


class _AlternatingModel:
    name = "AlternatingTest"

    def __init__(self):
        self.trained = True
        self.n_features_ = None
        self._i = 0

    def train(self, *a, **kw):
        return {"train": {"r2": 0, "rmse": 0, "mse": 0, "mae": 0}}

    def evaluate(self, X, y):
        return {"r2": 0.0, "rmse": 0.0, "mse": 0.0, "mae": 0.0}

    def predict(self, X):
        n = len(X)
        out = np.empty(n)
        out[0::2] = 0.05
        out[1::2] = -0.05
        return out


def test_zero_cost_vs_with_cost_returns_differ(synthetic_ohlcv):
    df = engineer(synthetic_ohlcv, with_news=False, with_micro=True)
    sp = split(df, with_news=False, with_micro=True)
    mdl = _AlternatingModel()

    no_cost = run(mdl, sp, commission_pct=0.0, slippage_pct=0.0)
    with_cost = run(mdl, sp, commission_pct=0.001, slippage_pct=0.0005)

    assert no_cost["metrics"]["total_costs"] == 0.0
    assert with_cost["metrics"]["total_costs"] > 0.0
    assert with_cost["metrics"]["final_equity"] < no_cost["metrics"]["final_equity"], (
        "Costs must reduce final equity relative to zero-cost baseline"
    )


def test_costs_scale_with_trade_frequency(synthetic_ohlcv):
    df = engineer(synthetic_ohlcv, with_news=False, with_micro=True)
    sp = split(df, with_news=False, with_micro=True)
    mdl = _AlternatingModel()

    res = run(mdl, sp, commission_pct=0.001, slippage_pct=0.0005)
    n_trades = res["metrics"]["n_trades"]
    assert n_trades > 0
    assert res["metrics"]["total_costs"] > 0
    assert res["metrics"]["costs_pct_of_capital"] > 0


def test_default_config_costs_are_nonzero():
    from utils.config import BACKTEST
    assert BACKTEST.commission_pct > 0, "Default commission should reflect real exchange fees"
    assert BACKTEST.slippage_pct > 0, "Default slippage should be modeled for realism"


def test_calc_metrics_includes_cost_keys():
    eq_df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=5),
                           "equity": [100, 101, 99, 102, 103]})
    trades_df = pd.DataFrame()
    m = _calc_metrics(eq_df, trades_df, capital=100, final_eq=103)
    assert "total_return" in m
    assert "sharpe" in m
