import numpy as np
import pandas as pd
from engine.backtester import run_portfolio, run_portfolio_all
from utils.types import Bundle, SplitDict
from utils.config import SIGNAL


class _MockModel:
    def __init__(self, pred_val: float):
        self.pred_val = pred_val

    def predict(self, X):
        return np.full(len(X), self.pred_val)


def _create_mock_bundle(symbol: str, pred_val: float, close_val: float, n: int = 50) -> Bundle:
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "close": np.full(n, close_val),
        "atr_pct": np.full(n, 0.01),
    }, index=idx)
    df.index.name = "date"
    df.attrs["symbol"] = symbol

    sp = SplitDict(**{
        "X_tr": np.zeros((n // 2, 5)),
        "y_tr": np.zeros(n // 2),
        "X_val": np.zeros((n // 4, 5)),
        "y_val": np.zeros(n // 4),
        "X_test": np.zeros((n // 4, 5)),
        "y_test": np.zeros(n // 4),
        "idx_tr": idx[:n // 2],
        "idx_val": idx[n // 2: 3 * n // 4],
        "idx_test": idx[3 * n // 4:],
        "df": df,
        "i_tr": n // 2,
        "i_va": 3 * n // 4,
        "split_idx": n // 2,
        "with_news": False,
        "with_micro": False,
        "with_cross_asset": False,
        "spec": None,
    })

    results = {
        "linear": {
            "model": _MockModel(pred_val),
            "metrics": {},
        }
    }

    return Bundle(**{
        "results": results,
        "split": sp,
        "symbol": symbol,
        "df": df,
        "with_news": False,
        "with_micro": False,
        "with_cross_asset": False,
        "spec": None,
        "sample": False,
    })


def test_portfolio_backtest_selects_best_candidate():
    btc_bundle = _create_mock_bundle("BTC/USDT", pred_val=0.03, close_val=100.0)
    eth_bundle = _create_mock_bundle("ETH/USDT", pred_val=0.04, close_val=100.0)

    bundles = {
        "BTC/USDT": btc_bundle,
        "ETH/USDT": eth_bundle,
    }

    res = run_portfolio(bundles, "linear", capital=1000.0)
    trades = res["trades"]
    
    assert len(trades) > 0
    buy_trade = trades.iloc[0]
    assert buy_trade["action"] == "BUY"
    assert buy_trade["symbol"] == "ETH/USDT"


def test_portfolio_holds_to_barrier_no_rotation():
    """Exit is triple-barrier only: an open position is held to its take-profit,
    stop, or time barrier -- it is NOT rotated out mid-trade when another symbol
    looks better, and it does NOT exit on a signal flip. Both of those exits
    churned the account into losses (the diagnosed failure mode), so they were
    removed. Here BTC is entered first and its price rises through the +1.5*ATR
    take-profit barrier; ETH becoming far more attractive mid-trade must not
    pull capital out early."""
    class _DynamicMockModel:
        def __init__(self, preds_list):
            self.preds_list = preds_list

        def predict(self, X):
            return np.array(self.preds_list[:len(X)])

    n = 40
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    # BTC price ramps up through the test region so the fixed +15% take-profit
    # barrier (SignalConfig.take_profit_pct) is touched; ETH stays flat.
    btc_close = np.concatenate([np.full(30, 100.0),
                                np.array([100.0, 103.0, 106.0, 109.0, 112.0,
                                          115.0, 118.0, 121.0, 124.0, 127.0])])
    df_btc = pd.DataFrame({"close": btc_close, "atr_pct": np.full(n, 0.01)}, index=idx)
    df_btc.attrs["symbol"] = "BTC/USDT"
    df_eth = pd.DataFrame({"close": np.full(n, 100.0), "atr_pct": np.full(n, 0.01)}, index=idx)
    df_eth.attrs["symbol"] = "ETH/USDT"

    # BTC leads at entry, then ETH becomes far more attractive (the old code
    # would have rotated); BTC pred also drops (the old code would have
    # signal-flip SELL'd).
    btc_preds = [0.05] + [-0.05] * 9
    eth_preds = [0.01] + [0.20] * 9

    sp_btc = SplitDict(
        X_tr=np.zeros((20, 5)), y_tr=np.zeros(20), X_val=np.zeros((10, 5)), y_val=np.zeros(10),
        X_test=np.zeros((10, 5)), y_test=np.zeros(10), idx_tr=idx[:20], idx_val=idx[20:30], idx_test=idx[30:],
        df=df_btc, i_tr=20, i_va=30, split_idx=20, with_news=False, with_micro=False, with_cross_asset=False, spec=None
    )
    sp_eth = SplitDict(
        X_tr=np.zeros((20, 5)), y_tr=np.zeros(20), X_val=np.zeros((10, 5)), y_val=np.zeros(10),
        X_test=np.zeros((10, 5)), y_test=np.zeros(10), idx_tr=idx[:20], idx_val=idx[20:30], idx_test=idx[30:],
        df=df_eth, i_tr=20, i_va=30, split_idx=20, with_news=False, with_micro=False, with_cross_asset=False, spec=None
    )

    btc_bundle = Bundle(
        results={"linear": {"model": _DynamicMockModel(btc_preds), "metrics": {}}},
        split=sp_btc, symbol="BTC/USDT", df=df_btc, with_news=False, with_micro=False, with_cross_asset=False, spec=None, sample=False
    )
    eth_bundle = Bundle(
        results={"linear": {"model": _DynamicMockModel(eth_preds), "metrics": {}}},
        split=sp_eth, symbol="ETH/USDT", df=df_eth, with_news=False, with_micro=False, with_cross_asset=False, spec=None, sample=False
    )

    bundles = {
        "BTC/USDT": btc_bundle,
        "ETH/USDT": eth_bundle,
    }

    res = run_portfolio(bundles, "linear", capital=1000.0)
    trades = res["trades"]

    # No churn exits of either kind.
    assert (trades["action"] == "ROTATE_EXIT").sum() == 0
    assert (trades["action"] == "SELL").sum() == 0
    # The position was entered on BTC and closed at its take-profit barrier.
    first_buy = trades[trades["action"] == "BUY"].iloc[0]
    assert first_buy["symbol"] == "BTC/USDT"
    tp_exits = trades[trades["action"] == "TAKE_PROFIT"]
    assert len(tp_exits) > 0
    assert tp_exits.iloc[0]["symbol"] == "BTC/USDT"
