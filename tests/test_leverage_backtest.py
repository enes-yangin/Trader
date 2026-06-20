"""Unit tests for leverage backtesting, funding costs, liquidation, and buffer stops."""
import pytest
import pandas as pd
import numpy as np
from engine.backtester import run
import utils.config
from utils.config import RiskConfig

class MockPredictor:
    def __init__(self, predictions):
        self.predictions = predictions
        self.name = "MockPredictor"

    def predict(self, X):
        return np.array(self.predictions)

def make_dummy_split(prices):
    n = len(prices)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "close": prices,
        "high": prices,
        "low": prices,
        "volume": [1000.0] * n,
        "atr_pct": [0.01] * n  # 1% ATR
    }, index=idx)
    df.index.name = "date"
    return {
        "X_test": np.zeros((n, 2)),
        "y_test": np.zeros(n),
        "idx_test": idx,
        "i_va": 0,
        "df": df
    }

def test_leverage_liquidation_trigger(monkeypatch):
    # Setup RiskConfig to use leverage
    new_risk = RiskConfig(
        use_leverage=True,
        max_leverage=10.0,
        maintenance_margin=0.005,
        liquidation_buffer=0.0,  # Disable buffer stop so we test raw liquidation first
        stop_loss_pct=0.50,       # Disable regular stop loss
        use_kelly_sizing=False,   # Disable Kelly
        max_position_pct=1.0      # Invest 100% of cash as margin
    )
    monkeypatch.setattr(utils.config, "RISK", new_risk)
    import engine.backtester
    monkeypatch.setattr(engine.backtester, "RISK", new_risk)
    import engine.risk
    monkeypatch.setattr(engine.risk, "RISK", new_risk)

    # 100 -> 100 -> 80 (should liquidate because 80 is below the ~90.25 liquidation price)
    prices = [100.0, 100.0, 80.0, 80.0]
    sp = make_dummy_split(prices)
    
    # Predict 0.1 on step 0 to trigger buy, then 0.0 to hold
    preds = [0.1, 0.0, 0.0, 0.0]
    mdl = MockPredictor(preds)
    
    res = run(
        mdl, sp, buy_th=0.02, sell_th=-0.02, capital=10000.0,
        commission_pct=0.0, slippage_pct=0.0
    )
    
    trades = res["trades"]
    assert not trades.empty
    
    # Check that a liquidation trade happened
    liq_trade = trades[trades["action"] == "LIQUIDATION"]
    assert len(liq_trade) == 1
    assert float(liq_trade.iloc[0]["pnl"]) == -1.0

def test_leverage_buffer_stop_trigger(monkeypatch):
    # Setup RiskConfig to use leverage and buffer stop
    new_risk = RiskConfig(
        use_leverage=True,
        max_leverage=10.0,
        maintenance_margin=0.005,
        liquidation_buffer=0.02,   # 2% buffer above liq
        stop_loss_pct=0.50,       # Disable regular stop loss
        use_kelly_sizing=False,
        max_position_pct=1.0
    )
    monkeypatch.setattr(utils.config, "RISK", new_risk)
    import engine.backtester
    monkeypatch.setattr(engine.backtester, "RISK", new_risk)
    import engine.risk
    monkeypatch.setattr(engine.risk, "RISK", new_risk)

    # 100 -> 100 -> 91.5
    # Liq price for 10x leverage (standard formula): 100 * (1 - 0.1 + 0.005) = 90.5
    # Buffer stop price (B3 fix): 90.5 * (1 + 0.02) = 92.31
    # So 91.5 is below 92.31 (triggers stop) but above 90.5 (not liquidated)
    prices = [100.0, 100.0, 91.5, 91.5]
    sp = make_dummy_split(prices)
    
    preds = [0.1, 0.0, 0.0, 0.0]
    mdl = MockPredictor(preds)
    
    res = run(
        mdl, sp, buy_th=0.02, sell_th=-0.02, capital=10000.0,
        commission_pct=0.0, slippage_pct=0.0
    )
    
    trades = res["trades"]
    assert not trades.empty
    
    # Check that a LIQ_PREVENT_STOP trade happened
    stop_trade = trades[trades["action"] == "LIQ_PREVENT_STOP"]
    assert len(stop_trade) == 1
    # PnL should be negative but better than -1.0 (not fully liquidated)
    pnl = float(stop_trade.iloc[0]["pnl"])
    assert -1.0 < pnl < 0.0

def test_leverage_funding_cost_deduction(monkeypatch):
    # Setup RiskConfig to use leverage and charge funding
    new_risk = RiskConfig(
        use_leverage=True,
        max_leverage=5.0,
        funding_rate_annual=0.10,  # 10% annual funding
        maintenance_margin=0.005,
        liquidation_buffer=0.0,
        stop_loss_pct=0.50,
        use_kelly_sizing=False,
        max_position_pct=1.0
    )
    monkeypatch.setattr(utils.config, "RISK", new_risk)
    import engine.backtester
    monkeypatch.setattr(engine.backtester, "RISK", new_risk)
    import engine.risk
    monkeypatch.setattr(engine.risk, "RISK", new_risk)

    # Price stays flat at 100, we hold position for 2 bars
    prices = [100.0, 100.0, 100.0, 100.0]
    sp = make_dummy_split(prices)
    
    preds = [0.1, 0.0, 0.0, 0.0]
    mdl = MockPredictor(preds)
    
    res = run(
        mdl, sp, buy_th=0.02, sell_th=-0.02, capital=10000.0,
        commission_pct=0.0, slippage_pct=0.0
    )
    
    # Verify that total_costs includes funding costs
    assert res["metrics"]["total_costs"] > 0.0
