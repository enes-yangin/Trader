"""Test Leverage Risk Engine (Feature 4)."""
import pytest
import pandas as pd
import numpy as np

from engine.risk import (
    liquidation_price, vol_adjusted_max_leverage,
    funding_cost, check_liquidation, liquidation_buffer_price,
)


class TestLiquidationPrice:
    def test_long_liquidation_below_entry(self):
        liq = liquidation_price(entry_px=30000.0, position_size=1000.0,
                                 leverage=5.0, side="long")
        assert liq < 30000.0  # Long likidasyon entry altında
        assert liq > 0

    def test_short_liquidation_above_entry(self):
        liq = liquidation_price(entry_px=30000.0, position_size=1000.0,
                                 leverage=5.0, side="short")
        assert liq > 30000.0  # Short likidasyon entry üstünde

    def test_higher_leverage_closer_liquidation(self):
        liq_5x = liquidation_price(30000.0, 1000.0, 5.0, "long")
        liq_10x = liquidation_price(30000.0, 1000.0, 10.0, "long")
        # 10x kaldıraçta likidasyon fiyatı entry'ye daha yakın
        assert liq_10x > liq_5x

    def test_no_leverage_no_liquidation(self):
        liq = liquidation_price(30000.0, 1000.0, 1.0, "long")
        # Spot benzeri (1x) — likidasyon çok düşük veya 0
        assert liq < 30000.0 * 0.5 or liq == 0


class TestVolAdjustedLeverage:
    def test_low_vol_high_leverage(self):
        lev = vol_adjusted_max_leverage(volatility=0.01, max_lev=10.0)
        # %1 günlük vol ≈ %19 yıllık → kaldıraç yüksek olmalı
        assert lev > 5.0

    def test_high_vol_low_leverage(self):
        lev = vol_adjusted_max_leverage(volatility=0.05, max_lev=10.0)
        # %5 günlük vol ≈ %96 yıllık → kaldıraç düşük olmalı
        assert lev < 5.0

    def test_clamped_to_range(self):
        lev = vol_adjusted_max_leverage(volatility=0.001, max_lev=10.0)
        assert 1.0 <= lev <= 10.0


class TestFundingCost:
    def test_positive_for_long_position(self):
        cost = funding_cost(10000.0, funding_rate_annual=0.10, holding_days=1.0)
        assert cost > 0
        assert cost == pytest.approx(10000 * 0.10 / 365, rel=0.01)

    def test_zero_funding_rate_no_cost(self):
        cost = funding_cost(10000.0, funding_rate_annual=0.0, holding_days=30.0)
        assert cost == 0.0


class TestLiquidationCheck:
    def test_long_not_liquidated_at_entry(self):
        hit, liq = check_liquidation(30000.0, 30000.0, 5.0, "long")
        assert not hit
        assert liq < 30000.0

    def test_long_liquidated_below_liq(self):
        liq = liquidation_price(30000.0, 1000.0, 10.0, "long")
        hit, _ = check_liquidation(30000.0, liq * 0.99, 10.0, "long")
        assert hit


class TestLiquidationBuffer:
    def test_buffer_between_entry_and_liq_long(self):
        liq = liquidation_price(30000.0, 1000.0, 10.0, "long")
        buf = liquidation_buffer_price(30000.0, 10.0, "long")
        # B3 fix: buffer = liq * (1 + buffer_pct), no more midpoint with entry
        assert liq < buf < 30000.0

    def test_buffer_between_entry_and_liq_short(self):
        liq = liquidation_price(30000.0, 1000.0, 10.0, "short")
        buf = liquidation_buffer_price(30000.0, 10.0, "short")
        # B3 fix: buffer = liq * (1 - buffer_pct)
        assert 30000.0 < buf < liq
