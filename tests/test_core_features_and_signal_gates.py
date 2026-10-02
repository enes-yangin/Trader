"""Plan items D/F/G: core feature set, cost-aware gating, vol-regime filter."""
import numpy as np
import pandas as pd
import pytest

from utils.types import FeatureSpec
from utils.config import BACKTEST, FEATURES, SIGNAL, RISK
from engine.risk import (
    expected_move_covers_costs, vol_regime_threshold, vol_regime_blocked,
)


# --------------------------------------------------------------------------- #
# D: core feature set                                                          #
# --------------------------------------------------------------------------- #

def test_core_spec_returns_only_core_columns():
    cols = FeatureSpec(core=True).feature_columns()
    assert cols == list(FEATURES.core_feature_cols)
    assert len(cols) == 5


def test_core_spec_with_reference_adds_funding_rate():
    cols = FeatureSpec(core=True, reference=True).feature_columns()
    assert cols[-1] == "funding_rate"
    assert len(cols) == len(FEATURES.core_feature_cols) + 1


def test_core_spec_without_reference_excludes_funding_rate():
    cols = FeatureSpec(core=True, reference=False).feature_columns()
    assert "funding_rate" not in cols


def test_non_core_spec_unaffected_by_core_field():
    # Default (non-core) behaviour must be identical whether or not the new
    # `core` field exists -- regression guard for the D short-circuit.
    assert FeatureSpec().feature_columns() == FeatureSpec(core=False).feature_columns()


def test_core_spec_engineer_end_to_end(synthetic_ohlcv):
    from data.indicators import engineer, get_features
    spec = FeatureSpec(core=True, micro=True, smooth=True)
    df = engineer(synthetic_ohlcv, spec=spec)
    X = get_features(df, spec=spec)
    assert list(X.columns) == list(FEATURES.core_feature_cols)
    assert np.all(np.isfinite(X.values))


# --------------------------------------------------------------------------- #
# F: cost-aware signal gating                                                  #
# --------------------------------------------------------------------------- #

def test_expected_move_covers_costs_above_threshold():
    # min_profit_target (default 20%) dominates the plain cost floor on any
    # realistic commission/slippage config, so "above threshold" means
    # above the 20% profit target, not just above break-even costs.
    big_move = SIGNAL.min_profit_target + 0.01
    assert expected_move_covers_costs(big_move)
    assert expected_move_covers_costs(-big_move)  # sign-agnostic (abs)


def test_expected_move_covers_costs_at_exact_threshold_is_false():
    # Strictly-greater-than: exactly at the break-even boundary must NOT pass
    # (a signal that only just clears costs isn't worth the round-trip risk).
    exact = 2.0 * (BACKTEST.commission_pct + BACKTEST.slippage_pct) + SIGNAL.dynamic_threshold_floor
    assert not expected_move_covers_costs(exact)


def test_expected_move_covers_costs_below_threshold_is_false():
    assert not expected_move_covers_costs(0.0001)


def test_expected_move_covers_costs_uses_min_profit_target_as_binding_floor():
    # User request: "only execute trades expected to yield >20% profit".
    # A move that clears round-trip costs but not the 20% target must still
    # be rejected -- min_profit_target is the stricter (binding) bar here.
    cost_floor = 2.0 * (BACKTEST.commission_pct + BACKTEST.slippage_pct) + SIGNAL.dynamic_threshold_floor
    move_clears_costs_only = cost_floor + 0.001  # e.g. ~3-4%, well below 20%
    assert move_clears_costs_only < SIGNAL.min_profit_target
    assert not expected_move_covers_costs(move_clears_costs_only)

    move_clears_both = SIGNAL.min_profit_target + 0.05
    assert expected_move_covers_costs(move_clears_both)


def test_expected_move_covers_costs_min_profit_target_override():
    assert expected_move_covers_costs(0.25, min_profit_target=0.20)
    assert not expected_move_covers_costs(0.15, min_profit_target=0.20)


def test_signalconfig_rejects_non_positive_min_profit_target():
    from utils.config import SignalConfig
    from utils.exceptions import ConfigError
    with pytest.raises(ConfigError):
        SignalConfig(min_profit_target=0.0)
    with pytest.raises(ConfigError):
        SignalConfig(min_profit_target=-0.1)


# --------------------------------------------------------------------------- #
# G: volatility-regime no-trade filter                                        #
# --------------------------------------------------------------------------- #

def test_vol_regime_threshold_matches_percentile():
    hist = np.arange(1, 101, dtype=float)  # 1..100
    th = vol_regime_threshold(hist, pctile=0.8)
    assert th == pytest.approx(np.percentile(hist, 80.0))


def test_vol_regime_threshold_empty_history_never_blocks():
    assert vol_regime_threshold(np.array([])) == float("inf")


def test_vol_regime_blocked_above_and_below_threshold():
    assert vol_regime_blocked(atr_pct=0.05, atr_threshold=0.03)
    assert not vol_regime_blocked(atr_pct=0.02, atr_threshold=0.03)


def test_riskconfig_rejects_invalid_vol_filter_pctile():
    from utils.config import RiskConfig
    from utils.exceptions import ConfigError
    with pytest.raises(ConfigError):
        RiskConfig(vol_filter_pctile=0.0)
    with pytest.raises(ConfigError):
        RiskConfig(vol_filter_pctile=1.0)
