import pytest
from utils.config import SPLIT, SplitConfig, SignalConfig, IndicatorConfig, BacktestConfig, OrderbookConfig, NewsConfig, ModelConfig, DataConfig, validate_all
from utils.exceptions import ConfigError


def test_default_config_validates():
    assert validate_all() is True


def test_split_ratios_sum_to_one():
    assert abs((SPLIT.train_ratio + SPLIT.val_ratio + SPLIT.test_ratio) - 1.0) < 1e-9


def test_split_config_rejects_bad_ratios():
    with pytest.raises(ConfigError):
        SplitConfig(train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)


def test_split_config_rejects_negative_min_rows():
    with pytest.raises(ConfigError):
        SplitConfig(min_train_rows=0)


def test_signal_config_rejects_positive_sell_threshold():
    with pytest.raises(ConfigError):
        SignalConfig(sell_threshold=0.01)


def test_signal_config_rejects_negative_buy_threshold():
    with pytest.raises(ConfigError):
        SignalConfig(buy_threshold=-0.01)


def test_signal_config_rejects_bad_confidence_max():
    with pytest.raises(ConfigError):
        SignalConfig(confidence_max=150)


def test_indicator_config_rejects_macd_fast_gte_slow():
    with pytest.raises(ConfigError):
        IndicatorConfig(macd_fast=26, macd_slow=12)


def test_indicator_config_rejects_zero_window():
    with pytest.raises(ConfigError):
        IndicatorConfig(rsi_window=0)


def test_backtest_config_rejects_negative_capital():
    with pytest.raises(ConfigError):
        BacktestConfig(initial_capital=-100)


def test_orderbook_config_rejects_bad_threshold():
    with pytest.raises(ConfigError):
        OrderbookConfig(imbalance_threshold=1.5)


def test_orderbook_config_rejects_bad_wall_factor():
    with pytest.raises(ConfigError):
        OrderbookConfig(wall_factor=1.0)


def test_news_config_rejects_unknown_backend():
    with pytest.raises(ConfigError):
        NewsConfig(sentiment_backend="bert-base")


def test_model_config_rejects_zero_seq_len():
    with pytest.raises(ConfigError):
        ModelConfig(seq_len=0)


def test_data_config_rejects_invalid_symbol_format():
    with pytest.raises(ConfigError):
        DataConfig(crypto_symbols=("BTCUSDT",))


def test_data_config_rejects_empty_symbols():
    with pytest.raises(ConfigError):
        DataConfig(crypto_symbols=())
