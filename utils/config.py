import os
from dataclasses import dataclass, field
from typing import Tuple, Dict
from utils.exceptions import ConfigError
from utils.dotenv import load_dotenv

# Load .env (project root) into os.environ before any config value is read,
# so a user's HF_TOKEN / API keys / exchange choice take effect everywhere.
load_dotenv()


def _env_str(name, default):
    return os.environ.get(name, default)


def _env_int(name, default):
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        raise ConfigError(f"Env var {name}={v!r} is not a valid int")


def _env_float(name, default):
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        raise ConfigError(f"Env var {name}={v!r} is not a valid float")


def _env_bool(name, default):
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class DataConfig:
    exchange_id: str = field(default_factory=lambda: _env_str("AI_TRADER_EXCHANGE", "kraken"))
    timeframe: str = "1d"
    ohlcv_limit: int = 500
    hist_years: int = field(default_factory=lambda: _env_int("AI_TRADER_HIST_YEARS", 5))

    live_timeframe: str = "1h"
    live_limit: int = field(default_factory=lambda: _env_int("AI_TRADER_LIVE_LIMIT", 168))
    live_refresh_s: int = field(default_factory=lambda: _env_int("AI_TRADER_LIVE_REFRESH_S", 60))

    data_dir: str = field(default_factory=lambda: _env_str("AI_TRADER_DATA_DIR", "datasets"))
    cache_max_age_h: float = field(default_factory=lambda: _env_float("AI_TRADER_CACHE_MAX_AGE_H", 12.0))
    cache_fmt: str = "parquet"

    crypto_symbols: Tuple[str, ...] = (
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT",
        "XRP/USDT", "ADA/USDT", "DOGE/USDT", "AVAX/USDT",
    )

    def __post_init__(self):
        if self.hist_years <= 0:
            raise ConfigError(f"hist_years must be positive, got {self.hist_years}")
        if self.ohlcv_limit <= 0:
            raise ConfigError(f"ohlcv_limit must be positive, got {self.ohlcv_limit}")
        if self.live_limit <= 0:
            raise ConfigError(f"live_limit must be positive, got {self.live_limit}")
        if self.live_refresh_s <= 0:
            raise ConfigError(f"live_refresh_s must be positive, got {self.live_refresh_s}")
        if self.cache_max_age_h <= 0:
            raise ConfigError(f"cache_max_age_h must be positive, got {self.cache_max_age_h}")
        if not self.crypto_symbols:
            raise ConfigError("crypto_symbols must not be empty")
        for s in self.crypto_symbols:
            if "/" not in s:
                raise ConfigError(f"Invalid symbol format (expected BASE/QUOTE): {s}")


@dataclass(frozen=True)
class FeatureConfig:
    target_col: str = "target"
    close_col: str = "close"
    date_col: str = "date"

    feature_cols: Tuple[str, ...] = (
        "ret_1d", "ret_5d", "ret_10d",
        "rsi", "macd_diff", "bb_pband",
        "atr_pct", "vol_chg", "vol_ratio",
        "ema20_dist", "sma50_dist", "ema_sma_dist",
        "rsi_bull_div", "rsi_bear_div",
    )
    micro_feature_cols: Tuple[str, ...] = (
        "vol_delta", "amihud_illiq", "roll_spread",
        "kyle_lambda", "vwap_dist", "hl_range",
    )
    news_feature_cols: Tuple[str, ...] = (
        "sentiment_avg", "sentiment_std", "news_volume", "sentiment_trend",
    )
    cross_asset_feature_cols: Tuple[str, ...] = (
        "eth_btc_chg", "btc_vol_share", "mkt_corr",
    )
    smoothing_feature_cols: Tuple[str, ...] = (
        "savgol_slope", "ema_slope",
    )
    reference_feature_cols: Tuple[str, ...] = (
        "dxy_chg", "stable_dom_chg", "liquidity_chg", "oi_chg",
        "funding_rate", "onchain_z", "fear_greed",
    )
    orderbook_feature_cols: Tuple[str, ...] = (
        "depth_imbalance", "wall_persistence", "microprice_drift", "spread_regime",
    )
    macro_event_feature_cols: Tuple[str, ...] = (
        "bars_to_next_event", "last_event_surprise", "event_importance", "event_day_flag",
    )
    social_feature_cols: Tuple[str, ...] = (
        "social_volume_z", "social_sentiment_avg", "social_polarity_shift",
    )

    use_micro: bool = field(default_factory=lambda: _env_bool("AI_TRADER_USE_MICRO", True))
    use_news: bool = field(default_factory=lambda: _env_bool("AI_TRADER_USE_NEWS", False))
    use_cross_asset: bool = field(default_factory=lambda: _env_bool("AI_TRADER_USE_CROSS_ASSET", True))
    use_smoothing: bool = field(default_factory=lambda: _env_bool("AI_TRADER_USE_SMOOTHING", True))
    use_reference: bool = field(default_factory=lambda: _env_bool("AI_TRADER_USE_REFERENCE", False))
    use_orderbook: bool = field(default_factory=lambda: _env_bool("AI_TRADER_USE_ORDERBOOK", True))
    use_macro_events: bool = field(default_factory=lambda: _env_bool("AI_TRADER_USE_MACRO_EVENTS", True))
    use_social: bool = field(default_factory=lambda: _env_bool("AI_TRADER_USE_SOCIAL", False))

    def __post_init__(self):
        if not self.feature_cols:
            raise ConfigError("feature_cols must not be empty")


@dataclass(frozen=True)
class CrossAssetConfig:
    ref_symbols: Tuple[str, ...] = ("BTC/USDT", "ETH/USDT", "BNB/USDT")
    eth_btc_symbol: str = "ETH/BTC"
    corr_window: int = 30
    vol_share_zscore_window: int = 60

    def __post_init__(self):
        if not self.ref_symbols:
            raise ConfigError("ref_symbols must not be empty")
        for s in self.ref_symbols:
            if "/" not in s:
                raise ConfigError(f"Invalid symbol format (expected BASE/QUOTE): {s}")
        if "/" not in self.eth_btc_symbol:
            raise ConfigError(f"Invalid eth_btc_symbol format: {self.eth_btc_symbol}")
        if self.corr_window <= 1:
            raise ConfigError(f"corr_window must be > 1, got {self.corr_window}")
        if self.vol_share_zscore_window <= 1:
            raise ConfigError(f"vol_share_zscore_window must be > 1, got {self.vol_share_zscore_window}")


@dataclass(frozen=True)
class ReferenceSeriesConfig:
    """External reference time-series (macro / derivatives / sentiment) merged
    into the price frame by date. Every series is shifted forward by shift_days
    so day-d's published value can only enter day-(d+shift) features -- the same
    no-same-day-leak discipline as news sentiment."""
    shift_days: int = 1
    zscore_window: int = 30
    dxy_clip: float = 0.1
    dom_clip: float = 0.1
    liquidity_clip: float = 0.1
    oi_clip: float = 0.5
    funding_clip: float = 0.01
    onchain_clip: float = 5.0
    fear_greed_clip: float = 1.0
    cache_ttl_s: int = 3600  # B7: cache expiry (1 hour default)
    request_timeout: int = 10
    fear_greed_limit: int = 0  # 0 = full history (index-independent, cache-safe)

    fear_greed_url: str = "https://api.alternative.me/fng/"
    defillama_stables_url: str = "https://stablecoins.llama.fi/stablecoincharts/all"
    fred_url: str = "https://api.stlouisfed.org/fred/series/observations"
    coinmetrics_url: str = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
    fred_api_key: str = field(default_factory=lambda: _env_str("FRED_API_KEY", ""))

    def __post_init__(self):
        if self.shift_days < 1:
            raise ConfigError(
                f"reference shift_days must be >= 1 to avoid same-day leak, "
                f"got {self.shift_days}"
            )
        if self.zscore_window <= 1:
            raise ConfigError(f"zscore_window must be > 1, got {self.zscore_window}")
        clips = {
            "dxy_clip": self.dxy_clip, "dom_clip": self.dom_clip,
            "liquidity_clip": self.liquidity_clip, "oi_clip": self.oi_clip,
            "funding_clip": self.funding_clip, "onchain_clip": self.onchain_clip,
            "fear_greed_clip": self.fear_greed_clip,
        }
        for name, v in clips.items():
            if v <= 0:
                raise ConfigError(f"{name} must be positive, got {v}")
        if self.cache_ttl_s <= 0:
            raise ConfigError(f"cache_ttl_s must be positive, got {self.cache_ttl_s}")
        if self.request_timeout <= 0:
            raise ConfigError(f"request_timeout must be positive, got {self.request_timeout}")
        if self.fear_greed_limit < 0:
            raise ConfigError(f"fear_greed_limit must be >= 0, got {self.fear_greed_limit}")


@dataclass(frozen=True)
class IndicatorConfig:
    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_window: int = 20
    bb_dev: float = 2.0
    atr_window: int = 14
    ema_window: int = 20
    sma_window: int = 50
    volume_sma_window: int = 20

    amihud_window: int = 5
    amihud_zscore_window: int = 60
    roll_spread_window: int = 20
    kyle_lambda_window: int = 20
    vwap_window: int = 20

    savgol_window: int = 11
    savgol_polyorder: int = 3
    ema_smooth_span: int = 10

    def __post_init__(self):
        windows = {
            "rsi_window": self.rsi_window, "macd_fast": self.macd_fast,
            "macd_slow": self.macd_slow, "macd_signal": self.macd_signal,
            "bb_window": self.bb_window, "atr_window": self.atr_window,
            "ema_window": self.ema_window, "sma_window": self.sma_window,
            "volume_sma_window": self.volume_sma_window,
            "amihud_window": self.amihud_window,
            "amihud_zscore_window": self.amihud_zscore_window,
            "roll_spread_window": self.roll_spread_window,
            "kyle_lambda_window": self.kyle_lambda_window,
            "vwap_window": self.vwap_window,
            "savgol_window": self.savgol_window,
            "ema_smooth_span": self.ema_smooth_span,
        }
        for name, val in windows.items():
            if val <= 0:
                raise ConfigError(f"{name} must be positive, got {val}")
        if self.macd_fast >= self.macd_slow:
            raise ConfigError(
                f"macd_fast ({self.macd_fast}) must be < macd_slow ({self.macd_slow})"
            )
        if self.bb_dev <= 0:
            raise ConfigError(f"bb_dev must be positive, got {self.bb_dev}")
        if self.savgol_window % 2 == 0:
            raise ConfigError(
                f"savgol_window must be odd, got {self.savgol_window}"
            )
        if self.savgol_polyorder >= self.savgol_window:
            raise ConfigError(
                f"savgol_polyorder ({self.savgol_polyorder}) must be < "
                f"savgol_window ({self.savgol_window})"
            )


@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float = 0.60
    val_ratio: float = 0.15
    test_ratio: float = 0.25
    min_train_rows: int = 252

    def __post_init__(self):
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ConfigError(
                f"train+val+test ratios must sum to 1.0, got {total} "
                f"({self.train_ratio}+{self.val_ratio}+{self.test_ratio})"
            )
        for name, val in [("train_ratio", self.train_ratio),
                          ("val_ratio", self.val_ratio),
                          ("test_ratio", self.test_ratio)]:
            if not (0 < val < 1):
                raise ConfigError(f"{name} must be in (0,1), got {val}")
        if self.min_train_rows <= 0:
            raise ConfigError(f"min_train_rows must be positive, got {self.min_train_rows}")


@dataclass(frozen=True)
class SignalConfig:
    buy_threshold: float = 0.02
    sell_threshold: float = -0.02
    confidence_scale: float = 50.0
    confidence_max: float = 99.0
    use_dynamic_threshold: bool = True
    atr_mult: float = 0.1

    def __post_init__(self):
        if self.buy_threshold <= 0:
            raise ConfigError(f"buy_threshold must be positive, got {self.buy_threshold}")
        if self.sell_threshold >= 0:
            raise ConfigError(f"sell_threshold must be negative, got {self.sell_threshold}")
        if self.confidence_scale <= 0:
            raise ConfigError(f"confidence_scale must be positive, got {self.confidence_scale}")
        if not (0 < self.confidence_max <= 100):
            raise ConfigError(f"confidence_max must be in (0,100], got {self.confidence_max}")
        if self.atr_mult <= 0:
            raise ConfigError(f"atr_mult must be positive, got {self.atr_mult}")


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 10000.0
    walkforward_train_size: int = 756
    walkforward_test_size: int = 189
    commission_pct: float = 0.001
    slippage_pct: float = 0.0005

    def __post_init__(self):
        if self.initial_capital <= 0:
            raise ConfigError(f"initial_capital must be positive, got {self.initial_capital}")
        if self.walkforward_train_size <= 0 or self.walkforward_test_size <= 0:
            raise ConfigError("walkforward train/test sizes must be positive")
        if not (0 <= self.commission_pct < 0.1):
            raise ConfigError(f"commission_pct must be in [0, 0.1), got {self.commission_pct}")
        if not (0 <= self.slippage_pct < 0.1):
            raise ConfigError(f"slippage_pct must be in [0, 0.1), got {self.slippage_pct}")


@dataclass(frozen=True)
class RiskConfig:
    stop_loss_pct: float = 0.05
    max_position_pct: float = 0.2
    min_position_pct: float = 0.1
    use_kelly_sizing: bool = True
    kelly_fraction: float = 0.25
    kelly_min_trades: int = 5

    # Kaldıraç / Likidasyon (Feature 4)
    maintenance_margin: float = 0.005         # %0.5 (Binance BTC tipik değeri)
    max_leverage: float = 10.0                # Maks kaldıraç
    funding_rate_annual: float = 0.10         # Yıllık funding oranı (%10)
    liquidation_buffer: float = 0.002         # Likidasyon fiyatına güvenlik buffer'ı
    use_leverage: bool = field(default_factory=lambda: _env_bool("AI_TRADER_USE_LEVERAGE", False))

    def __post_init__(self):
        if not (0 < self.stop_loss_pct < 1):
            raise ConfigError(f"stop_loss_pct must be in (0,1), got {self.stop_loss_pct}")
        if not (0 < self.max_position_pct <= 1):
            raise ConfigError(f"max_position_pct must be in (0,1], got {self.max_position_pct}")
        if not (0 < self.min_position_pct <= self.max_position_pct):
            raise ConfigError(
                f"min_position_pct ({self.min_position_pct}) must be in "
                f"(0, max_position_pct={self.max_position_pct}]"
            )
        if not (0 < self.kelly_fraction <= 1):
            raise ConfigError(f"kelly_fraction must be in (0,1], got {self.kelly_fraction}")
        if self.kelly_min_trades < 1:
            raise ConfigError(f"kelly_min_trades must be >= 1, got {self.kelly_min_trades}")
        if not (0 < self.maintenance_margin < 1):
            raise ConfigError(f"maintenance_margin must be in (0,1), got {self.maintenance_margin}")
        if self.max_leverage < 1:
            raise ConfigError(f"max_leverage must be >= 1, got {self.max_leverage}")
        if self.liquidation_buffer < 0:
            raise ConfigError(f"liquidation_buffer must be >= 0, got {self.liquidation_buffer}")


@dataclass(frozen=True)
class OrderbookConfig:
    depth: int = 50
    imbalance_levels: int = 20
    wall_levels: int = 50
    wall_factor: float = 3.0
    imbalance_threshold: float = 0.3      # enter a directional state above this
    imbalance_exit_threshold: float = 0.15  # leave it only below this (hysteresis)
    imbalance_smooth_alpha: float = 0.3   # EMA weight on the newest snapshot
    min_side_levels: int = 5              # need >= this many levels per side
    thin_book_ratio: float = 0.02         # weaker side must hold >= this of total

    def __post_init__(self):
        if self.depth <= 0 or self.imbalance_levels <= 0 or self.wall_levels <= 0:
            raise ConfigError("orderbook depth/levels must be positive")
        if self.wall_factor <= 1:
            raise ConfigError(f"wall_factor must be > 1, got {self.wall_factor}")
        if not (0 < self.imbalance_threshold < 1):
            raise ConfigError(
                f"imbalance_threshold must be in (0,1), got {self.imbalance_threshold}"
            )
        if not (0 <= self.imbalance_exit_threshold < self.imbalance_threshold):
            raise ConfigError(
                f"imbalance_exit_threshold must be in [0, imbalance_threshold), "
                f"got {self.imbalance_exit_threshold}"
            )
        if not (0 < self.imbalance_smooth_alpha <= 1):
            raise ConfigError(
                f"imbalance_smooth_alpha must be in (0,1], got {self.imbalance_smooth_alpha}"
            )
        if self.min_side_levels < 1:
            raise ConfigError(f"min_side_levels must be >= 1, got {self.min_side_levels}")
        if not (0 <= self.thin_book_ratio < 0.5):
            raise ConfigError(f"thin_book_ratio must be in [0, 0.5), got {self.thin_book_ratio}")


@dataclass(frozen=True)
class NewsConfig:
    news_days: int = 30
    sentiment_backend: str = field(default_factory=lambda: _env_str("AI_TRADER_SENTIMENT_BACKEND", "finbert"))
    sentiment_shift_days: int = 1
    sentiment_quantize: bool = field(default_factory=lambda: _env_bool("AI_TRADER_SENTIMENT_QUANTIZE", True))

    def __post_init__(self):
        if self.news_days <= 0:
            raise ConfigError(f"news_days must be positive, got {self.news_days}")
        if self.sentiment_backend not in ("finbert", "distilbert", "vader"):
            raise ConfigError(
                f"sentiment_backend must be 'finbert', 'distilbert' or 'vader', "
                f"got {self.sentiment_backend!r}"
            )
        if self.sentiment_shift_days < 0:
            raise ConfigError("sentiment_shift_days must be >= 0")


@dataclass(frozen=True)
class ModelConfig:
    pred_horizon: int = 5
    seq_len: int = 30
    xgb_params: Dict = field(default_factory=lambda: {
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "gamma": 0.1,
    })
    xgb_early_stopping_rounds: int = 20
    lstm_params: Dict = field(default_factory=lambda: {
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "epochs": 50,
        "batch_size": 32,
        "lr": 0.001,
        "weight_decay": 1e-5,
        "patience": 10,
    })

    def __post_init__(self):
        if self.pred_horizon <= 0:
            raise ConfigError(f"pred_horizon must be positive, got {self.pred_horizon}")
        if self.seq_len <= 0:
            raise ConfigError(f"seq_len must be positive, got {self.seq_len}")
        if self.seq_len >= self.pred_horizon * 100:
            pass
        req_xgb = {"n_estimators", "max_depth", "learning_rate", "subsample"}
        if not req_xgb.issubset(self.xgb_params):
            raise ConfigError(f"xgb_params missing keys: {req_xgb - set(self.xgb_params)}")
        if self.xgb_early_stopping_rounds < 0:
            raise ConfigError("xgb_early_stopping_rounds must be >= 0")
        req_lstm = {"hidden_dim", "num_layers", "dropout", "epochs", "batch_size", "lr",
                     "weight_decay", "patience"}
        if not req_lstm.issubset(self.lstm_params):
            raise ConfigError(f"lstm_params missing keys: {req_lstm - set(self.lstm_params)}")
        if self.lstm_params["epochs"] <= 0:
            raise ConfigError("lstm_params['epochs'] must be positive")
        if self.lstm_params["weight_decay"] < 0:
            raise ConfigError("lstm_params['weight_decay'] must be >= 0")
        if self.lstm_params["patience"] <= 0:
            raise ConfigError("lstm_params['patience'] must be positive")



@dataclass(frozen=True)
class OptimizationConfig:
    linear_n_trials: int = field(default_factory=lambda: _env_int("AI_TRADER_LINEAR_N_TRIALS", 20))
    xgb_n_trials: int = field(default_factory=lambda: _env_int("AI_TRADER_XGB_N_TRIALS", 20))
    lstm_n_trials: int = field(default_factory=lambda: _env_int("AI_TRADER_LSTM_N_TRIALS", 5))
    lstm_opt_epochs: int = field(default_factory=lambda: _env_int("AI_TRADER_LSTM_OPT_EPOCHS", 10))
    optuna_seed: int = 42
    feature_selection_top_k: int = 10
    ensemble_weighted: bool = field(default_factory=lambda: _env_bool("AI_TRADER_WEIGHTED_ENSEMBLE", True))
    ensemble_weight_metric: str = "val_r2"

    def __post_init__(self):
        for name, val in [("linear_n_trials", self.linear_n_trials),
                          ("xgb_n_trials", self.xgb_n_trials),
                          ("lstm_n_trials", self.lstm_n_trials),
                          ("lstm_opt_epochs", self.lstm_opt_epochs),
                          ("feature_selection_top_k", self.feature_selection_top_k)]:
            if val < 1:
                raise ConfigError(f"{name} must be >= 1, got {val}")
        if self.ensemble_weight_metric not in ("val_r2", "val_rmse"):
            raise ConfigError(
                f"ensemble_weight_metric must be 'val_r2' or 'val_rmse', "
                f"got {self.ensemble_weight_metric!r}"
            )


DATA = DataConfig()
FEATURES = FeatureConfig()
CROSS_ASSET = CrossAssetConfig()
REFERENCE = ReferenceSeriesConfig()
INDICATORS = IndicatorConfig()
SPLIT = SplitConfig()
SIGNAL = SignalConfig()
BACKTEST = BacktestConfig()
RISK = RiskConfig()
ORDERBOOK = OrderbookConfig()
NEWS = NewsConfig()
MODEL = ModelConfig()
OPTIMIZATION = OptimizationConfig()


def validate_all():
    for cfg in (DATA, FEATURES, CROSS_ASSET, REFERENCE, INDICATORS, SPLIT, SIGNAL,
                BACKTEST, RISK, ORDERBOOK, NEWS, MODEL, OPTIMIZATION):
        cfg.__post_init__()
    return True


validate_all()


