from dataclasses import dataclass
from typing import TypedDict, Literal, Dict, List, Any, Optional
import numpy as np
import pandas as pd

SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class FeatureSpec:
    """Single object carrying which optional feature families are active.

    Replaces the (with_news, with_micro, with_cross_asset) boolean triplet
    that was previously threaded individually through ~20 function signatures
    across data/indicators.py, engine/trainer.py, engine/predictor.py,
    engine/walkforward.py and engine/ensemble.py.
    """
    news: bool = False
    micro: bool = False
    cross_asset: bool = False

    @classmethod
    def from_bools(cls, with_news: bool = False, with_micro: bool = False,
                    with_cross_asset: bool = False) -> "FeatureSpec":
        return cls(news=with_news, micro=with_micro, cross_asset=with_cross_asset)

    @classmethod
    def default(cls) -> "FeatureSpec":
        from utils.config import FEATURES
        return cls(news=FEATURES.use_news, micro=FEATURES.use_micro, cross_asset=FEATURES.use_cross_asset)

    def feature_columns(self) -> List[str]:
        from utils.config import FEATURES
        cols = list(FEATURES.feature_cols)
        if self.micro:
            cols += list(FEATURES.micro_feature_cols)
        if self.cross_asset:
            cols += list(FEATURES.cross_asset_feature_cols)
        if self.news:
            cols += list(FEATURES.news_feature_cols)
        return cols

    @property
    def n_features(self) -> int:
        return len(self.feature_columns())


class _DictLike:
    """Mixin giving dataclasses a Mapping-like interface.

    Lets code written against the old dict/TypedDict-based Bundle and
    SplitDict (e.g. ``bundle["df"]``, ``sp.get("with_news")``) keep working
    unchanged, while construction now goes through the dataclass __init__
    -- which validates field names and required arguments at creation time
    instead of failing later with a KeyError deep in some consumer.
    """

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if key not in getattr(self, "__dataclass_fields__", {}):
            raise KeyError(key)
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key) and getattr(self, key) is not None

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self) -> List[str]:
        return list(getattr(self, "__dataclass_fields__", {}).keys())



class MetricsDict(TypedDict):
    mse: float
    rmse: float
    mae: float
    r2: float


class ClassMetrics(TypedDict):
    accuracy: float
    balanced_accuracy: float
    f1_macro: float
    precision_buy: float
    precision_sell: float
    n_buy_pred: int
    n_sell_pred: int
    n_hold_pred: int


class TrainResult(TypedDict, total=False):
    train: MetricsDict
    val: MetricsDict
    test: MetricsDict
    history: Dict[str, List[float]]
    best_iteration: int
    best_epoch: int


class ClassTrainResult(TypedDict, total=False):
    train: ClassMetrics
    val: ClassMetrics
    test: ClassMetrics
    best_iteration: int


@dataclass
class SplitDict(_DictLike):
    X_tr: np.ndarray
    y_tr: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    idx_tr: pd.Index
    idx_val: pd.Index
    idx_test: pd.Index
    df: pd.DataFrame
    i_tr: int
    i_va: int
    split_idx: int
    spec: FeatureSpec
    with_news: bool = False
    with_micro: bool = False
    with_cross_asset: bool = False


class ModelResult(TypedDict):
    model: Any
    metrics: TrainResult


class TrainSingleResult(ModelResult):
    split: SplitDict
    symbol: str


@dataclass
class Bundle(_DictLike):
    results: Dict[str, ModelResult]
    split: SplitDict
    symbol: str
    df: pd.DataFrame
    sample: bool
    with_news: bool = False
    with_micro: bool = False
    with_cross_asset: bool = False
    spec: Optional[FeatureSpec] = None
    news_analysis: Optional[str] = None


# Backward-compatible alias: news_analysis is now an optional field on Bundle
# itself rather than a separate total=False subclass.
BundleWithNews = Bundle


class SignalDict(TypedDict):
    signal: Literal["BUY", "HOLD", "SELL"]
    confidence: float
    predicted_return: float


class ModelSignal(SignalDict, total=False):
    model: str
    last_close: float
    symbol: str
    weight: float


class EnsembleSignal(TypedDict):
    consensus: Literal["BUY", "HOLD", "SELL"]
    avg_confidence: float
    avg_predicted_return: float
    votes: Dict[str, float]
    details: List[ModelSignal]
    last_close: float
    symbol: str


class WeightedEnsembleSignal(EnsembleSignal, total=False):
    weights: Dict[str, float]


class BacktestMetrics(TypedDict):
    total_return: float
    final_equity: float
    n_trades: int
    n_closed: int
    win_rate: float
    avg_pnl: float
    max_win: float
    max_loss: float
    sharpe: float
    max_drawdown: float
    total_costs: float
    costs_pct_of_capital: float
    n_stop_losses: int


class BacktestResult(TypedDict):
    metrics: BacktestMetrics
    trades: pd.DataFrame
    equity: pd.DataFrame
    model: str
    which: SplitName


class OrderBookSignal(TypedDict):
    symbol: str
    state: str
    imbalance: float
    confidence: float
    rel_spread: float
    mid_price: float
    microprice: float
    micro_skew: float
    bid_walls: int
    ask_walls: int
    walls: Dict[str, list]


class WalkForwardSummary(TypedDict, total=False):
    n_windows: int
    mean_r2: float
    std_r2: float
    mean_dir_acc: float
    std_dir_acc: float
    pct_r2_positive: float
    pct_dir_above_50: float


class WalkForwardResult(TypedDict):
    folds: pd.DataFrame
    summary: WalkForwardSummary
    n_windows: int
