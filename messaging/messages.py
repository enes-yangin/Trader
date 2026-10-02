"""
Message payload schemas for TraderAI v2 inter-service communication.

Every message that crosses the pub/sub bus uses a dataclass
so consumers can destructure safely without raw dict key errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class LivePricePayload:
    """Published on data.live.price.{symbol} after a live price fetch."""
    symbol: str
    close: float
    timestamp: str                  # ISO-8601
    df_json: str = ""               # full OHLCV DataFrame as JSON (optional)


@dataclass
class LiveOrderbookPayload:
    """Published on data.live.orderbook.{symbol} after an orderbook snapshot."""
    symbol: str
    imbalance: float = 0.0
    rel_spread: float = 0.0
    micro_skew: float = 0.0
    bid_walls: int = 0
    ask_walls: int = 0


@dataclass
class DatasetReadyPayload:
    """Published on data.dataset.ready.{symbol} after dataset.build() completes."""
    symbol: str
    row_count: int = 0
    source: str = "cache"           # "cache" | "fresh" | "fallback"


@dataclass
class TrainingStartedPayload:
    """Published on ml.training.started.{symbol} when training begins."""
    symbol: str
    feature_families: Dict[str, bool] = field(default_factory=dict)


@dataclass
class TrainingCompletePayload:
    """Published on ml.training.complete.{symbol} when training finishes."""
    symbol: str
    signal: Dict[str, Any] = field(default_factory=dict)
    # bundle is too large to serialize — stored in AppState instead;
    # this payload carries the signal (small) as an immediate UI push.
    # Consumers that need the full bundle call state.get_bundle(symbol).


@dataclass
class BacktestCompletePayload:
    """Published on backtest.complete.{symbol} after a backtest run."""
    symbol: str
    portfolio_mode: bool = False
    metrics_json: str = ""          # serialized metrics dict (for display)


@dataclass
class StatusPayload:
    """Published on ui.status for status-bar updates."""
    text: str
    level: str = "info"             # "info" | "warning" | "error"


@dataclass
class ErrorPayload:
    """Published on system.error for error propagation."""
    source: str                     # module/function name
    message: str
    detail: str = ""
