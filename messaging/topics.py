"""
ZeroMQ PUB/SUB topic constants for TraderAI v2 message bus.

Topic naming convention:
  {domain}.{subdomain}.{action}[.{symbol}]

Symbol-specific topics append the trading symbol as a suffix
so subscribers can filter with ZMQ prefix matching (e.g. "data.live.price"
matches all symbols; "data.live.price.BTC/USDT" matches only BTC).
"""

# ── Data Pipeline ───────────────────────────────────────────────────
LIVE_PRICE       = "data.live.price"          # + .{symbol}
LIVE_ORDERBOOK   = "data.live.orderbook"      # + .{symbol}
DATASET_READY    = "data.dataset.ready"       # + .{symbol}
FETCH_PROGRESS   = "data.fetch.progress"       # per-symbol completion count

# ── ML Pipeline ─────────────────────────────────────────────────────
TRAINING_STARTED  = "ml.training.started"      # + .{symbol}
TRAINING_COMPLETE = "ml.training.complete"     # + .{symbol}
PREDICTION_READY  = "ml.prediction.ready"      # + .{symbol}

# ── Backtest ────────────────────────────────────────────────────────
BACKTEST_COMPLETE = "backtest.complete"        # + .{symbol}

# ── Paper Trader ────────────────────────────────────────────────────
PAPER_ACCURACY    = "paper.accuracy"            # accuracy update from paper trader

# ── UI / System ─────────────────────────────────────────────────────
STATUS            = "ui.status"                 # status bar text + level
ERROR             = "system.error"              # error messages
SHUTDOWN          = "system.shutdown"           # graceful shutdown signal
