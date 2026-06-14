CRYPTO_SYMBOLS = ["BTC/USDT", "ETH/USDT"]
STOCK_SYMBOLS = ["AAPL", "MSFT"]

EXCHANGE_ID = "binance"
TIMEFRAME = "1d"
OHLCV_LIMIT = 500

STOCK_PERIOD = "2y"
STOCK_INTERVAL = "1d"

TARGET_COL = "target"
CLOSE_COL = "close"
DATE_COL = "date"

FEATURE_COLS = [
    "rsi", "macd", "macd_signal", "macd_diff",
    "bb_high", "bb_mid", "bb_low", "bb_pband",
    "atr", "ema_20", "sma_50", "volume_sma"
]

NEWS_FEATURE_COLS = [
    "sentiment_avg", "sentiment_std", "news_volume", "sentiment_trend"
]

USE_NEWS = True
NEWS_DAYS = 30
SENTIMENT_BACKEND = "finbert"

PRED_HORIZON = 5
TRAIN_RATIO = 0.8
SEQ_LEN = 30

SIGNAL_THRESH_BUY = 0.02
SIGNAL_THRESH_SELL = -0.02

XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
}

LSTM_PARAMS = {
    "hidden_dim": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "epochs": 50,
    "batch_size": 32,
    "lr": 0.001,
}
