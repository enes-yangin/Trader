<div align="center">

# AI Trader

### ML-Powered Crypto & Stock Prediction Engine

A fully local desktop application that fetches real-time market data,
engineers technical indicators, trains three ML models side-by-side,
and produces actionable **BUY / HOLD / SELL** signals with confidence scores.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-LSTM-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Regressor-006400?style=for-the-badge)
![scikit-learn](https://img.shields.io/badge/sklearn-Ridge-f7931e?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

</div>

---

## Screenshot

<div align="center">

> Replace with actual screenshot after first run:
> `python main.py` → Train & Predict → Backtest → take screenshot

![App Screenshot](assets/screenshot.png)

</div>

---

## Features

**Data Pipeline**
- Crypto data from Binance via `ccxt` (BTC/USDT, ETH/USDT)
- Stock data from Yahoo Finance via `yfinance` (AAPL, MSFT)
- Unified OHLCV interface with automatic source detection

**Feature Engineering — 12 Technical Indicators**
- Momentum: RSI (14)
- Trend: MACD (12/26/9), EMA-20, SMA-50
- Volatility: Bollinger Bands (20,2), ATR (14)
- Volume: 20-period Volume SMA

**Three ML Models Trained & Compared**

| Model | Type | Key Properties |
|-------|------|---------------|
| Ridge Regression | Linear baseline | StandardScaler preprocessing, L2 regularization |
| XGBoost | Gradient boosting | 200 estimators, depth 5, feature importance ranking |
| LSTM | Deep learning | 2-layer, 64 hidden units, 30-step sequences, PyTorch |

**Signal Generation**
- Per-model BUY / HOLD / SELL with confidence percentage
- Ensemble voting across all three models
- Configurable thresholds (default ±2% predicted return)

**Backtesting Engine**
- Simulated trading on historical validation data
- Metrics: total return, win rate, Sharpe ratio, max drawdown
- Per-trade log with entry/exit prices and PnL
- Equity curve visualization

**Dark-Themed Desktop GUI**
- Candlestick chart with Bollinger Bands and EMA overlay
- Volume bars, RSI oscillator panel
- Real-time signal card with vote breakdown
- Backtest results table with color-coded metrics

---

## Tech Stack

| Layer | Libraries | Purpose |
|-------|-----------|---------|
| Data | `ccxt` `yfinance` `pandas` `numpy` | Multi-source OHLCV fetching |
| Features | `ta` | 12 technical indicators |
| Models | `scikit-learn` `xgboost` `torch` | Ridge, XGBoost, LSTM |
| Visualization | `matplotlib` | Candlestick, equity curve, RSI |
| Interface | `tkinter` | Dark-themed desktop GUI |

---

## Project Structure

```
ai_trader/
│
├── data/
│   ├── fetcher.py            Binance (ccxt) + Yahoo Finance data fetching
│   └── indicators.py         12 technical indicators + target engineering
│
├── models/
│   ├── base_model.py         Abstract base: train / predict / evaluate / signal
│   ├── linear_model.py       Ridge Regression + StandardScaler
│   ├── xgb_model.py          XGBoost Regressor + feature importance
│   └── lstm_model.py         PyTorch LSTM with sliding-window sequences
│
├── engine/
│   ├── trainer.py            Training pipeline (single model or all)
│   ├── predictor.py          Ensemble signal generation + formatting
│   └── backtester.py         Historical simulation with full metrics
│
├── ui/
│   ├── app.py                Main Tkinter window + threading
│   ├── chart.py              Matplotlib candlestick + overlays
│   └── panels.py             Signal card, model details, metrics table
│
├── utils/
│   └── config.py             Symbols, hyperparameters, thresholds
│
├── main.py                   Entry point
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Internet connection (for initial data download)

### Setup

```bash
git clone https://github.com/yourusername/ai-trader.git
cd ai-trader

python -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### GPU Support (Optional)

The LSTM model automatically detects CUDA. For GPU acceleration:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Usage

### GUI Mode

```bash
python main.py
```

1. Select a symbol from the dropdown (**BTC/USDT**, **ETH/USDT**, **AAPL**, **MSFT**)
2. Click **▶ Train & Predict** — trains all 3 models, displays signal
3. Click **Backtest** — simulates trades on validation data, shows equity curve

### Programmatic Mode

```python
from engine.trainer import train_all, compare
from engine.predictor import predict_from_bundle, format_signal
from engine.backtester import run_all, format_report

bundle = train_all("BTC/USDT")
print(compare(bundle))

signal = predict_from_bundle(bundle)
print(format_signal(signal))

bt = run_all(bundle)
print(format_report(bt))
```

---

## Backtest Results

### Model Training Metrics (BTC/USDT — 500 days)

| Model | Train R² | Val R² | Train RMSE | Val RMSE |
|-------|----------|--------|------------|----------|
| Ridge Regression | 0.1154 | 0.0948 | 0.0355 | 0.0410 |
| XGBoost | 0.9849 | -0.1774 | 0.0046 | 0.0468 |
| LSTM | 0.3293 | 0.2241 | 0.0312 | 0.0359 |

### Backtest Performance ($10,000 initial capital)

| Model | Return | Final Equity | Trades | Win Rate | Sharpe | Max Drawdown |
|-------|--------|-------------|--------|----------|--------|--------------|
| Ridge Regression | +0.00% | $10,000 | 0 | — | 0.00 | 0.00% |
| XGBoost | **+4.85%** | **$10,485** | 3 | **100%** | 0.68 | -14.31% |
| LSTM | +4.10% | $10,410 | 1 | — | **0.71** | -10.15% |

### Signal Output Example

```
========================================
  BTC/USDT  |  HOLD  (50.2%)
  Predicted Return: -1.3830%
  Last Close: 122.73
  Votes: BUY=0 HOLD=2 SELL=1
========================================
    LinearRegression      HOLD    0.7%  ret=-0.019863
    XGBoost               HOLD   96.8%  ret=-0.000435
    LSTM                  SELL   53.0%  ret=-0.021193
```

> Results above use synthetic data for demonstration.
> Real market data produces different results depending on the symbol and time period.

---

## Architecture

```
                    ┌──────────────────┐
                    │   Binance (ccxt) │
                    │  Yahoo (yfinance)│
                    └────────┬─────────┘
                             │ OHLCV
                    ┌────────▼─────────┐
                    │  Feature Engine  │
                    │  12 indicators   │
                    │  + target column │
                    └────────┬─────────┘
                             │ Feature Matrix
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │  Ridge   │  │ XGBoost  │  │   LSTM   │
        │Regression│  │Regressor │  │ (PyTorch)│
        └────┬─────┘  └────┬─────┘  └────┬─────┘
             │              │              │
             └──────────────┼──────────────┘
                            │ Predictions
                   ┌────────▼─────────┐
                   │ Ensemble Voting  │
                   │ BUY / HOLD / SELL│
                   │ + Confidence %   │
                   └────────┬─────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼                           ▼
     ┌────────────────┐         ┌─────────────────┐
     │  Dark-themed   │         │   Backtester    │
     │  Tkinter GUI   │         │  PnL · Sharpe   │
     │  Candlestick   │         │  Equity Curve   │
     └────────────────┘         └─────────────────┘
```

---

## Configuration

All parameters are centralized in `utils/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PRED_HORIZON` | 5 | Prediction horizon (days) |
| `TRAIN_RATIO` | 0.8 | Train/validation split |
| `SEQ_LEN` | 30 | LSTM sequence length |
| `SIGNAL_THRESH_BUY` | 0.02 | BUY threshold (+2%) |
| `SIGNAL_THRESH_SELL` | -0.02 | SELL threshold (-2%) |
| `XGB_PARAMS` | depth=5, n=200 | XGBoost hyperparameters |
| `LSTM_PARAMS` | hidden=64, layers=2 | LSTM hyperparameters |

---

## Disclaimer

This project is for **educational and portfolio demonstration purposes only**.
It is not financial advice. Do not use these signals for real trading
without thorough validation and risk management.

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Models are trained locally. No external AI API used.**

</div>
