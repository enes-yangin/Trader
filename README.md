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

**Portfolio Tracking (Manual, No Auto-Trading)**
- The AI never places orders. It only produces signals; all real trades are
  entered by the user after the fact (symbol, side, quantity, price, fee, date).
- FIFO cost-basis matching computes realized PnL per closed lot and
  unrealized PnL on open positions using live prices.
- Monthly realized PnL summary (win rate, average return, best/worst trade).
- Signal-alignment report: compares the AI signal recorded at the time of
  entry against the realized outcome, to retrospectively evaluate signal quality.
- Ledger persists to a local JSON file -- no exchange API keys required or stored.

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
│   ├── fetcher.py            Binance (ccxt) OHLCV fetching + sample fallback
│   ├── indicators.py         Technical indicators + target engineering
│   ├── microstructure.py     Amihud, Kyle's lambda, Roll spread, VWAP
│   ├── cross_asset.py        BTC-dominance / ETH-BTC cross-asset features
│   ├── orderbook.py          Order-book imbalance + wall detection
│   ├── sentiment.py          FinBERT / VADER news sentiment
│   ├── news_features.py      News sentiment aggregated into features
│   ├── labeling.py           3-class direction + triple-barrier labeling
│   ├── dataset.py            Cache-aware dataset orchestration
│   └── ...                   cache_policy, enrichment, store, validation
│
├── models/
│   ├── base_model.py         Abstract regressor: train/predict/evaluate/signal
│   ├── linear_model.py       Ridge + StandardScaler
│   ├── xgb_model.py          XGBoost regressor (regularized + early stopping)
│   ├── lstm_model.py         PyTorch LSTM (early stopping + weight decay)
│   ├── base_classifier.py    Abstract classifier: predict_proba + metrics
│   └── classifier_models.py  LogisticRegression + XGBClassifier (direction)
│
├── engine/
│   ├── trainer.py                  Regression training pipeline
│   ├── classification_trainer.py   Classification training pipeline
│   ├── predictor.py                Ensemble signal generation
│   ├── backtester.py               Historical simulation with costs/metrics
│   ├── walkforward.py              Rolling-window eval + purging
│   ├── classification_walkforward.py  Hit-rate distribution across windows
│   ├── purging.py                  López de Prado purge/embargo helper
│   ├── validation_protocol.py      Sealed holdout + trial log + deflated p
│   ├── feature_ablation.py         Per-feature-family contribution study
│   ├── risk.py                     Kelly sizing + stop-loss
│   └── ...                         ensemble, optimizer, feature_selection
│
├── portfolio/
│   ├── position.py           Trade/Position dataclasses + FIFO matching
│   ├── ledger.py             JSON-backed trade ledger (CRUD)
│   └── pnl.py                Realized/unrealized PnL, monthly summary
│
├── ui/
│   ├── app.py                Main Tkinter window + threading
│   ├── chart.py              Matplotlib candlestick + overlays
│   ├── panels.py             Signal card, model details, metrics table
│   └── portfolio_panel.py    Manual trade ledger window
│
├── utils/
│   ├── config.py             Dataclass config groups (symbols, params, thresholds)
│   ├── types.py              TypedDicts + FeatureSpec + Bundle/SplitDict
│   ├── exceptions.py         Exception hierarchy
│   └── logger.py             Logging setup
│
├── scripts/
│   ├── evaluate_r2.py             Regression R² on real/synthetic data
│   ├── evaluate_classification.py Directional hit rate + walk-forward
│   └── evaluate_ablation.py       Feature-family ablation study
│
├── tests/                    240+ tests (pytest), incl. leakage/shield guards
│
├── main.py                   Entry point
├── bootstrap.py              Sets up venv, installs deps, launches the app
├── runtime_check.py          Environment diagnostic (Python, GPU, network)
├── install.sh / install.bat  One-command setup for end users
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- Internet connection (for initial dependency download and live market data)

### Quickstart (recommended)

A bootstrap script sets up an isolated environment, installs all dependencies,
runs a diagnostic check, and launches the app -- no manual venv steps needed.

```bash
git clone https://github.com/yourusername/ai-trader.git
cd ai-trader

# Linux / macOS
./install.sh

# Windows
install.bat
```

First run downloads dependencies (including PyTorch) into a local `.venv/`
and can take several minutes. Subsequent runs reuse the same environment and
start immediately. Useful flags:

```bash
./install.sh --check-only      # run diagnostics only, don't launch
./install.sh --skip-install     # skip dependency install (faster re-run)
./install.sh --force-reinstall  # reinstall all dependencies
```

### Manual Setup

```bash
python -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows

pip install -r requirements.txt
python runtime_check.py           # optional: verify environment
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

1. Select a symbol from the dropdown (**BTC/USDT**, **ETH/USDT**, **BNB/USDT**, **SOL/USDT**, and others)
2. Click **▶ Train & Predict** — trains all 3 models, displays signal
3. Click **Backtest** — simulates trades on validation data, shows equity curve
4. Click **Portfolio** — record your own real trades (FIFO PnL, no auto-trading)

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

### Out-of-Sample R² — What to Expect on Real Data

Predicting multi-day-ahead returns is close to predicting a near-efficient
market, so honest out-of-sample R² sits near zero and is often slightly
negative. This is expected, not a bug. The evaluation harness makes three
checks explicit:

- **Out-of-sample R² ≈ 0 (or mildly negative)** is the realistic regime for
  5-day return prediction. A test R² of 0.8 would almost certainly indicate a
  look-ahead leak rather than genuine predictive skill.
- **Train-vs-test R² gap** measures overfitting. Tree and LSTM models easily
  reach train R² > 0.9 while test R² stays near zero — fitting noise, not signal.
- **Directional accuracy near 0.50** (coin-flip). Values like 0.70+ on a test
  set are a red flag for leakage, not an edge.

**Limiting overfit on tree/sequence models.** XGBoost uses shallow trees
(`max_depth=3`), L1/L2 penalties, `min_child_weight`, and early stopping on
validation RMSE; LSTM uses weight decay and early stopping on validation loss
with the best-checkpoint restored. The effect: XGBoost train R² typically
drops from ~0.9 (unregularized) to ~0.0-0.05, landing close to test R², while
test R² itself is essentially unchanged -- the model stops memorizing noise
it could never have generalized from in the first place.

Run it yourself:

```bash
# On real fetched data (requires exchange access)
PYTHONPATH=. python scripts/evaluate_r2.py --symbols BTC/USDT ETH/USDT --years 4

# Offline, on realistic synthetic data (volatility clustering + fat tails)
PYTHONPATH=. python scripts/evaluate_r2.py --allow-sample
```

The `tests/test_r2_evaluation.py` suite encodes these expectations as
regression tests against realistic synthetic returns, so a future change that
silently introduces look-ahead bias (and an implausibly high R²) will fail CI.

### Classification Head — Predicting Direction Instead of Magnitude

Regression fights to predict an exact (mostly unpredictable) return. Reframing
the task as 3-class direction (SELL / HOLD / BUY) lets a classifier optimize
for what actually matters — which way price moves — and is measured directly by
directional hit rate rather than R². Two classifiers are provided
(`LogisticRegression`, `XGBClassifier`) with ATR-normalized labeling.

```bash
# Single-split directional hit rate
PYTHONPATH=. python scripts/evaluate_classification.py

# Walk-forward: hit-rate distribution across rolling windows
PYTHONPATH=. python scripts/evaluate_classification.py --walkforward
```

**Why walk-forward matters.** A single train/test split can look promising by
luck. Evaluating across rolling windows exposes whether an apparent edge is
consistent or occasional. On realistic synthetic data the classifiers land at
~0.50 mean hit rate with windows-above-50% hovering around 40-60% — i.e. no
stable edge, which is the honest result for near-efficient returns. The value
of this harness is not the score itself but catching the difference between
"52% on one lucky split" and "52% consistently across 11 windows." The
`tests/test_classification_walkforward.py` suite guards against an
implausibly high mean hit rate that would signal a look-ahead leak.

**Purging and embargoing.** Forward-looking labels (triple barrier, h-bar
returns) create a subtle leak even when no feature crosses the train/test
boundary: a training label at bar `t` depends on bars up to `t+h`, which may
fall inside the test window. Following López de Prado, the walk-forward purges
trailing training rows whose label horizon overlaps the test segment (with an
optional `embargo` buffer for residual autocorrelation). It defaults to the
label horizon and is verified by `tests/test_purging.py`, including an
end-to-end check that the last surviving training label cannot reference any
bar inside the test period.

### Feature Ablation — Does Each Feature Family Earn Its Place?

More features is not better. Each family widens the surface a model can overfit
to, so a family is only justified if it lifts walk-forward hit rate *without*
inflating its variance. The ablation harness runs the classification
walk-forward for each feature-group combination and compares them:

```bash
PYTHONPATH=. python scripts/evaluate_ablation.py
```

On realistic synthetic data the microstructure family (Amihud illiquidity,
Kyle's lambda, Roll spread, VWAP distance) helps on some symbols and hurts on
others, and tends to raise hit-rate variance — i.e. it adds noise more than
signal here, which is the honest result for near-random returns. This is the
disciplined version of "better features": rather than fabricating data streams
the project can't actually source (order-book depth, funding rate, on-chain),
it measures whether the families it *does* have are pulling their weight. The
same harness would quantify a genuinely new alpha source if one were wired in.

### Triple Barrier Labeling

The fixed-horizon label ("return exactly h bars later") ignores the path: a
trade that hits its profit target on day 2 and reverses by day 5 is scored on
the day-5 snapshot, not the win it actually was. Triple barrier labeling
(López de Prado) fixes this by placing three barriers from each entry — an
upper (profit-take) at `+pt_mult × ATR%`, a lower (stop-loss) at
`−sl_mult × ATR%`, and a vertical (time) barrier `h` bars out — and labeling by
which is touched first: upper → BUY, lower → SELL, neither → HOLD.

```bash
PYTHONPATH=. python scripts/evaluate_classification.py --walkforward --labeling triple_barrier
```

On realistic synthetic (near-random) data, triple barrier does not reliably
beat fixed-horizon hit rate — there is no path structure to exploit, so both
sit near coin-flip. Its advantage shows on real data, in **label quality**: the
labels speak the same language as actual trading (profit-take / stop-loss /
timeout) and reflect the path rather than an arbitrary snapshot. Barriers are
evaluated from future highs/lows, so the label is forward-looking by
construction and is used only as a prediction target, never as a model input —
`tests/test_triple_barrier.py` includes a perturbation test proving past bars
cannot change a label (no look-ahead leak into features).

---

## Architecture

```
                    ┌──────────────────┐
                    │  Binance (ccxt)  │
                    │  OHLCV / samples │
                    └────────┬─────────┘
                             │ OHLCV
                    ┌────────▼─────────┐
                    │  Feature Engine  │
                    │  technical +     │
                    │  microstructure  │
                    │  + cross-asset   │
                    └────────┬─────────┘
                             │ Feature Matrix
              ┌──────────────┴──────────────┐
              ▼                             ▼
   ┌────────────────────┐        ┌────────────────────┐
   │  Regression head   │        │ Classification head│
   │ Ridge·XGB·LSTM     │        │ Logistic·XGBClf    │
   │ predicts return    │        │ predicts direction │
   └─────────┬──────────┘        └─────────┬──────────┘
             │                             │
             │   labels: fixed-horizon  /  triple barrier
             └──────────────┬──────────────┘
                            │ signals
                   ┌────────▼──────────────────────┐
                   │     Validation Shield         │
                   │  sealed holdout · trial log   │
                   │  deflated-p · purge/embargo   │
                   │  walk-forward (hit-rate dist) │
                   └────────┬──────────────────────┘
                            │ honest metrics
              ┌─────────────┴─────────────┐
              ▼                           ▼
     ┌────────────────┐         ┌─────────────────┐
     │  Dark Tkinter  │         │   Backtester    │
     │  GUI + manual  │         │  PnL · Sharpe   │
     │  trade ledger  │         │  cost-aware     │
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
| `SIGNAL_THRESH_BUY` | 0.02 | Fallback BUY threshold (+2%), used when dynamic thresholding is off or ATR is unavailable |
| `SIGNAL_THRESH_SELL` | -0.02 | Fallback SELL threshold (-2%) |
| `SIGNAL_USE_DYNAMIC_THRESH` | True | Scale BUY/SELL thresholds by recent ATR% instead of using a fixed value |
| `SIGNAL_ATR_MULT` | 0.5 | Dynamic threshold = `ATR_MULT * ATR%`; widens in high-volatility regimes, narrows in low-volatility regimes |
| `KELLY_FRACTION` | 0.25 | Quarter-Kelly position sizing (fraction of full Kelly) |
| `MAX_POSITION_PCT` | 0.2 | Absolute cap on capital risked per trade (20%) |
| `XGB_PARAMS` | depth=3, n=200, reg_alpha=0.1, reg_lambda=1.0, gamma=0.1, min_child_weight=5 | XGBoost hyperparameters (regularized to limit overfitting) |
| `MODEL.xgb_early_stopping_rounds` | 20 | Stop boosting if validation RMSE doesn't improve for this many rounds |
| `LSTM_PARAMS` | hidden=64, layers=2, weight_decay=1e-5, patience=10 | LSTM hyperparameters; `patience` controls early stopping on validation loss |

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
