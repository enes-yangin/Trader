# TraderAI

**A leakage-safe ML research framework for crypto trading signals — and an honest negative result.**

TraderAI is a local desktop application (Python, Tkinter) that fetches daily crypto
OHLCV, engineers features, trains classifiers on triple-barrier labels, and
backtests the resulting signals with costs. Most of the engineering effort went
into the part that is easy to get wrong: **validation that does not fool
itself** (purged walk-forward, combinatorial purged CV, a sealed holdout, and
p-values deflated for the number of configurations tried).

> **Headline result: no tradable edge was found.** On five years of real
> Binance daily data across 8 symbols, the directional model is right 51.3% of
> the time, beats a "always predict the same direction" baseline on only 2 of 8
> symbols (by ~1 point), and is statistically significant on none. The
> pairs-trading (stat-arb) strategy passes the validation gate on 0 of 7 pairs.
> This is not a profitable trading bot, and nothing here is financial advice.

---

## What this project is — and is not

| It is | It is not |
|---|---|
| A working end-to-end pipeline: data → features → labels → models → signals → backtest → GUI | A profitable strategy |
| A set of leakage guards and overfitting controls, covered by tests | Validated in live or paper trading (13 predictions were logged, none were ever resolved) |
| A reproducible measurement that the tested strategies have no reliable edge | An HFT or order-book-driven system (daily bars only) |
| A manual trade ledger with FIFO PnL (the app never places orders) | Connected to any exchange account — public market data only |

---

## Results on real data

Everything in this section is produced by one script, on data fetched directly
from the exchange at run time (no cache, no synthetic fallback):

```bash
AI_TRADER_EXCHANGE=binance python scripts/evaluate_real.py
```

Raw output: [docs/results/real_data_eval_binance.json](docs/results/real_data_eval_binance.json),
[docs/results/real_data_eval_kraken.json](docs/results/real_data_eval_kraken.json).
Run on 2026-10-02. Re-running on a later date fetches newer bars, so numbers will shift.

### 1. Directional classifier

Setup: XGBoost classifier, triple-barrier labels (profit-take 2.5×ATR, stop
1.0×ATR, 20-bar time limit), 22 features (technical + microstructure proxies +
causal smoothing). Rolling walk-forward: 585-bar train, 175-bar test, 6 windows,
20-bar purge between train and test. Binance daily bars, 2021-10-04 → 2026-10-02
(1,825 per symbol); test span 2023-06-30 → 2026-05-14.

*Hit rate* = share of BUY/SELL signals whose direction matched the sign of the
20-bar forward return. *Best constant* = the hit rate of always predicting the
more common direction over the same test span, i.e. what zero skill scores.

| Symbol | Signals | Hit rate | Best constant | Windows > 50% | p (effective n) |
|---|---|---|---|---|---|
| BTC/USDT | 1,022 | 47.2% | 54.3% | 3 / 6 | 0.71 |
| ETH/USDT | 983 | 54.8% | 54.8% | 4 / 6 | 0.28 |
| BNB/USDT | 1,009 | 43.3% | 57.8% | 2 / 6 | 0.84 |
| SOL/USDT | 1,048 | 46.3% | 52.0% | 3 / 6 | 0.76 |
| XRP/USDT | 989 | 57.3% | 58.6% | 5 / 6 | 0.20 |
| ADA/USDT | 1,005 | 56.2% | 57.7% | 4 / 6 | 0.24 |
| DOGE/USDT | 1,016 | 55.7% | 54.5% | 5 / 6 | 0.24 |
| AVAX/USDT | 1,033 | 50.2% | 53.1% | 4 / 6 | 0.50 |
| **Pooled** | **8,105** | **51.3%** | | | |

How to read this:

- **The 55–57% rows are not an edge.** On XRP and ADA the market fell during
  most of the test span, so always saying SELL scores 58–59% — more than the
  model. Only ETH (+0.07 points) and DOGE (+1.2 points) beat the constant
  baseline.
- **The signals are not independent.** Each one is scored on a 20-bar forward
  return, so consecutive signals share 19 of 20 bars. A naive binomial test on
  ~1,000 signals reports p < 0.001 for some symbols; with the effective sample
  size (signals ÷ 20 ≈ 50) no symbol is below p = 0.19.
- **It is unstable.** On Kraken's shorter history (~2 years, test span
  2025-07 → 2026-07) BTC scores 56.7% and ETH 41.9% — the reverse of the
  ordering above. Pooled hit rate there is 51.3% as well, and one symbol of
  eight beats the constant baseline.

This evaluation scores signal direction only. It is not a PnL backtest and
includes no trading costs; costs can only make it worse.

### 2. Stat-arb (pairs vs BTC)

Setup: log prices, Engle-Granger cointegration test on the first 80%; z-score
entry/exit grid of 6 configurations selected by purged walk-forward on the first
80%; the chosen configuration is judged once on the sealed final 20%; p-value
deflated (Šidák) for the 6 trials; 0.3% round-trip cost.

| Pair (Binance, 5y) | Cointegrated at 5% | Holdout trades | Wins | Deflated p |
|---|---|---|---|---|
| ETH vs BTC | no | 1 | 1 | 0.98 |
| BNB vs BTC | no | 1 | 1 | 0.98 |
| SOL vs BTC | no | 1 | 1 | 0.98 |
| XRP vs BTC | no | 0 | 0 | 1.00 |
| ADA vs BTC | no | 1 | 0 | 1.00 |
| DOGE vs BTC | no | 1 | 0 | 1.00 |
| AVAX vs BTC | no | 1 | 1 | 0.98 |

No pair is cointegrated at the 5% level and no pair passes the gate
(deflated p < 0.05). With at most one trade per pair in the holdout there is
far too little evidence to say anything in either direction — the honest
reading is "untested at useful sample size", not "disproven".

### What was not measured

- No cost-inclusive portfolio PnL figure is reported. The backtester exists and
  runs, but an earlier set of backtest numbers was produced partly on a
  contaminated cache (see Known limitations) and has been removed rather than
  quoted.
- The optional feature families (news sentiment, cross-asset, macro reference
  series, order book, social) are not part of the evaluation above. None of them
  has been shown to improve out-of-sample results.

---

## Validation methodology

This is the part of the project intended to be reusable.

| Guard | What it prevents | Where |
|---|---|---|
| Purge + embargo | A training label at bar *t* depends on bars up to *t+h*; if those fall in the test window the model has seen the future. Rows whose label horizon overlaps the test segment are dropped. | `engine/purging.py` |
| Purged walk-forward | One lucky train/test split. Reports the distribution of hit rates over rolling windows. | `engine/walkforward.py`, `engine/classification_walkforward.py` |
| Combinatorial purged CV | Same, with test groups in the middle of the series (purged on both sides), giving several backtest paths instead of one. | `engine/cpcv.py` |
| Sealed holdout + trial log | Tuning on the test set. The final segment is only reachable through an explicit unseal call that counts every access, and every configuration tried is logged. | `engine/validation_protocol.py` |
| Deflated p-value | Picking the best of N configurations and reporting its raw p. Šidák correction for the number of trials. | `engine/validation_protocol.py` |
| Triple-barrier labels | Labels that ignore the path. Label = which is hit first: profit-take, stop-loss, or time limit (López de Prado). | `data/labeling.py` |
| Look-ahead tests | Silent leakage from future bars into features. Perturbation tests change future bars and assert past features and labels do not move. | `tests/test_lookahead_bias.py`, `tests/test_triple_barrier.py` |
| Feature lagging | Same-day leakage from external series (sentiment, macro) — shifted by at least one day. | `data/news_features.py`, `data/reference_series.py` |

In the GUI, "Train & Predict" computes a deflated p-value for the selected
model and shows a "NO EDGE" warning when it is above 0.05.

---

## Features

**Data**
- Daily OHLCV from any `ccxt` exchange (default Kraken; Binance recommended for
  history — see limitations). 8 symbols by default, configurable via `.env`.
- Parquet cache with freshness policy; data validation.

**Features (22 in the default evaluation set)**
- Technical: returns, RSI and RSI divergence, MACD, Bollinger %B, ATR, volume
  ratios, moving-average distances.
- Microstructure proxies computed from daily bars: Amihud illiquidity, Kyle's
  lambda, Roll spread, VWAP distance, volume delta.
- Causal smoothing: EMA slope and a one-sided Savitzky-Golay slope.
- Optional, off by default, not validated: news sentiment (FinBERT / VADER),
  cross-asset (ETH/BTC, market correlation), macro reference series (DXY,
  stablecoin supply, funding, open interest, Fear & Greed), social.

**Models**
- Classification (used by the GUI): XGBoost classifier on triple-barrier labels,
  with a confidence gate on predicted probability. Logistic regression is
  available as a baseline.
- Regression (scripts / library use): Ridge, XGBoost, PyTorch LSTM, with optional
  Optuna tuning.
- Stat-arb: Engle-Granger cointegration (ADF implemented in-repo), hedge ratio,
  z-score entry/exit.

**Backtesting and risk**
- Long-only single-asset and multi-asset backtests with commission, slippage,
  take-profit / stop-loss exits, fractional-Kelly sizing; optional leverage with
  an isolated-margin liquidation model; two-leg stat-arb backtest with funding
  cost. Seeded and reproducible.

**Application**
- Tkinter GUI: candlestick chart with indicators, signal card, validation card,
  backtest table and equity curve.
- Background services (data / training / backtest) communicate with the UI over
  a ZeroMQ in-process pub/sub bus.
- Manual portfolio ledger: FIFO cost basis, realised / unrealised PnL, monthly
  summary. The app never places orders and needs no exchange API keys.
- PyInstaller build script for a Windows executable.

---

## Architecture

```
  ccxt exchange (public OHLCV)
            │
   data/  fetch → validate → cache (parquet)
            │
   data/  feature engineering        ──►  labels (triple barrier / fixed horizon)
            │
   models/  XGBoost clf · Logistic · Ridge · XGBoost reg · LSTM
            │
   engine/  purged walk-forward · CPCV · sealed holdout · deflated p
            │
   engine/  backtester (costs, sizing, exits) · stat-arb
            │
   services/  data · training · backtest   ◄── ZeroMQ inproc pub/sub ──►   ui/ (Tkinter)
            │
   portfolio/  manual ledger, FIFO PnL
```

## Project structure

```
data/         fetching, caching, validation, feature families, labeling      (23 files)
models/       regressors and classifiers behind two small base classes       (7 files)
engine/       training, walk-forward, CPCV, validation protocol, backtester,
              stat-arb, risk, EV optimisation, Optuna tuning                 (22 files)
services/     background data / training / backtest services, app state      (5 files)
messaging/    ZeroMQ pub/sub bus, topics, message types                      (4 files)
portfolio/    positions, ledger, PnL                                         (4 files)
ui/           Tkinter app, chart, panels, portfolio window                   (5 files)
utils/        validated dataclass config, types, exchange pool, retry, logging
scripts/      evaluate_real.py (README numbers), evaluate_r2.py,
              evaluate_classification.py, evaluate_ablation.py,
              benchmark_accuracy.py, feature_edge_scan.py, build_exe.py, ...
tests/        418 pytest tests
docs/results/ raw output of scripts/evaluate_real.py
```

About 13,500 lines of application code and 5,300 lines of tests.

---

## Installation

Developed and run on Windows 11 with Python 3.14. `install.sh` exists for
Linux / macOS but has not been tested there.

```bash
# Windows
install.bat

# Linux / macOS (untested)
./install.sh
```

The bootstrap script creates `.venv/`, installs `requirements.txt` (including
PyTorch — several minutes on first run), runs an environment check and launches
the app. Manual setup:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Optional configuration goes in `.env` (copy `.env.example`): exchange id, symbol
list, and API keys for the optional feature families.

## Usage

```bash
python main.py                          # GUI
python scripts/evaluate_real.py         # real-data evaluation (README numbers)
python scripts/check_exchange.py        # which exchanges are reachable from your network
```

In the GUI: pick a symbol and a strategy ("ML Directional" or "Stat-Arb (vs BTC)"),
**Fetch Data**, **Train & Predict**, then **Backtest**. The "Synthetic" checkbox
allows generated sample data when the exchange is unreachable; results from it
are flagged as sample and mean nothing about real markets.

## Tests

```bash
python -m pytest tests/
```

418 tests, all passing on the last full run (2026-10-02, Windows 11, Python
3.14, about 9 minutes). Several tests attempt network fetches, and the run
overwrites some files in `datasets/` with synthetic data — delete that folder
afterwards (see Known limitations).

---

## Known limitations

Listed so nobody has to discover them the hard way.

**Evidence**
- No edge found (see Results). No forward / paper-trading track record.
- Daily bars only. Short selling is not modelled in the directional backtester
  (long-only), so SELL signals are never traded there.
- The 20-bar label horizon makes signals heavily overlapping; effective sample
  sizes are ~50 per symbol over three years of test data.

**Data**
- Kraken's public API returns only the most recent ~720 daily bars, so the
  default exchange yields ~2 years regardless of the 5-year setting. Use
  `AI_TRADER_EXCHANGE=binance` for full history.
- The on-disk dataset cache (`datasets/`) can be overwritten by the test suite
  with synthetic data that is not flagged as synthetic. `scripts/evaluate_real.py`
  bypasses the cache for this reason. Clear `datasets/` after running tests.
- Order-book features are always zero in training: snapshots can only be
  recorded live and there is no historical order-book data behind the daily bars.
- News sentiment covers only recent days via RSS unless a NewsAPI key is set.
  Macro "event surprise" is zero with the built-in static calendar.

**Modelling and simulation**
- Slippage, "market impact" and "latency" in the backtester are simple
  formulas of ATR and position size plus seeded noise — heuristics, not a
  model fitted to order-book data.
- The GUI's stat-arb path builds the spread from raw prices (cost scaled by the
  spread's gross notional), while `scripts/evaluate_real.py` uses log prices.
  The two can select different entry/exit thresholds; the numbers in this
  README are from the script.
- The validation card's "PBO" figure is `100 − (% of walk-forward windows above
  50% hit rate)`, not the formal Probability of Backtest Overfitting, and its
  "CPCV Sharpe" field is not computed from CPCV. CPCV itself is implemented and
  tested but only called from scripts, not from the GUI.
- The GUI's deflated p-value uses the naive signal count, without the
  overlapping-label correction applied in `scripts/evaluate_real.py`, so it is
  optimistic.
- Meta-labeling exists but is off by default and not validated out-of-sample.

**Engineering**
- `services/` and `messaging/` (the GUI's background layer) have no dedicated
  tests; coverage is concentrated in `data/`, `engine/`, `models/`, `portfolio/`.
- `ui/app.py` is a ~630-line class.
- Tested on one machine (Windows 11, Python 3.14). No CI.

---

## Development note

This project was built with substantial help from AI coding assistants
(Claude Code) for implementation, code review and test writing.

## Disclaimer

For education and research only. Not financial advice. The measured results
above show no reliable edge; do not trade real money on these signals.

## License

MIT — see [LICENSE](LICENSE).
