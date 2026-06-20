"""Expected-value optimization: the validation shield's primary purpose.

The question this module answers: is there ANY (pt_mult, sl_mult, horizon,
min_proba) combination for which the classifier produces signals with positive
expected value, after honest purged walk-forward estimation and Sidak-corrected
holdout judgment?

Honest expected answer on synthetic GARCH data: no. That is the point. The
shield was built so that a "no EV found" result is a provably rigorous
conclusion, not hand-waving. If the same code runs on real data and returns a
deflated p < 0.05, there is a genuine edge.

Protocol
--------
1. Split the data chronologically: trainval | holdout (the holdout is sealed).
2. Grid-search (pt_mult, sl_mult, horizon, min_proba) on trainval via purged
   walk-forward. Every combination is logged to a TrialLog.
3. Freeze the best-EV combination.
4. Refit on all trainval with those params and judge ONCE on the holdout:
   - Gate: deflated binomial p-value on per-trade win/loss.
   - Descriptive: per-trade Sharpe of realised forward returns.

EV definition
-------------
For each directional signal (predicted BUY or SELL) with confidence >= min_proba:
  - Win: model's predicted direction matches the sign of the h-period forward return.
  - Gain: the absolute forward return when winning.
  - Loss: the absolute forward return when losing.

  EV = win_rate * mean(gains) - (1 - win_rate) * mean(losses) - cost_per_trade

Using realised forward returns rather than theoretical barrier distances means
the EV number absorbs the actual path uncertainty (gaps, mean reversion after
the barrier, etc.) rather than the idealised bang-bang payout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from data.indicators import get_features
from data.labeling import triple_barrier_labels, CLASS_TARGET_COL, LABEL_BUY, LABEL_SELL
from engine.classification_trainer import build_classifier
from engine.purging import purge_window
from engine.validation_protocol import (
    TrialLog,
    binomial_p_value,
    deflated_p_value,
    expected_value,
)
from engine.walkforward import make_windows
from utils.config import FEATURES, MODEL
from utils.types import FeatureSpec

FloatArray = npt.NDArray[np.float64]

# ------------------------------------------------------------------ #
# Grid parameter type                                                  #
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class EVParams:
    """One point in the barrier geometry + threshold search space."""
    pt_mult: float
    sl_mult: float
    horizon: int
    min_proba: float = 0.5

    def __post_init__(self) -> None:
        if self.pt_mult <= 0:
            raise ValueError(f"pt_mult must be > 0, got {self.pt_mult}")
        if self.sl_mult <= 0:
            raise ValueError(f"sl_mult must be > 0, got {self.sl_mult}")
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if not (0.0 < self.min_proba <= 1.0):
            raise ValueError(f"min_proba must be in (0,1], got {self.min_proba}")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "pt_mult": self.pt_mult,
            "sl_mult": self.sl_mult,
            "horizon": self.horizon,
            "min_proba": self.min_proba,
        }


def default_grid() -> List[EVParams]:
    """Conservative default search space.  96 combinations.

    Kept small on purpose — the grid feeds the Sidak correction, so each extra
    trial inflates the deflated p-value. A denser grid needs more holdout
    trades to recover significance.
    """
    pts = (1.0, 1.5, 2.0, 2.5)
    sls = (0.75, 1.0, 1.5)
    horizons = (3, 5, 10)
    thresholds = (0.50, 0.55, 0.60, 0.65)
    return [EVParams(pt, sl, h, mp)
            for pt in pts for sl in sls for h in horizons for mp in thresholds]


# ------------------------------------------------------------------ #
# Per-window EV helper                                                 #
# ------------------------------------------------------------------ #

def _compute_ev(preds: npt.ArrayLike, proba: npt.ArrayLike,
                fwd: npt.ArrayLike, min_proba: float,
                cost: float) -> Dict[str, float]:
    """Compute EV and trade statistics for one test segment.

    Returns a dict with n_trades / wins / avg_gain / avg_loss / ev / sharpe.
    Trades with HOLD prediction or low confidence are excluded.
    """
    preds_a = np.asarray(preds, dtype=np.int64)
    proba_a = np.asarray(proba, dtype=np.float64)        # shape (n, 3)
    fwd_a = np.asarray(fwd, dtype=np.float64)
    n = min(len(preds_a), len(fwd_a))
    preds_a = preds_a[:n]
    proba_a = proba_a[:n]
    fwd_a = fwd_a[:n]

    max_proba = proba_a.max(axis=1)
    is_directional = (preds_a == LABEL_BUY) | (preds_a == LABEL_SELL)
    mask = is_directional & (max_proba >= min_proba) & np.isfinite(fwd_a)

    if mask.sum() == 0:
        return {"n_trades": 0, "wins": 0, "avg_gain": 0.0, "avg_loss": 0.0,
                "ev": -cost, "sharpe": 0.0}

    p = preds_a[mask]
    f = fwd_a[mask]

    # Win: model's directional prediction matches the sign of the forward return.
    long_win = (p == LABEL_BUY) & (f > 0)
    short_win = (p == LABEL_SELL) & (f < 0)
    win_mask = long_win | short_win

    gains = np.abs(f[win_mask])
    losses = np.abs(f[~win_mask])

    n_trades = int(mask.sum())
    wins = int(win_mask.sum())
    avg_gain = float(gains.mean()) if len(gains) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0

    wr = wins / n_trades
    ev = expected_value(wr, avg_gain, avg_loss, cost)

    # Per-trade signed return: +|fwd| when winning, -|fwd| when losing.
    # np.where needs same-shape operands — build via multiplier.
    sign_mult = np.where(win_mask, 1.0, -1.0)
    signed = np.abs(f) * sign_mult - cost
    sharpe = (float(signed.mean()) / float(signed.std(ddof=1))
               if len(signed) >= 2 and signed.std(ddof=1) > 0 else 0.0)

    return {
        "n_trades": n_trades,
        "wins": wins,
        "avg_gain": avg_gain,
        "avg_loss": avg_loss,
        "ev": ev,
        "sharpe": sharpe,
        "per_trade": signed.tolist(),
    }


# ------------------------------------------------------------------ #
# Walk-forward EV for one config                                       #
# ------------------------------------------------------------------ #

@dataclass
class EVWindowResult:
    """Aggregated trades and statistics over all walk-forward windows for one
    (pt_mult, sl_mult, horizon, min_proba) configuration."""
    params: EVParams
    n_windows: int = 0
    n_trades: int = 0
    wins: int = 0
    total_ev: float = 0.0
    per_trade_returns: List[float] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.wins / self.n_trades if self.n_trades else 0.0

    @property
    def mean_ev_per_trade(self) -> float:
        """Average EV per trade across windows (aggregate, not window-average)."""
        if not self.per_trade_returns:
            return 0.0
        return float(np.mean(self.per_trade_returns))

    @property
    def sharpe(self) -> float:
        if len(self.per_trade_returns) < 2:
            return 0.0
        arr = np.asarray(self.per_trade_returns)
        sd = float(arr.std(ddof=1))
        return float(arr.mean()) / sd if sd > 0 else 0.0


def ev_walkforward(df_engineered: pd.DataFrame, model_name: str,
                   params: EVParams, spec: Optional[FeatureSpec] = None,
                   train_size: int = 300, test_size: int = 60,
                   step: Optional[int] = None, cost: float = 0.0,
                   embargo: int = 0) -> EVWindowResult:
    """Walk-forward EV estimate for one (pt_mult, sl_mult, horizon, min_proba).

    Labels are computed with the supplied barrier geometry; the classifier fits
    on the purged train segment and scores the test segment. All label/feature
    computation uses only prices available at or before the bar being labelled;
    the forward look is contained in the label itself, which is excluded from
    the feature matrix -- the same discipline as the rest of the pipeline.
    """
    spec = spec or FeatureSpec(news=False, micro=True, cross_asset=False)
    h = params.horizon

    df = df_engineered.copy()
    df[CLASS_TARGET_COL] = triple_barrier_labels(
        df, h=h, pt_mult=params.pt_mult, sl_mult=params.sl_mult,
    )

    # Forward returns are computed from close prices, not from the labels.
    fwd = df[FEATURES.close_col].pct_change(h).shift(-h)
    df = df.assign(_fwd=fwd)
    df = df.dropna(subset=[CLASS_TARGET_COL, "_fwd"]).reset_index(drop=True)

    X = get_features(df, spec=spec).values
    y = df[CLASS_TARGET_COL].values.astype(int)
    fwd_arr = df["_fwd"].values

    wins_obj = make_windows(len(X), train_size, test_size, step)
    result = EVWindowResult(params=params)

    for w in wins_obj:
        a, b = purge_window(w["train"], w["test"], h, embargo)
        c, d = w["test"]
        if b - a < 4:
            continue
        mdl = build_classifier(model_name)
        mdl.train(X[a:b], y[a:b])
        preds = mdl.predict(X[c:d])
        proba = mdl.predict_proba(X[c:d])

        stats = _compute_ev(preds, proba, fwd_arr[c:d], params.min_proba, cost)
        if stats["n_trades"] == 0:
            result.n_windows += 1
            continue

        # Retrieve actual per-trade returns directly
        per_trade = stats["per_trade"]

        result.n_windows += 1
        result.n_trades += int(stats["n_trades"])
        result.wins += int(stats["wins"])
        result.total_ev += stats["ev"] * stats["n_trades"]
        result.per_trade_returns.extend(per_trade)

    return result


# ------------------------------------------------------------------ #
# Grid scan                                                            #
# ------------------------------------------------------------------ #

@dataclass
class EVScanResult:
    """Outcome of scanning the full grid on trainval."""
    log: TrialLog
    best_params: EVParams
    best_ev: float
    window_results: List[EVWindowResult] = field(default_factory=list)


def ev_scan(df_engineered: pd.DataFrame, model_name: str = "xgb_clf",
            grid: Optional[Sequence[EVParams]] = None,
            spec: Optional[FeatureSpec] = None,
            train_size: int = 300, test_size: int = 60,
            step: Optional[int] = None, cost: float = 0.0,
            embargo: int = 0) -> EVScanResult:
    """Scan a grid of barrier + threshold configs on trainval, log to TrialLog.

    Score per config: mean EV per trade pooled across all walk-forward windows.
    Favours configs that produce at least one trade. If no config produces any
    trade, the result's best_ev will be <= 0 and best_params will be grid[0].
    """
    if grid is None:
        grid = default_grid()
    if not grid:
        raise ValueError("grid must contain at least one EVParams")
    spec = spec or FeatureSpec(news=False, micro=True, cross_asset=False)

    log = TrialLog()
    best_ev = -np.inf
    best_params = grid[0]
    window_results: List[EVWindowResult] = []

    for params in grid:
        wf = ev_walkforward(df_engineered, model_name, params, spec=spec,
                            train_size=train_size, test_size=test_size,
                            step=step, cost=cost, embargo=embargo)
        ev = wf.mean_ev_per_trade
        log.record(params.as_dict(), ev)
        window_results.append(wf)
        if wf.n_trades > 0 and ev > best_ev:
            best_ev = ev
            best_params = params

    return EVScanResult(log=log, best_params=best_params, best_ev=best_ev,
                        window_results=window_results)


# ------------------------------------------------------------------ #
# Holdout judgment                                                     #
# ------------------------------------------------------------------ #

@dataclass
class EVHoldoutStats:
    """Statistics from running the best config on the sealed holdout."""
    n_trades: int
    wins: int
    win_rate: float
    avg_gain: float
    avg_loss: float
    ev: float
    sharpe: float
    raw_p: float
    deflated_p: float

    @property
    def significant(self) -> bool:
        return self.deflated_p < 0.05

    @property
    def positive_ev(self) -> bool:
        return self.ev > 0 and self.significant


def ev_judge_holdout(trainval_df: pd.DataFrame, holdout_df: pd.DataFrame,
                     params: EVParams, model_name: str = "xgb_clf",
                     n_trials: int = 1, spec: Optional[FeatureSpec] = None,
                     cost: float = 0.0) -> EVHoldoutStats:
    """Refit the best config on all of trainval, then evaluate once on holdout.

    This is the single sealed judgment. `n_trials` is the TrialLog count from
    the grid scan so the Sidak correction is applied correctly.
    """
    spec = spec or FeatureSpec(news=False, micro=True, cross_asset=False)
    h = params.horizon

    def _prep(df: pd.DataFrame) -> Tuple[FloatArray, FloatArray, FloatArray]:
        d = df.copy()
        d[CLASS_TARGET_COL] = triple_barrier_labels(
            d, h=h, pt_mult=params.pt_mult, sl_mult=params.sl_mult,
        )
        fwd = d[FEATURES.close_col].pct_change(h).shift(-h)
        d = d.assign(_fwd=fwd).dropna(subset=[CLASS_TARGET_COL, "_fwd"])
        X = get_features(d, spec=spec).values
        y = d[CLASS_TARGET_COL].values.astype(int)
        fwd_a = d["_fwd"].values
        return X, y, fwd_a

    X_tv, y_tv, _ = _prep(trainval_df)
    X_ho, _y_ho, fwd_ho = _prep(holdout_df)

    if len(X_tv) < 4 or len(X_ho) == 0:
        return EVHoldoutStats(n_trades=0, wins=0, win_rate=0.0, avg_gain=0.0,
                               avg_loss=0.0, ev=-cost, sharpe=0.0,
                               raw_p=1.0, deflated_p=1.0)

    mdl = build_classifier(model_name)
    mdl.train(X_tv, y_tv)
    preds_ho = mdl.predict(X_ho)
    proba_ho = mdl.predict_proba(X_ho)

    stats = _compute_ev(preds_ho, proba_ho, fwd_ho, params.min_proba, cost)
    n = int(stats["n_trades"])
    w = int(stats["wins"])
    raw_p = binomial_p_value(w, n)
    def_p = deflated_p_value(w, n, n_trials)

    return EVHoldoutStats(
        n_trades=n, wins=w,
        win_rate=w / n if n > 0 else 0.0,
        avg_gain=stats["avg_gain"], avg_loss=stats["avg_loss"],
        ev=stats["ev"], sharpe=stats["sharpe"],
        raw_p=raw_p, deflated_p=def_p,
    )


# ------------------------------------------------------------------ #
# Full protocol                                                        #
# ------------------------------------------------------------------ #

@dataclass
class EVOptimizationReport:
    """Complete EV optimisation outcome: grid scan + holdout judgment."""
    best_params: EVParams
    scan: EVScanResult
    holdout: EVHoldoutStats
    trainval_rows: int
    holdout_rows: int

    def summary(self) -> str:
        lines = [
            "=== EV Optimisation Report ===",
            f"  Grid size:        {self.scan.log.n_trials} configs",
            f"  Best config:      pt={self.best_params.pt_mult}  "
            f"sl={self.best_params.sl_mult}  "
            f"h={self.best_params.horizon}  "
            f"min_p={self.best_params.min_proba}",
            f"  Best trainval EV: {self.scan.best_ev:+.5f} per trade",
            "",
            "  --- Holdout judgment (unsealed once) ---",
            f"  Trades:           {self.holdout.n_trades}",
            f"  Win rate:         {self.holdout.win_rate:.1%}  "
            f"({self.holdout.wins}/{self.holdout.n_trades})",
            f"  Avg gain:         {self.holdout.avg_gain:+.4f}",
            f"  Avg loss:         {self.holdout.avg_loss:+.4f}",
            f"  EV per trade:     {self.holdout.ev:+.5f}",
            f"  Sharpe (trades):  {self.holdout.sharpe:+.3f}",
            f"  Raw p-value:      {self.holdout.raw_p:.4f}",
            f"  Deflated p:       {self.holdout.deflated_p:.4f}  "
            f"(Sidak, {self.scan.log.n_trials} trials)",
            f"  Significant:      {self.holdout.significant}",
            f"  Positive EV:      {self.holdout.positive_ev}",
        ]
        return "\n".join(lines)


def ev_optimize(df_engineered: pd.DataFrame, model_name: str = "xgb_clf",
                grid: Optional[Sequence[EVParams]] = None,
                spec: Optional[FeatureSpec] = None,
                train_size: int = 300, test_size: int = 60,
                step: Optional[int] = None, cost: float = 0.0,
                embargo: int = 0,
                trainval_frac: float = 0.8) -> EVOptimizationReport:
    """Full EV optimisation protocol on an engineered DataFrame.

    Parameters
    ----------
    df_engineered : engineered OHLCV (with atr_pct and feature columns).
    model_name    : classifier name recognised by build_classifier.
    grid          : list of EVParams to search; defaults to default_grid().
    spec          : feature spec; defaults to technical + micro features.
    train_size    : bars per walk-forward train segment.
    test_size     : bars per walk-forward test segment.
    step          : walk-forward step size; defaults to test_size.
    cost          : round-trip cost fraction (commission + slippage).
    embargo       : extra purge buffer beyond label horizon.
    trainval_frac : fraction of data reserved for trainval (sealed holdout
                    gets the remaining 1 - trainval_frac).
    """
    if not (0 < trainval_frac < 1):
        raise ValueError(f"trainval_frac must be in (0,1), got {trainval_frac}")

    n = len(df_engineered)
    tv_end = int(n * trainval_frac)
    if tv_end < train_size + test_size:
        raise ValueError(
            f"trainval block ({tv_end} rows) is smaller than "
            f"train_size + test_size ({train_size + test_size})"
        )
    if tv_end >= n:
        raise ValueError("trainval_frac leaves no rows for the holdout")

    trainval_df = df_engineered.iloc[:tv_end].copy()
    holdout_df = df_engineered.iloc[tv_end:].copy()

    if grid is None:
        grid = default_grid()

    scan = ev_scan(trainval_df, model_name, grid=list(grid), spec=spec,
                   train_size=train_size, test_size=test_size, step=step,
                   cost=cost, embargo=embargo)

    holdout_stats = ev_judge_holdout(
        trainval_df, holdout_df, scan.best_params, model_name=model_name,
        n_trials=scan.log.n_trials, spec=spec, cost=cost,
    )

    return EVOptimizationReport(
        best_params=scan.best_params,
        scan=scan,
        holdout=holdout_stats,
        trainval_rows=tv_end,
        holdout_rows=n - tv_end,
    )
