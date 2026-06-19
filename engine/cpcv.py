"""Combinatorial Purged Cross-Validation (CPCV).

Walk-forward gives a single backtest path: one train->test sweep, one Sharpe,
one hit-rate. That single number is fragile -- it could be lucky. CPCV (Lopez
de Prado, *Advances in Financial Machine Learning*, ch. 12) instead splits the
data into N contiguous groups and tests on every size-k combination of groups,
training on the rest. Reassembling the held-out predictions yields

    phi = C(N-1, k-1)

distinct full-length out-of-sample *paths*, so a strategy is judged by a
*distribution* of Sharpe / hit-rate rather than one point estimate. This is
what gives CPCV a lower Probability of Backtest Overfitting than walk-forward.

Purging vs. walk-forward
------------------------
In walk-forward the test block is always after the train block, so only the
train *tail* can leak into test (purged via engine.purging). In CPCV a test
group can sit in the *middle* of the data, so training samples on BOTH sides
need care:
  - Purge: a train label spanning [t, t+h] that overlaps a test group leaks
    future test prices into training -> drop train bars in [start-h, end).
  - Embargo: train bars right after a test group are serially correlated with
    it -> drop train bars in [end, end+embargo).

The same label-overlap logic as engine.purging, generalised to interior test
intervals on both sides.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import numpy.typing as npt
import pandas as pd

from data.indicators import get_features
from models.base_model import BaseModel
from utils.config import FEATURES, MODEL
from utils.exceptions import InsufficientDataError
from utils.types import FeatureSpec

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


# ------------------------------------------------------------------ #
# Group / combination geometry                                        #
# ------------------------------------------------------------------ #

def group_bounds(n: int, n_groups: int) -> List[Tuple[int, int]]:
    """Split [0, n) into n_groups contiguous, near-equal half-open intervals."""
    if n_groups < 2:
        raise ValueError(f"n_groups must be >= 2, got {n_groups}")
    if n_groups > n:
        raise ValueError(f"n_groups ({n_groups}) cannot exceed n ({n})")
    edges = np.linspace(0, n, n_groups + 1).astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_groups)]


def n_paths(n_groups: int, k_test: int) -> int:
    """Number of reconstructable backtest paths: C(N-1, k-1)."""
    if not (1 <= k_test < n_groups):
        raise ValueError(f"need 1 <= k_test < n_groups, got k={k_test}, N={n_groups}")
    return math.comb(n_groups - 1, k_test - 1)


def cpcv_combinations(n_groups: int, k_test: int) -> List[Tuple[int, ...]]:
    """All size-k combinations of group indices used as the test set."""
    if not (1 <= k_test < n_groups):
        raise ValueError(f"need 1 <= k_test < n_groups, got k={k_test}, N={n_groups}")
    return list(itertools.combinations(range(n_groups), k_test))


def train_indices(n: int, test_intervals: List[Tuple[int, int]],
                  label_horizon: int, embargo: int = 0) -> IntArray:
    """Indices usable for training given the test intervals, after purge+embargo.

    A bar is excluded if it is inside a test interval, if its forward label
    window would overlap a test interval (purge), or if it falls in the embargo
    window immediately after a test interval.
    """
    if label_horizon < 0:
        raise ValueError(f"label_horizon must be >= 0, got {label_horizon}")
    if embargo < 0:
        raise ValueError(f"embargo must be >= 0, got {embargo}")
    test_mask = np.zeros(n, dtype=bool)
    drop_mask = np.zeros(n, dtype=bool)
    for (s, e) in test_intervals:
        test_mask[s:e] = True
        lo = max(0, s - label_horizon)        # label-overlap purge (left)
        drop_mask[lo:e] = True
        hi = min(n, e + embargo)              # embargo (right)
        drop_mask[e:hi] = True
    usable = ~test_mask & ~drop_mask
    return np.where(usable)[0].astype(np.int64)


# ------------------------------------------------------------------ #
# Result containers                                                   #
# ------------------------------------------------------------------ #

@dataclass
class PathMetrics:
    """Performance of one reassembled out-of-sample path."""
    dir_acc: float
    r2: float
    sharpe: float
    n: int


@dataclass
class CPCVResult:
    """Distribution of metrics across all CPCV paths."""
    paths: List[PathMetrics]
    n_groups: int
    k_test: int
    n_combinations: int

    @property
    def n_paths(self) -> int:
        return len(self.paths)

    def _arr(self, field_name: str) -> FloatArray:
        return np.asarray([getattr(p, field_name) for p in self.paths],
                          dtype=np.float64)

    def summary(self) -> Dict[str, float]:
        if not self.paths:
            return {}
        da = self._arr("dir_acc")
        sh = self._arr("sharpe")
        r2 = self._arr("r2")
        return {
            "n_paths": float(len(self.paths)),
            "mean_dir_acc": float(da.mean()),
            "std_dir_acc": float(da.std(ddof=1)) if len(da) > 1 else 0.0,
            "min_dir_acc": float(da.min()),
            "max_dir_acc": float(da.max()),
            "pct_dir_above_50": float((da > 0.5).mean() * 100.0),
            "mean_sharpe": float(sh.mean()),
            "std_sharpe": float(sh.std(ddof=1)) if len(sh) > 1 else 0.0,
            "pct_sharpe_positive": float((sh > 0).mean() * 100.0),
            "mean_r2": float(r2.mean()),
        }

    def format_report(self, name: str = "CPCV") -> str:
        s = self.summary()
        if not s:
            return f"  {name}: no paths"
        return "\n".join([
            f"  [{name}] N={self.n_groups} k={self.k_test} -> "
            f"{self.n_combinations} combos, {int(s['n_paths'])} paths",
            f"    Dir Acc:  {s['mean_dir_acc']:.1%} "
            f"(std {s['std_dir_acc']:.1%}, "
            f"range {s['min_dir_acc']:.1%}-{s['max_dir_acc']:.1%})",
            f"    Dir>50%:  {s['pct_dir_above_50']:.0f}% of paths",
            f"    Sharpe:   {s['mean_sharpe']:+.3f} "
            f"(std {s['std_sharpe']:.3f}), "
            f"{s['pct_sharpe_positive']:.0f}% of paths > 0",
            f"    Mean R2:  {s['mean_r2']:+.4f}",
        ])


# ------------------------------------------------------------------ #
# Path metric helper                                                  #
# ------------------------------------------------------------------ #

def _path_metrics(preds: FloatArray, y: FloatArray) -> PathMetrics:
    n = min(len(preds), len(y))
    p = np.asarray(preds[:n], dtype=np.float64)
    t = np.asarray(y[:n], dtype=np.float64)
    if n == 0:
        return PathMetrics(dir_acc=0.0, r2=0.0, sharpe=0.0, n=0)

    dir_acc = float(np.mean(np.sign(p) == np.sign(t)))

    # A naive sign-following strategy: position = sign(prediction).
    pnl = np.sign(p) * t
    sd = float(pnl.std(ddof=1)) if n > 1 else 0.0
    sharpe = float(pnl.mean() / sd) if sd > 0 else 0.0

    ss_tot = float(((t - t.mean()) ** 2).sum())
    ss_res = float(((t - p) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return PathMetrics(dir_acc=dir_acc, r2=r2, sharpe=sharpe, n=n)


# ------------------------------------------------------------------ #
# CPCV runner                                                         #
# ------------------------------------------------------------------ #

def run_cpcv(df: pd.DataFrame, model_cls: Type[BaseModel],
             n_groups: int = 6, k_test: int = 2,
             spec: Optional[FeatureSpec] = None,
             purge: Optional[int] = None, embargo: int = 0,
             min_train: int = 30,
             model_kw: Optional[Dict[str, Any]] = None) -> CPCVResult:
    """Run CPCV on an engineered DataFrame and return the path metric distribution.

    For every size-k group combination, the model trains on the purged/embargoed
    complement and predicts each held-out group. Predictions are then reassembled
    into phi = C(N-1, k-1) full-length paths, each scored independently.

    `purge` defaults to the model prediction horizon (the label look-ahead).
    """
    if FEATURES.target_col not in df.columns:
        raise InsufficientDataError(
            f"run_cpcv expects an engineered DataFrame with the "
            f"'{FEATURES.target_col}' column. Call data.indicators.engineer first."
        )
    spec = spec or FeatureSpec.default()
    model_kw = model_kw or {}
    label_horizon = MODEL.pred_horizon if purge is None else purge

    X = get_features(df, spec=spec).values
    y = df[FEATURES.target_col].values.astype(np.float64)
    n = len(X)

    bounds = group_bounds(n, n_groups)
    combos = cpcv_combinations(n_groups, k_test)
    phi = n_paths(n_groups, k_test)

    # For each group: the list of prediction arrays, one per combination that
    # tests it, in combination order. Index p into these lists builds path p.
    preds_by_group: Dict[int, List[FloatArray]] = {g: [] for g in range(n_groups)}

    for combo in combos:
        test_intervals = [bounds[g] for g in combo]
        tr_idx = train_indices(n, test_intervals, label_horizon, embargo)
        if len(tr_idx) < min_train:
            raise InsufficientDataError(
                f"combination {combo} leaves only {len(tr_idx)} train rows "
                f"(< min_train={min_train}); use fewer groups, smaller k, or "
                f"more data."
            )
        mdl = model_cls(**model_kw)
        mdl.train(X[tr_idx], y[tr_idx])
        for g in combo:
            s, e = bounds[g]
            preds_by_group[g].append(np.asarray(mdl.predict(X[s:e]),
                                                 dtype=np.float64))

    # Each group must have been tested exactly phi times for paths to be square.
    for g in range(n_groups):
        if len(preds_by_group[g]) != phi:
            raise RuntimeError(
                f"group {g} tested {len(preds_by_group[g])} times, expected {phi}"
            )

    paths: List[PathMetrics] = []
    for p in range(phi):
        preds_parts: List[FloatArray] = []
        y_parts: List[FloatArray] = []
        for g in range(n_groups):
            s, e = bounds[g]
            pg = preds_by_group[g][p]
            yg = y[s:e]
            m = min(len(pg), len(yg))
            preds_parts.append(pg[:m])
            y_parts.append(yg[:m])
        preds_full = np.concatenate(preds_parts) if preds_parts else np.array([])
        y_full = np.concatenate(y_parts) if y_parts else np.array([])
        paths.append(_path_metrics(preds_full, y_full))

    return CPCVResult(paths=paths, n_groups=n_groups, k_test=k_test,
                      n_combinations=len(combos))


def run_cpcv_model(df: pd.DataFrame, model_name: str, **kw: Any) -> CPCVResult:
    """Convenience wrapper mirroring walkforward.run_model."""
    from models.linear_model import LinearModel
    from models.xgb_model import XGBModel
    from models.lstm_model import LSTMModel
    mp: Dict[str, Type[BaseModel]] = {
        "linear": LinearModel, "xgboost": XGBModel, "lstm": LSTMModel,
    }
    cls = mp[model_name.lower()]
    mk = {"epochs": MODEL.lstm_params["epochs"]} if model_name == "lstm" else {}
    return run_cpcv(df, cls, model_kw=mk, **kw)
