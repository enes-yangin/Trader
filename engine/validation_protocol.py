from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
import pandas as pd
from scipy import stats
from engine.purging import purged_train_end
from utils.exceptions import AITraderError


class HoldoutSealError(AITraderError):
    pass


@dataclass
class DataSplit:
    """Chronological three-way split. Holdout indices are kept private and can
    only be read by going through unseal(), which records that the seal was
    broken. The point is to make accidental holdout peeking impossible to do
    silently: any code path that touches the holdout leaves a trace.

    label_horizon (and embargo) purge trailing rows from `train` and `val`
    whose forward-looking labels would overlap the next segment (val and
    holdout respectively) -- see engine.purging for the rationale. Defaults
    to 0 (no purging) for generic use; callers working with horizon-h labels
    should pass label_horizon=h.
    """
    df: pd.DataFrame
    train_end: int
    val_end: int
    label_horizon: int = 0
    embargo: int = 0
    _holdout_unsealed: bool = field(default=False, init=False)
    _unseal_count: int = field(default=0, init=False)

    @property
    def train(self) -> pd.DataFrame:
        end = purged_train_end(0, self.train_end, self.train_end,
                               self.label_horizon, self.embargo)
        return self.df.iloc[:end]

    @property
    def val(self) -> pd.DataFrame:
        end = purged_train_end(self.train_end, self.val_end, self.val_end,
                               self.label_horizon, self.embargo)
        return self.df.iloc[self.train_end:end]

    @property
    def trainval(self) -> pd.DataFrame:
        end = purged_train_end(0, self.val_end, self.val_end,
                               self.label_horizon, self.embargo)
        return self.df.iloc[:end]

    def unseal_holdout(self, reason: str = "") -> pd.DataFrame:
        self._holdout_unsealed = True
        self._unseal_count += 1
        return self.df.iloc[self.val_end:]

    @property
    def holdout_was_touched(self) -> bool:
        return self._holdout_unsealed

    @property
    def unseal_count(self) -> int:
        return self._unseal_count

    def sizes(self) -> Dict[str, int]:
        return {
            "train": self.train_end,
            "val": self.val_end - self.train_end,
            "holdout": len(self.df) - self.val_end,
        }

    def n_purged(self) -> Dict[str, int]:
        return {
            "train": self.train_end - len(self.train),
            "val": (self.val_end - self.train_end) - len(self.val),
        }


def make_split(df: pd.DataFrame, train_frac: float = 0.6,
               val_frac: float = 0.2, label_horizon: int = 0,
               embargo: int = 0) -> DataSplit:
    if not (0 < train_frac < 1) or not (0 < val_frac < 1):
        raise AITraderError("train_frac and val_frac must be in (0,1)")
    if train_frac + val_frac >= 1.0:
        raise AITraderError(
            f"train_frac + val_frac must leave room for holdout, "
            f"got {train_frac}+{val_frac}={train_frac + val_frac}"
        )
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    if train_end == 0 or val_end == train_end or val_end == n:
        raise AITraderError(f"split produced an empty segment for n={n}")
    return DataSplit(df=df, train_end=train_end, val_end=val_end,
                     label_horizon=label_horizon, embargo=embargo)


@dataclass
class TrialLog:
    """Counts how many parameter combinations were evaluated on validation.
    Multiple testing inflates the best observed score: try enough knobs and
    one looks good by chance. This count feeds the significance deflation so
    we never read a lucky max as a real edge."""
    n_trials: int = 0
    records: List[Dict[str, Any]] = field(default_factory=list)

    def record(self, params: Dict[str, Any], score: float) -> None:
        self.n_trials += 1
        self.records.append({"params": dict(params), "score": score})

    def best(self) -> Optional[Dict[str, Any]]:
        if not self.records:
            return None
        return max(self.records, key=lambda r: r["score"])

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)


def binomial_p_value(wins: int, n: int, p_null: float = 0.5) -> float:
    """One-sided p: probability of >= wins successes under a fair coin."""
    if n == 0:
        return 1.0
    return float(stats.binomtest(wins, n, p_null, alternative="greater").pvalue)


def deflated_p_value(wins: int, n: int, n_trials: int, p_null: float = 0.5) -> float:
    """Sidak-style correction for having selected the best of n_trials configs.
    A raw p of 0.04 across 20 trials is not significant; this reflects that."""
    raw = binomial_p_value(wins, n, p_null)
    if n_trials <= 1:
        return raw
    return float(1.0 - (1.0 - raw) ** n_trials)


def breakeven_winrate(rr: float) -> float:
    """Win rate needed for zero EV given a reward:risk ratio."""
    if rr <= 0:
        return 1.0
    return 1.0 / (1.0 + rr)


def expected_value(win_rate: float, avg_gain: float, avg_loss: float,
                   cost: float = 0.0) -> float:
    return win_rate * avg_gain - (1.0 - win_rate) * avg_loss - cost
