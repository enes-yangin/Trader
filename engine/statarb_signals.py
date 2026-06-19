"""Spread z-score trading signals for a cointegrated pair, and the honest
out-of-sample evaluation of those signals.

Signal rule (mean reversion of the spread):
  z > +entry  -> SHORT the spread (sell y, buy beta·x); it is stretched high.
  z < -entry  -> LONG  the spread (buy y, sell beta·x); it is stretched low.
  |z| < exit  -> FLAT (close); the spread has reverted toward equilibrium.

The discipline that makes the backtest believable lives in `fit_and_signal`
and `statarb_walkforward`: the hedge ratio and the z-score's mean/std are
estimated on TRAIN ONLY and then frozen to score the TEST segment. Trades are
held across bars, so their outcomes overlap forward in time -- exactly the
label structure that needs purging between train and test (see engine.purging)
and trial-count deflation when a grid of thresholds is searched (see
engine.validation_protocol).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import numpy.typing as npt

from engine.statarb import hedge_ratio, spread, zscore
from engine.purging import purge_window
from engine.walkforward import make_windows
from engine.validation_protocol import TrialLog, deflated_p_value, binomial_p_value

FloatArray = npt.NDArray[np.float64]


@dataclass
class StatArbResult:
    """Outcome of running the spread strategy over one test segment."""
    positions: FloatArray
    per_bar_pnl: FloatArray
    trades: List[float]
    beta: float
    alpha: float

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> int:
        return int(sum(1 for t in self.trades if t > 0))

    @property
    def win_rate(self) -> float:
        return self.wins / self.n_trades if self.trades else 0.0

    @property
    def total_pnl(self) -> float:
        return float(sum(self.trades))

    @property
    def sharpe(self) -> float:
        """Per-trade Sharpe (unannualised): mean/std of round-trip PnL. A
        descriptive companion to the win-rate gate, not the gate itself."""
        if self.n_trades < 2:
            return 0.0
        arr = np.asarray(self.trades, dtype=np.float64)
        sd = float(arr.std(ddof=1))
        if sd == 0.0:
            return 0.0
        return float(arr.mean()) / sd


def generate_positions(z: npt.ArrayLike, entry: float = 2.0,
                       exit: float = 0.5) -> FloatArray:
    """Stateful position series in {-1, 0, +1} on the spread, from a z-score.

    position[t] is decided using z[t] only (no look-ahead); PnL is then earned
    on the spread move from t to t+1. A position opens at |z| >= entry and
    closes once the spread reverts to |z| < exit, holding in between."""
    za = np.asarray(z, dtype=np.float64)
    pos = np.zeros(za.size, dtype=np.float64)
    cur = 0.0
    for t in range(za.size):
        if cur == 0.0:
            if za[t] > entry:
                cur = -1.0          # short the (too-high) spread
            elif za[t] < -entry:
                cur = 1.0           # long the (too-low) spread
        elif abs(za[t]) < exit:
            cur = 0.0               # reverted -> close
        pos[t] = cur
    return pos


def simulate(spread_arr: npt.ArrayLike, positions: npt.ArrayLike,
             cost: float = 0.0) -> Tuple[List[float], FloatArray]:
    """Turn a position series into round-trip trade PnLs and a per-bar PnL.

    PnL during bar t (from t to t+1) accrues to the position held at t. A trade
    is a maximal run of equal non-zero positions; `cost` is charged once per
    round trip (entry+exit slippage/fees lumped together)."""
    sp = np.asarray(spread_arr, dtype=np.float64)
    pos = np.asarray(positions, dtype=np.float64)
    if sp.size != pos.size:
        raise ValueError(f"length mismatch: spread={sp.size}, pos={pos.size}")
    per_bar = np.zeros(pos.size, dtype=np.float64)
    if pos.size > 1:
        per_bar[:-1] = pos[:-1] * np.diff(sp)

    trades: List[float] = []
    t = 0
    n = pos.size
    while t < n:
        if pos[t] == 0.0:
            t += 1
            continue
        j = t
        while j + 1 < n and pos[j + 1] == pos[t]:
            j += 1
        trades.append(float(per_bar[t:j + 1].sum()) - cost)
        t = j + 1
    return trades, per_bar


def fit_and_signal(y_train: npt.ArrayLike, x_train: npt.ArrayLike,
                   y_test: npt.ArrayLike, x_test: npt.ArrayLike,
                   entry: float = 2.0, exit: float = 0.5,
                   cost: float = 0.0) -> StatArbResult:
    """Estimate the hedge ratio and spread normalisation on TRAIN, freeze them,
    then generate and simulate signals on TEST. This is the unit that keeps the
    evaluation out-of-sample: nothing from the test segment touches the fit."""
    beta, alpha = hedge_ratio(y_train, x_train)
    sp_train = spread(y_train, x_train, beta, alpha)
    mean = float(sp_train.mean())
    std = float(sp_train.std(ddof=0))

    sp_test = spread(y_test, x_test, beta, alpha)
    z = zscore(sp_test, mean=mean, std=std)
    pos = generate_positions(z, entry=entry, exit=exit)
    trades, per_bar = simulate(sp_test, pos, cost=cost)
    return StatArbResult(positions=pos, per_bar_pnl=per_bar, trades=trades,
                         beta=beta, alpha=alpha)


@dataclass
class StatArbWFResult:
    """Aggregated trades across all walk-forward windows of a single config."""
    trades: List[float] = field(default_factory=list)
    n_windows: int = 0

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> int:
        return int(sum(1 for t in self.trades if t > 0))

    @property
    def win_rate(self) -> float:
        return self.wins / self.n_trades if self.trades else 0.0

    @property
    def total_pnl(self) -> float:
        return float(sum(self.trades))

    @property
    def sharpe(self) -> float:
        if self.n_trades < 2:
            return 0.0
        arr = np.asarray(self.trades, dtype=np.float64)
        sd = float(arr.std(ddof=1))
        return float(arr.mean()) / sd if sd else 0.0

    @property
    def p_value(self) -> float:
        return binomial_p_value(self.wins, self.n_trades)


def statarb_walkforward(close_y: npt.ArrayLike, close_x: npt.ArrayLike,
                        train_size: int, test_size: int,
                        step: Optional[int] = None, entry: float = 2.0,
                        exit: float = 0.5, cost: float = 0.0,
                        purge: int = 1, embargo: int = 0) -> StatArbWFResult:
    """Walk forward over the pair: refit hedge ratio + z-score params on each
    train block, trade the following test block, pool the trades.

    `purge` drops that many bars from each train tail so the hedge-ratio fit
    and the spread normalisation do not lean on prices immediately adjacent to
    the test segment -- the spread is autocorrelated, so the boundary bars leak
    the test's opening spread state otherwise. This reuses engine.purging's
    purge_window with the holding-overlap horizon."""
    ya = np.asarray(close_y, dtype=np.float64)
    xa = np.asarray(close_x, dtype=np.float64)
    if ya.size != xa.size:
        raise ValueError(f"length mismatch: y={ya.size}, x={xa.size}")
    n = ya.size
    wins = make_windows(n, train_size, test_size, step)
    out = StatArbWFResult()
    for w in wins:
        (a, b) = purge_window(w["train"], w["test"], purge, embargo)
        c, d = w["test"]
        if b - a < 4:
            continue
        res = fit_and_signal(ya[a:b], xa[a:b], ya[c:d], xa[c:d],
                             entry=entry, exit=exit, cost=cost)
        out.trades.extend(res.trades)
        out.n_windows += 1
    return out


@dataclass
class ShieldReport:
    """Result of the full selection-then-sealed-judgment protocol."""
    best_params: Dict[str, float]
    n_trials: int
    holdout_trades: int
    holdout_wins: int
    holdout_win_rate: float
    holdout_sharpe: float
    holdout_total_pnl: float
    raw_p_value: float
    deflated_p_value: float
    trial_log: TrialLog

    @property
    def significant(self) -> bool:
        return self.deflated_p_value < 0.05


def select_and_judge(close_y: npt.ArrayLike, close_x: npt.ArrayLike,
                     grid: Sequence[Tuple[float, float]],
                     train_frac: float = 0.6, val_frac: float = 0.2,
                     wf_train: int = 120, wf_test: int = 40,
                     cost: float = 0.0, purge: int = 1,
                     embargo: int = 0,
                     select_by: str = "win_rate") -> ShieldReport:
    """The shield, end to end, for stat-arb signals.

    1. Split the pair chronologically into trainval and a sealed holdout.
    2. Search the (entry, exit) grid on trainval via purged walk-forward,
       logging every config to a TrialLog.
    3. Freeze the best config, refit on the whole trainval segment, and judge
       it ONCE on the holdout -- reporting a deflated p-value that pays for the
       number of configs tried (Sidak). The honest expected outcome on planted
       or real data is "not significant after deflation"."""
    ya = np.asarray(close_y, dtype=np.float64)
    xa = np.asarray(close_x, dtype=np.float64)
    if ya.size != xa.size:
        raise ValueError(f"length mismatch: y={ya.size}, x={xa.size}")
    if select_by not in ("win_rate", "sharpe", "total_pnl"):
        raise ValueError(f"unknown select_by={select_by!r}")
    n = ya.size
    tv_end = int(n * (train_frac + val_frac))
    if tv_end < wf_train + wf_test or tv_end >= n:
        raise ValueError("not enough data for trainval walk-forward + holdout")

    y_tv, x_tv = ya[:tv_end], xa[:tv_end]
    y_ho, x_ho = ya[tv_end:], xa[tv_end:]

    log = TrialLog()
    best_score = -np.inf
    best: Tuple[float, float] = grid[0]
    for entry, exit in grid:
        wf = statarb_walkforward(y_tv, x_tv, wf_train, wf_test, entry=entry,
                                 exit=exit, cost=cost, purge=purge,
                                 embargo=embargo)
        score = float(getattr(wf, select_by))
        log.record({"entry": entry, "exit": exit}, score)
        if score > best_score and wf.n_trades > 0:
            best_score = score
            best = (entry, exit)

    # Single sealed judgment on the holdout, refitting on all of trainval.
    e, x = best
    ho = fit_and_signal(y_tv, x_tv, y_ho, x_ho, entry=e, exit=x, cost=cost)
    raw_p = binomial_p_value(ho.wins, ho.n_trades)
    def_p = deflated_p_value(ho.wins, ho.n_trades, log.n_trials)
    return ShieldReport(
        best_params={"entry": e, "exit": x},
        n_trials=log.n_trials,
        holdout_trades=ho.n_trades,
        holdout_wins=ho.wins,
        holdout_win_rate=ho.win_rate,
        holdout_sharpe=ho.sharpe,
        holdout_total_pnl=ho.total_pnl,
        raw_p_value=raw_p,
        deflated_p_value=def_p,
        trial_log=log,
    )
