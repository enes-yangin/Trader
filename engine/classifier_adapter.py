"""Wraps a trained BaseClassifier as a BaseModel (plan item I).

Why: engine.backtester.run()/run_portfolio() and engine.predictor's ensemble
path are written against BaseModel's continuous-prediction interface
(predict() -> float array, scored against a dynamic +-threshold). Rather than
forking that code for classifiers, this adapter turns each classifier's
BUY/HOLD/SELL decision into a *signed pseudo-return* -- +pt_mult*atr_pct for
BUY, -sl_mult*atr_pct for SELL, 0.0 for HOLD -- so every existing consumer
(backtester, predict_single/predict_ensemble, compute_weights, meta-labeling's
hasattr(mdl, "predict_last") guard) keeps working unmodified.

Horizon scaling: atr_pct is a single-bar (1-day) volatility estimate, but the
triple-barrier label -- and the user's min_profit_target -- describe a move
over `horizon` bars. Assuming daily returns are roughly independent, expected
volatility over `horizon` days scales with sqrt(horizon) (the standard
random-walk time-scaling used for VaR/vol targets), so the pseudo-return is
pt_mult * atr_pct * sqrt(horizon), not the raw 1-day atr_pct. horizon=1
recovers the original single-day behaviour.

Cost-awareness (F) and the volatility-regime no-trade filter (G) are folded
into predict() itself, so the live signal path and the backtest path share
one gate -- a signal that wouldn't clear costs or that fires in a blocked
vol regime is forced to 0.0 (HOLD) in both places. The vol-regime check
itself still looks at the raw (unscaled) daily atr_pct, since it's asking
"is *today* unusually volatile", not "what's the n-day expected move".
"""
import math
from typing import Optional
import numpy as np
from models.base_model import BaseModel
from models.base_classifier import BaseClassifier
from data.labeling import LABEL_BUY, LABEL_SELL
from utils.config import FEATURES, RISK, SIGNAL
from utils.types import TrainResult
from engine.risk import expected_move_covers_costs, vol_regime_blocked


class ClassifierSignalAdapter(BaseModel):
    """name is inherited from the wrapped classifier so bundle["results"]
    keys/UI labels (e.g. "xgb_clf") read the same as before."""

    def __init__(self, clf: BaseClassifier, pt_mult: float = 1.5,
                 sl_mult: float = 1.0, atr_threshold: float = float("inf"),
                 horizon: int = 1, min_proba: Optional[float] = None):
        super().__init__(name=clf.name)
        self.clf = clf
        self.pt_mult = pt_mult
        self.sl_mult = sl_mult
        self.atr_threshold = atr_threshold
        # Confidence gate (see SignalConfig.min_proba): demote low-conviction
        # directional calls to HOLD. None -> take the configured default.
        self.min_proba = SIGNAL.min_proba if min_proba is None else min_proba
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        self.horizon = horizon
        self._horizon_scale = math.sqrt(horizon)
        self.trained = clf.trained
        self.n_features_ = clf.n_features_
        try:
            self._atr_idx: Optional[int] = list(FEATURES.core_feature_cols).index("atr_pct")
        except ValueError:
            self._atr_idx = None

    def train(self, X_tr: np.ndarray, y_tr: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> TrainResult:
        raise NotImplementedError(
            "ClassifierSignalAdapter wraps an already-trained classifier; "
            "train the BaseClassifier directly and pass it to __init__."
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_ready(X)
        labels = self.clf.predict(X)
        # Aligned [SELL, HOLD, BUY] columns, so proba[i, lbl] is the winning
        # class's own probability (LABEL_SELL=0, HOLD=1, BUY=2).
        proba = self.clf.predict_proba(X)
        atr = (X[:, self._atr_idx] if self._atr_idx is not None
               else np.zeros(len(X)))

        out = np.zeros(len(X), dtype=np.float64)
        for i, lbl in enumerate(labels):
            lbl = int(lbl)
            daily_atr = float(atr[i])
            if lbl == LABEL_BUY:
                move = self.pt_mult * daily_atr * self._horizon_scale
            elif lbl == LABEL_SELL:
                move = -self.sl_mult * daily_atr * self._horizon_scale
            else:
                continue  # HOLD stays 0.0
            # Confidence gate: below min_proba the directional call is too near
            # the 3-class chance line to trade -> HOLD.
            if float(proba[i, lbl]) < self.min_proba:
                continue
            if not expected_move_covers_costs(move):
                continue
            if RISK.use_vol_filter and vol_regime_blocked(daily_atr, self.atr_threshold):
                continue
            out[i] = move
        return out
