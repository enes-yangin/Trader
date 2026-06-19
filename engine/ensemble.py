from typing import Dict, Literal, cast
from engine.predictor import predict_single, spec_from_bundle
from utils.config import OPTIMIZATION
from utils.types import Bundle, WeightedEnsembleSignal
from utils.logger import get_logger

log = get_logger("ensemble")


def compute_weights(bundle: Bundle, metric: str = OPTIMIZATION.ensemble_weight_metric) -> Dict[str, float]:
    if metric not in ("val_r2", "val_rmse"):
        raise ValueError(f"Unknown metric: {metric!r} (expected val_r2 or val_rmse)")

    scores: Dict[str, float] = {}
    for name, r in bundle["results"].items():
        m = r["metrics"]
        val = m.get("val", {})
        if metric == "val_r2":
            r2 = val.get("r2", 0.0)
            scores[name] = max(r2, 0.0)
        else:
            rmse = val.get("rmse")
            scores[name] = (1.0 / rmse) if rmse and rmse > 0 else 0.0

    total = sum(scores.values())
    n = len(scores)
    if total <= 0:
        log.warning("All models have non-positive weight scores; falling back to equal weights")
        return {k: 1.0 / n for k in scores}
    return {k: v / total for k, v in scores.items()}


def predict_weighted_ensemble(bundle: Bundle,
                               metric: str = OPTIMIZATION.ensemble_weight_metric) -> WeightedEnsembleSignal:
    df = bundle["df"]
    spec = spec_from_bundle(bundle)
    weights = compute_weights(bundle, metric=metric)

    sigs = []
    sig_weights = []
    for name, r in bundle["results"].items():
        sig = predict_single(r["model"], df, spec=spec)
        sig["weight"] = round(weights[name], 4)
        sigs.append(sig)
        sig_weights.append(weights[name])

    avg_pred = sum(s["predicted_return"] * w for s, w in zip(sigs, sig_weights))
    avg_conf = sum(s["confidence"] * w for s, w in zip(sigs, sig_weights))

    vote_scores: Dict[str, float] = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
    for s, w in zip(sigs, sig_weights):
        vote_scores[s["signal"]] += w
    consensus = max(vote_scores, key=lambda k: vote_scores[k])

    return {
        "consensus": cast(Literal["BUY", "HOLD", "SELL"], consensus),
        "avg_confidence": round(avg_conf, 1),
        "avg_predicted_return": round(avg_pred, 6),
        "votes": {k: round(v, 4) for k, v in vote_scores.items()},
        "details": sigs,
        "last_close": sigs[0]["last_close"],
        "symbol": sigs[0]["symbol"],
        "weights": weights,
    }


def format_weights(weights: Dict[str, float]) -> str:
    lines = ["  Ensemble Weights", "  " + "-" * 24]
    for name, w in sorted(weights.items(), key=lambda x: -x[1]):
        bar = "#" * max(1, int(w * 40))
        lines.append(f"  {name:12s} {w:.3f}  {bar}")
    return "\n".join(lines)
