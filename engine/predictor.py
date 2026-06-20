from typing import List, Optional, Dict, Literal, cast
import numpy as np
import pandas as pd
import ccxt
from data.fetcher import fetch, generate_sample
from data.indicators import engineer, get_features
from models.base_model import BaseModel, dynamic_threshold
from utils.config import SIGNAL, OPTIMIZATION
from utils.logger import get_logger
from utils.types import Bundle, ModelSignal, EnsembleSignal, FeatureSpec

log = get_logger("predictor")


def predict_single(mdl: BaseModel, df: pd.DataFrame,
                    spec: Optional[FeatureSpec] = None,
                    selected_features: Optional[List[str]] = None) -> ModelSignal:
    spec = spec or FeatureSpec.default()
    feat_df = get_features(df, spec=spec)
    if selected_features is not None:
        feat_df = feat_df[selected_features]
    X = feat_df.values
    if hasattr(mdl, "predict_last"):
        pred = mdl.predict_last(X)
    else:
        pred = float(mdl.predict(X[-1:])[-1])
    atr_pct = float(df["atr_pct"].iloc[-1]) if "atr_pct" in df.columns else 0.0
    buy_th, sell_th = dynamic_threshold(atr_pct, SIGNAL.buy_threshold, SIGNAL.sell_threshold)
    sig: ModelSignal = mdl.signal(pred, buy_th=buy_th, sell_th=sell_th)
    sig["model"] = mdl.name
    sig["last_close"] = float(df["close"].iloc[-1])
    sig["symbol"] = df.attrs.get("symbol", "N/A")
    return sig


def predict_ensemble(models: List[BaseModel], df: pd.DataFrame,
                      spec: Optional[FeatureSpec] = None,
                      selected_features: Optional[List[str]] = None,
                      meta_model: Optional[Any] = None) -> EnsembleSignal:
    spec = spec or FeatureSpec.default()
    sigs = [predict_single(m, df, spec=spec, selected_features=selected_features) for m in models]
    preds = [s["predicted_return"] for s in sigs]
    avg_pred = float(np.mean(preds))

    votes: Dict[str, float] = {"BUY": 0, "SELL": 0, "HOLD": 0}
    for s in sigs:
        votes[s["signal"]] += 1
    consensus = max(votes, key=lambda k: votes[k])

    avg_conf = float(np.mean([s["confidence"] for s in sigs]))
    sig: EnsembleSignal = {
        "consensus": cast(Literal["BUY", "HOLD", "SELL"], consensus),
        "avg_confidence": round(avg_conf, 1),
        "avg_predicted_return": round(avg_pred, 6),
        "votes": votes,
        "details": sigs,
        "last_close": sigs[0]["last_close"],
        "symbol": sigs[0]["symbol"],
    }

    if meta_model is not None:
        from engine.meta_labeling import filter_signal_with_meta
        filter_signal_with_meta(sig, df, spec, meta_model, selected_features)

    return sig


def spec_from_bundle(bundle: Bundle) -> FeatureSpec:
    """Reconstruct FeatureSpec from bundle, preferring the stored spec object.

    B5 fix: Checks bundle.spec first, then bundle.split.spec (set by split()),
    then falls back to the bundle's legacy boolean fields.
    """
    spec = bundle.get("spec")
    if spec is not None:
        return spec
    # B5: split() now stores the full spec in SplitDict
    split_spec = bundle.get("split", {}).get("spec")
    if split_spec is not None:
        return split_spec
    # Legacy fallback
    return FeatureSpec.from_bools(
        with_news=bundle.get("with_news", False),
        with_micro=bundle.get("with_micro", False),
        with_cross_asset=bundle.get("with_cross_asset", False),
    )


def predict_from_bundle(bundle: Bundle) -> EnsembleSignal:
    if OPTIMIZATION.ensemble_weighted:
        from engine.ensemble import predict_weighted_ensemble
        return predict_weighted_ensemble(bundle)
    df = bundle["df"]
    spec = spec_from_bundle(bundle)
    selected_features = bundle.get("split", {}).get("selected_features")
    models = [r["model"] for r in bundle["results"].values()]
    meta_model = bundle.get("meta_model")
    return predict_ensemble(models, df, spec=spec, selected_features=selected_features, meta_model=meta_model)


def live_signal(sym: str, bundle: Bundle, src: str = "crypto") -> EnsembleSignal:
    spec = spec_from_bundle(bundle)
    sample = False
    try:
        df = fetch(sym, src=src)
    except (ccxt.BaseError, ConnectionError, TimeoutError, OSError) as e:
        log.warning(f"{sym}: live fetch failed ({type(e).__name__}: {e}), using sample data")
        df = generate_sample(sym)
        sample = True
    df = engineer(df, sym=sym, spec=spec, sample_news=sample)
    if OPTIMIZATION.ensemble_weighted:
        from engine.ensemble import predict_weighted_ensemble
        return predict_weighted_ensemble(bundle, df=df)
    models = [r["model"] for r in bundle["results"].values()]
    selected_features = bundle.get("split", {}).get("selected_features")
    meta_model = bundle.get("meta_model")
    return predict_ensemble(models, df, spec=spec, selected_features=selected_features, meta_model=meta_model)


def format_signal(sig: EnsembleSignal) -> str:
    lines = [
        f"{'='*40}",
        f"  {sig['symbol']}  |  {sig['consensus']}  ({sig['avg_confidence']:.1f}%)",
        f"  Predicted Return: {sig['avg_predicted_return']:+.4%}",
        f"  Last Close: {sig['last_close']:.2f}",
        f"  Votes: BUY={sig['votes']['BUY']} HOLD={sig['votes']['HOLD']} SELL={sig['votes']['SELL']}",
        f"{'='*40}",
    ]
    for d in sig["details"]:
        lines.append(f"    {d['model']:20s} {d['signal']:>5s}  {d['confidence']:5.1f}%  ret={d['predicted_return']:+.6f}")
    return "\n".join(lines)
