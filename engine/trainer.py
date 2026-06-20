from typing import Optional, Callable, Any
import pandas as pd
from data.indicators import engineer, get_features
from models.base_model import BaseModel
from models.linear_model import LinearModel
from models.xgb_model import XGBModel
from models.lstm_model import LSTMModel
from engine.purging import purged_train_end
from utils.config import FEATURES, MODEL, SPLIT
from utils.types import Bundle, BundleWithNews, SplitDict, TrainSingleResult, FeatureSpec


from engine.feature_selection import select_top_k
from engine.optimizer import build_optimized_model

MODEL_MAP = {
    "linear": LinearModel,
    "xgboost": XGBModel,
    "lstm": LSTMModel,
}


def _resolve_spec(
    spec: Optional[FeatureSpec],
    with_news: bool = False,
    with_micro: bool = False,
    with_cross_asset: bool = False,
    with_smoothing: bool = True,
    with_reference: bool = False,
    with_orderbook: bool = True,
    with_macro_events: bool = True,
    with_social: bool = True,
) -> FeatureSpec:
    """Return the explicit FeatureSpec if provided; otherwise build from the
    individual bool flags. Defaults for the 5 families not threaded through
    legacy callers (smoothing, reference, orderbook, macro_events, social)
    come from FEATURES config for consistency.

    B5 fix: Previously only 3 of 8 families were considered. Now all 8 are
    explicitly included and caller overrides are respected.
    """
    if spec is not None:
        return spec
    return FeatureSpec.from_bools(
        with_news=with_news,
        with_micro=with_micro,
        with_cross_asset=with_cross_asset,
        with_smoothing=with_smoothing,
        with_reference=with_reference,
        with_orderbook=with_orderbook,
        with_macro_events=with_macro_events,
        with_social=with_social,
    )


def load_data(sym: str, src: str = "crypto", horizon: int = MODEL.pred_horizon,
              spec: Optional[FeatureSpec] = None,
              with_news: bool = FEATURES.use_news, with_micro: bool = FEATURES.use_micro,
              with_cross_asset: bool = FEATURES.use_cross_asset,
              with_smoothing: bool = FEATURES.use_smoothing,
              with_reference: bool = FEATURES.use_reference,
              with_orderbook: bool = FEATURES.use_orderbook,
              with_macro_events: bool = FEATURES.use_macro_events,
              with_social: bool = FEATURES.use_social,
              use_cache: bool = True, allow_sample: bool = False,
              on_first: Optional[Callable[[str, int], None]] = None) -> pd.DataFrame:
    spec = _resolve_spec(spec, with_news, with_micro, with_cross_asset,
                         with_smoothing, with_reference, with_orderbook,
                         with_macro_events, with_social)
    from data import dataset
    if use_cache:
        raw = dataset.ensure(sym, src=src, with_news=spec.news,
                             allow_sample=allow_sample, on_first=on_first)
    else:
        raw = dataset.build(sym, src=src, with_news=spec.news,
                             force=True, allow_sample=allow_sample)
    sample = bool(raw.attrs.get("sample", False))
    df = engineer(raw, horizon=horizon, sym=sym, spec=spec, precomputed_news=True)
    df.attrs["symbol"] = sym
    df.attrs["sample"] = sample
    return df


def split(df: pd.DataFrame, spec: Optional[FeatureSpec] = None,
          with_news: bool = FEATURES.use_news, with_micro: bool = FEATURES.use_micro,
          with_cross_asset: bool = FEATURES.use_cross_asset,
          with_smoothing: bool = FEATURES.use_smoothing,
          with_reference: bool = FEATURES.use_reference,
          with_orderbook: bool = FEATURES.use_orderbook,
          with_macro_events: bool = FEATURES.use_macro_events,
          with_social: bool = FEATURES.use_social,
          tr: float = SPLIT.train_ratio, va: float = SPLIT.val_ratio) -> SplitDict:
    spec = _resolve_spec(spec, with_news, with_micro, with_cross_asset,
                         with_smoothing, with_reference, with_orderbook,
                         with_macro_events, with_social)
    X = get_features(df, spec=spec).values
    y = df[FEATURES.target_col].values
    idx = df.index
    n = len(X)
    i_tr = int(n * tr)
    i_va = int(n * (tr + va))

    # A1 & B6: Purge train and val endpoints to prevent forward-label leakage.
    # Labels at index t depend on bars [t+1, t+pred_horizon]. If a training
    # sample's label horizon reaches into the validation set, that sample must
    # be dropped from training. Same for val→test.
    label_horizon = MODEL.pred_horizon
    purged_i_tr = purged_train_end(0, i_tr, i_tr, label_horizon)
    purged_i_va = purged_train_end(i_tr, i_va, i_va, label_horizon)

    return SplitDict(**{
        "X_tr": X[:purged_i_tr], "y_tr": y[:purged_i_tr],
        "X_val": X[purged_i_tr:purged_i_va], "y_val": y[purged_i_tr:purged_i_va],
        "X_test": X[purged_i_va:], "y_test": y[purged_i_va:],
        "idx_tr": idx[:purged_i_tr], "idx_val": idx[purged_i_tr:purged_i_va],
        "idx_test": idx[purged_i_va:],
        "df": df, "i_tr": purged_i_tr, "i_va": purged_i_va,
        "split_idx": purged_i_tr, "with_news": spec.news, "with_micro": spec.micro,
        "with_cross_asset": spec.cross_asset, "spec": spec,
    })


def build_model(name: str, **kw: Any) -> BaseModel:
    cls = MODEL_MAP.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown model: {name}. Options: {list(MODEL_MAP.keys())}")
    return cls(**kw)


def train_single(sym: str, model_name: str, src: str = "crypto",
                 horizon: int = MODEL.pred_horizon, spec: Optional[FeatureSpec] = None,
                 with_news: bool = FEATURES.use_news, with_micro: bool = FEATURES.use_micro,
                 with_cross_asset: bool = FEATURES.use_cross_asset,
                 with_smoothing: bool = FEATURES.use_smoothing,
                 with_reference: bool = FEATURES.use_reference,
                 with_orderbook: bool = FEATURES.use_orderbook,
                 with_macro_events: bool = FEATURES.use_macro_events,
                 with_social: bool = FEATURES.use_social,
                 allow_sample: bool = False, **kw: Any) -> TrainSingleResult:
    spec = _resolve_spec(spec, with_news, with_micro, with_cross_asset,
                         with_smoothing, with_reference, with_orderbook,
                         with_macro_events, with_social)
    df = load_data(sym, src=src, horizon=horizon, spec=spec, allow_sample=allow_sample)
    sp = split(df, spec=spec)
    feature_names = spec.feature_columns()
    sp = select_top_k(sp, feature_names)
    mdl, res, best_params, study = build_optimized_model(model_name, sp, **kw)
    return {
        "model": mdl,
        "metrics": res,
        "split": sp,
        "symbol": sym,
    }


def train_all(sym: str, src: str = "crypto", horizon: int = MODEL.pred_horizon,
              spec: Optional[FeatureSpec] = None,
              with_news: bool = FEATURES.use_news, with_micro: bool = FEATURES.use_micro,
              with_cross_asset: bool = FEATURES.use_cross_asset,
              with_smoothing: bool = FEATURES.use_smoothing,
              with_reference: bool = FEATURES.use_reference,
              with_orderbook: bool = FEATURES.use_orderbook,
              with_macro_events: bool = FEATURES.use_macro_events,
              with_social: bool = FEATURES.use_social,
              allow_sample: bool = False) -> BundleWithNews:
    spec = _resolve_spec(spec, with_news, with_micro, with_cross_asset,
                         with_smoothing, with_reference, with_orderbook,
                         with_macro_events, with_social)
    df = load_data(sym, src=src, horizon=horizon, spec=spec, allow_sample=allow_sample)
    sp = split(df, spec=spec)
    feature_names = spec.feature_columns()
    sp = select_top_k(sp, feature_names)
    results: dict = {}
    for name in MODEL_MAP:
        mdl, res, best_params, study = build_optimized_model(name, sp)
        results[name] = {"model": mdl, "metrics": res}
    bundle = Bundle(**{
        "results": results, "split": sp, "symbol": sym, "df": df,
        "with_news": spec.news, "with_micro": spec.micro,
        "with_cross_asset": spec.cross_asset, "spec": spec,
        "sample": bool(df.attrs.get("sample", False)),
    })

    if spec.news and "sentiment_avg" in df.columns:
        from engine.news_analysis import report
        bundle["news_analysis"] = report(df)

    return bundle


def compare(bundle: Bundle) -> pd.DataFrame:
    rows = []
    for name, r in bundle["results"].items():
        m = r["metrics"]
        row = {"model": name}
        for part in ["train", "val", "test"]:
            if part in m:
                row[f"{part}_r2"] = m[part]["r2"]
                row[f"{part}_rmse"] = m[part]["rmse"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def split_summary(sp: SplitDict) -> pd.DataFrame:
    def rng(idx: pd.Index) -> str:
        if len(idx) == 0:
            return "—"
        return f"{idx[0].date()} → {idx[-1].date()}"

    def yrs(idx: pd.Index) -> float:
        if len(idx) < 2:
            return 0.0
        return (idx[-1] - idx[0]).days / 365.0

    rows = [
        {"set": "train", "rows": len(sp["idx_tr"]),
         "range": rng(sp["idx_tr"]), "years": round(yrs(sp["idx_tr"]), 2)},
        {"set": "val", "rows": len(sp["idx_val"]),
         "range": rng(sp["idx_val"]), "years": round(yrs(sp["idx_val"]), 2)},
        {"set": "test", "rows": len(sp["idx_test"]),
         "range": rng(sp["idx_test"]), "years": round(yrs(sp["idx_test"]), 2)},
    ]
    return pd.DataFrame(rows).set_index("set")
