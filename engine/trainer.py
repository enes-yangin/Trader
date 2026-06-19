from typing import Optional, Callable, Any
import pandas as pd
from data.indicators import engineer, get_features
from models.base_model import BaseModel
from models.linear_model import LinearModel
from models.xgb_model import XGBModel
from models.lstm_model import LSTMModel
from utils.config import FEATURES, MODEL, SPLIT
from utils.types import Bundle, BundleWithNews, SplitDict, TrainSingleResult, FeatureSpec


MODEL_MAP = {
    "linear": LinearModel,
    "xgboost": XGBModel,
    "lstm": LSTMModel,
}


def _resolve_spec(spec: Optional[FeatureSpec], with_news: bool, with_micro: bool,
                   with_cross_asset: bool) -> FeatureSpec:
    if spec is not None:
        return spec
    return FeatureSpec.from_bools(with_news, with_micro, with_cross_asset)


def load_data(sym: str, src: str = "crypto", horizon: int = MODEL.pred_horizon,
              spec: Optional[FeatureSpec] = None,
              with_news: bool = FEATURES.use_news, with_micro: bool = FEATURES.use_micro,
              with_cross_asset: bool = FEATURES.use_cross_asset,
              use_cache: bool = True, allow_sample: bool = False,
              on_first: Optional[Callable[[str, int], None]] = None) -> pd.DataFrame:
    spec = _resolve_spec(spec, with_news, with_micro, with_cross_asset)
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
          tr: float = SPLIT.train_ratio, va: float = SPLIT.val_ratio) -> SplitDict:
    spec = _resolve_spec(spec, with_news, with_micro, with_cross_asset)
    X = get_features(df, spec=spec).values
    y = df[FEATURES.target_col].values
    idx = df.index
    n = len(X)
    i_tr = int(n * tr)
    i_va = int(n * (tr + va))
    return SplitDict(**{
        "X_tr": X[:i_tr], "y_tr": y[:i_tr],
        "X_val": X[i_tr:i_va], "y_val": y[i_tr:i_va],
        "X_test": X[i_va:], "y_test": y[i_va:],
        "idx_tr": idx[:i_tr], "idx_val": idx[i_tr:i_va], "idx_test": idx[i_va:],
        "df": df, "i_tr": i_tr, "i_va": i_va,
        "split_idx": i_tr, "with_news": spec.news, "with_micro": spec.micro,
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
                 allow_sample: bool = False, **kw: Any) -> TrainSingleResult:
    spec = _resolve_spec(spec, with_news, with_micro, with_cross_asset)
    df = load_data(sym, src=src, horizon=horizon, spec=spec, allow_sample=allow_sample)
    sp = split(df, spec=spec)
    mdl = build_model(model_name, **kw)
    res = mdl.train(sp["X_tr"], sp["y_tr"], sp["X_val"], sp["y_val"])
    res["test"] = mdl.evaluate(sp["X_test"], sp["y_test"])
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
              allow_sample: bool = False) -> BundleWithNews:
    spec = _resolve_spec(spec, with_news, with_micro, with_cross_asset)
    df = load_data(sym, src=src, horizon=horizon, spec=spec, allow_sample=allow_sample)
    sp = split(df, spec=spec)
    results: dict = {}
    for name, cls in MODEL_MAP.items():
        if name == "lstm":
            mdl = cls(epochs=MODEL.lstm_params["epochs"])
        else:
            mdl = cls()
        res = mdl.train(sp["X_tr"], sp["y_tr"], sp["X_val"], sp["y_val"])
        res["test"] = mdl.evaluate(sp["X_test"], sp["y_test"])
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
