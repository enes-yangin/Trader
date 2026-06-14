import numpy as np
import pandas as pd
from data.fetcher import fetch, generate_sample
from data.indicators import engineer, get_features
from models.linear_model import LinearModel
from models.xgb_model import XGBModel
from models.lstm_model import LSTMModel
from utils.config import (
    TARGET_COL, TRAIN_RATIO, PRED_HORIZON, LSTM_PARAMS, USE_NEWS
)


MODEL_MAP = {
    "linear": LinearModel,
    "xgboost": XGBModel,
    "lstm": LSTMModel,
}


def load_data(sym, src="auto", horizon=PRED_HORIZON, with_news=USE_NEWS):
    sample = False
    try:
        df = fetch(sym, src=src)
    except Exception:
        df = generate_sample(sym)
        sample = True
    df = engineer(df, horizon=horizon, sym=sym, with_news=with_news,
                  sample_news=sample)
    return df


def split(df, ratio=TRAIN_RATIO, with_news=USE_NEWS):
    X = get_features(df, with_news=with_news).values
    y = df[TARGET_COL].values
    idx = df.index
    sp = int(len(X) * ratio)
    return {
        "X_tr": X[:sp], "y_tr": y[:sp],
        "X_val": X[sp:], "y_val": y[sp:],
        "idx_tr": idx[:sp], "idx_val": idx[sp:],
        "df": df, "split_idx": sp, "with_news": with_news,
    }


def build_model(name, **kw):
    cls = MODEL_MAP.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown model: {name}. Options: {list(MODEL_MAP.keys())}")
    return cls(**kw)


def train_single(sym, model_name, src="auto", horizon=PRED_HORIZON,
                 with_news=USE_NEWS, **kw):
    df = load_data(sym, src=src, horizon=horizon, with_news=with_news)
    sp = split(df, with_news=with_news)
    mdl = build_model(model_name, **kw)
    res = mdl.train(sp["X_tr"], sp["y_tr"], sp["X_val"], sp["y_val"])
    return {
        "model": mdl,
        "metrics": res,
        "split": sp,
        "symbol": sym,
    }


def train_all(sym, src="auto", horizon=PRED_HORIZON, with_news=USE_NEWS):
    df = load_data(sym, src=src, horizon=horizon, with_news=with_news)
    sp = split(df, with_news=with_news)
    results = {}
    for name, cls in MODEL_MAP.items():
        if name == "lstm":
            mdl = cls(epochs=LSTM_PARAMS["epochs"])
        else:
            mdl = cls()
        res = mdl.train(sp["X_tr"], sp["y_tr"], sp["X_val"], sp["y_val"])
        results[name] = {"model": mdl, "metrics": res}
    return {"results": results, "split": sp, "symbol": sym, "df": df,
            "with_news": with_news}


def compare(bundle):
    rows = []
    for name, r in bundle["results"].items():
        m = r["metrics"]
        row = {"model": name}
        for k in ["r2", "rmse", "mae"]:
            row[f"train_{k}"] = m["train"][k]
            if "val" in m:
                row[f"val_{k}"] = m["val"][k]
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")
