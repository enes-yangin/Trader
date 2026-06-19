import os
import json
import time
import pandas as pd
from utils.config import DATA


def _root():
    os.makedirs(DATA.data_dir, exist_ok=True)
    return DATA.data_dir


def _slug(sym):
    return sym.replace("/", "_").replace(" ", "")


def _paths(sym, kind):
    r = _root()
    s = _slug(sym)
    return {
        "data": os.path.join(r, f"{s}_{kind}.parquet"),
        "meta": os.path.join(r, f"{s}_{kind}.meta.json"),
    }


def save(sym, df, kind="price", extra=None):
    p = _paths(sym, kind)
    out = df.copy()
    out.to_parquet(p["data"])
    meta = {
        "symbol": sym,
        "kind": kind,
        "rows": int(len(df)),
        "saved_at": time.time(),
        "saved_human": time.strftime("%Y-%m-%d %H:%M:%S"),
        "start": str(df.index.min()) if len(df) else None,
        "end": str(df.index.max()) if len(df) else None,
        "cols": list(df.columns),
    }
    if extra:
        meta.update(extra)
    with open(p["meta"], "w") as f:
        json.dump(meta, f, indent=2)
    return p["data"]


def load(sym, kind="price"):
    p = _paths(sym, kind)
    if not os.path.exists(p["data"]):
        return None
    df = pd.read_parquet(p["data"])
    return df


def meta(sym, kind="price"):
    p = _paths(sym, kind)
    if not os.path.exists(p["meta"]):
        return None
    with open(p["meta"]) as f:
        return json.load(f)


def age_hours(sym, kind="price"):
    m = meta(sym, kind)
    if m is None:
        return None
    return (time.time() - m["saved_at"]) / 3600.0


def is_fresh(sym, kind="price", max_age=DATA.cache_max_age_h):
    a = age_hours(sym, kind)
    if a is None:
        return False
    return a <= max_age


def exists(sym, kind="price"):
    return os.path.exists(_paths(sym, kind)["data"])


def delete(sym, kind=None):
    kinds = [kind] if kind else ["price", "news", "dataset", "dataset_news"]
    for k in kinds:
        p = _paths(sym, k)
        for fp in p.values():
            if os.path.exists(fp):
                os.remove(fp)


def list_datasets():
    r = _root()
    rows = []
    for f in os.listdir(r):
        if f.endswith(".meta.json"):
            with open(os.path.join(r, f)) as fh:
                m = json.load(fh)
            rows.append(m)
    if not rows:
        return pd.DataFrame(columns=["symbol", "kind", "rows", "start", "end", "saved_human"])
    df = pd.DataFrame(rows)
    keep = ["symbol", "kind", "rows", "start", "end", "saved_human"]
    return df[[c for c in keep if c in df.columns]].sort_values(["symbol", "kind"])


def cache_size_mb():
    r = _root()
    total = 0
    for f in os.listdir(r):
        total += os.path.getsize(os.path.join(r, f))
    return total / 1e6
