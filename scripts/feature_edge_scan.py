"""Gerçek BTC verisinde feature-edge taraması.

Her feature için 5-günlük forward getiriyle Spearman IC hesaplar.
p-değerleri örtüşmesiz (her 5. bar) örneklemden alınır ve feature sayısı
kadar Sidak-deflate edilir — çoklu-test şansı 'kenar' sanmayalım diye.
Ek: purged XGB gain-importance + CPCV (full vs technical-only) dağılımı.

Çalıştır: .venv\\Scripts\\python.exe scripts\\feature_edge_scan.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats

from utils.types import FeatureSpec
from utils.config import FEATURES, MODEL
from data.indicators import engineer, get_features


def load_raw():
    try:
        from data import dataset
        raw = dataset.load_cached("BTC/USDT", with_news=False)
        if raw is not None and len(raw) > 500:
            return raw, "cache api"
    except Exception:
        pass
    return pd.read_parquet("datasets/BTC_USDT_dataset.parquet"), "parquet"


raw, src = load_raw()
print(f"Gercek BTC verisi: {len(raw)} satir ({src}), "
      f"{raw.index.min().date()} -> {raw.index.max().date()}")

spec = FeatureSpec(micro=True, smooth=True)  # sadece gercek (yerel) aileler
df = engineer(raw, spec=spec)
X = get_features(df, spec=spec)
y = df[FEATURES.target_col].values
h = MODEL.pred_horizon
n_feat = X.shape[1]

# --- Feature basina IC (bilgi katsayisi) --------------------------------
rows = []
for c in X.columns:
    x = X[c].values
    ic_full = float(stats.spearmanr(x, y).statistic)
    xs, ys = x[::h], y[::h]                      # ortusmesiz hedefler
    n = len(xs)
    ic = float(stats.spearmanr(xs, ys).statistic)
    t = ic * np.sqrt((n - 2) / max(1e-12, 1 - ic * ic))
    p = float(2 * stats.t.sf(abs(t), n - 2))
    p_defl = float(1 - (1 - p) ** n_feat)        # Sidak: n_feat deneme
    rows.append((c, ic_full, ic, p, p_defl))

rows.sort(key=lambda r: -abs(r[2]))
print(f"\n{'feature':18s} {'IC(full)':>9s} {'IC(5d-no)':>10s} {'p':>8s} {'p_defl':>8s}  verdict")
sig = 0
for c, icf, ic, p, pd_ in rows:
    v = "**EDGE?**" if pd_ < 0.05 else ("zayif" if p < 0.05 else "gurultu")
    sig += pd_ < 0.05
    print(f"{c:18s} {icf:+9.3f} {ic:+10.3f} {p:8.3f} {pd_:8.3f}  {v}")
print(f"\nOrtusmesiz n={len(y[::h])}, feature={n_feat}, "
      f"deflasyon sonrasi anlamli: {sig}/{n_feat}")

# --- Purged XGB gain-importance -----------------------------------------
from xgboost import XGBRegressor
cut = int(len(X) * 0.7)
m = XGBRegressor(**MODEL.xgb_params, verbosity=0, random_state=42)
m.fit(X.values[: cut - h], y[: cut - h])         # h-bar purge
print("\nXGB gain-importance (purged train, ilk 8):")
for c, w in sorted(zip(X.columns, m.feature_importances_),
                   key=lambda z: -z[1])[:8]:
    print(f"  {c:18s} {w:.3f}")

# --- CPCV: full vs technical-only ---------------------------------------
from engine.cpcv import run_cpcv
from models.linear_model import LinearModel
print()
print(run_cpcv(df, LinearModel, n_groups=6, k_test=2,
               spec=spec, purge=h).format_report("full(tech+micro+smooth)"))
print(run_cpcv(df, LinearModel, n_groups=6, k_test=2,
               spec=FeatureSpec(), purge=h).format_report("technical_only"))
