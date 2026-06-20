"""
Order Book Features — bar-başı özellikler kaydedilmiş order book geçmişinden.

Leak-free: Her bar için sadece o bar kapanışına kadarki defter durumu kullanılır.
Gelecek barların order book durumu özellik hesaplamasına SIZMAZ.

Özellikler:
  - depth_imbalance: bid/ask hacim dengesizliği (bar sonundaki)
  - wall_persistence: duvar fiyatlarının bar içinde ne kadar süre dayandığı
  - microprice_drift: microprice'ın mid price'dan sapma trendi
  - spread_regime: spread'in genişleme/daralma rejim göstergesi
"""

from __future__ import annotations

import hashlib
import pandas as pd
import numpy as np

from utils.config import ORDERBOOK
from utils.logger import get_logger

log = get_logger("orderbook_features")

ORDERBOOK_FEATURE_COLS = [
    "depth_imbalance", "wall_persistence", "microprice_drift", "spread_regime",
]


def _normalize_tz(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Tüm zaman dilimlerini naive yap (karşılaştırma güvenliği)."""
    if hasattr(idx, 'tz') and idx.tz is not None:
        return idx.tz_localize(None)
    return idx


def _resample_to_bar(ob_hist: pd.DataFrame, bar_idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Order book snapshot'larını bar periyotlarına grupla.

    Her bar için, o bar aralığına düşen son snapshot'ı temsilci olarak alır.
    Bar = [bar_start, bar_end) yarı-açık aralık.
    """
    if ob_hist.empty:
        return pd.DataFrame(columns=ob_hist.columns)

    bar_idx = _normalize_tz(bar_idx)
    ob = ob_hist.copy()
    ob.index = _normalize_tz(pd.to_datetime(ob.index))
    ob["_bar_end"] = pd.NaT

    for i, bar_end in enumerate(bar_idx):
        if i == 0:
            bar_start = bar_idx[0] - (bar_idx[1] - bar_idx[0])
        else:
            bar_start = bar_idx[i - 1]
        mask = (ob.index > bar_start) & (ob.index <= bar_end)
        ob.loc[mask, "_bar_end"] = bar_end

    ob = ob.dropna(subset=["_bar_end"])
    if ob.empty:
        return pd.DataFrame(columns=ob.columns.drop("_bar_end", errors="ignore"))

    result = ob.groupby("_bar_end").last()
    result.index.name = "date"
    return result


def depth_imbalance(ob_hist: pd.DataFrame, bar_idx: pd.DatetimeIndex) -> pd.Series:
    """Bar kapanışındaki bid/ask hacim dengesizliği.

    Pozitif = alış baskısı, negatif = satış baskısı.
    Zaten orderbook.imbalance() tarafından hesaplanıp kaydediliyor;
    burada sadece bar seviyesine indirgiyoruz.
    """
    bar_ob = _resample_to_bar(ob_hist, bar_idx)
    result = bar_ob.get("imbalance", pd.Series(np.nan, index=bar_idx))
    return result.reindex(bar_idx).fillna(0)


def wall_persistence(ob_hist: pd.DataFrame, bar_idx: pd.DatetimeIndex,
                     min_size: float = 3.0) -> pd.Series:
    """Bar içinde büyük duvarların kalıcılık skoru.

    Her snapshot'ta detect_walls ile bulunan duvar sayısının bar boyunca
    ağırlıklı ortalaması — yüksek değer = kalıcı duvar baskısı.
    """
    if ob_hist.empty:
        return pd.Series(0.0, index=bar_idx)

    bar_idx = _normalize_tz(bar_idx)
    ob = ob_hist.copy()
    ob.index = _normalize_tz(pd.to_datetime(ob.index))
    ob["_bar_end"] = pd.NaT

    for i, bar_end in enumerate(bar_idx):
        bar_start = bar_idx[i - 1] if i > 0 else bar_idx[0] - (bar_idx[1] - bar_idx[0])
        mask = (ob.index > bar_start) & (ob.index <= bar_end)
        ob.loc[mask, "_bar_end"] = bar_end

    ob = ob.dropna(subset=["_bar_end"])

    # Wall skoru = bid_walls + ask_walls (zaten snapshot'ta var)
    ob["_wall_score"] = ob.get("bid_walls", 0) + ob.get("ask_walls", 0)

    # Bar başına ortalama wall skoru
    result = ob.groupby("_bar_end")["_wall_score"].mean()
    return result.reindex(bar_idx).fillna(0)


def microprice_drift(ob_hist: pd.DataFrame, bar_idx: pd.DatetimeIndex) -> pd.Series:
    """Microprice'ın mid price'dan kümülatif sapması (bar içi drift).

    Her bar için bar başı ve sonu arasındaki micro_skew değişimi.
    Pozitif = microprice mid'den yukarı kaymış (alış tarafı ağırlıklı).
    """
    if ob_hist.empty:
        return pd.Series(0.0, index=bar_idx)

    bar_ob = _resample_to_bar(ob_hist, bar_idx)
    skew = bar_ob.get("micro_skew", pd.Series(np.nan, index=bar_idx))
    skew = skew.reindex(bar_idx)

    # Drift = skew'in bar'lar arası değişimi
    drift = skew.diff().fillna(0)
    return drift.fillna(0)


def spread_regime(ob_hist: pd.DataFrame, bar_idx: pd.DatetimeIndex,
                  window: int = 20) -> pd.Series:
    """Spread rejimi: genişleyen/daralan piyasa göstergesi.

    rel_spread'in son N barlık Z-score'u.
    Pozitif = spread normalden geniş (belirsizlik yüksek).
    Negatif = spread normalden dar (likidite bol).
    """
    if ob_hist.empty:
        return pd.Series(0.0, index=bar_idx)

    bar_ob = _resample_to_bar(ob_hist, bar_idx)
    rel_spread = bar_ob.get("rel_spread", pd.Series(np.nan, index=bar_idx))
    rel_spread = rel_spread.reindex(bar_idx)

    mu = rel_spread.rolling(window, min_periods=3).mean()
    sd = rel_spread.rolling(window, min_periods=3).std().replace(0, np.nan)
    regime = ((rel_spread - mu) / sd).fillna(0)
    return regime.fillna(0)


def _make_synthetic_ob_history_for_index(index: pd.Index, symbol: str, mid_start: float) -> pd.DataFrame:
    """Generate deterministic synthetic orderbook history aligned with target index for testing/backtesting."""
    import hashlib
    snapshots = []
    sym_hash = int(hashlib.md5(symbol.encode()).hexdigest(), 16)
    rng = np.random.default_rng(42 + sym_hash % 10000)
    for i, t in enumerate(index):
        if i == 0:
            dt = pd.Timedelta("1D")
        else:
            dt = index[i] - index[i - 1]
        
        # Spaced 5 snapshots per bar interval
        for j in range(5):
            snap_time = t - dt * (1.0 - (j + 1) / 5.0)
            imbalance = rng.uniform(-0.6, 0.6)
            rel_spread = rng.uniform(0.0001, 0.002)
            micro_skew = imbalance * 0.0005
            snapshots.append({
                "timestamp": snap_time,
                "imbalance": imbalance,
                "rel_spread": rel_spread,
                "micro_skew": micro_skew,
                "bid_walls": int(rng.integers(0, 3)),
                "ask_walls": int(rng.integers(0, 3)),
            })
    df_ob = pd.DataFrame(snapshots)
    df_ob.set_index("timestamp", inplace=True)
    return df_ob


def add_orderbook_features(price_df: pd.DataFrame,
                           ob_hist: pd.DataFrame,
                           allow_sample: bool = False) -> pd.DataFrame:
    """Fiyat DataFrame'ine order book özelliklerini ekle (leak-free).

    price_df: günlük OHLCV DataFrame'i (datetime index)
    ob_hist: Recorder'dan gelen snapshot geçmişi (datetime index)
    allow_sample: veri bulunmadığında sentetik geçmiş oluşturulmasına izin verilsin mi?

    Her özellik SADECE o barın kapanış anına kadarki defter verisini kullanır —
    gelecek bar'ın defter durumu hiçbir şekilde sızmaz.
    """
    df = price_df.copy()
    bar_idx = pd.to_datetime(df.index)

    if ob_hist is None or ob_hist.empty:
        if allow_sample:
            symbol = df.attrs.get("symbol", "BTC/USDT")
            mid_start = float(df["close"].iloc[0]) if not df.empty else 30000.0
            ob_hist = _make_synthetic_ob_history_for_index(bar_idx, symbol, mid_start)
        else:
            df["depth_imbalance"] = 0.0
            df["wall_persistence"] = 0.0
            df["microprice_drift"] = 0.0
            df["spread_regime"] = 0.0
            return df

    # Bar periyodu tespiti
    if len(bar_idx) >= 2:
        bar_freq = bar_idx[1] - bar_idx[0]
    else:
        bar_freq = pd.Timedelta("1D")

    df["depth_imbalance"] = depth_imbalance(ob_hist, bar_idx).values
    df["wall_persistence"] = wall_persistence(ob_hist, bar_idx).values
    df["microprice_drift"] = microprice_drift(ob_hist, bar_idx).values
    df["spread_regime"] = spread_regime(ob_hist, bar_idx).values

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df
