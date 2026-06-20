"""
Olay Özellikleri — ekonomik takvim olaylarından leak-free sinyaller.

Kritik ayrım:
  - bars_to_next_event: İLERİYE-DÖNÜK olarak meşrudur — olay tarihleri
    önceden duyurulur, sızıntı DEĞİLDİR.
  - last_event_surprise: MUTLAKA shift(1) ile geçmiş olaydan gelir —
    olay sonucu (actual−forecast) ancak açıklandığı an bilinir hale gelir.

Bu ayrım mimarinin çekirdeğidir; ilk prensip ihlal edilirse backtest
gerçeği yansıtmaz.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import numpy as np

from utils.logger import get_logger

log = get_logger("event_features")

MACRO_EVENT_FEATURE_COLS = [
    "bars_to_next_event", "last_event_surprise", "event_importance",
    "event_day_flag",
]


def _nearest_event_features(
    price_df: pd.DataFrame,
    calendar: pd.DataFrame,
    shift_surprise: int = 1,
) -> pd.DataFrame:
    """Her bar için en yakın olay tabanlı özellikleri hesapla.

    Parameters
    ----------
    price_df : pd.DataFrame
        DatetimeIndex'li OHLCV DataFrame'i.
    calendar : pd.DataFrame
        econ_calendar.fetch_econ_calendar() çıktısı.
    shift_surprise : int
        surprise'ı kaç bar ileri kaydıracağımız (1 = yarınki bar'da görünür).

    Returns
    -------
    pd.DataFrame
        bars_to_next_event, last_event_surprise, event_importance, event_day_flag
    """
    price_idx = pd.to_datetime(price_df.index)
    result = pd.DataFrame(index=price_idx)

    cal = calendar.copy()
    if cal.empty:
        # B2: No synthetic data — features default to neutral when calendar is empty
        log.debug("Economic calendar empty, filling event features with neutral defaults")
        result["bars_to_next_event"] = -1
        result["last_event_surprise"] = 0.0
        result["event_importance"] = 0.0
        result["event_day_flag"] = 0
        return result

    cal["date"] = pd.to_datetime(cal["date"])
    cal = cal.sort_values("date")

    # Her bar için sonraki olaya kalan bar sayısı (ileriye-dönük, SIZINTI DEĞİL)
    bars_ahead = np.full(len(price_idx), -1, dtype=float)
    importance_arr = np.zeros(len(price_idx), dtype=float)
    event_flag = np.zeros(len(price_idx), dtype=int)

    cal_dates = cal["date"].values
    cal_importance = cal.get("importance", pd.Series(1, index=cal.index)).values

    for i, bar_time in enumerate(price_idx):
        # Bu bar gününde olay var mı?
        day_mask = cal_dates == bar_time
        if day_mask.any():
            event_flag[i] = int(day_mask.sum())
            importance_arr[i] = float(cal_importance[day_mask].max())

        # Sonraki olaya kalan bar sayısı
        future_events = cal_dates >= bar_time
        if future_events.any():
            next_ev_date = cal_dates[future_events][0]
            bars_diff = (next_ev_date - bar_time).days
            bars_ahead[i] = max(0, bars_diff)

            # Önem skoru: en yakın olayın önemi
            if importance_arr[i] == 0:
                next_idx = np.where(future_events)[0][0]
                importance_arr[i] = float(cal_importance[next_idx])

    result["bars_to_next_event"] = bars_ahead
    result["event_importance"] = importance_arr
    result["event_day_flag"] = event_flag

    # Surprise hesaplama: actual − forecast (sadece actual değeri olan olaylar)
    cal_with_actual = cal[cal["actual"].notna() & cal["forecast"].notna()].copy()
    if not cal_with_actual.empty:
        cal_with_actual["surprise"] = (
            cal_with_actual["actual"].astype(float)
            - cal_with_actual["forecast"].astype(float)
        )
        daily_surprise = cal_with_actual.groupby("date")["surprise"].mean().sort_index()
        surprise_series = pd.Series(0.0, index=price_idx)
        for d, val in daily_surprise.items():
            if d in surprise_series.index:
                surprise_series.loc[d] = float(val)

        # SURPRISE SHIFT: olay sonucu ancak AÇIKLANDIKTAN SONRA bilinir
        # shift(1) → bugünkü bar'da DÜN açıklanan olayın sürprizi görünür
        surprise_series = surprise_series.shift(shift_surprise).fillna(0.0)
        result["last_event_surprise"] = surprise_series.values
    else:
        result["last_event_surprise"] = 0.0

    return result


def add_event_features(
    price_df: pd.DataFrame,
    calendar: Optional[pd.DataFrame] = None,
    shift_surprise: int = 1,
    allow_sample: bool = False,
) -> pd.DataFrame:
    """Fiyat DataFrame'ine makro olay özelliklerini ekle.

    Parameters
    ----------
    price_df : pd.DataFrame
        Günlük OHLCV DataFrame'i.
    calendar : pd.DataFrame veya None
        Ekonomik takvim. None ise statik takvim otomatik çekilir (allow_sample=True ise).
    shift_surprise : int
        Surprise shift gün sayısı (varsayılan 1 = leak-free).
    allow_sample : bool
        API veya statik takvim fallback'ine izin verilsin mi? False ise özellikler 0/varsayılan olarak doldurulur.

    Returns
    -------
    pd.DataFrame
        Orijinal df + 4 olay özelliği eklenmiş.
    """
    df = price_df.copy()

    if calendar is None:
        if allow_sample:
            from data.econ_calendar import fetch_econ_calendar
            start = str(df.index.min().strftime("%Y-%m-%d"))
            end = str(df.index.max().strftime("%Y-%m-%d"))
            calendar = fetch_econ_calendar(start=start, end=end)
        else:
            df["bars_to_next_event"] = -1.0
            df["last_event_surprise"] = 0.0
            df["event_importance"] = 0.0
            df["event_day_flag"] = 0
            return df

    event_df = _nearest_event_features(df, calendar, shift_surprise=shift_surprise)

    for col in MACRO_EVENT_FEATURE_COLS:
        if col in event_df.columns:
            df[col] = event_df[col].values

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df
