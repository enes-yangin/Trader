"""
Ekonomik Takvim — planlı makro olayları çeker (FOMC, CPI, NFP vb.).

Ücretsiz kaynaklardan veya statik takvimden beslenir.
Olay tarihleri ÖNCEDEN bilinir → sızıntı DEĞİLDİR.
Olay sonuçları (actual, forecast) yalnızca AÇIKLANDIKTAN sonra bilinir.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any

import pandas as pd
import numpy as np

from utils.config import DATA
from utils.logger import get_logger

log = get_logger("econ_calendar")

CACHE_FILE = "econ_calendar.json"
CACHE_MAX_AGE_H = 24.0

# Statik takvim — önemli ABD makro olaylarının tipik tarih paterni
# (ayın belirli günleri / haftaları). Canlı API olmadığında fallback.
STATIC_EVENTS: List[Dict[str, Any]] = [
    {"name": "FOMC", "country": "US", "importance": 3,
     "frequency": "6w", "description": "Fed faiz kararı"},
    {"name": "CPI", "country": "US", "importance": 3,
     "frequency": "monthly", "day_hint": "10-15",
     "description": "Tüketici Fiyat Endeksi"},
    {"name": "PPI", "country": "US", "importance": 2,
     "frequency": "monthly", "day_hint": "10-15",
     "description": "Üretici Fiyat Endeksi"},
    {"name": "NFP", "country": "US", "importance": 3,
     "frequency": "monthly", "day_hint": "first_friday",
     "description": "Tarım Dışı İstihdam"},
    {"name": "GDP", "country": "US", "importance": 3,
     "frequency": "quarterly", "description": "Gayri Safi Yurtiçi Hasıla"},
    {"name": "FOMC_Minutes", "country": "US", "importance": 2,
     "frequency": "6w", "offset_days": 21,
     "description": "FOMC toplantı tutanakları"},
    {"name": "Retail_Sales", "country": "US", "importance": 2,
     "frequency": "monthly", "day_hint": "mid_month",
     "description": "Perakende Satışlar"},
    {"name": "Unemployment_Claims", "country": "US", "importance": 2,
     "frequency": "weekly", "day_of_week": 4,
     "description": "İşsizlik Başvuruları"},
]

EVENT_WEIGHTS = {1: 0.5, 2: 1.0, 3: 2.0}


def _build_static_calendar(start: str, end: str) -> pd.DataFrame:
    """Statik takvimden belirli tarih aralığı için olay takvimi üret."""
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    rows: List[Dict[str, Any]] = []

    current = start_dt
    while current <= end_dt:
        for ev in STATIC_EVENTS:
            freq = ev.get("frequency", "monthly")
            hint = ev.get("day_hint", "")

            # Aylık olaylar — ayın ortasında (10-15 arası)
            if freq == "monthly":
                if hint == "first_friday":
                    # Ayın ilk Cuma'sı
                    first_day = current.replace(day=1)
                    days_to_fri = (4 - first_day.dayofweek) % 7
                    event_day = first_day + timedelta(days=days_to_fri)
                elif hint == "mid_month":
                    event_day = current.replace(day=14)
                elif hint == "10-15":
                    event_day = current.replace(day=12)
                else:
                    event_day = current.replace(day=14)

                if event_day <= end_dt and start_dt <= event_day:
                    rows.append({
                        "date": event_day.strftime("%Y-%m-%d"),
                        "name": ev["name"], "country": ev["country"],
                        "importance": ev["importance"],
                        "description": ev["description"],
                        "forecast": None, "actual": None,
                    })

            # Haftalık olaylar
            elif freq == "weekly":
                dow = ev.get("day_of_week", 4)  # Perşembe
                days_ahead = (dow - current.dayofweek) % 7
                event_day = current + timedelta(days=days_ahead)
                if event_day <= end_dt and start_dt <= event_day:
                    rows.append({
                        "date": event_day.strftime("%Y-%m-%d"),
                        "name": ev["name"], "country": ev["country"],
                        "importance": ev["importance"],
                        "description": ev["description"],
                        "forecast": None, "actual": None,
                    })

            # 6 haftalık FOMC döngüsü — yaklaşık
            elif freq == "6w":
                offset = ev.get("offset_days", 0)
                # FOMC toplantıları ~6 haftada bir, yıl başından itibaren say
                week_num = (current.isocalendar()[1] - 1)
                if week_num % 6 == 0 and current.dayofweek == 2:  # Salı
                    event_day = current + timedelta(days=offset)
                    if event_day <= end_dt and start_dt <= event_day:
                        rows.append({
                            "date": event_day.strftime("%Y-%m-%d"),
                            "name": ev["name"], "country": ev["country"],
                            "importance": ev["importance"],
                            "description": ev["description"],
                            "forecast": None, "actual": None,
                        })

            # Çeyreklik
            elif freq == "quarterly":
                if current.month in (1, 4, 7, 10) and current.day == 28:
                    event_day = current
                    if event_day <= end_dt and start_dt <= event_day:
                        rows.append({
                            "date": event_day.strftime("%Y-%m-%d"),
                            "name": ev["name"], "country": ev["country"],
                            "importance": ev["importance"],
                            "description": ev["description"],
                            "forecast": None, "actual": None,
                        })

        current += timedelta(days=1)

    if not rows:
        return pd.DataFrame(columns=[
            "date", "name", "country", "importance", "description",
            "forecast", "actual",
        ])

    df = pd.DataFrame(rows).drop_duplicates(subset=["date", "name"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_econ_calendar(
    start: Optional[str] = None,
    end: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Ekonomik takvimi getir (canlı API → statik fallback).

    Parameters
    ----------
    start : str
        Başlangıç tarihi (YYYY-MM-DD). Yoksa 1 yıl öncesi.
    end : str
        Bitiş tarihi (YYYY-MM-DD). Yoksa 3 ay sonrası.
    use_cache : bool
        Disk cache kullanılsın mı?

    Returns
    -------
    pd.DataFrame
        Columns: date, name, country, importance, description, forecast, actual
    """
    now = pd.Timestamp.now()
    start = start or (now - timedelta(days=365)).strftime("%Y-%m-%d")
    end = end or (now + timedelta(days=90)).strftime("%Y-%m-%d")

    cache_path = os.path.join(DATA.data_dir, CACHE_FILE)

    # Cache kontrol
    if use_cache and os.path.exists(cache_path):
        try:
            age_h = (time.time() - os.path.getmtime(cache_path)) / 3600.0
            if age_h < CACHE_MAX_AGE_H:
                df = pd.read_json(cache_path)
                df["date"] = pd.to_datetime(df["date"])
                start_dt = pd.Timestamp(start)
                end_dt = pd.Timestamp(end)
                return df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]
        except (ValueError, KeyError, OSError) as e:
            log.warning(f"Cache okunamadı: {e}")

    # Canlı API dene (ör: free tier economic calendar)
    df = None
    try:
        df = _fetch_from_api(start, end)
    except Exception as e:
        log.info(f"API ekonomik takvim alınamadı ({e}), statik fallback")

    if df is None or df.empty:
        df = _build_static_calendar(start, end)

    if df.empty:
        return df

    # Cache'e yaz
    try:
        os.makedirs(DATA.data_dir, exist_ok=True)
        df.to_json(cache_path, date_format="iso")
    except OSError:
        pass

    return df


def _fetch_from_api(start: str, end: str) -> Optional[pd.DataFrame]:
    """Ücretsiz ekonomik takvim API'sinden veri çekmeyi dene.

    Birden fazla kaynağı sırayla dener: Trading Economics, FXStreet RSS.
    Gerçek API key yoksa None döner — sessizce statik takvime geçilir.
    """
    # TODO: Ücretsiz API endpoint eklendiğinde buraya gerçek HTTP isteği gelecek.
    # Şimdilik statik takvime sessizce geçiyoruz.
    return None
