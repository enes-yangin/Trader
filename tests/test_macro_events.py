"""Test Ekonomik Takvim + Olay Özellikleri (Feature 2)."""
import pandas as pd
import numpy as np

from data.econ_calendar import fetch_econ_calendar, _build_static_calendar, STATIC_EVENTS
from data.event_features import (
    add_event_features, _nearest_event_features, MACRO_EVENT_FEATURE_COLS,
)


class TestEconCalendar:
    def test_static_calendar_has_expected_events(self):
        df = _build_static_calendar("2024-01-01", "2024-03-01")
        assert len(df) > 0
        assert "FOMC" in df["name"].values or "CPI" in df["name"].values
        assert df["importance"].max() == 3

    def test_fetch_returns_dataframe(self):
        df = fetch_econ_calendar("2024-01-01", "2024-03-01", use_cache=False)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        for col in ["date", "name", "country", "importance"]:
            assert col in df.columns

    def test_fetch_respects_date_range(self):
        df = fetch_econ_calendar("2024-06-01", "2024-06-07", use_cache=False)
        if len(df) > 0:
            assert df["date"].min() >= pd.Timestamp("2024-06-01")
            assert df["date"].max() <= pd.Timestamp("2024-06-07")

    def test_static_events_not_empty(self):
        assert len(STATIC_EVENTS) >= 5
        important = [e for e in STATIC_EVENTS if e["importance"] == 3]
        assert len(important) >= 3  # FOMC, CPI, NFP minimum


class TestEventFeatures:
    @staticmethod
    def _make_price():
        idx = pd.date_range("2024-01-01", periods=90, freq="D")
        df = pd.DataFrame({
            "open": 30000.0, "high": 31000.0, "low": 29000.0,
            "close": np.linspace(29000, 35000, 90),
            "volume": 1000.0,
        }, index=idx)
        return df

    def test_nearest_event_features_returns_all_cols(self):
        price = self._make_price()
        cal = fetch_econ_calendar("2024-01-01", "2024-04-01", use_cache=False)
        feat = _nearest_event_features(price, cal, shift_surprise=1)
        for col in MACRO_EVENT_FEATURE_COLS:
            assert col in feat.columns
        assert len(feat) == 90

    def test_bars_to_next_event_nonnegative(self):
        price = self._make_price()
        cal = _build_static_calendar("2024-01-01", "2024-06-01")
        feat = _nearest_event_features(price, cal)
        # bars_to_next_event -1 olabilir (hiç olay yoksa), ama NaN olmamalı
        assert not feat["bars_to_next_event"].isna().any()
        # Olay varken -1 dışındaki değerler >= 0
        non_missing = feat["bars_to_next_event"][feat["bars_to_next_event"] != -1]
        assert (non_missing >= 0).all()

    def test_event_day_flag_binary(self):
        price = self._make_price()
        cal = _build_static_calendar("2024-01-01", "2024-04-01")
        feat = _nearest_event_features(price, cal)
        # event_day_flag = o gün kaç olay var (0-N arası)
        assert (feat["event_day_flag"] >= 0).all()

    def test_last_event_surprise_shifted(self):
        """Surprise değerleri shift'lenmeli: bugünkü bar'da geçmiş olayın
        sürprizi görünmeli, yarınki olayın değil."""
        price = self._make_price()
        # Surprise'ı olan yapay takvim
        cal = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-15", "2024-02-15"]),
            "name": ["CPI", "CPI"],
            "country": ["US", "US"],
            "importance": [3, 3],
            "forecast": [3.1, 3.2],
            "actual": [3.4, 2.9],
            "description": ["test", "test"],
        })
        feat = _nearest_event_features(price, cal, shift_surprise=1)

        # 2024-01-15'teki CPI sürprizi 2024-01-16'da görünmeli
        jan15 = pd.Timestamp("2024-01-15")
        jan16 = pd.Timestamp("2024-01-16")

        if jan15 in feat.index:
            assert feat.loc[jan15, "last_event_surprise"] == 0.0, (
                "Olay gününde surprise 0 olmalı (henüz bilinmiyor)"
            )
        if jan16 in feat.index:
            # 3.4 - 3.1 = 0.3
            assert abs(feat.loc[jan16, "last_event_surprise"] - 0.3) < 0.01

    def test_add_event_features_integrates(self):
        price = self._make_price()
        cal = fetch_econ_calendar("2024-01-01", "2024-04-01", use_cache=False)
        result = add_event_features(price, cal)
        for col in MACRO_EVENT_FEATURE_COLS:
            assert col in result.columns
        assert len(result) == len(price)

    def test_empty_calendar_no_crash(self):
        price = self._make_price()
        empty_cal = pd.DataFrame(columns=["date", "name", "importance"])
        result = add_event_features(price, empty_cal)
        assert len(result) == 90
        assert (result["bars_to_next_event"] == -1).all() or (result["event_importance"] == 0).all()
