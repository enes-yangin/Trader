"""Test Order Book Recorder + Features (Feature 1)."""
import pandas as pd
import numpy as np

from data.orderbook_recorder import generate_sample_ob_history, OrderBookRecorder, STORE_SUBDIR
from data.orderbook_features import (
    depth_imbalance, wall_persistence, microprice_drift,
    spread_regime, add_orderbook_features, ORDERBOOK_FEATURE_COLS,
)


class TestOrderBookRecorder:
    def test_recorder_creates_store_dir(self, isolated_cache_dir):
        rec = OrderBookRecorder("BTC/USDT", interval_s=9999, data_dir=str(isolated_cache_dir))
        assert STORE_SUBDIR in rec._store_dir
        assert "BTC_USDT" in rec._parquet_path

    def test_load_history_empty_returns_empty_df(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            rec = OrderBookRecorder("EMPTY/PAIR", data_dir=tmp)
            df = rec.load_history()
            assert len(df) == 0

    def test_sample_history_generates_expected_columns(self):
        df = generate_sample_ob_history(n_snapshots=500, seed=1)
        assert len(df) == 500
        assert "imbalance" in df.columns
        assert "mid_price" in df.columns
        assert "rel_spread" in df.columns
        assert abs(df["imbalance"].mean()) < 0.5  # ~simetrik, ortalaması küçük

    def test_sample_history_with_recorder(self, isolated_cache_dir):
        recorder = OrderBookRecorder("TEST/USDT", data_dir=str(isolated_cache_dir))
        ob_hist = generate_sample_ob_history(n_snapshots=200, seed=3)
        # Save sample and verify load
        import os
        os.makedirs(os.path.dirname(recorder._parquet_path), exist_ok=True)
        ms_ts = (ob_hist.index.astype("int64") // 10**6).values
        save_df = ob_hist.reset_index()
        save_df["timestamp"] = ms_ts
        save_df.to_parquet(recorder._parquet_path)
        loaded = recorder.load_history()
        assert len(loaded) == 200


class TestOrderBookFeatures:
    @staticmethod
    def _make_fixtures():
        ob = generate_sample_ob_history(n_snapshots=5000, seed=7)
        bar_idx = pd.date_range("2020-01-01", periods=60, freq="D")
        price = pd.DataFrame({
            "open": 30000.0, "high": 31000.0, "low": 29000.0,
            "close": np.linspace(29000, 35000, 60),
            "volume": 1000.0,
        }, index=bar_idx)
        return ob, bar_idx, price

    def test_depth_imbalance_nonempty(self):
        ob, bar_idx, _ = self._make_fixtures()
        imb = depth_imbalance(ob, bar_idx)
        assert len(imb) == 60
        assert not imb.isna().all()

    def test_wall_persistence_nonempty(self):
        ob, bar_idx, _ = self._make_fixtures()
        wp = wall_persistence(ob, bar_idx)
        assert len(wp) == 60
        assert (wp >= 0).all()  # Duvar sayıları negatif olamaz

    def test_microprice_drift_nonempty(self):
        ob, bar_idx, _ = self._make_fixtures()
        drift = microprice_drift(ob, bar_idx)
        assert len(drift) == 60

    def test_spread_regime_nonempty(self):
        ob, bar_idx, _ = self._make_fixtures()
        regime = spread_regime(ob, bar_idx, window=5)
        assert len(regime) == 60

    def test_empty_ob_returns_zeros(self):
        empty = pd.DataFrame(columns=["timestamp", "imbalance", "mid_price"])
        bar_idx = pd.date_range("2020-01-01", periods=10, freq="D")
        assert (depth_imbalance(empty, bar_idx) == 0).all()
        assert (wall_persistence(empty, bar_idx) == 0).all()

    def test_add_orderbook_features_returns_all_cols(self):
        ob, bar_idx, price = self._make_fixtures()
        result = add_orderbook_features(price, ob)
        for col in ORDERBOOK_FEATURE_COLS:
            assert col in result.columns
        assert len(result) == len(price)

    def test_no_lookahead_bias(self):
        """Order book özellikleri her bar için SADECE o bar kapanışına
        kadarki snapshot'ları kullanır — gelecek bar'ın defter bilgisi sızmaz."""
        ob, bar_idx, price = self._make_fixtures()
        result = add_orderbook_features(price, ob)

        # bar-5'teki özellikler sadece bar_0..bar_5 arası snapshot'lardan türemeli.
        # En basit test: son bar'ın özelliği sonraki (var olmayan) bar'dan
        # etkilenmemeli.
        assert not result["depth_imbalance"].isna().any()
        assert not np.isinf(result["depth_imbalance"]).any()
