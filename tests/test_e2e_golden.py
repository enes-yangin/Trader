import numpy as np
import pytest
from unittest.mock import patch
from data.indicators import engineer
from engine.trainer import split, build_model, train_all, compare, split_summary
from engine.predictor import predict_from_bundle, format_signal
from engine.backtester import run_all, summary_table, format_report
from engine.walkforward import run_model
from data import orderbook
from utils.types import Bundle


@pytest.mark.parametrize("model_name", ["linear", "xgboost"])
def test_single_model_pipeline_e2e(synthetic_ohlcv, model_name):
    df = engineer(synthetic_ohlcv, with_news=False, with_micro=True)
    sp = split(df, with_news=False, with_micro=True)

    mdl = build_model(model_name)
    res = mdl.train(sp["X_tr"], sp["y_tr"], sp["X_val"], sp["y_val"])
    res["test"] = mdl.evaluate(sp["X_test"], sp["y_test"])

    assert "train" in res and "test" in res
    for part in ("train", "val", "test"):
        for k in ("mse", "rmse", "mae", "r2"):
            assert np.isfinite(res[part][k])

    bundle = Bundle(**{
        "results": {model_name: {"model": mdl, "metrics": res}},
        "split": sp, "symbol": "BTC/USDT", "df": df,
        "with_news": False, "with_micro": True, "sample": False,
    })

    sig = predict_from_bundle(bundle)
    assert sig["consensus"] in ("BUY", "HOLD", "SELL")
    assert format_signal(sig)

    bt = run_all(bundle)
    assert model_name in bt
    table = summary_table(bt)
    assert "total_return" in table.columns
    assert format_report(bt)


def test_full_three_model_pipeline_e2e(mock_ccxt_exchange):
    with patch("ccxt.kraken", return_value=mock_ccxt_exchange):
        bundle = train_all("BTC/USDT", with_news=False)

        assert set(bundle["results"].keys()) == {"linear", "xgboost", "lstm"}
        assert bundle["with_micro"] is True

        cmp = compare(bundle)
        assert len(cmp) == 3
        for col in ["train_r2", "test_r2"]:
            assert col in cmp.columns
            assert cmp[col].notna().all()

        summ = split_summary(bundle["split"])
        assert set(summ.index) == {"train", "val", "test"}

        sig = predict_from_bundle(bundle)
        assert sig["consensus"] in ("BUY", "HOLD", "SELL")
        assert len(sig["details"]) == 3

        bt = run_all(bundle)
        assert len(bt) == 3
        for name, r in bt.items():
            assert np.isfinite(r["metrics"]["total_return"])
            assert r["which"] == "test"

        wf = run_model(bundle["df"], "linear", train_size=400, test_size=100, with_news=False)
        assert wf["n_windows"] > 0

        ob_sig = orderbook.live_signal("BTC/USDT")
        assert ob_sig["state"] in ("BUY PRESSURE", "SELL PRESSURE", "BALANCED")


def test_pipeline_handles_minimal_data_gracefully(synthetic_ohlcv):
    small = synthetic_ohlcv.iloc[:100]
    df = engineer(small, with_news=False, with_micro=True)
    assert len(df) > 0
    sp = split(df, with_news=False, with_micro=True)
    assert len(sp["X_tr"]) + len(sp["X_val"]) + len(sp["X_test"]) == len(df)
