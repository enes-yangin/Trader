import numpy as np
from data.indicators import engineer, get_features
from engine.trainer import split, build_model, train_all
from engine.optimizer import optimize_linear, optimize_xgb, build_optimized_model
from engine.feature_selection import rank_features, select_top_k, format_importance
from engine.ensemble import compute_weights, predict_weighted_ensemble, format_weights


def _split(synthetic_ohlcv):
    df = engineer(synthetic_ohlcv, with_news=False, with_micro=True)
    return split(df, with_news=False, with_micro=True), df


def test_optimize_linear_returns_valid_alpha(synthetic_ohlcv):
    sp, _ = _split(synthetic_ohlcv)
    best_params, study = optimize_linear(sp, n_trials=3)
    assert "alpha" in best_params
    assert best_params["alpha"] > 0
    assert study.best_value >= 0


def test_optimize_xgb_returns_valid_params(synthetic_ohlcv):
    sp, _ = _split(synthetic_ohlcv)
    best_params, study = optimize_xgb(sp, n_trials=3)
    for key in ("n_estimators", "max_depth", "learning_rate", "subsample"):
        assert key in best_params
    assert 2 <= best_params["max_depth"] <= 8
    assert study.best_value >= 0


def test_build_optimized_model_linear(synthetic_ohlcv):
    sp, _ = _split(synthetic_ohlcv)
    mdl, res, best_params, study = build_optimized_model("linear", sp, n_trials=3)
    assert mdl.trained is True
    assert "test" in res
    assert np.isfinite(res["test"]["r2"])
    assert mdl.model.alpha == best_params["alpha"]


def test_build_optimized_model_xgb(synthetic_ohlcv):
    sp, _ = _split(synthetic_ohlcv)
    mdl, res, best_params, study = build_optimized_model("xgboost", sp, n_trials=3)
    assert mdl.trained is True
    assert "test" in res
    assert np.isfinite(res["test"]["r2"])


def test_optimization_reproducible_with_seed(synthetic_ohlcv):
    sp, _ = _split(synthetic_ohlcv)
    p1, _ = optimize_linear(sp, n_trials=3, seed=123)
    p2, _ = optimize_linear(sp, n_trials=3, seed=123)
    assert p1 == p2


def test_rank_features_returns_all_features(synthetic_ohlcv):
    sp, df = _split(synthetic_ohlcv)
    # B5: Use the same spec from split() to ensure feature count consistency
    feature_names = list(get_features(df, spec=sp["spec"]).columns)
    ranked = rank_features(sp, feature_names)

    assert len(ranked) == len(feature_names)
    assert set(name for name, _ in ranked) == set(feature_names)
    importances = [imp for _, imp in ranked]
    assert importances == sorted(importances, reverse=True)


def test_rank_features_dimension_mismatch_raises(synthetic_ohlcv):
    sp, df = _split(synthetic_ohlcv)
    wrong_names = ["a", "b", "c"]
    try:
        rank_features(sp, wrong_names)
        assert False, "should have raised ValueError"
    except ValueError:
        pass


def test_select_top_k_reduces_features(synthetic_ohlcv):
    sp, df = _split(synthetic_ohlcv)
    # B5: Use the same spec from split() for consistent feature count
    feature_names = list(get_features(df, spec=sp["spec"]).columns)
    k = 5
    new_sp = select_top_k(sp, feature_names, k=k)

    assert new_sp["X_tr"].shape[1] == k
    assert new_sp["X_val"].shape[1] == k
    assert new_sp["X_test"].shape[1] == k
    assert len(new_sp["selected_features"]) == k
    assert new_sp["X_tr"].shape[0] == sp["X_tr"].shape[0]


def test_select_top_k_caps_at_available_features(synthetic_ohlcv):
    sp, df = _split(synthetic_ohlcv)
    feature_names = list(get_features(df, spec=sp["spec"]).columns)
    new_sp = select_top_k(sp, feature_names, k=1000)
    assert new_sp["X_tr"].shape[1] == len(feature_names)


def test_selected_model_trains_on_reduced_features(synthetic_ohlcv):
    sp, df = _split(synthetic_ohlcv)
    feature_names = list(get_features(df, spec=sp["spec"]).columns)
    new_sp = select_top_k(sp, feature_names, k=5)

    mdl = build_model("linear")
    mdl.train(new_sp["X_tr"], new_sp["y_tr"], new_sp["X_val"], new_sp["y_val"])
    assert mdl.n_features_ == 5
    preds = mdl.predict(new_sp["X_test"])
    assert len(preds) == len(new_sp["y_test"])


def test_format_importance_no_crash(synthetic_ohlcv):
    sp, df = _split(synthetic_ohlcv)
    feature_names = list(get_features(df, spec=sp["spec"]).columns)
    ranked = rank_features(sp, feature_names)
    out = format_importance(ranked)
    assert "Feature Importance" in out


def test_compute_weights_sum_to_one(mock_ccxt_exchange):
    from unittest.mock import patch
    with patch("ccxt.kraken", return_value=mock_ccxt_exchange):
        bundle = train_all("BTC/USDT", with_news=False)
    weights = compute_weights(bundle)
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert set(weights.keys()) == set(bundle["results"].keys())
    for w in weights.values():
        assert w >= 0


def test_compute_weights_better_model_gets_more_weight(mock_ccxt_exchange):
    from unittest.mock import patch
    with patch("ccxt.kraken", return_value=mock_ccxt_exchange):
        bundle = train_all("BTC/USDT", with_news=False)

    for r in bundle["results"].values():
        r["metrics"].setdefault("val", {})

    bundle["results"]["linear"]["metrics"]["val"] = {"r2": 0.5, "rmse": 0.01, "mse": 0.0001, "mae": 0.005}
    bundle["results"]["xgboost"]["metrics"]["val"] = {"r2": 0.1, "rmse": 0.02, "mse": 0.0004, "mae": 0.01}
    bundle["results"]["lstm"]["metrics"]["val"] = {"r2": -0.2, "rmse": 0.03, "mse": 0.0009, "mae": 0.02}

    weights = compute_weights(bundle, metric="val_r2")
    assert weights["linear"] > weights["xgboost"]
    assert weights["lstm"] == 0.0


def test_compute_weights_all_negative_falls_back_equal(mock_ccxt_exchange):
    from unittest.mock import patch
    with patch("ccxt.kraken", return_value=mock_ccxt_exchange):
        bundle = train_all("BTC/USDT", with_news=False)

    for r in bundle["results"].values():
        r["metrics"]["val"] = {"r2": -0.5, "rmse": 0.05, "mse": 0.0025, "mae": 0.04}

    weights = compute_weights(bundle, metric="val_r2")
    n = len(weights)
    for w in weights.values():
        assert abs(w - 1.0 / n) < 1e-9


def test_predict_weighted_ensemble_valid_signal(mock_ccxt_exchange):
    from unittest.mock import patch
    with patch("ccxt.kraken", return_value=mock_ccxt_exchange):
        bundle = train_all("BTC/USDT", with_news=False)

    sig = predict_weighted_ensemble(bundle)
    assert sig["consensus"] in ("BUY", "HOLD", "SELL")
    assert "weights" in sig
    assert abs(sum(sig["weights"].values()) - 1.0) < 1e-9
    for d in sig["details"]:
        assert "weight" in d


def test_format_weights_no_crash(mock_ccxt_exchange):
    from unittest.mock import patch
    with patch("ccxt.kraken", return_value=mock_ccxt_exchange):
        bundle = train_all("BTC/USDT", with_news=False)
    weights = compute_weights(bundle)
    out = format_weights(weights)
    assert "Ensemble Weights" in out


def test_invalid_ensemble_metric_raises(mock_ccxt_exchange):
    from unittest.mock import patch
    with patch("ccxt.kraken", return_value=mock_ccxt_exchange):
        bundle = train_all("BTC/USDT", with_news=False)
    try:
        compute_weights(bundle, metric="bogus")
        assert False, "should have raised ValueError"
    except ValueError:
        pass
