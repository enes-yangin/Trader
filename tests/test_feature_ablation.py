import warnings
import pandas as pd
import pytest
from tests.realistic_data import make_realistic_ohlcv
from data.indicators import engineer
from engine.feature_ablation import run_ablation, summarize_ablation, _spec_variants
from utils.types import FeatureSpec

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def engineered():
    return engineer(make_realistic_ohlcv(n=1300, seed=0), with_news=False, with_micro=True)


def test_spec_variants_have_different_feature_counts():
    variants = _spec_variants()
    counts = {name: spec.n_features for name, spec in variants.items()}
    assert counts["technical_only"] < counts["technical+micro"]


def test_ablation_returns_row_per_variant(engineered):
    table = run_ablation(engineered, model_name="logistic",
                          train_size=400, test_size=100)
    assert isinstance(table, pd.DataFrame)
    assert len(table) == len(_spec_variants())
    for col in ("variant", "n_features", "mean_hit_rate", "std_hit_rate"):
        assert col in table.columns


def test_ablation_hit_rates_plausible(engineered):
    table = run_ablation(engineered, model_name="xgb_clf",
                          train_size=400, test_size=100)
    valid = table["mean_hit_rate"].dropna()
    assert ((valid > 0.2) & (valid < 0.8)).all(), \
        "ablation hit rates outside plausible band (possible leak)"


def test_ablation_feature_counts_match_spec(engineered):
    table = run_ablation(engineered, model_name="logistic",
                          train_size=400, test_size=100)
    tech_only = table[table["variant"] == "technical_only"].iloc[0]
    assert tech_only["n_features"] == FeatureSpec(micro=False).n_features


def test_summarize_runs(engineered):
    table = run_ablation(engineered, model_name="logistic",
                          train_size=400, test_size=100)
    txt = summarize_ablation(table)
    assert "Ablation" in txt
    assert "hit" in txt


def test_ablation_insufficient_data_marks_nan():
    df = engineer(make_realistic_ohlcv(n=200, seed=0), with_news=False, with_micro=True)
    table = run_ablation(df, model_name="logistic", train_size=400, test_size=100)
    assert table["mean_hit_rate"].isna().all()
