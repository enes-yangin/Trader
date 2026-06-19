from data.indicators import engineer
from engine.trainer import split
from utils.config import SPLIT


def test_split_ratios_match_config(synthetic_ohlcv):
    df = engineer(synthetic_ohlcv, with_news=False, with_micro=True)
    sp = split(df, with_news=False, with_micro=True)

    n = len(df)
    expected_tr = int(n * SPLIT.train_ratio)
    expected_va = int(n * (SPLIT.train_ratio + SPLIT.val_ratio))

    assert len(sp["X_tr"]) == expected_tr
    assert len(sp["X_val"]) == expected_va - expected_tr
    assert len(sp["X_test"]) == n - expected_va


def test_split_is_chronological_no_overlap(synthetic_ohlcv):
    df = engineer(synthetic_ohlcv, with_news=False, with_micro=True)
    sp = split(df, with_news=False, with_micro=True)

    tr_end = sp["idx_tr"][-1]
    va_start = sp["idx_val"][0]
    va_end = sp["idx_val"][-1]
    te_start = sp["idx_test"][0]

    assert tr_end < va_start, "train must end before val starts"
    assert va_end < te_start, "val must end before test starts"

    all_idx = list(sp["idx_tr"]) + list(sp["idx_val"]) + list(sp["idx_test"])
    assert all_idx == sorted(all_idx), "combined split index must be monotonically increasing"
    assert len(set(all_idx)) == len(all_idx), "no duplicate dates across splits"


def test_split_covers_full_dataset(synthetic_ohlcv):
    df = engineer(synthetic_ohlcv, with_news=False, with_micro=True)
    sp = split(df, with_news=False, with_micro=True)

    total = len(sp["X_tr"]) + len(sp["X_val"]) + len(sp["X_test"])
    assert total == len(df)


def test_split_X_y_alignment(synthetic_ohlcv):
    df = engineer(synthetic_ohlcv, with_news=False, with_micro=True)
    sp = split(df, with_news=False, with_micro=True)

    for part in ["tr", "val", "test"]:
        assert len(sp[f"X_{part}"]) == len(sp[f"y_{part}"])
        assert len(sp[f"X_{part}"]) == len(sp[f"idx_{part}"])
