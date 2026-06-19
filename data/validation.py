import numpy as np
import pandas as pd
from utils.exceptions import DataValidationError, InsufficientDataError
from utils.logger import get_logger

log = get_logger("validation")

REQUIRED_COLS = ["open", "high", "low", "close", "volume"]
MIN_ROWS_TRAIN = 252
MAX_GAP_RATIO = 0.05
MAX_ZERO_VOL_RATIO = 0.10


def check_columns(df):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")


def check_min_length(df, min_rows=MIN_ROWS_TRAIN):
    if len(df) < min_rows:
        raise InsufficientDataError(
            f"Only {len(df)} rows available, need at least {min_rows} "
            f"for reliable train/val/test split"
        )


def check_ohlc_consistency(df):
    bad = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    )
    n_bad = int(bad.sum())
    if n_bad > 0:
        ratio = n_bad / len(df)
        log.warning(f"OHLC consistency violations: {n_bad} rows ({ratio:.1%})")
        if ratio > 0.05:
            raise DataValidationError(
                f"{n_bad} rows ({ratio:.1%}) violate OHLC ordering "
                f"(high >= open/close/low, low <= open/close)"
            )
    return n_bad


def check_gaps(df, expected_freq="D"):
    if len(df) < 2:
        return 0
    full_range = pd.date_range(df.index.min(), df.index.max(), freq=expected_freq)
    missing = full_range.difference(df.index)
    ratio = len(missing) / len(full_range) if len(full_range) > 0 else 0.0
    if ratio > MAX_GAP_RATIO:
        log.warning(
            f"Date gaps: {len(missing)}/{len(full_range)} "
            f"({ratio:.1%}) missing periods"
        )
    return len(missing)


def check_zero_volume(df):
    if "volume" not in df.columns or len(df) == 0:
        return 0
    n_zero = int((df["volume"] == 0).sum())
    ratio = n_zero / len(df)
    if ratio > MAX_ZERO_VOL_RATIO:
        log.warning(f"Zero-volume rows: {n_zero} ({ratio:.1%})")
    return n_zero


def check_no_nan_inf(df, cols=None):
    if cols is None:
        cols = list(df.columns)
    else:
        cols = list(cols)
    cols = [c for c in cols if c in df.columns]
    sub = df[cols]
    n_nan = int(sub.isna().sum().sum())
    n_inf = int(np.isinf(sub.select_dtypes(include=[np.number]).values).sum())
    if n_nan > 0 or n_inf > 0:
        raise DataValidationError(
            f"Feature matrix contains {n_nan} NaN and {n_inf} inf values"
        )
    return True


def validate_ohlcv(df, min_rows=MIN_ROWS_TRAIN, strict=False):
    check_columns(df)
    check_min_length(df, min_rows=min_rows)
    n_bad_ohlc = check_ohlc_consistency(df)
    n_gaps = check_gaps(df)
    n_zero_vol = check_zero_volume(df)

    report = {
        "rows": len(df),
        "start": str(df.index.min()),
        "end": str(df.index.max()),
        "ohlc_violations": n_bad_ohlc,
        "missing_periods": n_gaps,
        "zero_volume_rows": n_zero_vol,
    }
    log.info(f"OHLCV validation passed: {report}")
    return report


def validate_features(df, feature_cols):
    check_no_nan_inf(df, cols=feature_cols)
    return True
