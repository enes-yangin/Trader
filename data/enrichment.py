from typing import Tuple
import pandas as pd
from data.fetcher import fetch_hist, generate_sample
from data.validation import validate_ohlcv
from utils.config import DATA, NEWS, SPLIT
from utils.exceptions import DataFetchError, InsufficientDataError, DataValidationError
from utils.logger import get_logger

log = get_logger("enrichment")

NEWS_FEATURE_OUTPUT_COLS = ["sentiment_avg", "sentiment_std", "news_volume", "sentiment_trend"]


def fetch_and_validate(sym: str, src: str = "crypto", years: int = DATA.hist_years,
                        allow_sample: bool = False) -> Tuple[pd.DataFrame, bool]:
    """Fetch historical OHLCV, falling back to synthetic sample data if the
    real fetch fails and allow_sample=True. Returns (price_df, sample_flag).

    Raises DataFetchError if the fetch fails and allow_sample=False, or
    InsufficientDataError/DataValidationError if the resulting data fails
    validation regardless of source.
    """
    sample = False
    try:
        price = fetch_hist(sym, src=src, years=years)
        if price is None or len(price) == 0:
            raise DataFetchError(f"No data returned for {sym}")
    except (DataFetchError, InsufficientDataError):
        raise
    except (ConnectionError, TimeoutError, OSError) as e:
        if not allow_sample:
            raise DataFetchError(
                f"Real data fetch failed for {sym}: {type(e).__name__}: {e}. "
                f"Check internet/exchange access, or pass allow_sample=True "
                f"to use synthetic data for testing."
            ) from e
        log.warning(f"{sym}: network error, falling back to SYNTHETIC sample data")
        price = generate_sample(sym, n=years * 365)
        sample = True
    except Exception as e:
        if not allow_sample:
            raise DataFetchError(
                f"Real data fetch failed for {sym}: {type(e).__name__}: {e}. "
                f"Check internet/exchange access, or pass allow_sample=True "
                f"to use synthetic data for testing."
            ) from e
        log.warning(
            f"{sym}: unexpected fetch error ({type(e).__name__}: {e}), "
            f"falling back to SYNTHETIC sample data"
        )
        price = generate_sample(sym, n=years * 365)
        sample = True

    try:
        validate_ohlcv(price, min_rows=SPLIT.min_train_rows)
    except (InsufficientDataError, DataValidationError) as e:
        log.error(f"{sym}: validation failed: {e}")
        raise

    return price, sample


def enrich_with_news(price: pd.DataFrame, sym: str, sample: bool,
                      days: int = NEWS.news_days) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Merge news-sentiment features into price data.

    Returns (merged_df, news_only_df) where news_only_df holds just the
    sentiment columns, suitable for separate caching.
    """
    from data.news_features import add_news_features
    df = add_news_features(price, sym, days=days, sample=sample)
    news_only = df[[c for c in NEWS_FEATURE_OUTPUT_COLS if c in df.columns]].copy()
    return df, news_only
