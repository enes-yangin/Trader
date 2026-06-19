from typing import Optional, Dict, Any
import pandas as pd
from data import store
from utils.config import FEATURES


def dataset_kind(with_news: bool = FEATURES.use_news) -> str:
    """Cache bucket name for the engineered-ready dataset.

    with_news=True datasets carry merged sentiment columns; with_news=False
    datasets are price-only. These must be cached under separate keys --
    reusing one for the other would silently return a dataset missing (or
    carrying unwanted) sentiment columns.
    """
    return "dataset_news" if with_news else "dataset"


def load_cached_dataset(sym: str, with_news: bool = FEATURES.use_news) -> Optional[pd.DataFrame]:
    df = store.load(sym, dataset_kind(with_news))
    if df is not None:
        df.attrs["symbol"] = sym
        df.attrs["cached"] = True
    return df


def has_cached_dataset(sym: str, with_news: bool = FEATURES.use_news) -> bool:
    return store.exists(sym, dataset_kind(with_news))


def is_dataset_fresh(sym: str, with_news: bool = FEATURES.use_news) -> bool:
    return store.is_fresh(sym, dataset_kind(with_news))


def save_dataset(sym: str, df: pd.DataFrame, with_news: bool, years: int, sample: bool) -> None:
    store.save(sym, df, kind=dataset_kind(with_news),
               extra={"years": years, "with_news": with_news, "sample": sample})


def save_price(sym: str, price: pd.DataFrame, years: int, sample: bool) -> None:
    store.save(sym, price, kind="price", extra={"years": years, "sample": sample})


def save_news(sym: str, news_only: pd.DataFrame, sample: bool) -> None:
    store.save(sym, news_only, kind="news", extra={"sample": sample})


def dataset_info(sym: str) -> Dict[str, Any]:
    return {
        "price": store.meta(sym, "price"),
        "news": store.meta(sym, "news"),
        "dataset": store.meta(sym, "dataset"),
        "dataset_news": store.meta(sym, "dataset_news"),
        "age_h": store.age_hours(sym, "dataset"),
        "age_h_news": store.age_hours(sym, "dataset_news"),
        "fresh": store.is_fresh(sym, "dataset"),
        "fresh_news": store.is_fresh(sym, "dataset_news"),
    }
