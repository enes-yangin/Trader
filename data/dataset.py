from typing import Callable, Optional, Dict, Any, List
import pandas as pd
from data import cache_policy
from data.enrichment import fetch_and_validate, enrich_with_news
from utils.config import DATA, FEATURES, NEWS
from utils.exceptions import DataFetchError, InsufficientDataError, DataValidationError
from utils.logger import get_logger

log = get_logger("dataset")


def build(sym: str, src: str = "crypto", years: int = DATA.hist_years, with_news: bool = FEATURES.use_news,
          force: bool = False, allow_sample: bool = False) -> pd.DataFrame:
    if not force and cache_policy.is_dataset_fresh(sym, with_news):
        df = cache_policy.load_cached_dataset(sym, with_news)
        if df is not None:
            log.info(f"{sym}: loaded from cache ({len(df)} rows, with_news={with_news})")
            return df

    price, sample = fetch_and_validate(sym, src=src, years=years, allow_sample=allow_sample)
    cache_policy.save_price(sym, price, years=years, sample=sample)

    if with_news:
        df, news_only = enrich_with_news(price, sym, sample=sample, days=NEWS.news_days)
        cache_policy.save_news(sym, news_only, sample=sample)
    else:
        df = price

    df.attrs["symbol"] = sym
    df.attrs["cached"] = False
    df.attrs["sample"] = sample
    cache_policy.save_dataset(sym, df, with_news=with_news, years=years, sample=sample)
    log.info(f"{sym}: built dataset ({len(df)} rows, sample={sample}, with_news={with_news})")
    return df


def load_cached(sym: str, with_news: bool = FEATURES.use_news) -> Optional[pd.DataFrame]:
    return cache_policy.load_cached_dataset(sym, with_news)


def has_cache(sym: str, with_news: bool = FEATURES.use_news) -> bool:
    return cache_policy.has_cached_dataset(sym, with_news)


def ensure(sym: str, src: str = "crypto", years: int = DATA.hist_years, with_news: bool = FEATURES.use_news,
           allow_sample: bool = False,
           on_first: Optional[Callable[[str, int], None]] = None) -> pd.DataFrame:
    if cache_policy.has_cached_dataset(sym, with_news):
        df = cache_policy.load_cached_dataset(sym, with_news)
        if df is not None:
            if not cache_policy.is_dataset_fresh(sym, with_news):
                log.warning(f"{sym}: returning stale cached dataset (with_news={with_news})")
            return df
    if on_first:
        on_first(sym, years)
    return build(sym, src=src, years=years, with_news=with_news,
                 force=True, allow_sample=allow_sample)


def info(sym: str) -> Dict[str, Any]:
    return cache_policy.dataset_info(sym)


def build_many(syms: List[str], src: str = "crypto", years: int = DATA.hist_years,
               with_news: bool = FEATURES.use_news, force: bool = False,
               progress: Optional[Callable[[int, int, str], None]] = None,
               allow_sample: bool = False) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    errors: Dict[str, str] = {}
    for i, s in enumerate(syms):
        if progress:
            progress(i, len(syms), s)
        try:
            out[s] = build(s, src=src, years=years, with_news=with_news,
                           force=force, allow_sample=allow_sample)
        except (DataFetchError, InsufficientDataError, DataValidationError) as e:
            log.error(f"{s}: {type(e).__name__}: {e}")
            errors[s] = str(e)
    if progress:
        progress(len(syms), len(syms), "done")
    if errors:
        out["_errors"] = errors
    return out
