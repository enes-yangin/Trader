from typing import Dict, Optional
import numpy as np
import pandas as pd
from utils.config import CROSS_ASSET, DATA, FEATURES
from utils.exceptions import DataFetchError, InsufficientDataError, DataValidationError
from utils.logger import get_logger

log = get_logger("cross_asset")


def _fetch_ref_price(sym: str, years: int = DATA.hist_years,
                      allow_sample: bool = False) -> Optional[pd.DataFrame]:
    from data import dataset
    try:
        df = dataset.build(sym, with_news=False, years=years, allow_sample=allow_sample)
        out = df[["close", "volume"]].copy()
        out.index = pd.to_datetime(out.index).normalize()
        out = out[~out.index.duplicated(keep="last")]
        return out
    except (DataFetchError, InsufficientDataError, DataValidationError) as e:
        log.warning(f"cross-asset reference {sym} unavailable: {type(e).__name__}: {e}")
        return None


def fetch_reference_data(years: int = DATA.hist_years,
                          allow_sample: bool = False) -> Dict[str, pd.DataFrame]:
    data = {}
    for s in CROSS_ASSET.ref_symbols:
        ref = _fetch_ref_price(s, years=years, allow_sample=allow_sample)
        if ref is not None:
            data[s] = ref
    ethbtc = _fetch_ref_price(CROSS_ASSET.eth_btc_symbol, years=years, allow_sample=allow_sample)
    if ethbtc is not None:
        data[CROSS_ASSET.eth_btc_symbol] = ethbtc
    return data


def _align(series: pd.Series, target_index: pd.Index) -> pd.Series:
    s = series.copy()
    s.index = pd.to_datetime(s.index).normalize()
    s = s[~s.index.duplicated(keep="last")]
    ti = pd.to_datetime(target_index).normalize()
    aligned = s.reindex(ti, method="ffill")
    aligned.index = target_index
    return aligned


def add_cross_asset_features(df: pd.DataFrame, sym: str,
                              ref_data: Optional[Dict[str, pd.DataFrame]] = None,
                              years: int = DATA.hist_years, allow_sample: bool = False,
                              corr_window: int = CROSS_ASSET.corr_window,
                              vol_share_zwindow: int = CROSS_ASSET.vol_share_zscore_window) -> pd.DataFrame:
    df = df.copy()
    if ref_data is None:
        ref_data = fetch_reference_data(years=years, allow_sample=allow_sample)

    if CROSS_ASSET.eth_btc_symbol in ref_data:
        ethbtc_ret = ref_data[CROSS_ASSET.eth_btc_symbol]["close"].pct_change()
        ethbtc_ret = ethbtc_ret.clip(-0.5, 0.5)
        df["eth_btc_chg"] = _align(ethbtc_ret, df.index)
    else:
        df["eth_btc_chg"] = 0.0

    vol_cols = {s: ref_data[s]["volume"] for s in CROSS_ASSET.ref_symbols if s in ref_data}
    if "BTC/USDT" in vol_cols and len(vol_cols) >= 2:
        vol_df = pd.DataFrame(vol_cols).dropna(how="any")
        total = vol_df.sum(axis=1).replace(0, np.nan)
        share = vol_df["BTC/USDT"] / total
        mu = share.rolling(vol_share_zwindow).mean()
        sd = share.rolling(vol_share_zwindow).std().replace(0, np.nan)
        share_z = ((share - mu) / sd).fillna(0)
        df["btc_vol_share"] = _align(share_z, df.index)
    else:
        df["btc_vol_share"] = 0.0

    ret_cols = {}
    for s in CROSS_ASSET.ref_symbols:
        if s in ref_data:
            ret_cols[s] = ref_data[s]["close"].pct_change()
    if ret_cols:
        ret_df = pd.DataFrame(ret_cols)
        basket_avg = ret_df.mean(axis=1)
        basket_aligned = _align(basket_avg, df.index)
        asset_ret = df[FEATURES.close_col].pct_change()
        df["mkt_corr"] = asset_ret.rolling(corr_window).corr(basket_aligned)
    else:
        df["mkt_corr"] = 0.0

    new_cols = ["eth_btc_chg", "btc_vol_share", "mkt_corr"]
    df[new_cols] = df[new_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df
