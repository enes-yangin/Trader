from typing import Tuple
import pandas as pd
from utils.config import FEATURES


def sentiment_price_corr(df: pd.DataFrame, horizons: Tuple[int, ...] = (1, 3, 5, 10)) -> pd.DataFrame:
    if "sentiment_avg" not in df.columns:
        return pd.DataFrame()
    rows = []
    for h in horizons:
        fwd = df[FEATURES.close_col].pct_change(h).shift(-h)
        c = df["sentiment_avg"].corr(fwd)
        c_vol = df["news_volume"].corr(fwd.abs())
        rows.append({
            "horizon": h,
            "sent_vs_return": round(c, 4) if pd.notna(c) else 0.0,
            "volume_vs_volatility": round(c_vol, 4) if pd.notna(c_vol) else 0.0,
        })
    return pd.DataFrame(rows)


def top_sentiment_days(df: pd.DataFrame, n: int = 10, kind: str = "pos") -> pd.DataFrame:
    if "sentiment_avg" not in df.columns:
        return pd.DataFrame()
    d = df[df["news_volume"] > 0].copy()
    d["fwd_5d"] = d[FEATURES.close_col].pct_change(5).shift(-5)
    asc = kind == "neg"
    d = d.sort_values("sentiment_avg", ascending=asc).head(n)
    return d[["sentiment_avg", "news_volume", FEATURES.close_col, "fwd_5d"]].round(4)


def sentiment_buckets(df: pd.DataFrame) -> pd.DataFrame:
    if "sentiment_avg" not in df.columns:
        return pd.DataFrame()
    d = df[df["news_volume"] > 0].copy()
    d["fwd_5d"] = d[FEATURES.close_col].pct_change(5).shift(-5)
    d["bucket"] = pd.cut(
        d["sentiment_avg"],
        bins=[-1.01, -0.3, -0.05, 0.05, 0.3, 1.01],
        labels=["very_neg", "neg", "neutral", "pos", "very_pos"],
    )
    g = d.groupby("bucket", observed=True)["fwd_5d"]
    out = pd.DataFrame({
        "avg_fwd_5d": g.mean().round(4),
        "win_rate": g.apply(lambda x: (x > 0).mean()).round(3),
        "count": g.count(),
    })
    return out


def report(df: pd.DataFrame) -> str:
    lines = ["=" * 55, "  NEWS-PRICE CORRELATION ANALYSIS", "=" * 55]
    corr = sentiment_price_corr(df)
    if len(corr) > 0:
        lines.append("\n  Sentiment vs Forward Return:")
        for _, r in corr.iterrows():
            lines.append(
                f"    {int(r['horizon']):2d}d: corr={r['sent_vs_return']:+.4f}  "
                f"vol/volat={r['volume_vs_volatility']:+.4f}"
            )
    buckets = sentiment_buckets(df)
    if len(buckets) > 0:
        lines.append("\n  Forward 5d return by sentiment bucket:")
        for idx, r in buckets.iterrows():
            lines.append(
                f"    {str(idx):10s}: avg={r['avg_fwd_5d']:+.4f}  "
                f"win={r['win_rate']:.1%}  n={int(r['count'])}"
            )
    lines.append("\n" + "=" * 55)
    return "\n".join(lines)
