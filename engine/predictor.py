import numpy as np
import pandas as pd
from data.fetcher import fetch, generate_sample
from data.indicators import engineer, get_features
from utils.config import SIGNAL_THRESH_BUY, SIGNAL_THRESH_SELL, USE_NEWS


def predict_single(mdl, df, with_news=USE_NEWS):
    X = get_features(df, with_news=with_news).values
    if hasattr(mdl, "predict_last"):
        pred = mdl.predict_last(X)
    else:
        pred = float(mdl.predict(X[-1:])[-1])
    sig = mdl.signal(pred, buy_th=SIGNAL_THRESH_BUY, sell_th=SIGNAL_THRESH_SELL)
    sig["model"] = mdl.name
    sig["last_close"] = float(df["close"].iloc[-1])
    sig["symbol"] = df.attrs.get("symbol", "N/A")
    return sig


def predict_ensemble(models, df, with_news=USE_NEWS):
    sigs = [predict_single(m, df, with_news=with_news) for m in models]
    preds = [s["predicted_return"] for s in sigs]
    avg_pred = float(np.mean(preds))

    votes = {"BUY": 0, "SELL": 0, "HOLD": 0}
    for s in sigs:
        votes[s["signal"]] += 1
    consensus = max(votes, key=votes.get)

    avg_conf = float(np.mean([s["confidence"] for s in sigs]))
    return {
        "consensus": consensus,
        "avg_confidence": round(avg_conf, 1),
        "avg_predicted_return": round(avg_pred, 6),
        "votes": votes,
        "details": sigs,
        "last_close": sigs[0]["last_close"],
        "symbol": sigs[0]["symbol"],
    }


def predict_from_bundle(bundle):
    df = bundle["df"]
    wn = bundle.get("with_news", USE_NEWS)
    models = [r["model"] for r in bundle["results"].values()]
    ens = predict_ensemble(models, df, with_news=wn)
    return ens


def live_signal(sym, bundle, src="auto"):
    wn = bundle.get("with_news", USE_NEWS)
    sample = False
    try:
        df = fetch(sym, src=src)
    except Exception:
        df = generate_sample(sym)
        sample = True
    df = engineer(df, sym=sym, with_news=wn, sample_news=sample)
    models = [r["model"] for r in bundle["results"].values()]
    return predict_ensemble(models, df, with_news=wn)


def format_signal(sig):
    lines = [
        f"{'='*40}",
        f"  {sig['symbol']}  |  {sig['consensus']}  ({sig['avg_confidence']:.1f}%)",
        f"  Predicted Return: {sig['avg_predicted_return']:+.4%}",
        f"  Last Close: {sig['last_close']:.2f}",
        f"  Votes: BUY={sig['votes']['BUY']} HOLD={sig['votes']['HOLD']} SELL={sig['votes']['SELL']}",
        f"{'='*40}",
    ]
    for d in sig["details"]:
        lines.append(f"    {d['model']:20s} {d['signal']:>5s}  {d['confidence']:5.1f}%  ret={d['predicted_return']:+.6f}")
    return "\n".join(lines)
