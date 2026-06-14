import warnings
import threading
warnings.filterwarnings("ignore")

FINBERT_MODEL = "ProsusAI/finbert"
_finbert = None
_vader = None
_backend = None
_lock = threading.Lock()


def _load_finbert():
    global _finbert, _backend
    try:
        from transformers import pipeline
        _finbert = pipeline("sentiment-analysis", model=FINBERT_MODEL,
                            tokenizer=FINBERT_MODEL, truncation=True, max_length=512)
        _backend = "finbert"
        return True
    except Exception:
        return False


def _load_vader():
    global _vader, _backend
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader = SentimentIntensityAnalyzer()
        _backend = "vader"
        return True
    except Exception:
        return False


def init(prefer="finbert"):
    global _backend
    if _backend is not None:
        return _backend
    with _lock:
        if _backend is not None:
            return _backend
        if prefer == "finbert" and _load_finbert():
            return _backend
        if _load_vader():
            return _backend
        _backend = "none"
        return _backend


def score_text(txt):
    if _backend is None:
        init()
    if not txt or not txt.strip():
        return 0.0
    if _backend == "finbert":
        r = _finbert(txt[:512])[0]
        lbl = r["label"].lower()
        s = r["score"]
        if lbl == "positive":
            return s
        if lbl == "negative":
            return -s
        return 0.0
    if _backend == "vader":
        return _vader.polarity_scores(txt)["compound"]
    return 0.0


def score_batch(texts):
    if _backend is None:
        init()
    if not texts:
        return []
    if _backend == "finbert":
        out = []
        res = _finbert([t[:512] for t in texts])
        for r in res:
            lbl = r["label"].lower()
            s = r["score"]
            out.append(s if lbl == "positive" else -s if lbl == "negative" else 0.0)
        return out
    return [score_text(t) for t in texts]


def get_backend():
    if _backend is None:
        init()
    return _backend
