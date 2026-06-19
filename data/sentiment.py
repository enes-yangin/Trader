import warnings
import threading
from utils.config import NEWS
from utils.logger import get_logger

warnings.filterwarnings("ignore")

log = get_logger("sentiment")

FINBERT_MODEL = "ProsusAI/finbert"
DISTILBERT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

_finbert = None
_distilbert = None
_vader = None
_backend = None
_lock = threading.Lock()


def _quantize(model):
    if not NEWS.sentiment_quantize:
        return model
    try:
        import torch
        return torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    except (ImportError, RuntimeError, AttributeError) as e:
        log.warning(f"Dynamic quantization unavailable ({type(e).__name__}: {e}), using full precision")
        return model


def _load_transformer_pipeline(model_name):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model = _quantize(model)
    return pipeline("sentiment-analysis", model=model, tokenizer=tok,  # type: ignore[call-overload]
                    truncation=True, max_length=512)


def _load_finbert():
    global _finbert, _backend
    try:
        _finbert = _load_transformer_pipeline(FINBERT_MODEL)
        _backend = "finbert"
        log.info(f"Sentiment backend: FinBERT loaded (quantized={NEWS.sentiment_quantize})")
        return True
    except (ImportError, OSError, RuntimeError) as e:
        log.warning(f"FinBERT unavailable ({type(e).__name__}: {e}), trying fallback")
        return False


def _load_distilbert():
    global _distilbert, _backend
    try:
        _distilbert = _load_transformer_pipeline(DISTILBERT_MODEL)
        _backend = "distilbert"
        log.info(f"Sentiment backend: DistilBERT loaded (quantized={NEWS.sentiment_quantize})")
        return True
    except (ImportError, OSError, RuntimeError) as e:
        log.warning(f"DistilBERT unavailable ({type(e).__name__}: {e}), trying fallback")
        return False


def _load_vader():
    global _vader, _backend
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader = SentimentIntensityAnalyzer()
        _backend = "vader"
        log.info("Sentiment backend: VADER loaded")
        return True
    except (ImportError, OSError) as e:
        log.warning(f"VADER unavailable ({type(e).__name__}: {e})")
        return False


_LOADERS = {
    "finbert": _load_finbert,
    "distilbert": _load_distilbert,
    "vader": _load_vader,
}

_FALLBACK_ORDER = ["finbert", "distilbert", "vader"]


def init(prefer=NEWS.sentiment_backend):
    global _backend
    if _backend is not None:
        return _backend
    with _lock:
        if _backend is not None:
            return _backend

        order = [prefer] + [b for b in _FALLBACK_ORDER if b != prefer]
        for name in order:
            loader = _LOADERS.get(name)
            if loader and loader():
                return _backend

        _backend = "none"
        log.warning("No sentiment backend available, using neutral (0.0) scores")
        return _backend


def _pipeline_for_backend():
    if _backend == "finbert":
        return _finbert
    if _backend == "distilbert":
        return _distilbert
    return None


def score_text(txt):
    if _backend is None:
        init()
    if not txt or not txt.strip():
        return 0.0
    pipe = _pipeline_for_backend()
    if pipe is not None:
        r = pipe(txt[:512])[0]
        lbl = r["label"].lower()
        s = r["score"]
        if lbl in ("positive", "pos"):
            return s
        if lbl in ("negative", "neg"):
            return -s
        return 0.0
    if _backend == "vader":
        assert _vader is not None
        return _vader.polarity_scores(txt)["compound"]
    return 0.0


def score_batch(texts):
    if _backend is None:
        init()
    if not texts:
        return []
    pipe = _pipeline_for_backend()
    if pipe is not None:
        out = []
        res = pipe([t[:512] for t in texts])
        for r in res:
            lbl = r["label"].lower()
            s = r["score"]
            if lbl in ("positive", "pos"):
                out.append(s)
            elif lbl in ("negative", "neg"):
                out.append(-s)
            else:
                out.append(0.0)
        return out
    return [score_text(t) for t in texts]


def get_backend():
    if _backend is None:
        init()
    return _backend
