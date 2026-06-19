import dataclasses
import importlib
from unittest.mock import patch
import pytest
from utils.config import NEWS
from utils.exceptions import ConfigError


def _fresh_sentiment_module():
    import data.sentiment as sentiment
    importlib.reload(sentiment)
    return sentiment


def test_init_falls_back_to_vader_without_network():
    sentiment = _fresh_sentiment_module()
    backend = sentiment.init(prefer="finbert")
    assert backend in ("finbert", "distilbert", "vader", "none")
    assert sentiment.get_backend() == backend


def test_score_text_empty_string_returns_zero():
    sentiment = _fresh_sentiment_module()
    assert sentiment.score_text("") == 0.0
    assert sentiment.score_text("   ") == 0.0


def test_score_batch_empty_list_returns_empty():
    sentiment = _fresh_sentiment_module()
    assert sentiment.score_batch([]) == []


def test_init_dispatch_order_with_mocked_loaders():
    sentiment = _fresh_sentiment_module()
    sentiment._backend = None

    calls = []

    def fake_finbert():
        calls.append("finbert")
        return False

    def fake_distilbert():
        calls.append("distilbert")
        sentiment._backend = "distilbert"
        return True

    def fake_vader():
        calls.append("vader")
        return True

    with patch.dict(sentiment._LOADERS, {
            "finbert": fake_finbert,
            "distilbert": fake_distilbert,
            "vader": fake_vader,
        }):
        backend = sentiment.init(prefer="finbert")

    assert backend == "distilbert"
    assert calls[0] == "finbert"
    assert calls[1] == "distilbert"
    assert "vader" not in calls


def test_init_prefer_vader_skips_transformers():
    sentiment = _fresh_sentiment_module()
    sentiment._backend = None

    calls = []

    def fake_vader():
        calls.append("vader")
        sentiment._backend = "vader"
        return True

    def fail(*a, **kw):
        calls.append("transformer")
        return False

    with patch.dict(sentiment._LOADERS, {
        "finbert": fail,
        "distilbert": fail,
        "vader": fake_vader,
    }):
        backend = sentiment.init(prefer="vader")

    assert backend == "vader"
    assert calls == ["vader"]


def test_init_all_backends_unavailable_returns_none():
    sentiment = _fresh_sentiment_module()
    sentiment._backend = None

    def fail(*a, **kw):
        return False

    with patch.dict(sentiment._LOADERS, {
        "finbert": fail, "distilbert": fail, "vader": fail,
    }):
        backend = sentiment.init(prefer="finbert")

    assert backend == "none"
    assert sentiment.score_text("good news") == 0.0


def test_quantize_handles_failure_gracefully():
    sentiment = _fresh_sentiment_module()

    class _FakeModel:
        pass

    with patch("torch.quantization.quantize_dynamic", side_effect=RuntimeError("boom")):
        out = sentiment._quantize(_FakeModel())
    assert isinstance(out, _FakeModel)


def test_quantize_disabled_returns_model_unchanged():
    sentiment = _fresh_sentiment_module()
    disabled = dataclasses.replace(NEWS, sentiment_quantize=False)
    with patch("data.sentiment.NEWS", disabled):
        sentinel = object()
        assert sentiment._quantize(sentinel) is sentinel


def test_news_config_accepts_distilbert():
    from utils.config import NewsConfig
    cfg = NewsConfig(sentiment_backend="distilbert")
    cfg.__post_init__()
    assert cfg.sentiment_backend == "distilbert"


def test_news_config_rejects_unknown_backend_strict():
    from utils.config import NewsConfig
    with pytest.raises(ConfigError):
        NewsConfig(sentiment_backend="gpt-oracle")


def test_news_config_quantize_default_true():
    from utils.config import NEWS
    assert isinstance(NEWS.sentiment_quantize, bool)
