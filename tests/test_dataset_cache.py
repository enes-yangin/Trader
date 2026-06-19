from unittest.mock import patch


def test_dataset_cache_separates_with_news_flag(mock_ccxt_exchange, isolated_cache_dir):
    from data import dataset

    with patch("ccxt.binance", return_value=mock_ccxt_exchange):
        no_news = dataset.build("ETH/USDT", with_news=False, allow_sample=True)
        assert "sentiment_avg" not in no_news.columns

        with_news = dataset.build("ETH/USDT", with_news=True, allow_sample=True)
        assert "sentiment_avg" in with_news.columns

        no_news_again = dataset.ensure("ETH/USDT", with_news=False, allow_sample=True)
        assert "sentiment_avg" not in no_news_again.columns

        with_news_again = dataset.ensure("ETH/USDT", with_news=True, allow_sample=True)
        assert "sentiment_avg" in with_news_again.columns


def test_dataset_has_cache_and_load_cached_respect_with_news(mock_ccxt_exchange, isolated_cache_dir):
    from data import dataset

    with patch("ccxt.binance", return_value=mock_ccxt_exchange):
        dataset.build("BNB/USDT", with_news=False, allow_sample=True)

        assert dataset.has_cache("BNB/USDT", with_news=False) is True
        assert dataset.has_cache("BNB/USDT", with_news=True) is False

        cached = dataset.load_cached("BNB/USDT", with_news=False)
        assert cached is not None
        assert "sentiment_avg" not in cached.columns

        assert dataset.load_cached("BNB/USDT", with_news=True) is None


def test_train_all_with_news_then_without_does_not_cross_contaminate(mock_ccxt_exchange, isolated_cache_dir):
    from engine.trainer import train_all

    with patch("ccxt.binance", return_value=mock_ccxt_exchange):
        b_no_news = train_all("ADA/USDT", with_news=False, with_micro=True, allow_sample=True)
        assert "sentiment_avg" not in b_no_news["df"].columns

        b_news = train_all("ADA/USDT", with_news=True, with_micro=True, allow_sample=True)
        assert "sentiment_avg" in b_news["df"].columns
        assert "news_analysis" in b_news

        b_no_news_again = train_all("ADA/USDT", with_news=False, with_micro=True, allow_sample=True)
        assert "sentiment_avg" not in b_no_news_again["df"].columns
