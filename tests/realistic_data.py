import numpy as np
import pandas as pd


def make_realistic_ohlcv(n=900, seed=0, mu=0.0003, base_vol=0.025,
                          vol_persist=0.92, vol_of_vol=0.35,
                          momentum=0.04, reversion=-0.03, t_df=4.0,
                          start_price=30000.0, symbol="BTC/USDT"):
    """Generate OHLCV whose daily returns carry the statistical signature of
    real crypto: volatility clustering (GARCH-like), heavy tails (Student-t),
    and only faint serial structure (small momentum + mean reversion).

    The point is realism for R2 testing: a correctly built model should land
    near zero out-of-sample R2 on this data, exactly as it would on real
    returns, because the predictable component is tiny relative to noise.
    Pure IID random walk understates achievable R2; this slightly overstates
    it via the small AR terms, bracketing the real-data regime.
    """
    rng = np.random.default_rng(seed)

    log_var = np.log(base_vol ** 2)
    rets = np.zeros(n)
    h = log_var
    prev = 0.0
    for i in range(n):
        h = vol_persist * h + (1 - vol_persist) * log_var + vol_of_vol * rng.standard_normal()
        sigma = np.sqrt(np.exp(h))
        shock = rng.standard_t(t_df) / np.sqrt(t_df / (t_df - 2))
        r = mu + momentum * prev + reversion * prev + sigma * shock
        rets[i] = r
        prev = r

    price = start_price * np.exp(np.cumsum(rets))
    idx = pd.date_range(end=pd.Timestamp.now().normalize(), periods=n, freq="D")

    intraday = np.abs(rng.standard_normal(n)) * (base_vol * 0.6)
    close = price
    open_ = np.empty(n)
    open_[0] = start_price
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) * (1 + intraday)
    low = np.minimum(open_, close) * (1 - intraday)
    vol = rng.lognormal(mean=10.0, sigma=0.8, size=n)

    df = pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": vol,
    }, index=idx)
    df.index.name = "date"
    df.attrs["symbol"] = symbol
    df.attrs["source"] = "realistic_synthetic"
    return df


def _ohlcv_from_close(close, idx, rng, base_vol, symbol):
    """Wrap a close series in plausible OHLCV so it matches the shape the rest
    of the pipeline expects. Only `close` carries the cointegration structure;
    O/H/L are cosmetic intraday wiggle around it."""
    n = len(close)
    intraday = np.abs(rng.standard_normal(n)) * base_vol
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) * (1 + intraday)
    low = np.minimum(open_, close) * (1 - intraday)
    vol = rng.lognormal(mean=10.0, sigma=0.8, size=n)
    df = pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": vol,
    }, index=idx)
    df.index.name = "date"
    df.attrs["symbol"] = symbol
    df.attrs["source"] = "realistic_synthetic"
    return df


def make_cointegrated_pair(n=900, seed=0, beta=0.065, alpha=0.0,
                            trend_vol=300.0, spread_half_life=15.0,
                            spread_vol=40.0, start_x=30000.0,
                            sym_x="BTC/USDT", sym_y="ETH/USDT"):
    """Generate two aligned price series that are cointegrated.

    Construction (Engle-Granger ground truth):
      x_t = a random walk  -> I(1), the shared stochastic trend.
      s_t = alpha + AR(1) mean-reverting process -> I(0), the spread.
      y_t = beta * x_t + s_t.
    Then y_t - beta*x_t = s_t is stationary, so (x, y) are cointegrated with
    hedge ratio beta and a spread that reverts to its mean.

    We need this because the sandbox cannot pull two real synchronized symbol
    feeds; the only honest way to exercise the cointegration -> signal ->
    purged-evaluation chain is to plant a known cointegration and check the
    machinery recovers it. The edge is easy to find *because* we planted it --
    the value on display is the methodology, not the alpha.

    `spread_half_life` sets how fast the spread reverts (phi = 2**(-1/HL));
    `trend_vol`/`spread_vol` set the magnitudes of the common trend vs. the
    mean-reverting deviation. Returns (df_x, df_y) as OHLCV frames; the true
    parameters are stashed in df_y.attrs for tests to assert against.
    """
    rng = np.random.default_rng(seed)

    # Shared stochastic trend: a random walk (integrated, non-stationary).
    x = start_x + np.cumsum(rng.standard_normal(n) * trend_vol)
    x = np.maximum(x, 1.0)

    # Mean-reverting spread: AR(1) with the requested half-life.
    phi = 2.0 ** (-1.0 / spread_half_life)
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = phi * s[t - 1] + spread_vol * rng.standard_normal()

    y = alpha + beta * x + s
    y = np.maximum(y, 1.0)

    idx = pd.date_range(end=pd.Timestamp.now().normalize(), periods=n, freq="D")
    df_x = _ohlcv_from_close(x, idx, rng, base_vol=0.015, symbol=sym_x)
    df_y = _ohlcv_from_close(y, idx, rng, base_vol=0.015, symbol=sym_y)
    df_y.attrs["true_beta"] = beta
    df_y.attrs["true_alpha"] = alpha
    df_y.attrs["spread_half_life"] = spread_half_life
    return df_x, df_y


def make_independent_walks(n=900, seed=0, trend_vol=300.0,
                            start_x=30000.0, start_y=2000.0,
                            sym_x="BTC/USDT", sym_y="ETH/USDT"):
    """Two independent random walks -- NOT cointegrated. A correct
    cointegration test must FAIL to reject the unit root on their spread.
    Used as the negative control for the stat-arb tests."""
    rng = np.random.default_rng(seed)
    x = np.maximum(start_x + np.cumsum(rng.standard_normal(n) * trend_vol), 1.0)
    y = np.maximum(start_y + np.cumsum(rng.standard_normal(n) * trend_vol * 0.07), 1.0)
    idx = pd.date_range(end=pd.Timestamp.now().normalize(), periods=n, freq="D")
    df_x = _ohlcv_from_close(x, idx, rng, base_vol=0.015, symbol=sym_x)
    df_y = _ohlcv_from_close(y, idx, rng, base_vol=0.015, symbol=sym_y)
    return df_x, df_y


def return_stats(df):
    r = df["close"].pct_change().dropna()
    return {
        "mean": float(r.mean()),
        "std": float(r.std()),
        "skew": float(r.skew()),
        "kurtosis": float(r.kurtosis()),
        "acf1": float(r.autocorr(lag=1)),
        "vol_clustering": float(r.abs().autocorr(lag=1)),
    }
