"""Point-in-time adjusted closing prices and regime-switching synthetic assets."""
import numpy as np
import pandas as pd


def synthetic_prices(days=900, seed=42):
    if days < 180:
        raise ValueError("days must be >=180")
    rng = np.random.default_rng(seed)
    names = ["equity", "bonds", "gold", "commodities"]
    mu = np.array([[.0006, .00012, .0002, .0003], [-.0010, .0003, .00055, -.0008]])
    sigma = np.array([[.009, .0035, .007, .012], [.021, .008, .013, .024]])
    corr = np.array([[1, -.1, .15, .55], [-.1, 1, -.05, -.15], [.15, -.05, 1, .25], [.55, -.15, .25, 1.]])
    ret = np.zeros((days, len(names)))
    regime = 0
    for t in range(days):
        if rng.random() < .025:
            regime = 1 - regime
        covariance = np.outer(sigma[regime], sigma[regime]) * corr
        ret[t] = rng.multivariate_normal(mu[regime], covariance)
    return pd.DataFrame(100*np.exp(np.cumsum(ret, axis=0)),
                        index=pd.bdate_range("2018-01-01", periods=days), columns=names)


def load_prices(path):
    df = pd.read_csv(path)
    if "date" not in df.columns or len(df.columns) < 3:
        raise ValueError("Need date and at least two adjusted price columns")
    dates = pd.to_datetime(df.pop("date"), utc=True, errors="raise")
    prices = df.apply(pd.to_numeric, errors="raise")
    x = prices.to_numpy(float)
    if (len(prices) < 180 or dates.isna().any() or dates.duplicated().any() or
        not dates.is_monotonic_increasing or not np.isfinite(x).all() or (x <= 0).any()):
        raise ValueError("Need >=180 positive finite prices and strictly increasing unique dates")
    prices.index = dates
    return prices


def daily_returns(prices):
    values = prices.to_numpy(dtype=float)
    if (values.ndim != 2 or values.shape[1] < 2 or len(values) < 40 or
        not np.isfinite(values).all() or np.any(values <= 0) or
        not prices.index.is_monotonic_increasing or not prices.index.is_unique):
        raise ValueError("Need ordered, positive daily prices for two or more assets")
    return pd.DataFrame(values[1:]/values[:-1]-1., index=prices.index[1:], columns=prices.columns)
