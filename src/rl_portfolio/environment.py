"""Long-only fully-invested historical portfolio environment.

At decision t, features use returns [t-lookback,t), not return t. The order is
assumed filled at the beginning of return t with a proportional turnover cost.
Actual open execution, spreads and impact are *not* modeled.
"""
import numpy as np


def logits_to_weights(action, n_assets):
    """Independent N-1 bounded logits with anchor 0, mapped into an N-simplex."""
    z = np.asarray(action, float)
    if z.shape != (n_assets-1,) or not np.isfinite(z).all():
        raise ValueError("Expected finite N-1 logits")
    z = np.r_[np.clip(z, -1, 1)*3, 0.]
    z -= z.max()
    v = np.exp(z)
    return v/v.sum()


class PortfolioEnv:
    def __init__(self, returns, lookback=20, start=None, end=None,
                 cost_bps=5., episode_length=None, seed=42):
        self.returns = np.asarray(returns, float)
        if (self.returns.ndim != 2 or self.returns.shape[1] < 2 or
            not np.isfinite(self.returns).all() or np.any(self.returns <= -1)):
            raise ValueError("Return matrix must be finite T x N, N>=2 and individual returns >-100%")
        self.n_assets = self.returns.shape[1]
        self.lookback = int(lookback)
        self.start = self.lookback if start is None else int(start)
        self.end = len(self.returns) if end is None else int(end)
        if (self.lookback < 5 or self.start < self.lookback or self.end > len(self.returns) or
            self.end <= self.start or cost_bps < 0 or not np.isfinite(cost_bps)):
            raise ValueError("Invalid environment boundaries")
        if episode_length is not None and (episode_length < 1 or episode_length > self.end-self.start):
            raise ValueError("Invalid episode length")
        self.episode_length = episode_length
        self.cost_bps = float(cost_bps)
        self.rng = np.random.default_rng(seed)
        self.t = self.start
        self.stop = self.end
        self.weights = np.ones(self.n_assets)/self.n_assets
        self.wealth = 1.
        self.observation_size = 4*self.n_assets
        self.action_size = self.n_assets - 1

    def observe(self):
        x = self.returns[self.t-self.lookback:self.t]
        # Fixed scales from a declared constant, never fitted on test observations.
        mean = np.clip(x.mean(axis=0)/.02, -5, 5)
        vol = np.clip(x.std(axis=0)/.02, 0, 5)
        last = np.clip(x[-1]/.02, -5, 5)
        return np.r_[mean, vol, last, self.weights].astype('float32')

    def reset(self, random_start=False):
        self.t = self.start
        if random_start and self.episode_length is not None:
            self.t = int(self.rng.integers(self.start, self.end-self.episode_length+1))
        self.stop = min(self.end, self.t+self.episode_length) if self.episode_length else self.end
        self.weights = np.ones(self.n_assets)/self.n_assets
        self.wealth = 1.
        return self.observe()

    def step_weights(self, target):
        target = np.asarray(target, float)
        if (target.shape != (self.n_assets,) or not np.isfinite(target).all() or
            np.any(target < -1e-10) or not np.isclose(target.sum(), 1, atol=1e-7)):
            raise ValueError("Portfolio weights must be long-only and sum to 1")
        if self.t >= self.stop:
            raise RuntimeError("Episode finished")
        turnover = float(np.abs(target-self.weights).sum())
        cost = turnover*self.cost_bps/10000
        asset_returns = self.returns[self.t]
        gross = float(target@asset_returns)
        net = gross-cost
        if net <= -1 or gross <= -1:
            raise ValueError("Portfolio bankrupt")
        self.weights = target*(1+asset_returns)/(1+gross)
        self.wealth *= (1+net)
        self.t += 1
        done = self.t >= self.stop
        state = None if done else self.observe()
        return state, float(100*net), bool(done), {
            "gross_return": gross, "net_return": net, "cost": cost,
            "turnover": turnover, "target_weights": target.copy(),
            "wealth": float(self.wealth), "t": self.t-1}

    def step(self, action):
        return self.step_weights(logits_to_weights(action, self.n_assets))
