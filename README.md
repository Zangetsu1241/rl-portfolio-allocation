# PPO vs SAC: Reinforcement Learning Portfolio Allocation

Test-driven quantitative research that compares custom PyTorch **Proximal Policy Optimization (PPO)** and **Soft Actor-Critic (SAC)** agents with **equal weight**, **inverse volatility**, and **equal-risk-contribution** portfolios.

## Research question
Do policy-gradient and off-policy continuous-control agents produce superior held-out, cost-adjusted portfolio outcomes relative to simple allocation rules? No outperformance is assumed.

## Install and run
Python 3.10+:
```bash
python -m pip install -e '.[test]'
python -m pytest -q
rl-portfolio --days 900 --ppo-steps 2048 --sac-steps 2048 --output results
```

Short CI smoke experiment:
```bash
rl-portfolio --days 220 --lookback 10 --ppo-steps 32 --sac-steps 40 --hidden 8 --episode-length 15 --output results
```

Optional CSV: `date` plus **at least two point-in-time adjusted, positive daily close columns**, in ascending date order.

```bash
rl-portfolio --csv adjusted_prices.csv --output results-market
```

Outputs: `walk_forward.csv`, `metrics.json`, `equity_curves.png`.

## Research design
- Four correlated synthetic assets with changing means and volatility regimes or a user-provided adjusted-price panel.
- 60% chronological training, 20% validation and 20% held-out testing. No training or checkpoint selection on test periods.
- State at decision t: trailing mean, volatility, previous daily return and drifted portfolio holdings. Features use only returns through t-1.
- PPO: clipped policy updates and generalized advantage estimation. SAC: two Q critics, replay buffer, stochastic actor, target networks and Polyak averaging.
- N-1 continuous bounded logits are transformed by anchored softmax to nonnegative, fully invested N-asset portfolios. SAC entropy is defined in latent bounded-logit action space, not simplex weights.
- Random training episodes are confined to the training period. Deterministic evaluation is used on validation and test.
- Equal-weight, inverse-volatility and equal-risk-contribution baselines share the same test dates.
- Portfolio drift, absolute turnover and per-trade proportional costs are modeled. Learning reward is 100 times net daily simple return.

## Synthetic benchmark
```bash
rl-portfolio --days 650 --ppo-steps 600 --sac-steps 600 --hidden 32 --output results
```

| Strategy | Test total return | Test Sharpe | Test maximum drawdown |
|---|---:|---:|---:|
| PPO | -7.24% | -1.045 | -11.98% |
| SAC | -6.61% | -0.823 | -12.00% |
| Equal weight | -8.02% | -1.032 | -13.39% |
| Inverse volatility | -5.65% | -0.979 | -9.40% |
| Risk parity | -4.75% | -0.850 | -8.20% |

Risk parity had the least-negative return on this particular **synthetic** held-out test. These results are not empirical or live investment results; a short RL training budget is only a research demonstration.

## Testing
15 tests validate simplex constraints, historical-only features, portfolio accounting, terminal transitions, actor log probabilities, PPO/SAC training, replay buffer, equal-risk contributions, shared evaluation, input validation and CLI integration. GitHub Actions targets Python 3.10 and 3.12.

## Limitations
The backtest assumes an order fill at the beginning of a close-to-close return period. Trading after observing that closing price cannot execute at the same close. Real execution requires next-open or order-cutoff modeling, fees, spreads, slippage, market impact, financing and taxes. Synthetic market behavior, architecture, reward design and hyperparameters are simplified. Results are seed- and sample-dependent. No guarantee of convergence or outperformance. Real datasets require point-in-time universes, delistings, calendar and corporate-action adjustments.

MIT licensed.
