import json
import numpy as np
import pandas as pd
import pytest
import torch
from rl_portfolio.data import synthetic_prices,load_prices,daily_returns
from rl_portfolio.environment import PortfolioEnv,logits_to_weights
from rl_portfolio.agents import PPO,SAC,ReplayBuffer
from rl_portfolio.benchmark import Config,equal_risk_weights,evaluate_agent,performance,run_experiment
from rl_portfolio.cli import main


def test_softmax_simplex_and_logit_invariance():
    w=logits_to_weights([-.5,.1,.9],4)
    assert w.shape==(4,) and np.all(w>0) and np.isclose(w.sum(),1.)
    with pytest.raises(ValueError):logits_to_weights([0.,0.],4)


def test_no_future_returns_in_observations():
    r=daily_returns(synthetic_prices(240)).to_numpy()
    changed=r.copy();changed[180:]+=0.08
    a=PortfolioEnv(r,lookback=20,start=100,end=170)
    b=PortfolioEnv(changed,lookback=20,start=100,end=170)
    assert np.array_equal(a.reset(),b.reset())
    for _ in range(50):
        sa,ra,da,_=a.step(np.zeros(3))
        sb,rb,db,_=b.step(np.zeros(3))
        assert ra==rb and da==db
        assert np.array_equal(sa,sb)


def test_turnover_costs_and_wealth():
    r=np.tile(np.array([.02,0.]),(80,1))
    e=PortfolioEnv(r,lookback=10,start=10,end=12,cost_bps=10.)
    e.reset()
    _,_,_,info=e.step_weights([1.,0.])
    assert np.isclose(info['turnover'],1.)
    assert np.isclose(info['net_return'],.02-.001)
    assert np.isclose(info['wealth'],1.019)


def test_portfolio_drift_and_done():
    r=np.tile([.1,0.],(40,1))
    e=PortfolioEnv(r,lookback=6,start=6,end=8)
    e.reset()
    _,_,done,_=e.step_weights([.5,.5])
    assert not done and e.weights[0]>.5
    _,_,done,_=e.step_weights([.5,.5])
    assert done
    with pytest.raises(RuntimeError):e.step_weights([.5,.5])


def test_terminal_and_invalid_inputs():
    with pytest.raises(ValueError):PortfolioEnv(np.zeros((20,1)))
    e=PortfolioEnv(np.zeros((60,2)),start=20)
    with pytest.raises(ValueError):e.step_weights([-.2,1.2])
    with pytest.raises(ValueError):e.step_weights([.1,.1])


def test_actor_log_probability_finite():
    p=PPO(8,1,hidden=8)
    x=torch.randn(4,8)
    actions,lp=p.actor.sample(x)
    assert actions.shape==(4,1) and lp.shape==(4,)
    assert torch.isfinite(lp).all()
    assert torch.allclose(lp,p.actor.evaluate_log_prob(x,actions),atol=1e-4)


def test_sac_policy_bounded_and_critics():
    a=SAC(8,1,hidden=8,seed=3)
    obs=np.zeros(8,dtype=np.float32)
    action=a.act(obs)
    assert len(action)==1 and np.max(np.abs(action))<=1
    assert np.isfinite(a.q1(torch.zeros(2,9)).detach().numpy()).all()


def test_replay_buffer_shapes():
    buf=ReplayBuffer(seed=3)
    for _ in range(12):buf.add(np.ones(8),[.1],.02,np.zeros(8),False)
    s,a,r,ns,d=buf.sample(5)
    assert s.shape==(5,8) and a.shape==(5,1) and d.shape==(5,)


def test_equal_risk_budget():
    covariance=np.array([[.09,.01,.012],[.01,.04,.007],[.012,.007,.025]])
    w=equal_risk_weights(covariance)
    rc=w*(covariance@w)
    assert np.allclose(rc/rc.sum(),1/3,atol=1e-5)


def test_ppo_short_training():
    r=daily_returns(synthetic_prices(180)).to_numpy()
    env=PortfolioEnv(r,lookback=10,start=10,end=110,episode_length=20)
    p=PPO(16,3,hidden=8)
    log=p.train(env,steps=48,rollout=16,epochs=1,batch_size=16)
    assert log['steps']==48
    assert np.isfinite(p.act(env.reset())).all()


def test_sac_short_training():
    r=daily_returns(synthetic_prices(180)).to_numpy()
    env=PortfolioEnv(r,lookback=10,start=10,end=110,episode_length=20)
    agent=SAC(16,3,hidden=8)
    log=agent.train(env,steps=40,warmup=8,batch_size=8)
    assert log['updates']>0
    assert np.isfinite(agent.act(env.reset())).all()


def test_shared_test_range_and_costs():
    prices=synthetic_prices(220)
    r=daily_returns(prices).to_numpy()
    result=evaluate_agent('Equal weight',r,start=165,end=210,lookback=10,cost_bps=5,method='Equal weight')
    assert len(result)==45
    assert np.allclose(result.net_return,result.gross_return-result.turnover*.0005)
    assert np.allclose(result[[f'w_{i}' for i in range(4)]].sum(axis=1),1)


def test_performance_zero_and_bad():
    assert performance(np.zeros(20))['sharpe']==0
    with pytest.raises(ValueError):performance([-1.2,.1])


def test_csv_validation(tmp_path):
    p=tmp_path/'bad.csv'
    p.write_text('date,a,b\n2020-01-03,100,105\n2020-01-03,101,104\n')
    with pytest.raises(ValueError):load_prices(p)


def test_full_experiment_and_cli(tmp_path):
    cfg=Config(lookback=10,ppo_steps=32,sac_steps=40,hidden=8,episode_length=15,seed=12)
    report,results=run_experiment(synthetic_prices(220),cfg)
    assert len(report['metrics'])==5
    assert set(results.split)=={'validation','test'}
    assert results.groupby(['split','strategy']).size().nunique()==1
    assert np.isfinite(results.net_return).all()
    main(['--days','220','--lookback','10','--ppo-steps','32',
          '--sac-steps','40','--hidden','8','--episode-length','15','--output',str(tmp_path)])
    for file in ('walk_forward.csv','metrics.json','equity_curves.png'):
        assert (tmp_path/file).exists()
    assert len(json.loads((tmp_path/'metrics.json').read_text())['research']['metrics'])==5
