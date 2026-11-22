"""Uniform out-of-sample comparison; no test-data model selection."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .data import daily_returns
from .agents import PPO,SAC
from .environment import PortfolioEnv,logits_to_weights


@dataclass(frozen=True)
class Config:
    lookback: int=20
    train_fraction: float=.60
    val_fraction: float=.20
    cost_bps: float=5.
    ppo_steps: int=2048
    sac_steps: int=2048
    hidden: int=64
    seed: int=42
    episode_length: int=64


def _risk_parity_cov(history):
    cov=np.atleast_2d(np.cov(history,rowvar=False))
    cov=.8*cov+.2*np.diag(np.diag(cov))+np.eye(cov.shape[0])*1e-9
    return cov


def equal_risk_weights(cov):
    """ERC weights by a numerically scaled convex log-barrier program."""
    cov=np.asarray(cov,float)
    n=len(cov)
    cov=cov/np.mean(np.diag(cov))
    def fun(z):
        x=np.exp(np.clip(z,-30,30))
        return .5*x@cov@x-z.sum()
    def grad(z):
        x=np.exp(np.clip(z,-30,30))
        return x*(cov@x)-1
    sol=minimize(fun,np.zeros(n),jac=grad,method='BFGS',options={'gtol':1e-9,'maxiter':500})
    w=np.exp(np.clip(sol.x,-30,30))
    w=w/w.sum()
    rc=w*(cov@w)
    if not np.all(np.isfinite(w)) or np.max(np.abs(rc/rc.sum()-1/n))>1e-5:
        raise RuntimeError('Risk parity optimization failed')
    return w


def _baseline_target(name,env):
    x=env.returns[env.t-env.lookback:env.t]
    if name=='Equal weight':return np.ones(env.n_assets)/env.n_assets
    cov=_risk_parity_cov(x)
    if name=='Inverse volatility':
        inv=1/np.sqrt(np.diag(cov))
        return inv/inv.sum()
    if name=='Risk parity':return equal_risk_weights(cov)
    raise ValueError('Unknown baseline')


def evaluate_agent(agent,returns,start,end,lookback,cost_bps,method):
    env=PortfolioEnv(returns,lookback,start,end,cost_bps)
    state=env.reset()
    rows=[]
    while True:
        if isinstance(agent,str):
            target=_baseline_target(agent,env)
            ns,reward,done,info=env.step_weights(target)
        else:
            action=agent.act(state,deterministic=True)
            ns,reward,done,info=env.step(action)
        rows.append({'t':info['t'],'strategy':method,'net_return':info['net_return'],
                     'gross_return':info['gross_return'],'turnover':info['turnover'],
                     'wealth':info['wealth'],**{f'w_{i}':v for i,v in enumerate(info['target_weights'])}})
        if done:break
        state=ns
    return pd.DataFrame(rows)


def performance(simple_returns):
    r=np.asarray(simple_returns,float)
    if r.ndim!=1 or not len(r) or not np.isfinite(r).all() or np.any(r<=-1):
        raise ValueError('Expected valid nonbankrupt simple returns')
    growth=np.cumprod(1+r)
    peak=np.maximum.accumulate(np.r_[1.,growth])[1:]
    std=float(r.std(ddof=1)) if len(r)>1 else 0.
    return {'total_return':float(growth[-1]-1),
            'annualized_return':float(growth[-1]**(252/len(r))-1),
            'sharpe':float(r.mean()/std*np.sqrt(252)) if std>0 else 0.,
            'max_drawdown':float(np.min(growth/peak-1)),
            'annualized_volatility':float(std*np.sqrt(252))}


def run_experiment(prices,cfg=Config()):
    if not .4<=cfg.train_fraction<=.8 or not .05<=cfg.val_fraction<=.3 or cfg.train_fraction+cfg.val_fraction>=.9:
        raise ValueError('Invalid fractions')
    if cfg.lookback<5 or cfg.ppo_steps<1 or cfg.sac_steps<1 or cfg.hidden<4 or cfg.episode_length<5:
        raise ValueError('Invalid training configuration')
    returns_df=daily_returns(prices)
    r=returns_df.to_numpy()
    train_end=int(len(r)*cfg.train_fraction)
    val_end=int(len(r)*(cfg.train_fraction+cfg.val_fraction))
    if train_end-cfg.lookback<cfg.episode_length or len(r)-val_end<20 or val_end-train_end<20:
        raise ValueError('Insufficient observations for train/validation/test')
    def training_env(seed):
        return PortfolioEnv(r,lookback=cfg.lookback,start=cfg.lookback,end=train_end,
                            cost_bps=cfg.cost_bps,episode_length=cfg.episode_length,seed=seed)
    obs_dim=4*r.shape[1]
    act_dim=r.shape[1]-1
    ppo=PPO(obs_dim,act_dim,hidden=cfg.hidden,seed=cfg.seed)
    ppo_training=ppo.train(training_env(cfg.seed),steps=cfg.ppo_steps)
    sac=SAC(obs_dim,act_dim,hidden=cfg.hidden,seed=cfg.seed)
    sac_training=sac.train(training_env(cfg.seed+1),steps=cfg.sac_steps,
                           warmup=min(128,max(0,cfg.sac_steps//4)),batch_size=min(64,max(2,cfg.sac_steps//4)))
    methods={'PPO':ppo,'SAC':sac,'Equal weight':'Equal weight',
             'Inverse volatility':'Inverse volatility','Risk parity':'Risk parity'}
    pieces=[]
    stats={}
    for split,start,end in [('validation',train_end,val_end),('test',val_end,len(r))]:
        for name,agent in methods.items():
            frame=evaluate_agent(agent,r,start,end,cfg.lookback,cfg.cost_bps,name)
            frame['split']=split
            frame['date']=returns_df.index[frame['t']].to_numpy()
            pieces.append(frame)
            stats.setdefault(name,{})[split]={**performance(frame['net_return']),
                                                'total_turnover':float(frame.turnover.sum())}
    return {'train_end':str(returns_df.index[train_end-1]),
            'validation_end':str(returns_df.index[val_end-1]),
            'split_rows':{'train':train_end,'validation':val_end-train_end,'test':len(r)-val_end},
            'training':{'PPO':ppo_training,'SAC':sac_training},'metrics':stats},pd.concat(pieces,ignore_index=True)
