"""End-to-end research runner with synthetic or adjusted price data."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import synthetic_prices,load_prices
from .benchmark import Config,run_experiment


def main(argv=None):
    p=argparse.ArgumentParser(description='PPO vs SAC portfolio allocation research')
    p.add_argument('--csv',help='Date column followed by 2+ adjusted prices')
    p.add_argument('--days',type=int,default=900)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--lookback',type=int,default=20)
    p.add_argument('--ppo-steps',type=int,default=2048)
    p.add_argument('--sac-steps',type=int,default=2048)
    p.add_argument('--hidden',type=int,default=64)
    p.add_argument('--episode-length',type=int,default=64)
    p.add_argument('--cost-bps',type=float,default=5.)
    p.add_argument('--output',default='results')
    args=p.parse_args(argv)
    import torch
    torch.set_num_threads(1)
    prices=load_prices(args.csv) if args.csv else synthetic_prices(args.days,args.seed)
    cfg=Config(lookback=args.lookback,ppo_steps=args.ppo_steps,sac_steps=args.sac_steps,
               hidden=args.hidden,seed=args.seed,episode_length=args.episode_length,cost_bps=args.cost_bps)
    metrics,results=run_experiment(prices,cfg)
    dest=Path(args.output)
    dest.mkdir(parents=True,exist_ok=True)
    results.to_csv(dest/'walk_forward.csv',index=False)
    payload={'data':'user adjusted-price CSV' if args.csv else 'synthetic regime-switching assets',
             'configuration':vars(args),'research':metrics,
             'limitations':'Held-out historical simulation; synthetic results are not evidence of live trading performance. Close-to-close cost and fill approximations.'}
    (dest/'metrics.json').write_text(json.dumps(payload,indent=2))
    fig,ax=plt.subplots(figsize=(11,6))
    for name,df in results[results.split=='test'].groupby('strategy'):
        ax.plot(df['date'],df['wealth'],label=name)
    ax.set(xlabel='Date',ylabel='Wealth from 1',title='Held-out test portfolio wealth (illustrative)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(dest/'equity_curves.png',dpi=150)
    plt.close(fig)
    print(json.dumps({'test':{k:v['test'] for k,v in metrics['metrics'].items()}},indent=2))
    return payload

if __name__=='__main__':main()
