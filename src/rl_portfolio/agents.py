"""Small, inspectable implementations of PPO and entropy-regularized SAC.

Actions are tanh-transformed independent logits in [-1,1]^(N-1). The environment
maps these into fully invested, long-only weights using anchored softmax.
Entropy is defined in the latent *action* space, not the weight simplex.
"""
from collections import deque
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
import torch.nn.functional as F


def mlp(inputs, outputs, hidden):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, outputs))


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.net = mlp(obs_dim, 2*act_dim, hidden)
        self.act_dim = act_dim

    def params(self, obs):
        raw = self.net(obs)
        mean, log_std = torch.chunk(raw, 2, dim=-1)
        return mean, torch.clamp(log_std, -5, 1)

    def sample(self, obs, deterministic=False):
        mean, log_std = self.params(obs)
        dist = Normal(mean, log_std.exp())
        raw = mean if deterministic else dist.rsample()
        action = raw.tanh()
        log_prob = (dist.log_prob(raw) - torch.log(1-action.square()+1e-6)).sum(dim=-1)
        return action, log_prob

    def evaluate_log_prob(self, obs, actions):
        raw = torch.atanh(torch.clamp(actions, -1+1e-6, 1-1e-6))
        mean, log_std = self.params(obs)
        return (Normal(mean, log_std.exp()).log_prob(raw) -
                torch.log(1-actions.square()+1e-6)).sum(dim=-1)


class PPO:
    def __init__(self, obs_dim, act_dim, hidden=64, lr=3e-4, seed=42):
        torch.manual_seed(seed)
        self.actor = Actor(obs_dim, act_dim, hidden)
        self.critic = mlp(obs_dim, 1, hidden)
        self.optim = torch.optim.Adam(list(self.actor.parameters())+list(self.critic.parameters()), lr=lr)

    def act(self, state, deterministic=True):
        x = torch.as_tensor(state,dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action,_ = self.actor.sample(x, deterministic)
        return action.squeeze(0).numpy()

    def train(self, env, steps=2048, rollout=128, epochs=4, batch_size=64,
              gamma=.99, gae_lambda=.95, clip=.2):
        if steps < 1 or rollout < 2 or epochs < 1 or batch_size < 1:
            raise ValueError("Invalid PPO hyperparameters")
        state = env.reset(random_start=True)
        totals=[]
        episode_reward=0.
        completed=0
        while completed < steps:
            records=[]
            for _ in range(min(rollout, steps-completed)):
                s=torch.as_tensor(state,dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    action, lp=self.actor.sample(s)
                    value=self.critic(s).squeeze(-1)
                next_state, reward, done, _=env.step(action.squeeze(0).numpy())
                records.append((state.copy(),action.squeeze(0).numpy(),float(lp.item()),float(value.item()), reward,done))
                episode_reward+=reward
                completed+=1
                if done:
                    totals.append(episode_reward)
                    episode_reward=0.
                    state=env.reset(random_start=True)
                else:
                    state=next_state
            with torch.no_grad():
                bootstrap=0. if records[-1][-1] else float(self.critic(torch.as_tensor(state,dtype=torch.float32).unsqueeze(0)).item())
            advantages=np.zeros(len(records),dtype=np.float32)
            returns=np.zeros(len(records),dtype=np.float32)
            gae=0.
            for j in reversed(range(len(records))):
                _,_,_,v,reward,done=records[j]
                next_value=(bootstrap if j==len(records)-1 else records[j+1][3])
                delta=reward +gamma*next_value*(1-int(done)) -v
                gae=delta+gamma*gae_lambda*(1-int(done))*gae
                advantages[j]=gae
                returns[j]=gae+v
            states=torch.tensor(np.stack([r[0] for r in records]),dtype=torch.float32)
            actions=torch.tensor(np.stack([r[1] for r in records]),dtype=torch.float32)
            old_lp=torch.tensor([r[2] for r in records],dtype=torch.float32)
            targets=torch.tensor(returns,dtype=torch.float32)
            advantage=torch.tensor(advantages,dtype=torch.float32)
            advantage=(advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)
            for _ in range(epochs):
                perm=torch.randperm(len(records))
                for index in perm.split(batch_size):
                    ratio=torch.exp(self.actor.evaluate_log_prob(states[index],actions[index])-old_lp[index])
                    obj=torch.minimum(ratio*advantage[index], torch.clamp(ratio,1-clip,1+clip)*advantage[index])
                    v_pred=self.critic(states[index]).squeeze(-1)
                    loss=-obj.mean()+.5*F.mse_loss(v_pred,targets[index])
                    self.optim.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(list(self.actor.parameters())+list(self.critic.parameters()),1.)
                    self.optim.step()
        return {"steps":completed,"finished_episodes":len(totals),"average_episode_reward_pct":float(np.mean(totals)) if totals else None}


class ReplayBuffer:
    def __init__(self, capacity=100_000,seed=42):
        self.data=deque(maxlen=capacity)
        self.rng=np.random.default_rng(seed)

    def add(self,s,a,r,ns,done):
        self.data.append((np.asarray(s,np.float32),np.asarray(a,np.float32),float(r),
                         np.asarray(ns,np.float32),float(done)))

    def sample(self,batch):
        idx=self.rng.choice(len(self.data),size=batch,replace=False)
        tuples=[self.data[int(i)] for i in idx]
        return [torch.as_tensor(np.stack(group),dtype=torch.float32) for group in zip(*tuples)]

    def __len__(self): return len(self.data)


class SAC:
    def __init__(self,obs_dim,act_dim,hidden=64,lr=3e-4,alpha=.1,tau=.01,seed=42):
        if alpha<0 or not 0<tau<=1:raise ValueError('Invalid SAC entropy/target parameters')
        torch.manual_seed(seed)
        self.actor=Actor(obs_dim,act_dim,hidden)
        self.q1=mlp(obs_dim+act_dim,1,hidden)
        self.q2=mlp(obs_dim+act_dim,1,hidden)
        self.target1=deepcopy(self.q1)
        self.target2=deepcopy(self.q2)
        self.policy_opt=torch.optim.Adam(self.actor.parameters(),lr=lr)
        self.q1_opt=torch.optim.Adam(self.q1.parameters(),lr=lr)
        self.q2_opt=torch.optim.Adam(self.q2.parameters(),lr=lr)
        self.alpha=alpha
        self.tau=tau
        self.buffer=ReplayBuffer(seed=seed)
        self.rng=np.random.default_rng(seed)

    def act(self,state,deterministic=True):
        with torch.no_grad():
            x=torch.as_tensor(state,dtype=torch.float32).unsqueeze(0)
            a,_=self.actor.sample(x,deterministic)
        return a.squeeze(0).numpy()

    def _update(self,batch,gamma=.99):
        s,a,r,ns,done=self.buffer.sample(batch)
        with torch.no_grad():
            a_next, lp_next=self.actor.sample(ns)
            q_next=torch.minimum(self.target1(torch.cat([ns,a_next],1)),
                                 self.target2(torch.cat([ns,a_next],1))).squeeze(-1)
            target=r+gamma*(1-done)*(q_next-self.alpha*lp_next)
        for q,opt in ((self.q1,self.q1_opt),(self.q2,self.q2_opt)):
            value=q(torch.cat([s,a],1)).squeeze(-1)
            loss=F.mse_loss(value,target)
            opt.zero_grad();loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(),1.)
            opt.step()
        new_actions,lp=self.actor.sample(s)
        q_min=torch.minimum(self.q1(torch.cat([s,new_actions],1)),
                            self.q2(torch.cat([s,new_actions],1))).squeeze(-1)
        actor_loss=(self.alpha*lp-q_min).mean()
        self.policy_opt.zero_grad();actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(),1.)
        self.policy_opt.step()
        with torch.no_grad():
            for original,target_net in [(self.q1,self.target1),(self.q2,self.target2)]:
                for p,tp in zip(original.parameters(),target_net.parameters()):
                    tp.mul_(1-self.tau).add_(self.tau*p)

    def train(self,env,steps=2048,warmup=128,batch_size=64,gamma=.99):
        if steps<1 or warmup<0 or batch_size<2:raise ValueError('Invalid SAC training settings')
        state=env.reset(random_start=True)
        updates=0
        for step in range(steps):
            a=self.rng.uniform(-1,1,size=env.action_size).astype(np.float32) if step<warmup else self.act(state,False)
            next_state,reward,done,_=env.step(a)
            ns=np.zeros_like(state) if done else next_state
            self.buffer.add(state,a,reward,ns,done)
            state=env.reset(random_start=True) if done else next_state
            if len(self.buffer)>=batch_size and step>=warmup:
                self._update(batch_size,gamma)
                updates+=1
        return {"steps":steps,"updates":updates,"replay_size":len(self.buffer)}
