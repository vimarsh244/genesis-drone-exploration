from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import genesis as gs


def mlp(sizes, act=nn.ReLU, out_act=None):
    layers = []
    for j in range(len(sizes) - 1):
        act_layer = act if j < len(sizes) - 2 else (out_act if out_act is not None else nn.Identity)
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act_layer()]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, hidden: Tuple[int, int] = (256, 256)):
        super().__init__()
        self.net = mlp([obs_dim, hidden[0], hidden[1], 3], act=nn.ReLU)

    def forward(self, obs):
        # outputs: pitch [0,1.5], yaw [-10,10], roll [-1,1]
        x = self.net(obs)
        pitch = torch.sigmoid(x[..., 0:1]) * 1.5
        yaw = torch.tanh(x[..., 1:2]) * 10.0
        roll = torch.tanh(x[..., 2:3]) * 1.0
        return torch.cat([pitch, yaw, roll], dim=-1)


class Critic1(nn.Module):
    def __init__(self, obs_dim: int, hidden: Tuple[int, int] = (256, 256)):
        super().__init__()
        self.q = mlp([obs_dim + 1, hidden[0], hidden[1], 1], act=nn.ReLU)

    def forward(self, obs, a_scalar):
        return self.q(torch.cat([obs, a_scalar], dim=-1))


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: Tuple[int, int] = (256, 256)):
        super().__init__()
        self.q = mlp([obs_dim + act_dim, hidden[0], hidden[1], 1], act=nn.ReLU)

    def forward(self, obs, act):
        return self.q(torch.cat([obs, act], dim=-1))


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, size: int = 100_000):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros((size, 3), dtype=np.float32)  # r1, r2, r3
        self.done_buf = np.zeros((size, 1), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew_components, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew_components
        self.obs2_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=256):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs], act=self.act_buf[idxs], rew=self.rew_buf[idxs],
                     obs2=self.obs2_buf[idxs], done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}


@dataclass
class DDPGConfig:
    obs_dim: int
    act_dim: int = 3
    gamma: float = 0.99
    polyak: float = 0.995
    lr: float = 1e-3
    batch_size: int = 256
    noise_std: float = 0.1
    multi_critic: bool = True


class DDPGAgent:
    def __init__(self, cfg: DDPGConfig):
        self.cfg = cfg
        self.actor = Actor(cfg.obs_dim)
        if cfg.multi_critic:
            self.critic_pitch = Critic1(cfg.obs_dim)
            self.critic_yaw = Critic1(cfg.obs_dim)
            self.critic_roll = Critic1(cfg.obs_dim)
            self.target_critic_pitch = Critic1(cfg.obs_dim)
            self.target_critic_yaw = Critic1(cfg.obs_dim)
            self.target_critic_roll = Critic1(cfg.obs_dim)
        else:
            self.critic = Critic(cfg.obs_dim, cfg.act_dim)
            self.target_critic = Critic(cfg.obs_dim, cfg.act_dim)
        self.target_actor = Actor(cfg.obs_dim)

        # copy weights
        self.target_actor.load_state_dict(self.actor.state_dict())
        if cfg.multi_critic:
            self.target_critic_pitch.load_state_dict(self.critic_pitch.state_dict())
            self.target_critic_yaw.load_state_dict(self.critic_yaw.state_dict())
            self.target_critic_roll.load_state_dict(self.critic_roll.state_dict())
        else:
            self.target_critic.load_state_dict(self.critic.state_dict())

        # move models to genesis device
        self.actor.to(gs.device)
        self.target_actor.to(gs.device)
        if cfg.multi_critic:
            self.critic_pitch.to(gs.device)
            self.critic_yaw.to(gs.device)
            self.critic_roll.to(gs.device)
            self.target_critic_pitch.to(gs.device)
            self.target_critic_yaw.to(gs.device)
            self.target_critic_roll.to(gs.device)
        else:
            self.critic.to(gs.device)
            self.target_critic.to(gs.device)

        params = list(self.actor.parameters())
        if cfg.multi_critic:
            self.critic_params = list(self.critic_pitch.parameters()) + list(self.critic_yaw.parameters()) + list(self.critic_roll.parameters())
        else:
            self.critic_params = list(self.critic.parameters())
        self.act_opt = torch.optim.Adam(params, lr=cfg.lr)
        self.critic_opt = torch.optim.Adam(self.critic_params, lr=cfg.lr)
        self.loss_fn = nn.MSELoss()

    @torch.no_grad()
    def act(self, obs: torch.Tensor, add_noise=True):
        # ensure obs is on the same device as the model
        obs = obs.to(gs.device)
        a = self.actor(obs)
        if add_noise and self.cfg.noise_std > 0:
            noise = torch.randn_like(a) * torch.tensor([0.1, 2.0, 0.2], device=a.device)
            a = a + noise
        # clip to bounds
        a[..., 0] = torch.clamp(a[..., 0], 0.0, 1.5)
        a[..., 1] = torch.clamp(a[..., 1], -10.0, 10.0)
        a[..., 2] = torch.clamp(a[..., 2], -1.0, 1.0)
        return a

    def _polyak_update(self, source: nn.Module, target: nn.Module):
        with torch.no_grad():
            for p, p_targ in zip(source.parameters(), target.parameters()):
                p_targ.data.mul_(self.cfg.polyak)
                p_targ.data.add_((1 - self.cfg.polyak) * p.data)

    def update(self, batch):
        # move batch to device
        batch = {k: v.to(gs.device) for k, v in batch.items()}
        o, a, r_comp, o2, d = batch["obs"], batch["act"], batch["rew"], batch["obs2"], batch["done"]
        # target actor action
        with torch.no_grad():
            a2 = self.target_actor(o2)
        
        metrics = {}
        
        if self.cfg.multi_critic:
            # per-dimension q
            q1 = self.critic_pitch(o, a[:, 0:1])
            q2 = self.critic_yaw(o, a[:, 1:2])
            q3 = self.critic_roll(o, a[:, 2:3])
            with torch.no_grad():
                q1_t = self.target_critic_pitch(o2, a2[:, 0:1])
                q2_t = self.target_critic_yaw(o2, a2[:, 1:2])
                q3_t = self.target_critic_roll(o2, a2[:, 2:3])
            # Bellman targets for each component using respective reward component
            y1 = r_comp[:, 0:1] + self.cfg.gamma * (1 - d) * q1_t
            y2 = r_comp[:, 1:2] + self.cfg.gamma * (1 - d) * q2_t
            y3 = r_comp[:, 2:3] + self.cfg.gamma * (1 - d) * q3_t
            
            # individual critic losses
            critic_loss_1 = self.loss_fn(q1, y1)
            critic_loss_2 = self.loss_fn(q2, y2)
            critic_loss_3 = self.loss_fn(q3, y3)
            critic_loss = critic_loss_1 + critic_loss_2 + critic_loss_3
            
            # store individual metrics
            metrics['critic_loss_pitch'] = critic_loss_1.item()
            metrics['critic_loss_yaw'] = critic_loss_2.item()
            metrics['critic_loss_roll'] = critic_loss_3.item()
            metrics['q_value_pitch'] = q1.mean().item()
            metrics['q_value_yaw'] = q2.mean().item()
            metrics['q_value_roll'] = q3.mean().item()
        else:
            q = self.critic(o, a)
            with torch.no_grad():
                q_t = self.target_critic(o2, a2)
            # sum of reward components
            r = torch.sum(r_comp, dim=-1, keepdim=True)
            y = r + self.cfg.gamma * (1 - d) * q_t
            critic_loss = self.loss_fn(q, y)
            metrics['q_value'] = q.mean().item()

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # actor loss: minimize -Q for each critic and sum
        a_pi = self.actor(o)
        if self.cfg.multi_critic:
            q1_pi = self.critic_pitch(o, a_pi[:, 0:1])
            q2_pi = self.critic_yaw(o, a_pi[:, 1:2])
            q3_pi = self.critic_roll(o, a_pi[:, 2:3])
            act_loss = -(q1_pi + q2_pi + q3_pi).mean()
            
            # store policy q-values
            metrics['policy_q_pitch'] = q1_pi.mean().item()
            metrics['policy_q_yaw'] = q2_pi.mean().item()
            metrics['policy_q_roll'] = q3_pi.mean().item()
        else:
            q_pi = self.critic(o, a_pi)
            act_loss = -q_pi.mean()
            metrics['policy_q'] = q_pi.mean().item()

        self.act_opt.zero_grad()
        act_loss.backward()
        self.act_opt.step()

        # target updates
        self._polyak_update(self.actor, self.target_actor)
        if self.cfg.multi_critic:
            self._polyak_update(self.critic_pitch, self.target_critic_pitch)
            self._polyak_update(self.critic_yaw, self.target_critic_yaw)
            self._polyak_update(self.critic_roll, self.target_critic_roll)
        else:
            self._polyak_update(self.critic, self.target_critic)

        metrics['critic_loss'] = critic_loss.item()
        metrics['actor_loss'] = act_loss.item()
        return metrics
