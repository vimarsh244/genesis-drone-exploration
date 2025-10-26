import argparse
import os
import time

import torch

import genesis as gs

from maze_env import MazeEnv
from ddpg_agents import DDPGConfig, DDPGAgent, ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


try:
    from rsl_rl.runners import OnPolicyRunner
    RSL_AVAILABLE = True
except Exception:
    RSL_AVAILABLE = False


def make_env(show_viewer=False, n_envs=1, episode_length_s=90.0):
    return MazeEnv(num_envs=n_envs, grid_size=(11, 11), seed=1, show_viewer=show_viewer, episode_length_s=episode_length_s)


def train_ddpg(args):
    env = make_env(show_viewer=args.vis, n_envs=1, episode_length_s=args.episode_length_s)
    log_dir = os.path.join("logs", "drone-exploration-ddpg")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    obs, _ = env.reset()
    obs_dim = env.num_obs
    cfg = DDPGConfig(obs_dim=obs_dim, multi_critic=(args.algo == "ddpg-mc"), lr=args.lr, batch_size=args.batch_size)
    agent = DDPGAgent(cfg)
    buf = ReplayBuffer(obs_dim=obs_dim, act_dim=3, size=args.buffer_size)

    total_steps = args.steps
    start_steps = min(1000, total_steps // 10)
    update_after = 1000
    update_every = 50

    o = obs.clone().cpu().numpy().squeeze(0)
    ep_ret = 0.0
    ep_len = 0

    last_losses = None
    for t in range(1, total_steps + 1):
        if t < start_steps:
            a = torch.tensor([[1.5 * torch.rand(1), 20.0 * (torch.rand(1) - 0.5), 2.0 * (torch.rand(1) - 0.5)]], dtype=torch.float32).squeeze(0)
        else:
            with torch.no_grad():
                a = agent.act(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0), add_noise=True).squeeze(0)
        next_obs, rew, done, extras = env.step(a.unsqueeze(0))
        comps = extras.get("components", torch.zeros((1, 3)))
        r_components = comps.reshape(-1)
        o2 = next_obs.cpu().numpy().squeeze(0)
        d = float(done.item())
        buf.store(o, a.cpu().numpy(), r_components.cpu().numpy(), o2, d)
        o = o2
        ep_ret += rew.item()
        ep_len += 1
        if done:
            writer.add_scalar("train/ep_return", ep_ret, t)
            writer.add_scalar("train/ep_length", ep_len, t)
            obs, _ = env.reset()
            o = obs.cpu().numpy().squeeze(0)
            ep_ret = 0.0
            ep_len = 0

        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                batch = buf.sample_batch(args.batch_size)
                last_losses = agent.update(batch)
            if last_losses is not None:
                writer.add_scalar("loss/critic", last_losses.get("critic_loss", 0.0), t)
                writer.add_scalar("loss/actor", last_losses.get("actor_loss", 0.0), t)

        if t % 1000 == 0:
            print(f"step {t}/{total_steps}")

    print("training complete")
    writer.flush()
    writer.close()


def train_ppo(args):
    if not RSL_AVAILABLE:
        raise ImportError("PPO option requires rsl-rl-lib installed.")
    env = make_env(show_viewer=args.vis, n_envs=args.num_envs, episode_length_s=args.episode_length_s)

    # minimal PPO config leveraging rsl-rl runner
    train_cfg = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.004,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 3e-4,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "policy": {
            "activation": "tanh",
            "actor_hidden_dims": [128, 128],
            "critic_hidden_dims": [128, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": "drone-exploration-ppo",
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": args.ppo_iters,
            "record_interval": args.record_interval,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 100,
        "save_interval": 100,
        "empirical_normalization": None,
        "seed": 1,
    }

    log_dir = os.path.join("logs", "drone-exploration-ppo")
    os.makedirs(log_dir, exist_ok=True)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.learn(num_learning_iterations=args.ppo_iters, init_at_random_ep_len=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, choices=["ddpg-mc", "ddpg-sc", "ppo"], default="ddpg-mc")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--buffer_size", type=int, default=200_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ppo_iters", type=int, default=301)
    parser.add_argument("-B", "--num_envs", type=int, default=2048)
    parser.add_argument("--episode_length_s", type=float, default=90.0)
    parser.add_argument("--record_interval", type=int, default=-1)
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    if args.algo.startswith("ddpg"):
        train_ddpg(args)
    else:
        train_ppo(args)
