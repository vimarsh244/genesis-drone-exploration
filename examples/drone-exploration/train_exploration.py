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


def make_env(show_viewer=False, n_envs=1, episode_length_s=90.0, visualize_camera=False):
    return MazeEnv(num_envs=n_envs, grid_size=(11, 11), seed=1, show_viewer=show_viewer, episode_length_s=episode_length_s, visualize_camera=visualize_camera)


def train_ddpg(args):
    # ensure genesis uses gpu so tensors and sim run on the same device
    if not getattr(gs, "_initialized", False):
        gs.init(backend=gs.gpu, logging_level="warning")

    env = make_env(show_viewer=args.vis, n_envs=1, episode_length_s=args.episode_length_s)
    log_dir = os.path.join("logs", "drone-exploration-ddpg", args.exp_name)
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

    # keep current observation as a tensor on simulation/device
    o = obs.squeeze(0)
    ep_ret = 0.0
    ep_len = 0
    ep_start_time = time.time()
    
    # tracking for individual reward components
    ep_r1 = 0.0
    ep_r2 = 0.0
    ep_r3 = 0.0
    
    # tracking for moving averages
    episode_returns = []
    episode_lengths = []
    episode_times = []

    last_losses = None
    for t in range(1, total_steps + 1):
        if t < start_steps:
            a = torch.tensor([[1.5 * torch.rand(1, device=gs.device), 20.0 * (torch.rand(1, device=gs.device) - 0.5), 2.0 * (torch.rand(1, device=gs.device) - 0.5)]], dtype=torch.float32, device=gs.device).squeeze(0)
        else:
            with torch.no_grad():
                a = agent.act(o.unsqueeze(0), add_noise=True).squeeze(0)
        next_obs, rew, done, extras = env.step(a.unsqueeze(0))
        comps = extras.get("components", torch.zeros((1, 3), device=gs.device))
        r_components = comps.reshape(-1)
        o2 = next_obs.squeeze(0)
        d = float(done.item())
        # store cpu copies in the replay buffer
        buf.store(o.detach().cpu().numpy(), a.detach().cpu().numpy(), r_components.detach().cpu().numpy(), o2.detach().cpu().numpy(), d)
        o = o2
        ep_ret += rew.item()
        ep_len += 1
        
        # accumulate individual reward components
        ep_r1 += r_components[0].item()
        ep_r2 += r_components[1].item()
        ep_r3 += r_components[2].item()
        
        if done:
            ep_time = time.time() - ep_start_time
            
            # log episode metrics
            writer.add_scalar("train/ep_return", ep_ret, t)
            writer.add_scalar("train/ep_length", ep_len, t)
            writer.add_scalar("train/ep_time", ep_time, t)
            
            # log individual reward components
            writer.add_scalar("train/ep_reward_r1_pitch", ep_r1, t)
            writer.add_scalar("train/ep_reward_r2_yaw", ep_r2, t)
            writer.add_scalar("train/ep_reward_r3_roll", ep_r3, t)
            
            # update moving averages
            episode_returns.append(ep_ret)
            episode_lengths.append(ep_len)
            episode_times.append(ep_time)
            
            # keep only last 100 episodes for moving average
            if len(episode_returns) > 100:
                episode_returns.pop(0)
                episode_lengths.pop(0)
                episode_times.pop(0)
            
            # log moving averages
            writer.add_scalar("train/mean_ep_return", sum(episode_returns) / len(episode_returns), t)
            writer.add_scalar("train/mean_ep_length", sum(episode_lengths) / len(episode_lengths), t)
            writer.add_scalar("train/mean_ep_time", sum(episode_times) / len(episode_times), t)
            
            obs, _ = env.reset()
            o = obs.squeeze(0)
            ep_ret = 0.0
            ep_len = 0
            ep_r1 = 0.0
            ep_r2 = 0.0
            ep_r3 = 0.0
            ep_start_time = time.time()

        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                batch = buf.sample_batch(args.batch_size)
                last_losses = agent.update(batch)
            if last_losses is not None:
                # log total losses
                writer.add_scalar("loss/critic", last_losses.get("critic_loss", 0.0), t)
                writer.add_scalar("loss/actor", last_losses.get("actor_loss", 0.0), t)
                
                # log individual critic losses (multi-critic only)
                if cfg.multi_critic:
                    writer.add_scalar("loss/critic_pitch", last_losses.get("critic_loss_pitch", 0.0), t)
                    writer.add_scalar("loss/critic_yaw", last_losses.get("critic_loss_yaw", 0.0), t)
                    writer.add_scalar("loss/critic_roll", last_losses.get("critic_loss_roll", 0.0), t)
                    
                    # log q-values for each critic
                    writer.add_scalar("value/q_pitch", last_losses.get("q_value_pitch", 0.0), t)
                    writer.add_scalar("value/q_yaw", last_losses.get("q_value_yaw", 0.0), t)
                    writer.add_scalar("value/q_roll", last_losses.get("q_value_roll", 0.0), t)
                    
                    # log policy q-values
                    writer.add_scalar("value/policy_q_pitch", last_losses.get("policy_q_pitch", 0.0), t)
                    writer.add_scalar("value/policy_q_yaw", last_losses.get("policy_q_yaw", 0.0), t)
                    writer.add_scalar("value/policy_q_roll", last_losses.get("policy_q_roll", 0.0), t)
                else:
                    writer.add_scalar("value/q_value", last_losses.get("q_value", 0.0), t)
                    writer.add_scalar("value/policy_q", last_losses.get("policy_q", 0.0), t)

        if t % 1000 == 0:
            print(f"step {t}/{total_steps}")
            if last_losses is not None:
                print(f"  critic_loss: {last_losses.get('critic_loss', 0.0):.4f}, actor_loss: {last_losses.get('actor_loss', 0.0):.4f}")
                if len(episode_returns) > 0:
                    print(f"  mean_return: {sum(episode_returns) / len(episode_returns):.2f}, mean_ep_len: {sum(episode_lengths) / len(episode_lengths):.1f}")

    print("training complete")
    writer.flush()
    writer.close()


def train_ppo_custom(args):
    """Train with custom multi-critic PPO implementation."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    from genesis_drone.models.ppo import PPO
    
    # ensure genesis uses gpu so tensors and sim run on the same device
    if not getattr(gs, "_initialized", False):
        gs.init(backend=gs.gpu, logging_level="warning")
    
    env = make_env(show_viewer=args.vis, n_envs=1, episode_length_s=args.episode_length_s)
    log_dir = os.path.join("logs", "drone-exploration-ppo-mc", args.exp_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    # configure multi-critic PPO
    ppo_config = {
        'state_dim': env.num_obs,
        'action_dim': 3,
        'hidden_dim': 128,
        'num_layers': 2,
        'batch_size': 64,
        'gamma': 0.99,
        'clip_param': 0.2,
        'actor_lr': 3e-4,
        'critic_lr': 3e-4,
        'multi_critic': True,
        'gae_lambda': 0.95,
        'value_loss_coef': 0.5,
        'entropy_coef': 0.01,
        'max_grad_norm': 0.5,
        'num_mini_batches': 4,
        'num_epochs': 10,
        'max_pitch': 1.5,
        'max_yaw': 10.0,
        'max_roll': 1.0,
    }
    
    agent = PPO(ppo_config)
    
    # training loop
    total_steps = 0
    episode_returns = []
    episode_lengths = []
    episode_times = []
    
    for iteration in range(args.ppo_iters):
        # reset environment
        obs, _ = env.reset()
        o = obs.squeeze(0).cpu().numpy()
        
        ep_ret = 0.0
        ep_len = 0
        ep_r1 = 0.0
        ep_r2 = 0.0
        ep_r3 = 0.0
        ep_start_time = time.time()
        done = False
        
        # collect rollout
        rollout_steps = 0
        max_rollout_steps = args.rollout_length
        
        while not done and rollout_steps < max_rollout_steps:
            # select action
            action = agent.select_action(o, deterministic=False)
            
            # step environment
            action_tensor = torch.tensor([action], dtype=torch.float32, device=gs.device)
            next_obs, rew, done_tensor, extras = env.step(action_tensor)
            
            # extract reward components
            comps = extras.get("components", torch.zeros((1, 3), device=gs.device))
            r_components = comps.reshape(-1).cpu().numpy()
            
            # process step
            agent.process_step(r_components, done_tensor.item())
            
            # update tracking
            o = next_obs.squeeze(0).cpu().numpy()
            ep_ret += rew.item()
            ep_len += 1
            ep_r1 += r_components[0]
            ep_r2 += r_components[1]
            ep_r3 += r_components[2]
            done = done_tensor.item()
            rollout_steps += 1
            total_steps += 1
        
        # train agent
        metrics = agent.train()
        
        # log episode metrics
        ep_time = time.time() - ep_start_time
        
        writer.add_scalar("train/ep_return", ep_ret, iteration)
        writer.add_scalar("train/ep_length", ep_len, iteration)
        writer.add_scalar("train/ep_time", ep_time, iteration)
        writer.add_scalar("train/ep_reward_r1_pitch", ep_r1, iteration)
        writer.add_scalar("train/ep_reward_r2_yaw", ep_r2, iteration)
        writer.add_scalar("train/ep_reward_r3_roll", ep_r3, iteration)
        
        # update moving averages
        episode_returns.append(ep_ret)
        episode_lengths.append(ep_len)
        episode_times.append(ep_time)
        
        if len(episode_returns) > 100:
            episode_returns.pop(0)
            episode_lengths.pop(0)
            episode_times.pop(0)
        
        # log moving averages
        writer.add_scalar("train/mean_ep_return", sum(episode_returns) / len(episode_returns), iteration)
        writer.add_scalar("train/mean_ep_length", sum(episode_lengths) / len(episode_lengths), iteration)
        writer.add_scalar("train/mean_ep_time", sum(episode_times) / len(episode_times), iteration)
        
        # log training metrics
        writer.add_scalar("loss/actor", metrics['actor_loss'], iteration)
        writer.add_scalar("loss/critic", metrics['critic_loss'], iteration)
        writer.add_scalar("loss/entropy", metrics['entropy'], iteration)
        
        # log individual critic metrics for multi-critic
        if 'critic_loss_pitch' in metrics:
            writer.add_scalar("loss/critic_pitch", metrics['critic_loss_pitch'], iteration)
            writer.add_scalar("loss/critic_yaw", metrics['critic_loss_yaw'], iteration)
            writer.add_scalar("loss/critic_roll", metrics['critic_loss_roll'], iteration)
            writer.add_scalar("value/value_pitch", metrics['value_pitch'], iteration)
            writer.add_scalar("value/value_yaw", metrics['value_yaw'], iteration)
            writer.add_scalar("value/value_roll", metrics['value_roll'], iteration)
        
        # print progress
        if iteration % 10 == 0:
            print(f"Iteration {iteration}/{args.ppo_iters}")
            print(f"  actor_loss: {metrics['actor_loss']:.4f}, critic_loss: {metrics['critic_loss']:.4f}, entropy: {metrics['entropy']:.4f}")
            if len(episode_returns) > 0:
                print(f"  mean_return: {sum(episode_returns) / len(episode_returns):.2f}, mean_ep_len: {sum(episode_lengths) / len(episode_lengths):.1f}")
        
        # save model periodically
        if iteration % 100 == 0:
            model_path = os.path.join(log_dir, f"model_{iteration}.pt")
            agent.save(model_path)
    
    print("training complete")
    writer.flush()
    writer.close()

def train_ppo(args):
    if not RSL_AVAILABLE:
        raise ImportError("PPO option requires rsl-rl-lib installed.")
    # ensure genesis uses gpu so tensors and sim run on the same device
    if not getattr(gs, "_initialized", False):
        gs.init(backend=gs.gpu, logging_level="warning")
    # enable offscreen camera if we intend to record
    env = make_env(show_viewer=args.vis, n_envs=args.num_envs, episode_length_s=args.episode_length_s, visualize_camera=(args.record_interval and args.record_interval > 0))

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

    # set run name from exp_name while keeping base logs directory unchanged
    train_cfg["runner"]["run_name"] = args.exp_name

    log_dir = os.path.join("logs", "drone-exploration-ppo")
    os.makedirs(log_dir, exist_ok=True)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.learn(num_learning_iterations=args.ppo_iters, init_at_random_ep_len=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, choices=["ddpg-mc", "ddpg-sc", "ppo", "ppo-mc"], default="ddpg-mc")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--buffer_size", type=int, default=200_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ppo_iters", type=int, default=301)
    parser.add_argument("--rollout_length", type=int, default=1000, help="Max steps per rollout for PPO-MC")
    parser.add_argument("-B", "--num_envs", type=int, default=2048)
    parser.add_argument("--episode_length_s", type=float, default=90.0)
    parser.add_argument("--record_interval", type=int, default=-1, help="Every N iterations, record a rollout if env has camera")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    # accept both --exp_name and --run_name, mapping to the same destination
    parser.add_argument("-e", "--exp_name", dest="exp_name", type=str, default="drone-exploration")
    parser.add_argument("--run_name", dest="exp_name", type=str, help="alias for --exp_name")
    args = parser.parse_args()

    if args.algo.startswith("ddpg"):
        train_ddpg(args)
    elif args.algo == "ppo-mc":
        train_ppo_custom(args)
    elif args.algo == "ppo":
        train_ppo(args)
    else:
        raise ValueError(f"Invalid algorithm: {args.algo}")
