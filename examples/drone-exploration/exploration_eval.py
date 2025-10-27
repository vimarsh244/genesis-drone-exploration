import argparse
import os
from importlib import metadata

import torch

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from maze_env import MazeEnv


def build_train_cfg(exp_name: str, max_iterations: int):
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
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
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
    return train_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="drone-exploration-ppo")
    parser.add_argument("--ckpt", type=int, default=300)
    parser.add_argument("--record", action="store_true", default=False)
    parser.add_argument("-B", "--num_envs", type=int, default=1)
    parser.add_argument("--episode_length_s", type=float, default=90.0)
    args = parser.parse_args()

    gs.init()

    log_dir = os.path.join("logs", args.exp_name)
    train_cfg = build_train_cfg(args.exp_name, max_iterations=1)

    env = MazeEnv(
        num_envs=args.num_envs,
        grid_size=(11, 11),
        seed=1,
        show_viewer=True,
        episode_length_s=args.episode_length_s,
        visualize_camera=args.record,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    if not os.path.exists(resume_path):
        raise FileNotFoundError(f"checkpoint not found: {resume_path}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)

    obs, _ = env.reset()

    # visualize up to the episode length at viewer FPS
    max_sim_step = int(args.episode_length_s * env.max_FPS)
    with torch.no_grad():
        if args.record and env.cam is not None:
            env.cam.start_recording()
            for _ in range(max_sim_step):
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)
                env.cam.render()
            env.cam.stop_recording(save_to_filename="video.mp4", fps=env.max_FPS)
        else:
            for _ in range(max_sim_step):
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)


if __name__ == "__main__":
    main()


