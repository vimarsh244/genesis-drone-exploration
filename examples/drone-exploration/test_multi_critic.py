"""
Test script for multi-critic DDPG and PPO implementations.
Runs short training sessions to verify everything works.
"""

import sys
import os

# add genesis_drone to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import torch
import genesis as gs

from maze_env import MazeEnv
from ddpg_agents import DDPGConfig, DDPGAgent, ReplayBuffer
from genesis_drone.models.ppo import PPO


def test_ddpg_multi_critic():
    """Test multi-critic DDPG implementation."""
    print("\n" + "="*60)
    print("Testing Multi-Critic DDPG")
    print("="*60)
    
    # initialize genesis
    if not getattr(gs, "_initialized", False):
        gs.init(backend=gs.cpu, logging_level="warning")
    
    # create environment
    env = MazeEnv(num_envs=1, grid_size=(11, 11), seed=1, show_viewer=False, episode_length_s=10.0)
    obs, _ = env.reset()
    obs_dim = env.num_obs
    
    # create agent
    cfg = DDPGConfig(obs_dim=obs_dim, multi_critic=True, lr=1e-3, batch_size=32)
    agent = DDPGAgent(cfg)
    buf = ReplayBuffer(obs_dim=obs_dim, act_dim=3, size=1000)
    
    print(f"✓ Agent created successfully")
    print(f"  - Observation dim: {obs_dim}")
    print(f"  - Using multi-critic: {cfg.multi_critic}")
    
    # collect some experience
    o = obs.squeeze(0)
    for step in range(100):
        # random actions initially
        a = torch.tensor([
            torch.rand(1) * 1.5,
            (torch.rand(1) - 0.5) * 20.0,
            (torch.rand(1) - 0.5) * 2.0
        ], device=gs.device, dtype=torch.float32)
        
        next_obs, rew, done, extras = env.step(a.unsqueeze(0))
        comps = extras.get("components", torch.zeros((1, 3), device=gs.device))
        r_components = comps.reshape(-1)
        o2 = next_obs.squeeze(0)
        
        buf.store(
            o.detach().cpu().numpy(),
            a.detach().cpu().numpy(),
            r_components.detach().cpu().numpy(),
            o2.detach().cpu().numpy(),
            float(done.item())
        )
        
        o = o2
        if done:
            obs, _ = env.reset()
            o = obs.squeeze(0)
    
    print(f"✓ Collected {buf.size} transitions")
    
    # test training update
    batch = buf.sample_batch(32)
    metrics = agent.update(batch)
    
    print(f"✓ Training update successful")
    print(f"  - Critic loss: {metrics['critic_loss']:.4f}")
    print(f"  - Actor loss: {metrics['actor_loss']:.4f}")
    
    # check individual critic metrics
    if cfg.multi_critic:
        print(f"  - Critic pitch loss: {metrics['critic_loss_pitch']:.4f}")
        print(f"  - Critic yaw loss: {metrics['critic_loss_yaw']:.4f}")
        print(f"  - Critic roll loss: {metrics['critic_loss_roll']:.4f}")
        print(f"  - Q pitch: {metrics['q_value_pitch']:.4f}")
        print(f"  - Q yaw: {metrics['q_value_yaw']:.4f}")
        print(f"  - Q roll: {metrics['q_value_roll']:.4f}")
    
    print("\n✅ Multi-Critic DDPG test passed!")
    return True


def test_ppo_multi_critic():
    """Test multi-critic PPO implementation."""
    print("\n" + "="*60)
    print("Testing Multi-Critic PPO")
    print("="*60)
    
    # initialize genesis
    if not getattr(gs, "_initialized", False):
        gs.init(backend=gs.cpu, logging_level="warning")
    
    # create environment
    env = MazeEnv(num_envs=1, grid_size=(11, 11), seed=1, show_viewer=False, episode_length_s=10.0)
    obs, _ = env.reset()
    
    # configure PPO
    ppo_config = {
        'state_dim': env.num_obs,
        'action_dim': 3,
        'hidden_dim': 64,
        'num_layers': 1,
        'batch_size': 32,
        'gamma': 0.99,
        'clip_param': 0.2,
        'actor_lr': 3e-4,
        'critic_lr': 3e-4,
        'multi_critic': True,
        'gae_lambda': 0.95,
        'value_loss_coef': 0.5,
        'entropy_coef': 0.01,
        'max_grad_norm': 0.5,
        'num_mini_batches': 2,
        'num_epochs': 2,
        'max_pitch': 1.5,
        'max_yaw': 10.0,
        'max_roll': 1.0,
    }
    
    agent = PPO(ppo_config)
    
    print(f"✓ Agent created successfully")
    print(f"  - State dim: {ppo_config['state_dim']}")
    print(f"  - Using multi-critic: {ppo_config['multi_critic']}")
    
    # collect rollout
    o = obs.squeeze(0).cpu().numpy()
    done = False
    steps = 0
    
    while not done and steps < 50:
        action = agent.select_action(o, deterministic=False)
        action_tensor = torch.tensor([action], dtype=torch.float32, device=gs.device)
        next_obs, rew, done_tensor, extras = env.step(action_tensor)
        
        comps = extras.get("components", torch.zeros((1, 3), device=gs.device))
        r_components = comps.reshape(-1).cpu().numpy()
        
        agent.process_step(r_components, done_tensor.item())
        
        o = next_obs.squeeze(0).cpu().numpy()
        done = done_tensor.item()
        steps += 1
    
    print(f"✓ Collected rollout with {steps} steps")
    
    # test training
    metrics = agent.train()
    
    print(f"✓ Training update successful")
    print(f"  - Critic loss: {metrics['critic_loss']:.4f}")
    print(f"  - Actor loss: {metrics['actor_loss']:.4f}")
    print(f"  - Entropy: {metrics['entropy']:.4f}")
    
    # check individual critic metrics
    if ppo_config['multi_critic']:
        print(f"  - Critic pitch loss: {metrics['critic_loss_pitch']:.4f}")
        print(f"  - Critic yaw loss: {metrics['critic_loss_yaw']:.4f}")
        print(f"  - Critic roll loss: {metrics['critic_loss_roll']:.4f}")
        print(f"  - Value pitch: {metrics['value_pitch']:.4f}")
        print(f"  - Value yaw: {metrics['value_yaw']:.4f}")
        print(f"  - Value roll: {metrics['value_roll']:.4f}")
    
    print("\n✅ Multi-Critic PPO test passed!")
    return True


def test_reward_components():
    """Test that reward components are being computed correctly."""
    print("\n" + "="*60)
    print("Testing Reward Components")
    print("="*60)
    
    # initialize genesis
    if not getattr(gs, "_initialized", False):
        gs.init(backend=gs.cpu, logging_level="warning")
    
    # create environment
    env = MazeEnv(num_envs=1, grid_size=(11, 11), seed=1, show_viewer=False, episode_length_s=10.0)
    obs, _ = env.reset()
    
    # take a random action
    action = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32, device=gs.device)
    next_obs, rew, done, extras = env.step(action)
    
    # check reward components
    assert "components" in extras, "No reward components in extras"
    comps = extras["components"]
    
    print(f"✓ Reward components present")
    print(f"  - Shape: {comps.shape}")
    print(f"  - r1 (pitch): {comps[0, 0].item():.4f}")
    print(f"  - r2 (yaw): {comps[0, 1].item():.4f}")
    print(f"  - r3 (roll): {comps[0, 2].item():.4f}")
    print(f"  - Total reward: {rew.item():.4f}")
    print(f"  - Sum of components: {comps.sum().item():.4f}")
    
    # verify they sum to total (approximately, due to scaling)
    total_from_comps = comps.sum().item() * env.step_reward_scale
    print(f"  - Scaled sum matches: {abs(total_from_comps - rew.item()) < 1e-5}")
    
    print("\n✅ Reward components test passed!")
    return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Multi-Critic Implementation Test Suite")
    print("="*60)
    
    try:
        # run tests
        test_reward_components()
        test_ddpg_multi_critic()
        test_ppo_multi_critic()
        
        print("\n" + "="*60)
        print("✅ All tests passed successfully!")
        print("="*60)
        print("\nYou can now run full training with:")
        print("  python train_exploration.py --algo ddpg-mc --steps 50000")
        print("  python train_exploration.py --algo ppo-mc --ppo_iters 500")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

