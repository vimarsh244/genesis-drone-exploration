# Multi-Critic DDPG and PPO Implementation

This document describes the multi-critic reinforcement learning implementations for drone navigation, based on the approach from `ddpg_new.py`.

## Overview

The multi-critic approach uses **separate critic networks** for each action dimension (pitch, yaw, roll), with each critic evaluating its corresponding action component using a separate reward signal.

### Key Concepts

1. **Three Action Dimensions:**
   - **Pitch** (forward/backward movement): Range [0, 1.5]
   - **Yaw** (rotation): Range [-10, 10]
   - **Roll** (sideways movement): Range [-1, 1]

2. **Three Reward Components:**
   - **r1 (Pitch reward)**: Encourages maintaining safe distance from front obstacles
   - **r2 (Yaw reward)**: Promotes balanced navigation between left/right obstacles
   - **r3 (Roll reward)**: Encourages staying centered in corridors

3. **Three Critics:**
   - **Critic_pitch**: Evaluates pitch actions using r1 reward
   - **Critic_yaw**: Evaluates yaw actions using r2 reward
   - **Critic_roll**: Evaluates roll actions using r3 reward

## Implementations

### 1. Multi-Critic DDPG (ddpg-mc)

Located in: `Genesis/examples/drone-exploration/`

#### Architecture

**Actor Network:**
- Input: State (F, R, L) - front, right, left distances
- Output: 3 actions (pitch, yaw, roll)

**Critic Networks (3 separate networks):**
- Each takes: (state, single action component)
- Each outputs: Q-value for that action component
- Each is trained with its corresponding reward component

#### Training Algorithm

```python
# For each timestep:
1. Actor selects action: a = (pitch, yaw, roll)
2. Environment returns: s', [r1, r2, r3], done
3. Update critics individually:
   - Critic_pitch: Q_pitch(s, pitch) → target: r1 + γ * Q_target_pitch(s', pitch')
   - Critic_yaw: Q_yaw(s, yaw) → target: r2 + γ * Q_target_yaw(s', yaw')
   - Critic_roll: Q_roll(s, roll) → target: r3 + γ * Q_target_roll(s', roll')
4. Update actor to maximize: Q_pitch + Q_yaw + Q_roll
5. Soft update target networks
```

#### Usage

```bash
# Train multi-critic DDPG
python train_exploration.py --algo ddpg-mc --steps 50000 --vis

# Train single-critic DDPG (baseline)
python train_exploration.py --algo ddpg-sc --steps 50000 --vis

# Custom parameters
python train_exploration.py \
    --algo ddpg-mc \
    --steps 100000 \
    --batch_size 256 \
    --lr 1e-3 \
    --buffer_size 200000 \
    --episode_length_s 90.0 \
    --exp_name my_experiment
```

#### Logged Metrics

**Episode Metrics:**
- `train/ep_return`: Total episode return
- `train/ep_length`: Episode length (steps)
- `train/ep_time`: Episode duration (seconds)
- `train/ep_reward_r1_pitch`: Cumulative r1 (pitch) reward
- `train/ep_reward_r2_yaw`: Cumulative r2 (yaw) reward
- `train/ep_reward_r3_roll`: Cumulative r3 (roll) reward
- `train/mean_ep_return`: Moving average (last 100 episodes)
- `train/mean_ep_length`: Moving average episode length
- `train/mean_ep_time`: Moving average episode time

**Training Metrics:**
- `loss/critic`: Total critic loss (sum of all 3)
- `loss/critic_pitch`: Pitch critic loss
- `loss/critic_yaw`: Yaw critic loss
- `loss/critic_roll`: Roll critic loss
- `loss/actor`: Actor loss

**Value Metrics:**
- `value/q_pitch`: Average Q-value from pitch critic
- `value/q_yaw`: Average Q-value from yaw critic
- `value/q_roll`: Average Q-value from roll critic
- `value/policy_q_pitch`: Q-value for policy actions (pitch)
- `value/policy_q_yaw`: Q-value for policy actions (yaw)
- `value/policy_q_roll`: Q-value for policy actions (roll)

### 2. Multi-Critic PPO (ppo-mc)

Located in: `genesis_drone/models/ppo.py`

#### Architecture

**Actor Network:**
- Input: State (F, R, L)
- Output: Action mean and log std for each action dimension

**Critic Networks (3 separate networks):**
- Each takes: state
- Each outputs: Value estimate for that reward component
- Each is trained with its corresponding reward component

#### Training Algorithm

```python
# For each rollout:
1. Collect trajectories with current policy
2. For each reward component (r1, r2, r3):
   - Compute returns and advantages using corresponding critic
3. Update actor using combined advantages
4. Update each critic separately with its reward component
```

#### Usage

```bash
# Train multi-critic PPO
python train_exploration.py --algo ppo-mc --ppo_iters 500 --rollout_length 1000 --vis

# Train standard PPO (baseline, uses rsl-rl)
python train_exploration.py --algo ppo --ppo_iters 500 --num_envs 2048 --vis

# Custom parameters
python train_exploration.py \
    --algo ppo-mc \
    --ppo_iters 1000 \
    --rollout_length 2000 \
    --episode_length_s 90.0 \
    --exp_name my_ppo_experiment
```

#### Logged Metrics

**Episode Metrics:**
- Same as DDPG (see above)

**Training Metrics:**
- `loss/actor`: Actor loss
- `loss/critic`: Total critic loss (average of all 3)
- `loss/critic_pitch`: Pitch critic loss
- `loss/critic_yaw`: Yaw critic loss
- `loss/critic_roll`: Roll critic loss
- `loss/entropy`: Policy entropy

**Value Metrics:**
- `value/value_pitch`: Average value estimate from pitch critic
- `value/value_yaw`: Average value estimate from yaw critic
- `value/value_roll`: Average value estimate from roll critic

## Reward Functions

Based on the paper implementation in `ddpg_new.py`:

### R1: Front Distance Reward (Pitch)

```python
def reward_1(F, F_threshold=1.5):
    if F > F_threshold:
        return 0.2 * (F - F_threshold)^2
    else:
        return -30 * exp(1/(F - F_threshold))
```

Encourages maintaining safe distance from front obstacles.

### R2: Balance Reward (Yaw)

```python
def reward_2(balance_t1, balance_t0):
    delta_b = balance_t1 - balance_t0
    if balance_t1 < 8.89 and delta_b < 0:
        return 4 * 8.89 / balance_t1
    elif balance_t1 < 8.89 and delta_b > 0:
        return 8.89 / balance_t1
    else:
        return -19 * exp(-1/(balance_t1 - 26.25))
```

Where balance is the ratio of repulsive potentials from left/right obstacles. Promotes balanced navigation.

### R3: Attraction Reward (Roll)

```python
def reward_3(R, L):
    attraction = -0.1 * (R - L)^2 + 2.53
    return 2 * attraction  # Doubled to encourage centering
```

Encourages staying centered between obstacles.

## Implementation Details

### Observation Space

- **F**: Minimum distance in front sector (80-100 degrees)
- **R**: Minimum distance in right sector (101-181 degrees)
- **L**: Minimum distance in left sector (0-80 degrees)

All distances clamped to [0, 6] meters.

### Action Space

- **Pitch**: [0, 1.5] - forward movement magnitude
- **Yaw**: [-10, 10] - rotation rate
- **Roll**: [-1, 1] - sideways movement magnitude

### Hyperparameters

**DDPG:**
- Learning rate: 1e-3
- Batch size: 256
- Replay buffer: 200,000
- Gamma: 0.99
- Polyak: 0.995
- Exploration noise: Gaussian with adaptive std

**PPO:**
- Learning rate: 3e-4
- Batch size: 64
- Gamma: 0.99
- GAE lambda: 0.95
- Clip param: 0.2
- Entropy coef: 0.01
- Epochs: 10
- Mini-batches: 4

## Monitoring Training

### TensorBoard

```bash
# View training logs
tensorboard --logdir Genesis/examples/drone-exploration/logs/
```

Navigate to http://localhost:6006 to view:
- Episode returns and lengths
- Individual reward components
- Critic and actor losses
- Q-values and value estimates
- Moving averages

### Console Output

The training script prints progress every 1000 steps (DDPG) or 10 iterations (PPO):

```
step 10000/50000
  critic_loss: 0.2345, actor_loss: -12.3456
  mean_return: 123.45, mean_ep_len: 456.7
```

## Comparison with Single-Critic

### Advantages of Multi-Critic:

1. **Specialized Learning**: Each critic focuses on one action dimension with its specific reward signal
2. **Better Credit Assignment**: Clear attribution of rewards to actions
3. **Improved Exploration**: Independent critics can learn different aspects simultaneously
4. **More Stable Training**: Separate losses prevent interference between different objectives

### When to Use:

- **Multi-Critic**: When rewards naturally decompose by action dimensions
- **Single-Critic**: When rewards are inherently coupled or for simpler tasks

## File Structure

```
Genesis/examples/drone-exploration/
├── ddpg_agents.py          # DDPG implementations (multi & single critic)
├── train_exploration.py    # Training script for all algorithms
├── maze_env.py            # Environment with reward decomposition
└── MULTI_CRITIC_README.md # This file

genesis_drone/models/
├── ppo.py                 # PPO implementation (multi & single critic)
└── networks.py            # Neural network architectures
```

## References

1. Original implementation: `old_drone_sim/drone_sim/ddpg_new.py`
2. Paper: "Improved Stability and Sample Efficiency with Multi-Critic DDPG for Autonomous Navigation in Indoor Exploration" (based on IS_DDPG_TAI_submitted_paper.pdf)

## Troubleshooting

### Common Issues:

1. **Import errors for PPO-MC:**
   - Ensure `genesis_drone` is in your Python path
   - The training script automatically adds it, but you may need to install in development mode:
     ```bash
     cd genesis_drone && pip install -e .
     ```

2. **CUDA out of memory:**
   - Reduce batch size: `--batch_size 128`
   - Reduce rollout length (PPO): `--rollout_length 500`

3. **Training instability:**
   - Reduce learning rate: `--lr 5e-4`
   - Increase replay buffer (DDPG): `--buffer_size 500000`

4. **Slow training:**
   - Disable visualization: remove `--vis` flag
   - Reduce episode length: `--episode_length_s 60.0`

## Future Work

- Add multi-critic support to rsl-rl for multi-environment PPO
- Implement prioritized experience replay for DDPG
- Add curriculum learning for maze complexity
- Implement hindsight experience replay (HER)

