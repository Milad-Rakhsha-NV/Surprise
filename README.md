# Surprise Estimation Scripts

Scripts for training and evaluating surprise estimators for anomaly detection in RL policies.

## Pipeline

1. **collect_rollout_data.py** - Collect nominal (o_t, a_t, o_{t+1}) transitions from trained policy
2. **train_forward_model.py** - Train probabilistic forward dynamics model with NLL loss
3. **evaluate_surprise_online.py** - Real-time surprise computation with physical disturbances

## Usage

### Step 1: Collect Data

```bash
cd ~/Documents/repos/IsaacLab
conda activate IsaacLab

# Bipedal policy (Newton backend)
./isaaclab.sh -p /path/to/unitree_rl_lab/scripts/surprise_estimation/collect_rollout_data.py \
    --task Unitree-Go2-Bipedal-Walk-Rough \
    --checkpoint /path/to/unitree_rl_lab/logs/rsl_rl/unitree_go2_bipedal_walk_rough/0_best/best.pt \
    --presets newton \
    --num_envs 64 --num_steps 10000 --headless

# Quadruped policy
./isaaclab.sh -p /path/to/unitree_rl_lab/scripts/surprise_estimation/collect_rollout_data.py \
    --task Unitree-Go2-Velocity-Flat \
    --checkpoint /path/to/quadruped/checkpoint.pt \
    --presets newton \
    --num_envs 64 --num_steps 10000 --headless
```

### Step 2: Train Forward Model

```bash
python /path/to/unitree_rl_lab/scripts/surprise_estimation/train_forward_model.py \
    --data_path logs/rsl_rl/unitree_go2_bipedal_walk_rough/0_best/rollout_data/rollout_data.npz \
    --num_epochs 100 \
    --batch_size 256
```

### Step 3: Evaluate with Disturbances

```bash
./isaaclab.sh -p /path/to/unitree_rl_lab/scripts/surprise_estimation/evaluate_surprise_online.py \
    --task Unitree-Go2-Bipedal-Walk-Rough \
    --checkpoint /path/to/unitree_rl_lab/logs/rsl_rl/unitree_go2_bipedal_walk_rough/0_best/best.pt \
    --forward_model logs/rsl_rl/unitree_go2_bipedal_walk_rough/0_best/rollout_data/forward_model/forward_model_best.pt \
    --presets newton \
    --disturbance_type push --push_velocity 1.0 \
    --plot --headless
```

## Disturbance Types

- `none` - No disturbance (baseline)
- `push` - Instantaneous velocity impulse (typical: 0.5-2.0 m/s)
- `external_force` - Continuous force on base (typical: 20-100 N)
- `external_torque` - Continuous torque on base (typical: 5-20 Nm)

## Output

- `rollout_data.npz` - Collected transitions
- `normalization_stats.npz` - Input normalization statistics
- `forward_model_best.pt` - Trained forward model
- `surprise_results_*.npz` - Per-environment surprise traces
- `surprise_plot_*.png` - Visualization of surprise over time
