# Bipedal Go2 - Flat Terrain Forward Model

Forward model trained on flat terrain data collected with the rough-terrain policy.

## Training Data
- **Policy**: `unitree_go2_bipedal_walk_rough/0_best/best.pt` (rough terrain policy)
- **Environment**: `Unitree-Go2-Bipedal-Walk` (flat terrain)
- **Transitions**: 607,268 (after filtering first 50 warmup steps per episode)
- **obs_dim**: 135, **action_dim**: 12

## Model
- Architecture: 147 → [256, 256] → 135
- Parameters: 174,094
- Final val_nll: -3.85
- Final val_mse: 0.068

## Usage
```python
from train_forward_model import ForwardModel
import torch

# Load model
checkpoint = torch.load("forward_model_best.pt")
model = ForwardModel(
    obs_dim=135,
    action_dim=12,
    hidden_dims=[256, 256]
)
model.load_state_dict(checkpoint["model_state_dict"])

# Normalization stats are included in checkpoint
obs_mean = checkpoint["obs_mean"]
obs_std = checkpoint["obs_std"]
action_mean = checkpoint["action_mean"]
action_std = checkpoint["action_std"]
```

## Evaluation Script
```bash
cd ~/repos/Surprise
./scripts/run_bipedal_flat.sh --num_envs 16 --push_velocity 1.5
```
