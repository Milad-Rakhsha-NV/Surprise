# Quadruped Go2 - Velocity Rough Terrain Forward Model

Forward model trained on rough terrain data with the quadruped velocity tracking policy.

## Training Data
- **Policy**: `unitree_go2_velocity_rough/0_best/model.pt`
- **Environment**: `Unitree-Go2-Velocity-Rough-v0`
- **obs_dim**: 48, **action_dim**: 12

## Usage
```bash
cd ~/repos/Surprise
./scripts/run_quadruped_rough.sh --num_envs 16 --push_velocity 1.5
```
