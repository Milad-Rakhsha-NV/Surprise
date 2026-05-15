# Bipedal Go2 - Rough Terrain Forward Model

Forward model trained on rough terrain data with the bipedal walking policy.

## Training Data
- **Policy**: `unitree_go2_bipedal_walk_rough/0_best/best.pt`
- **Environment**: `Unitree-Go2-Bipedal-Walk-Rough`
- **obs_dim**: 135, **action_dim**: 12
- Version v4: with min_std=0.01 clamping and filtered normalization stats

## Notes
- First 50 steps of each episode filtered (standup phase)
- Dimension 25 is constant zero (handled with min_std clamp)

## Usage
```bash
cd ~/repos/Surprise
./scripts/run_bipedal_rough.sh --num_envs 16 --push_velocity 1.5
```
