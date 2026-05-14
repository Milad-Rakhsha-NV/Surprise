#!/usr/bin/env python3
"""Diagnose observation distribution mismatch between training data and online rollout."""

import numpy as np
import torch
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", required=True, help="Path to training rollout data")
    parser.add_argument("--forward_model", required=True, help="Path to forward model checkpoint")
    parser.add_argument("--min_episode_step", type=int, default=50)
    args = parser.parse_args()

    # Load training data
    print("Loading training data...")
    train_data = np.load(args.train_data)
    mask = train_data['episode_steps'] >= args.min_episode_step
    train_obs = train_data['obs'][mask]
    train_actions = train_data['actions'][mask]
    train_next_obs = train_data['next_obs'][mask]

    print(f"Training obs shape: {train_obs.shape}")
    print(f"Training actions shape: {train_actions.shape}")

    # Load forward model normalization stats
    print("\nLoading forward model...")
    fm = torch.load(args.forward_model, map_location='cpu', weights_only=False)
    obs_mean = fm['obs_mean'].numpy()
    obs_std = fm['obs_std'].numpy()
    action_mean = fm['action_mean'].numpy()
    action_std = fm['action_std'].numpy()

    # Check if normalization stats match training data
    print("\n=== Normalization Stats Check ===")
    train_obs_mean = train_obs.mean(axis=0)
    train_obs_std = train_obs.std(axis=0)
    train_action_mean = train_actions.mean(axis=0)
    train_action_std = train_actions.std(axis=0)

    obs_mean_match = np.allclose(obs_mean, train_obs_mean, atol=0.01)
    obs_std_match = np.allclose(obs_std, train_obs_std + 1e-6, atol=0.01)
    print(f"Obs mean matches training data: {obs_mean_match}")
    print(f"Obs std matches training data: {obs_std_match}")

    if not obs_mean_match:
        diff = np.abs(obs_mean - train_obs_mean)
        worst_dims = np.argsort(diff)[-5:][::-1]
        print(f"Worst obs mean mismatch dims: {worst_dims}")
        for d in worst_dims:
            print(f"  dim {d}: model={obs_mean[d]:.4f}, train={train_obs_mean[d]:.4f}, diff={diff[d]:.4f}")

    # Check normalized statistics
    print("\n=== Normalized Training Data Stats ===")
    obs_std_safe = np.where(obs_std > 1e-6, obs_std, 1.0)
    train_obs_n = (train_obs - obs_mean) / obs_std_safe
    print(f"Normalized obs mean: {train_obs_n.mean():.6f} (should be ~0)")
    print(f"Normalized obs std: {train_obs_n.std():.6f} (should be ~1)")

    # Per-dimension analysis
    print("\n=== Per-dimension normalized stats ===")
    for i in range(train_obs_n.shape[1]):
        mean_n = train_obs_n[:, i].mean()
        std_n = train_obs_n[:, i].std()
        if abs(mean_n) > 0.1 or abs(std_n - 1.0) > 0.2:
            print(f"dim {i}: mean={mean_n:.4f}, std={std_n:.4f} (unusual)")

    # Check prediction residuals in normalized space
    print("\n=== Prediction Residual Analysis ===")
    train_next_n = (train_next_obs - obs_mean) / obs_std_safe
    delta_n = train_next_n - train_obs_n
    print(f"Delta (next - current) in normalized space:")
    print(f"  mean: {delta_n.mean():.6f}")
    print(f"  std: {delta_n.std():.6f}")
    print(f"  MSE (trivial baseline): {(delta_n**2).mean():.6f}")

    # The model's test MSE should be much lower than this trivial baseline
    print(f"\nModel test MSE: ~0.044 (from training log)")
    print(f"Trivial baseline MSE: {(delta_n**2).mean():.6f}")
    print(f"Model improvement factor: {(delta_n**2).mean() / 0.044:.1f}x")


if __name__ == "__main__":
    main()
