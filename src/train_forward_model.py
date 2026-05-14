# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train a forward model for observation prediction and surprise computation.

The forward model learns to predict o_{t+1} given (o_t, a_t) with uncertainty estimation.
This implements a probabilistic forward dynamics model using a Gaussian likelihood.
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import datetime
from typing import Tuple


class ForwardModel(nn.Module):
    """Neural network that predicts next observation given current observation and action.
    
    Architecture: MLP with LayerNorm that outputs both mean and log-variance 
    for a diagonal Gaussian distribution over the next observation.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list[int] | None = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 256]

        # shared feature extractor
        layers = []
        in_dim = obs_dim + action_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim

        self.feature_net = nn.Sequential(*layers)

        # separate heads for mean and log-variance
        self.mean_head = nn.Linear(hidden_dims[-1], obs_dim)
        self.logvar_head = nn.Linear(hidden_dims[-1], obs_dim)

        # initialize log-variance head to output small variance initially
        nn.init.constant_(self.logvar_head.bias, -2.0)
        nn.init.zeros_(self.logvar_head.weight)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            obs: Current observation (batch_size, obs_dim)
            action: Current action (batch_size, action_dim)
            
        Returns:
            Tuple of (predicted_mean, predicted_logvar) for next observation
        """
        x = torch.cat([obs, action], dim=-1)
        features = self.feature_net(x)
        mean = self.mean_head(features)
        logvar = self.logvar_head(features)
        # clamp logvar for numerical stability
        logvar = torch.clamp(logvar, min=-10.0, max=2.0)
        return mean, logvar


class TransitionDataset(Dataset):
    """Dataset for (o_t, a_t, o_{t+1}) transitions."""

    def __init__(self, obs: np.ndarray, actions: np.ndarray, next_obs: np.ndarray, 
                 dones: np.ndarray, normalize: bool = True, stats: dict | None = None,
                 episode_steps: np.ndarray | None = None, min_episode_step: int = 0):
        # filter out transitions that end with a reset (next_obs is from new episode)
        valid_mask = ~dones
        
        # filter out warmup/standup phase if requested
        if episode_steps is not None and min_episode_step > 0:
            warmup_mask = episode_steps >= min_episode_step
            valid_mask = valid_mask & warmup_mask
            print(f"[INFO]: Filtering to episode steps >= {min_episode_step}: "
                  f"{valid_mask.sum():,} / {len(dones):,} samples ({100*valid_mask.mean():.1f}%)")
        
        self.obs = torch.from_numpy(obs[valid_mask]).float()
        self.actions = torch.from_numpy(actions[valid_mask]).float()
        self.next_obs = torch.from_numpy(next_obs[valid_mask]).float()

        # normalization statistics
        self.normalize = normalize
        if normalize:
            if stats is None:
                self.obs_mean = self.obs.mean(dim=0)
                # Use robust normalization: clip std to avoid explosion from outliers
                # Also use percentile-based std for robustness
                obs_std_raw = self.obs.std(dim=0)
                # Clip extreme values: cap at 10 std from mean
                self.obs_std = obs_std_raw.clamp(min=0.01)
                
                self.action_mean = self.actions.mean(dim=0)
                self.action_std = self.actions.std(dim=0).clamp(min=0.01)
                
                # Report constant dimensions
                const_dims = (obs_std_raw < 0.001).nonzero().squeeze().tolist()
                if const_dims:
                    if isinstance(const_dims, int):
                        const_dims = [const_dims]
                    print(f"[INFO]: Constant observation dimensions (std < 0.001): {const_dims}")
            else:
                self.obs_mean = torch.from_numpy(stats["obs_mean"]).float()
                self.obs_std = torch.from_numpy(stats["obs_std"]).float()
                self.action_mean = torch.from_numpy(stats["action_mean"]).float()
                self.action_std = torch.from_numpy(stats["action_std"]).float()

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        obs = self.obs[idx]
        action = self.actions[idx]
        next_obs = self.next_obs[idx]

        if self.normalize:
            obs = (obs - self.obs_mean) / self.obs_std
            action = (action - self.action_mean) / self.action_std
            next_obs = (next_obs - self.obs_mean) / self.obs_std

        return obs, action, next_obs


def gaussian_nll_loss(pred_mean: torch.Tensor, pred_logvar: torch.Tensor, 
                      target: torch.Tensor) -> torch.Tensor:
    """Gaussian negative log-likelihood loss (averaged over batch and dimensions).
    
    NLL = 0.5 * (log(var) + (target - mean)^2 / var)
    """
    var = pred_logvar.exp()
    loss = 0.5 * (pred_logvar + (target - pred_mean).pow(2) / var)
    return loss.mean()


def compute_metrics(model: ForwardModel, dataloader: DataLoader, device: str, obs_dim: int) -> dict:
    """Compute detailed metrics on a dataset."""
    model.eval()
    
    all_nll_avg = []  # averaged over dimensions (comparable to training loss)
    all_nll_sum = []  # summed over dimensions (used for online surprise)
    all_mse = []
    all_mae = []
    all_pred_std = []
    all_pred_logvar = []
    per_dim_mse = []
    per_dim_nll = []
    
    with torch.no_grad():
        for obs_batch, action_batch, next_obs_batch in dataloader:
            obs_batch = obs_batch.to(device)
            action_batch = action_batch.to(device)
            next_obs_batch = next_obs_batch.to(device)
            
            pred_mean, pred_logvar = model(obs_batch, action_batch)
            var = pred_logvar.exp()
            
            # NLL per sample per dimension
            nll_per_dim = 0.5 * (pred_logvar + (next_obs_batch - pred_mean).pow(2) / var)
            
            # averaged NLL (same as training loss)
            all_nll_avg.append(nll_per_dim.mean(dim=-1).cpu())
            
            # summed NLL (used for online surprise)
            all_nll_sum.append(nll_per_dim.sum(dim=-1).cpu())
            
            # MSE per sample
            mse = ((pred_mean - next_obs_batch) ** 2).mean(dim=-1)
            all_mse.append(mse.cpu())
            
            # MAE per sample
            mae = (pred_mean - next_obs_batch).abs().mean(dim=-1)
            all_mae.append(mae.cpu())
            
            # predicted std
            all_pred_std.append(var.sqrt().mean(dim=-1).cpu())
            all_pred_logvar.append(pred_logvar.mean(dim=-1).cpu())
            
            # per-dimension metrics
            per_dim_mse.append(((pred_mean - next_obs_batch) ** 2).cpu())
            per_dim_nll.append(nll_per_dim.cpu())
    
    all_nll_avg = torch.cat(all_nll_avg)
    all_nll_sum = torch.cat(all_nll_sum)
    all_mse = torch.cat(all_mse)
    all_mae = torch.cat(all_mae)
    all_pred_std = torch.cat(all_pred_std)
    all_pred_logvar = torch.cat(all_pred_logvar)
    per_dim_mse = torch.cat(per_dim_mse, dim=0)
    per_dim_nll = torch.cat(per_dim_nll, dim=0)
    
    return {
        # NLL averaged over dimensions (comparable to training loss)
        "nll_avg_mean": all_nll_avg.mean().item(),
        "nll_avg_std": all_nll_avg.std().item(),
        # NLL summed over dimensions (what online surprise uses)
        "nll_sum_mean": all_nll_sum.mean().item(),
        "nll_sum_std": all_nll_sum.std().item(),
        # prediction errors
        "mse_mean": all_mse.mean().item(),
        "mse_std": all_mse.std().item(),
        "mae_mean": all_mae.mean().item(),
        "mae_std": all_mae.std().item(),
        # uncertainty estimates
        "pred_std_mean": all_pred_std.mean().item(),
        "pred_logvar_mean": all_pred_logvar.mean().item(),
        # per-dimension breakdown
        "per_dim_mse": per_dim_mse.mean(dim=0).numpy(),
        "per_dim_nll": per_dim_nll.mean(dim=0).numpy(),
        # percentiles for threshold setting
        "nll_sum_percentile_95": np.percentile(all_nll_sum.numpy(), 95),
        "nll_sum_percentile_99": np.percentile(all_nll_sum.numpy(), 99),
        # obs_dim for reference
        "obs_dim": obs_dim,
    }


def print_progress_bar(epoch: int, num_epochs: int, train_loss: float, val_loss: float, 
                       val_mse: float, lr: float, best_loss: float, is_best: bool):
    """Print a formatted progress bar."""
    bar_length = 30
    progress = (epoch + 1) / num_epochs
    filled = int(bar_length * progress)
    bar = "=" * filled + ">" + "." * (bar_length - filled - 1)
    
    best_marker = " *BEST*" if is_best else ""
    
    print(f"\rEpoch [{epoch+1:3d}/{num_epochs}] [{bar}] "
          f"train_nll: {train_loss:7.4f} | val_nll: {val_loss:7.4f} | "
          f"val_mse: {val_mse:.6f} | lr: {lr:.2e} | best: {best_loss:.4f}{best_marker}")


def plot_learning_curves(history: dict, output_path: str):
    """Plot and save learning curves."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        
        # Loss curves
        ax = axes[0, 0]
        ax.plot(epochs, history["train_loss"], label="Train NLL", linewidth=2)
        ax.plot(epochs, history["val_loss"], label="Val NLL", linewidth=2)
        ax.axhline(y=min(history["val_loss"]), color='r', linestyle='--', alpha=0.5, 
                   label=f'Best Val: {min(history["val_loss"]):.4f}')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Negative Log-Likelihood")
        ax.set_title("Training and Validation Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # MSE curves
        ax = axes[0, 1]
        ax.plot(epochs, history["train_mse"], label="Train MSE", linewidth=2)
        ax.plot(epochs, history["val_mse"], label="Val MSE", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean Squared Error")
        ax.set_title("Prediction MSE")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Learning rate
        ax = axes[1, 0]
        ax.plot(epochs, history["lr"], linewidth=2, color='green')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Loss gap (overfitting indicator)
        ax = axes[1, 1]
        gap = np.array(history["val_loss"]) - np.array(history["train_loss"])
        ax.plot(epochs, gap, linewidth=2, color='purple')
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.fill_between(epochs, 0, gap, where=(gap > 0), alpha=0.3, color='red', label='Overfitting')
        ax.fill_between(epochs, 0, gap, where=(gap <= 0), alpha=0.3, color='green', label='Underfitting')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val Loss - Train Loss")
        ax.set_title("Generalization Gap")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        print(f"\n[INFO]: Learning curves saved to {output_path}")
        
    except ImportError:
        print("\n[WARNING]: matplotlib not available, skipping learning curve plot")


def plot_test_analysis(test_metrics: dict, per_dim_names: list[str] | None, output_path: str):
    """Plot test set analysis."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Per-dimension MSE
        ax = axes[0]
        per_dim_mse = test_metrics["per_dim_mse"]
        x = np.arange(len(per_dim_mse))
        bars = ax.bar(x, per_dim_mse, color='steelblue', alpha=0.8)
        ax.set_xlabel("Observation Dimension")
        ax.set_ylabel("MSE")
        ax.set_title("Per-Dimension Prediction Error")
        
        if per_dim_names is not None and len(per_dim_names) == len(per_dim_mse):
            ax.set_xticks(x)
            ax.set_xticklabels(per_dim_names, rotation=45, ha='right', fontsize=8)
        
        ax.grid(True, alpha=0.3, axis='y')
        
        # Highlight worst dimensions
        worst_idx = np.argsort(per_dim_mse)[-3:]
        for idx in worst_idx:
            bars[idx].set_color('coral')
        
        # Text summary
        ax = axes[1]
        ax.axis('off')
        
        summary_text = f"""
Test Set Metrics Summary
========================

NLL (avg per dim, comparable to train loss):
  Mean:          {test_metrics['nll_avg_mean']:.4f}
  Std:           {test_metrics['nll_avg_std']:.4f}

NLL (sum, used for online surprise):
  Mean:          {test_metrics['nll_sum_mean']:.4f}
  95th %ile:     {test_metrics['nll_sum_percentile_95']:.4f}
  99th %ile:     {test_metrics['nll_sum_percentile_99']:.4f}

Prediction Error:
  MSE Mean:      {test_metrics['mse_mean']:.6f}
  MSE Std:       {test_metrics['mse_std']:.6f}
  MAE Mean:      {test_metrics['mae_mean']:.6f}
  MAE Std:       {test_metrics['mae_std']:.6f}

Uncertainty Estimation:
  Avg Pred Std:  {test_metrics['pred_std_mean']:.6f}

Worst Predicted Dimensions: {worst_idx.tolist()}
"""
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        print(f"[INFO]: Test analysis saved to {output_path}")
        
    except ImportError:
        print("[WARNING]: matplotlib not available, skipping test analysis plot")


def train_forward_model(
    data_path: str,
    output_dir: str,
    hidden_dims: list[int] | None = None,
    batch_size: int = 256,
    lr: float = 1e-3,
    num_epochs: int = 100,
    val_split: float = 0.1,
    test_split: float = 0.1,
    device: str = "cuda",
    seed: int = 42,
    early_stopping_patience: int = 20,
    min_episode_step: int = 0,
):
    """Train the forward model with full validation and testing.
    
    Args:
        data_path: Path to rollout_data.npz
        output_dir: Directory to save trained model
        hidden_dims: Hidden layer dimensions
        batch_size: Training batch size
        lr: Learning rate
        num_epochs: Number of training epochs
        val_split: Fraction of data for validation
        test_split: Fraction of data for final testing
        device: Device to train on
        seed: Random seed
        early_stopping_patience: Stop if no improvement for this many epochs
        min_episode_step: Minimum episode step to include in training (for filtering warmup/standup)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if hidden_dims is None:
        hidden_dims = [256, 256, 256]

    print("=" * 70)
    print("FORWARD MODEL TRAINING FOR SURPRISE ESTIMATION")
    print("=" * 70)

    # load data
    print(f"\n[1/5] Loading data from {data_path}")
    data = np.load(data_path)
    obs = data["obs"]
    actions = data["actions"]
    next_obs = data["next_obs"]
    dones = data["dones"]
    episode_steps = data["episode_steps"] if "episode_steps" in data else None
    obs_dim = int(data["obs_dim"])
    action_dim = int(data["action_dim"])

    print(f"      obs_dim={obs_dim}, action_dim={action_dim}")
    print(f"      Total samples: {len(obs):,}")
    print(f"      Valid transitions: {(~dones).sum():,} ({100*(~dones).mean():.1f}%)")
    if min_episode_step > 0:
        print(f"      Filtering warmup: min_episode_step={min_episode_step}")

    # load normalization stats (only if NOT filtering warmup - otherwise compute from filtered data)
    stats_path = os.path.join(os.path.dirname(data_path), "normalization_stats.npz")
    if min_episode_step == 0 and os.path.exists(stats_path):
        stats = dict(np.load(stats_path))
        print(f"      Loaded normalization statistics from {stats_path}")
    else:
        stats = None
        if min_episode_step > 0:
            print(f"      Will compute normalization stats from filtered data (warmup excluded)")
        else:
            print("      Computing normalization statistics from data")

    # create dataset
    print(f"\n[2/5] Creating dataset splits (train/val/test: "
          f"{100*(1-val_split-test_split):.0f}/{100*val_split:.0f}/{100*test_split:.0f}%)")
    
    dataset = TransitionDataset(obs, actions, next_obs, dones, normalize=True, stats=stats,
                                  episode_steps=episode_steps, min_episode_step=min_episode_step)
    
    # train/val/test split
    total_size = len(dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size - test_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    print(f"      Train: {len(train_dataset):,} | Val: {len(val_dataset):,} | Test: {len(test_dataset):,}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    # create model
    print(f"\n[3/5] Creating model")
    model = ForwardModel(obs_dim, action_dim, hidden_dims).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"      Architecture: {obs_dim + action_dim} -> {hidden_dims} -> {obs_dim}")
    print(f"      Parameters: {num_params:,}")
    print(f"      Device: {device}")

    # training loop
    print(f"\n[4/5] Training for {num_epochs} epochs")
    print("-" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    
    history = {
        "train_loss": [], "val_loss": [], "train_mse": [], "val_mse": [], "lr": []
    }

    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # training
        model.train()
        train_loss = 0.0
        train_mse = 0.0
        num_batches = 0
        
        for obs_batch, action_batch, next_obs_batch in train_loader:
            obs_batch = obs_batch.to(device)
            action_batch = action_batch.to(device)
            next_obs_batch = next_obs_batch.to(device)

            pred_mean, pred_logvar = model(obs_batch, action_batch)
            loss = gaussian_nll_loss(pred_mean, pred_logvar, next_obs_batch)
            mse = ((pred_mean - next_obs_batch) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_mse += mse.item()
            num_batches += 1

        train_loss /= num_batches
        train_mse /= num_batches

        # validation
        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for obs_batch, action_batch, next_obs_batch in val_loader:
                obs_batch = obs_batch.to(device)
                action_batch = action_batch.to(device)
                next_obs_batch = next_obs_batch.to(device)

                pred_mean, pred_logvar = model(obs_batch, action_batch)
                loss = gaussian_nll_loss(pred_mean, pred_logvar, next_obs_batch)
                mse = ((pred_mean - next_obs_batch) ** 2).mean()

                val_loss += loss.item()
                val_mse += mse.item()
                num_val_batches += 1

        val_loss /= num_val_batches
        val_mse /= num_val_batches

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)
        history["lr"].append(current_lr)

        # check for best model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "obs_dim": obs_dim,
                "action_dim": action_dim,
                "hidden_dims": hidden_dims,
                "obs_mean": dataset.obs_mean,
                "obs_std": dataset.obs_std,
                "action_mean": dataset.action_mean,
                "action_std": dataset.action_std,
                "epoch": epoch + 1,
                "val_loss": val_loss,
            }, os.path.join(output_dir, "forward_model_best.pt"))
        else:
            patience_counter += 1

        # print progress
        print_progress_bar(epoch, num_epochs, train_loss, val_loss, val_mse, 
                          current_lr, best_val_loss, is_best)
        
        # early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\n\n[INFO]: Early stopping at epoch {epoch + 1} (no improvement for {early_stopping_patience} epochs)")
            break

    training_time = time.time() - start_time
    print("\n" + "-" * 70)
    print(f"Training completed in {training_time:.1f}s ({training_time/60:.1f} min)")
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")

    # save final model
    torch.save({
        "model_state_dict": model.state_dict(),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "hidden_dims": hidden_dims,
        "obs_mean": dataset.obs_mean,
        "obs_std": dataset.obs_std,
        "action_mean": dataset.action_mean,
        "action_std": dataset.action_std,
        "epoch": epoch + 1,
        "val_loss": val_loss,
    }, os.path.join(output_dir, "forward_model_final.pt"))

    # plot learning curves
    plot_learning_curves(history, os.path.join(output_dir, "learning_curves.png"))

    # convergence diagnostics
    print(f"\n[5/5] Convergence Diagnostics & Test Evaluation")
    print("-" * 70)
    
    # analyze convergence
    train_losses = np.array(history["train_loss"])
    val_losses = np.array(history["val_loss"])
    
    # compute convergence metrics
    final_epochs = min(10, len(val_losses))
    recent_val_loss = val_losses[-final_epochs:]
    recent_train_loss = train_losses[-final_epochs:]
    
    val_loss_change = (recent_val_loss[-1] - recent_val_loss[0]) / (abs(recent_val_loss[0]) + 1e-8) * 100
    train_val_gap = (val_losses[-1] - train_losses[-1])
    relative_gap = train_val_gap / (abs(train_losses[-1]) + 1e-8) * 100
    
    print("\nConvergence Analysis:")
    print(f"      Final train loss:       {train_losses[-1]:.4f}")
    print(f"      Final val loss:         {val_losses[-1]:.4f}")
    print(f"      Best val loss:          {best_val_loss:.4f} (epoch {best_epoch})")
    print(f"      Train-Val gap:          {train_val_gap:.4f} ({relative_gap:+.1f}%)")
    print(f"      Val loss change (last {final_epochs} epochs): {val_loss_change:+.2f}%")
    
    # convergence verdict
    converged = True
    warnings = []
    
    if abs(val_loss_change) > 5:
        converged = False
        warnings.append(f"Validation loss still changing ({val_loss_change:+.1f}%)")
    
    if relative_gap > 20:
        warnings.append(f"Large train-val gap suggests overfitting ({relative_gap:.1f}%)")
    elif relative_gap < -20:
        warnings.append(f"Val loss lower than train - possible data leakage or noise")
    
    if best_epoch < len(val_losses) - early_stopping_patience:
        warnings.append(f"Best model was {len(val_losses) - best_epoch} epochs ago")
    
    print(f"\n      Convergence Status: {'CONVERGED' if converged and not warnings else 'CHECK WARNINGS'}")
    if warnings:
        for w in warnings:
            print(f"      [!] {w}")
    else:
        print("      [OK] No issues detected")
    
    # test set evaluation
    print(f"\nTest Set Evaluation ({len(test_dataset):,} held-out samples):")
    
    # load best model for testing
    best_checkpoint = torch.load(os.path.join(output_dir, "forward_model_best.pt"))
    model.load_state_dict(best_checkpoint["model_state_dict"])
    
    test_metrics = compute_metrics(model, test_loader, device, obs_dim)
    
    print(f"      NLL (avg/dim):  {test_metrics['nll_avg_mean']:.4f} +/- {test_metrics['nll_avg_std']:.4f}  <- comparable to training loss")
    print(f"      NLL (sum):      {test_metrics['nll_sum_mean']:.4f} +/- {test_metrics['nll_sum_std']:.4f}  <- used for online surprise")
    print(f"      MSE:            {test_metrics['mse_mean']:.6f} +/- {test_metrics['mse_std']:.6f}")
    print(f"      MAE:            {test_metrics['mae_mean']:.6f} +/- {test_metrics['mae_std']:.6f}")
    print(f"      Pred Std:       {test_metrics['pred_std_mean']:.6f}")
    print(f"      Pred LogVar:    {test_metrics['pred_logvar_mean']:.4f}")
    
    # check test vs validation consistency
    test_val_diff = abs(test_metrics['nll_avg_mean'] - best_val_loss) / (abs(best_val_loss) + 1e-8) * 100
    print(f"\n      Test-Val NLL difference: {test_val_diff:.1f}%", end="")
    if test_val_diff < 10:
        print(" [OK - consistent]")
    else:
        print(" [WARNING - may indicate data issues]")
    
    print(f"\n      Surprise thresholds (for online use):")
    print(f"        95th percentile: {test_metrics['nll_sum_percentile_95']:.4f}")
    print(f"        99th percentile: {test_metrics['nll_sum_percentile_99']:.4f}")
    
    # save test metrics
    np.savez(os.path.join(output_dir, "test_metrics.npz"), **test_metrics)
    
    # plot test analysis
    plot_test_analysis(test_metrics, None, os.path.join(output_dir, "test_analysis.png"))

    # save training history
    np.savez(os.path.join(output_dir, "training_history.npz"), **history)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"  - forward_model_best.pt  (best validation loss)")
    print(f"  - forward_model_final.pt (last epoch)")
    print(f"  - learning_curves.png")
    print(f"  - test_analysis.png")
    print(f"  - test_metrics.npz")
    print(f"  - training_history.npz")
    print("=" * 70)

    return model, test_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train forward model for surprise estimation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data_path", type=str, required=True, 
                        help="Path to rollout_data.npz")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save trained model")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 256, 256],
                        help="Hidden layer dimensions")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--num_epochs", type=int, default=100, help="Maximum number of epochs")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--test_split", type=float, default=0.1, help="Test split ratio")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--early_stopping", type=int, default=20, 
                        help="Early stopping patience (epochs)")
    parser.add_argument("--min_episode_step", type=int, default=0,
                        help="Minimum episode step to include (filter warmup/standup, e.g. 50)")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.data_path), "forward_model")

    train_forward_model(
        data_path=args.data_path,
        output_dir=args.output_dir,
        hidden_dims=args.hidden_dims,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.num_epochs,
        val_split=args.val_split,
        test_split=args.test_split,
        device=args.device,
        seed=args.seed,
        early_stopping_patience=args.early_stopping,
        min_episode_step=args.min_episode_step,
    )


if __name__ == "__main__":
    main()
