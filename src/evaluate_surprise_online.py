# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Script to evaluate surprise online using a trained forward model.

This script runs the policy while computing surprise in real-time.
It can inject disturbances (external forces/torques, push velocities) to validate 
that surprise increases under perturbations.

Supports video recording to visualize when the robot gets disturbed and correlate
with surprise spikes.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate surprise online with optional disturbances.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Unitree-Go2-v0", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the policy checkpoint.")
parser.add_argument("--forward_model", type=str, required=True, help="Path to trained forward model.")
parser.add_argument("--num_steps", type=int, default=2000, help="Number of steps to run.")
parser.add_argument("--disturbance_type", type=str, default="none", 
                    choices=["none", "push", "external_force", "external_torque"],
                    help="Type of disturbance to inject.")
parser.add_argument("--force_magnitude", type=float, default=20.0, 
                    help="Magnitude of external force (N). Typical: 10-50N")
parser.add_argument("--torque_magnitude", type=float, default=5.0, 
                    help="Magnitude of external torque (Nm). Typical: 2-10Nm")
parser.add_argument("--push_velocity", type=float, default=0.5, 
                    help="Push velocity magnitude (m/s). Typical: 0.3-1.0 m/s")
parser.add_argument("--disturbance_start", type=int, default=200,
                    help="Step at which to start disturbance.")
parser.add_argument("--disturbance_duration", type=int, default=20,
                    help="Duration of disturbance in steps. At 50Hz, 20 steps = 0.4s")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--flat_terrain", action="store_true", help="Override terrain to flat (for rough policy on flat ground).")
parser.add_argument("--plot", action="store_true", help="Plot surprise over time.")
parser.add_argument("--video", action="store_true", help="Record video of the evaluation.")
parser.add_argument("--video_resolution", type=int, nargs=2, default=[1920, 1080], 
                    metavar=('WIDTH', 'HEIGHT'), help="Video resolution (default: 1920 1080).")
parser.add_argument("--real_time", action="store_true", help="Run at real-time speed for visualization.")
parser.add_argument("--output_dir", type=str, default=None, help="Directory to save results.")
parser.add_argument("--ignore_warmup_steps", type=int, default=0,
                    help="Number of initial steps per episode to ignore in surprise computation (e.g., 50 for bipedal standup).")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# enable cameras if recording video
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401

# Import unitree_rl_lab tasks (must be after SimulationApp)
try:
    from unitree_rl_lab.tasks.locomotion.robots import go2  # noqa: F401
    print("[INFO]: Loaded unitree_rl_lab.tasks.locomotion.robots.go2")
except ImportError as e:
    print(f"[WARNING]: Could not import unitree_rl_lab tasks: {e}")

from isaaclab_tasks.utils.hydra import hydra_task_config


class ForwardModel(nn.Module):
    """Forward model for observation prediction."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list[int] | None = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 256]

        layers = []
        in_dim = obs_dim + action_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim

        self.feature_net = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dims[-1], obs_dim)
        self.logvar_head = nn.Linear(hidden_dims[-1], obs_dim)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, action], dim=-1)
        features = self.feature_net(x)
        mean = self.mean_head(features)
        logvar = self.logvar_head(features)
        logvar = torch.clamp(logvar, min=-10.0, max=2.0)
        return mean, logvar


class SurpriseEstimator:
    """Online surprise estimation using a trained forward model."""

    def __init__(self, model_path: str, device: str = "cuda", ema_alpha: float = 0.1):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        obs_dim = checkpoint["obs_dim"]
        action_dim = checkpoint["action_dim"]
        hidden_dims = checkpoint["hidden_dims"]
        
        self.model = ForwardModel(obs_dim, action_dim, hidden_dims).to(device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
        self.obs_mean = checkpoint["obs_mean"].to(device)
        self.obs_std = checkpoint["obs_std"].to(device)
        self.action_mean = checkpoint["action_mean"].to(device)
        self.action_std = checkpoint["action_std"].to(device)
        
        self.device = device
        self.ema_alpha = ema_alpha
        self.surprise_ema = None
        
    def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self.obs_mean) / self.obs_std
    
    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return (action - self.action_mean) / self.action_std
    
    @torch.no_grad()
    def compute_surprise(self, obs: torch.Tensor, action: torch.Tensor, 
                         next_obs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute surprise for a batch of transitions.
        
        Args:
            obs: Current observations (num_envs, obs_dim)
            action: Actions taken (num_envs, action_dim)
            next_obs: Resulting observations (num_envs, obs_dim)
            
        Returns:
            Dictionary with surprise values per environment
        """
        obs_n = self.normalize_obs(obs)
        action_n = self.normalize_action(action)
        next_obs_n = self.normalize_obs(next_obs)
        
        # Clip normalized values to avoid extreme outliers affecting predictions
        obs_n = obs_n.clamp(-10, 10)
        action_n = action_n.clamp(-10, 10)
        next_obs_n_clipped = next_obs_n.clamp(-10, 10)
        
        pred_mean, pred_logvar = self.model(obs_n, action_n)
        var = pred_logvar.exp()
        
        # per-sample surprise (summed over dimensions) - shape: (num_envs,)
        surprise = 0.5 * (pred_logvar + (next_obs_n_clipped - pred_mean).pow(2) / var).sum(dim=-1)
        
        # prediction error per environment (normalized space, clipped)
        pred_error = (next_obs_n_clipped - pred_mean).pow(2).mean(dim=-1)
        
        # unnormalized prediction error (for interpretability)
        pred_unnorm = pred_mean * self.obs_std + self.obs_mean
        pred_error_raw = (next_obs - pred_unnorm).pow(2).mean(dim=-1)
        
        # update EMA per environment
        if self.surprise_ema is None:
            self.surprise_ema = surprise.clone()
        else:
            self.surprise_ema = self.ema_alpha * surprise + (1 - self.ema_alpha) * self.surprise_ema
        
        return {
            "surprise": surprise,
            "surprise_ema": self.surprise_ema.clone(),
            "pred_error": pred_error,
            "pred_error_raw": pred_error_raw,
        }
    
    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset EMA for specific environments."""
        if self.surprise_ema is not None and env_ids is not None:
            self.surprise_ema[env_ids] = 0.0


def apply_external_force_torque(robot, force: torch.Tensor, torque: torch.Tensor, body_ids: list[int]):
    """Apply external force and torque to robot bodies.
    
    Args:
        robot: The robot articulation
        force: External forces (num_envs, num_bodies, 3)
        torque: External torques (num_envs, num_bodies, 3)
        body_ids: List of body indices to apply forces to
    """
    robot.set_external_force_and_torque(force, torque, body_ids=body_ids)


def apply_push_velocity(robot, velocity: torch.Tensor):
    """Apply push by setting root velocity.
    
    Args:
        robot: The robot articulation
        velocity: Velocity to add (num_envs, 6) - [lin_vel (3), ang_vel (3)]
    """
    # Handle both torch tensors and Newton ProxyArrays
    current_vel = robot.data.root_vel_w
    if hasattr(current_vel, 'clone'):
        current_vel = current_vel.clone()
    else:
        # Newton ProxyArray - convert to tensor
        current_vel = torch.as_tensor(current_vel, device=velocity.device)
    new_vel = current_vel + velocity
    # Use the new index-based API for Newton compatibility
    robot.write_root_velocity_to_sim_index(root_velocity=new_vel)


def load_policy_from_checkpoint(checkpoint_path: str, obs_dim: int, action_dim: int, device: str):
    """Load policy directly from checkpoint, bypassing RSL-RL runner config issues.
    
    This is a fallback for when the agent config doesn't match the current RSL-RL version.
    """
    from rsl_rl.modules import MLP, GaussianDistribution
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Check checkpoint format
    if "actor_state_dict" in checkpoint:
        actor_state = checkpoint["actor_state_dict"]
        
        # Infer architecture from weights
        # Look for mlp weights: mlp.0.weight, mlp.2.weight, etc.
        hidden_dims = []
        layer_idx = 0
        while f"mlp.{layer_idx * 2}.weight" in actor_state:
            w = actor_state[f"mlp.{layer_idx * 2}.weight"]
            if layer_idx == 0:
                input_dim = w.shape[1]
            hidden_dims.append(w.shape[0])
            layer_idx += 1
        
        # Last hidden dim is actually the output (action) dim
        output_dim = hidden_dims.pop()
        
        print(f"[INFO]: Inferred actor architecture: {input_dim} -> {hidden_dims} -> {output_dim}")
        
        # Build MLP actor
        mlp = MLP(input_dim, output_dim, hidden_dims, activation="elu").to(device)
        
        # Load MLP weights (need to strip 'mlp.' prefix)
        mlp_state = {k.replace('mlp.', ''): v for k, v in actor_state.items() if k.startswith("mlp.")}
        mlp.load_state_dict(mlp_state)
        
        # Get std params for distribution
        if "distribution.std_param" in actor_state:
            std_param = actor_state["distribution.std_param"].to(device)
        else:
            std_param = torch.zeros(output_dim, device=device)
        
        mlp.eval()
        
        # Add a dummy reset method that MLP doesn't have
        mlp.reset = lambda x=None: None
        
        def inference_policy(obs_dict):
            """Wrapper to match the expected interface."""
            obs = obs_dict["policy"]
            return mlp(obs)
        
        return inference_policy, mlp
    else:
        raise ValueError(f"Unknown checkpoint format. Keys: {list(checkpoint.keys())}")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Run online surprise evaluation with physical disturbances."""
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    
    # Override terrain to flat if requested (keeps observations the same)
    if args_cli.flat_terrain:
        print("[INFO]: Overriding terrain to flat ground")
        env_cfg.scene.terrain.terrain_type = "plane"
        env_cfg.scene.terrain.terrain_generator = None
        # Disable terrain curriculum
        if hasattr(env_cfg, 'curriculum') and hasattr(env_cfg.curriculum, 'terrain_levels'):
            env_cfg.curriculum.terrain_levels = None
    
    # set video resolution if recording
    if args_cli.video:
        env_cfg.viewer.resolution = tuple(args_cli.video_resolution)
        print(f"[INFO]: Video resolution set to {args_cli.video_resolution[0]}x{args_cli.video_resolution[1]}")

    # setup output directory
    if args_cli.output_dir is None:
        checkpoint_dir = os.path.dirname(args_cli.checkpoint)
        output_dir = os.path.join(checkpoint_dir, "surprise_evaluation")
    else:
        output_dir = args_cli.output_dir
    os.makedirs(output_dir, exist_ok=True)

    env_cfg.log_dir = output_dir

    # create environment with video recording support
    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    
    # wrap for video recording if requested
    if args_cli.video:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_folder = os.path.join(output_dir, "videos")
        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step == 0,  # record from start
            "video_length": args_cli.num_steps,
            "disable_logger": True,
            "name_prefix": f"surprise_eval_{args_cli.disturbance_type}_{timestamp}",
        }
        print("[INFO]: Recording video during evaluation.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    # get step dt for real-time playback
    dt = env.unwrapped.step_dt

    # get the underlying Isaac Lab environment and robot
    isaac_env = env.unwrapped
    robot = isaac_env.scene["robot"]
    num_envs = isaac_env.num_envs
    device = isaac_env.device
    
    # find body index for base/root body (different robots have different names)
    base_body_names = ["base", "pelvis", "trunk", "torso", "body"]
    base_body_ids = None
    for body_name in base_body_names:
        try:
            base_body_ids = robot.find_bodies(body_name)[0]
            if isinstance(base_body_ids, int):
                base_body_ids = [base_body_ids]
            print(f"[INFO]: Found base body '{body_name}' with IDs: {base_body_ids}")
            break
        except ValueError:
            continue
    
    if base_body_ids is None:
        # fallback to first body
        base_body_ids = [0]
        print(f"[INFO]: Using first body as base, IDs: {base_body_ids}")

    # load policy
    resume_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[INFO]: Loading policy from: {resume_path}")
    
    # Try standard RSL-RL loading first, fallback to direct loading if config mismatch
    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=device)
        
        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            policy_nn = runner.alg.actor_critic
        print("[INFO]: Policy loaded via OnPolicyRunner")
    except (KeyError, TypeError, ValueError) as e:
        print(f"[WARNING]: OnPolicyRunner failed ({e}), loading policy directly from checkpoint")
        obs_dim = env.observation_space["policy"].shape[0]
        action_dim = env.action_space.shape[0]
        policy, policy_nn = load_policy_from_checkpoint(resume_path, obs_dim, action_dim, device)

    # load forward model
    print(f"[INFO]: Loading forward model from: {args_cli.forward_model}")
    surprise_estimator = SurpriseEstimator(args_cli.forward_model, device=device)

    # compute disturbance window
    disturbance_end = args_cli.disturbance_start + args_cli.disturbance_duration

    # storage for logging - per environment
    surprise_history = []  # (num_steps, num_envs)
    surprise_ema_history = []
    pred_error_history = []
    pred_error_raw_history = []
    disturbance_active_history = []
    episode_step_history = []  # track episode step for warmup filtering

    # track episode step per environment (for warmup filtering)
    episode_steps = torch.zeros(num_envs, dtype=torch.int32, device=device)

    # run evaluation
    obs = env.get_observations()

    print(f"[INFO]: Running evaluation for {args_cli.num_steps} steps with {num_envs} environments")
    print(f"[INFO]: Disturbance type: {args_cli.disturbance_type}")
    print(f"[INFO]: Disturbance window: steps {args_cli.disturbance_start} to {disturbance_end}")
    if args_cli.ignore_warmup_steps > 0:
        print(f"[INFO]: Ignoring first {args_cli.ignore_warmup_steps} steps per episode (warmup/standup)")
    if args_cli.disturbance_type == "external_force":
        print(f"[INFO]: Force magnitude: {args_cli.force_magnitude} N")
    elif args_cli.disturbance_type == "external_torque":
        print(f"[INFO]: Torque magnitude: {args_cli.torque_magnitude} Nm")
    elif args_cli.disturbance_type == "push":
        print(f"[INFO]: Push velocity: {args_cli.push_velocity} m/s")

    for step in range(args_cli.num_steps):
        step_start_time = time.time()
        
        # check if disturbance should be active
        disturbance_is_active = (args_cli.disturbance_type != "none" and 
                                  args_cli.disturbance_start <= step < disturbance_end)
        
        with torch.inference_mode():
            # extract the policy observation tensor
            obs_policy = obs["policy"]
            
            # get action from policy
            actions = policy(obs)
            
            # apply disturbance before stepping if active
            if disturbance_is_active:
                if args_cli.disturbance_type == "external_force":
                    # random force direction, fixed magnitude
                    force_dir = torch.randn(num_envs, 1, 3, device=device)
                    force_dir = force_dir / (force_dir.norm(dim=-1, keepdim=True) + 1e-8)
                    force = force_dir * args_cli.force_magnitude
                    torque = torch.zeros(num_envs, 1, 3, device=device)
                    apply_external_force_torque(robot, force, torque, base_body_ids)
                    
                elif args_cli.disturbance_type == "external_torque":
                    # random torque direction, fixed magnitude
                    force = torch.zeros(num_envs, 1, 3, device=device)
                    torque_dir = torch.randn(num_envs, 1, 3, device=device)
                    torque_dir = torque_dir / (torque_dir.norm(dim=-1, keepdim=True) + 1e-8)
                    torque = torque_dir * args_cli.torque_magnitude
                    apply_external_force_torque(robot, force, torque, base_body_ids)
                    
                elif args_cli.disturbance_type == "push":
                    # random horizontal push
                    push_dir = torch.randn(num_envs, 2, device=device)
                    push_dir = push_dir / (push_dir.norm(dim=-1, keepdim=True) + 1e-8)
                    velocity = torch.zeros(num_envs, 6, device=device)
                    velocity[:, 0] = push_dir[:, 0] * args_cli.push_velocity
                    velocity[:, 1] = push_dir[:, 1] * args_cli.push_velocity
                    apply_push_velocity(robot, velocity)
            else:
                # clear any external forces when not disturbing
                if args_cli.disturbance_type in ["external_force", "external_torque"]:
                    force = torch.zeros(num_envs, 1, 3, device=device)
                    torque = torch.zeros(num_envs, 1, 3, device=device)
                    apply_external_force_torque(robot, force, torque, base_body_ids)
            
            # step environment
            obs_next, _, dones, _ = env.step(actions)
            obs_policy_next = obs_next["policy"]
            
            # compute surprise for all environments
            surprise_result = surprise_estimator.compute_surprise(
                obs_policy, actions, obs_policy_next
            )
            
            # log per-environment values
            surprise_history.append(surprise_result["surprise"].cpu().numpy())
            surprise_ema_history.append(surprise_result["surprise_ema"].cpu().numpy())
            pred_error_history.append(surprise_result["pred_error"].cpu().numpy())
            pred_error_raw_history.append(surprise_result["pred_error_raw"].cpu().numpy())
            disturbance_active_history.append(1.0 if disturbance_is_active else 0.0)
            episode_step_history.append(episode_steps.cpu().numpy().copy())
            
            # DEBUG: Log observation statistics at specific steps
            if step == 100:  # During normal walking
                obs_np = obs_policy.cpu().numpy()
                print(f"\n[DEBUG] Step {step} observation stats:")
                print(f"  Mean: {obs_np.mean():.4f}, Std: {obs_np.std():.4f}")
                print(f"  Min: {obs_np.min():.4f}, Max: {obs_np.max():.4f}")
                print(f"  Per-dim mean range: [{obs_np.mean(axis=0).min():.4f}, {obs_np.mean(axis=0).max():.4f}]")
                print(f"  Dim 25 value: {obs_np[:, 25].mean():.6f}")
            
            # update episode step counters
            episode_steps = episode_steps + 1
            
            # reset surprise EMA for terminated environments
            dones_bool = dones.bool() if hasattr(dones, 'bool') else dones
            if dones_bool.any():
                surprise_estimator.reset(dones_bool.nonzero().squeeze(-1))
                # reset episode steps for terminated envs
                episode_steps[dones_bool] = 0
            
            policy_nn.reset(dones_bool)
            obs = obs_next

        # real-time delay for visualization
        if args_cli.real_time:
            elapsed = time.time() - step_start_time
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        if (step + 1) % 500 == 0:
            recent_surprise = np.mean([s.mean() for s in surprise_history[-100:]])
            recent_ema = np.mean([s.mean() for s in surprise_ema_history[-100:]])
            print(f"Step {step + 1}: mean_surprise={recent_surprise:.4f}, mean_ema={recent_ema:.4f}, "
                  f"disturbance={disturbance_is_active}")

    # convert to arrays
    surprise_history = np.stack(surprise_history)  # (num_steps, num_envs)
    surprise_ema_history = np.stack(surprise_ema_history)
    pred_error_history = np.stack(pred_error_history)
    disturbance_active_history = np.array(disturbance_active_history)
    episode_step_history = np.stack(episode_step_history)  # (num_steps, num_envs)

    # create warmup mask (True = past warmup phase)
    warmup_mask = episode_step_history >= args_cli.ignore_warmup_steps  # (num_steps, num_envs)
    
    # compute mean across environments for summary (excluding warmup)
    if args_cli.ignore_warmup_steps > 0:
        # mask warmup values with NaN, then use nanmean
        surprise_masked = np.where(warmup_mask, surprise_history, np.nan)
        surprise_mean = np.nanmean(surprise_masked, axis=1)
        surprise_std = np.nanstd(surprise_masked, axis=1)
        # replace any remaining NaN (all envs in warmup) with 0
        surprise_mean = np.nan_to_num(surprise_mean, nan=0.0)
        surprise_std = np.nan_to_num(surprise_std, nan=0.0)
    else:
        surprise_mean = surprise_history.mean(axis=1)
        surprise_std = surprise_history.std(axis=1)
    
    # save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "surprise_per_env": surprise_history,
        "surprise_ema_per_env": surprise_ema_history,
        "pred_error_per_env": pred_error_history,
        "surprise_mean": surprise_mean,
        "surprise_std": surprise_std,
        "disturbance_active": disturbance_active_history,
        "episode_steps": episode_step_history,
        "warmup_mask": warmup_mask,
        "ignore_warmup_steps": args_cli.ignore_warmup_steps,
        "disturbance_type": args_cli.disturbance_type,
        "force_magnitude": args_cli.force_magnitude,
        "torque_magnitude": args_cli.torque_magnitude,
        "push_velocity": args_cli.push_velocity,
        "disturbance_start": args_cli.disturbance_start,
        "disturbance_end": disturbance_end,
        "num_envs": num_envs,
    }
    
    results_path = os.path.join(output_dir, f"surprise_results_{timestamp}.npz")
    np.savez(results_path, **results)
    print(f"\n[INFO]: Results saved to {results_path}")

    # compute and print statistics
    pre_idx = slice(0, args_cli.disturbance_start)
    during_idx = slice(args_cli.disturbance_start, disturbance_end)
    post_idx = slice(disturbance_end, None)
    
    print("\n" + "=" * 70)
    print("SURPRISE STATISTICS (averaged across all environments)")
    print("=" * 70)
    
    pre_surprise = surprise_mean[pre_idx]
    during_surprise = surprise_mean[during_idx]
    post_surprise = surprise_mean[post_idx]
    
    if len(pre_surprise) > 0:
        print(f"Pre-disturbance:    mean={pre_surprise.mean():.4f}, std={pre_surprise.std():.4f}")
    if len(during_surprise) > 0:
        print(f"During disturbance: mean={during_surprise.mean():.4f}, std={during_surprise.std():.4f}")
    if len(post_surprise) > 0:
        print(f"Post-disturbance:   mean={post_surprise.mean():.4f}, std={post_surprise.std():.4f}")
    
    if len(pre_surprise) > 0 and len(during_surprise) > 0:
        increase_factor = during_surprise.mean() / (pre_surprise.mean() + 1e-8)
        print(f"\nSurprise increase factor: {increase_factor:.2f}x")
    
    # per-environment statistics during disturbance
    print(f"\nPer-environment surprise during disturbance:")
    env_surprise_during = surprise_history[during_idx].mean(axis=0)
    print(f"  Min:  {env_surprise_during.min():.4f}")
    print(f"  Max:  {env_surprise_during.max():.4f}")
    print(f"  Mean: {env_surprise_during.mean():.4f}")
    print(f"  Std:  {env_surprise_during.std():.4f}")
    print("=" * 70)

    # plotting
    if args_cli.plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            # compute time axis (for video synchronization)
            time_axis = np.arange(len(surprise_mean)) * dt
            disturbance_start_time = args_cli.disturbance_start * dt
            disturbance_end_time = disturbance_end * dt
            
            fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
            
            # surprise with confidence band
            ax = axes[0]
            ax.fill_between(time_axis, surprise_mean - surprise_std, surprise_mean + surprise_std, 
                           alpha=0.3, color='blue', label='Std across envs')
            ax.plot(time_axis, surprise_mean, linewidth=2, color='blue', label='Mean surprise')
            ax.axvspan(disturbance_start_time, disturbance_end_time, 
                      alpha=0.2, color='red', label='Disturbance window')
            ax.axvline(disturbance_start_time, color='red', linestyle='--', linewidth=2, 
                      label=f'Disturbance start (t={disturbance_start_time:.2f}s)')
            ax.set_ylabel("Surprise (NLL sum)")
            ax.legend(loc='upper right')
            ax.set_title(f"Online Surprise - {args_cli.disturbance_type} disturbance ({num_envs} envs)")
            ax.grid(True, alpha=0.3)
            
            # individual environment traces (sample)
            ax = axes[1]
            num_show = min(10, num_envs)
            for i in range(num_show):
                ax.plot(time_axis, surprise_history[:, i], alpha=0.5, linewidth=0.8)
            ax.axvspan(disturbance_start_time, disturbance_end_time, alpha=0.2, color='red')
            ax.axvline(disturbance_start_time, color='red', linestyle='--', linewidth=2)
            ax.set_ylabel("Surprise (per env)")
            ax.set_title(f"Sample of {num_show} individual environment traces")
            ax.grid(True, alpha=0.3)
            
            # prediction error
            ax = axes[2]
            pred_error_mean = pred_error_history.mean(axis=1)
            ax.plot(time_axis, pred_error_mean, linewidth=2, color='green')
            ax.axvspan(disturbance_start_time, disturbance_end_time, alpha=0.2, color='red')
            ax.axvline(disturbance_start_time, color='red', linestyle='--', linewidth=2)
            ax.set_ylabel("Prediction Error (MSE)")
            ax.set_xlabel("Time (seconds)")
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = os.path.join(output_dir, f"surprise_plot_{timestamp}.png")
            plt.savefig(plot_path, dpi=150)
            print(f"[INFO]: Plot saved to {plot_path}")
            
            # histogram of surprise during disturbance
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.hist(env_surprise_during, bins=30, edgecolor='black', alpha=0.7)
            ax2.axvline(env_surprise_during.mean(), color='red', linestyle='--', 
                       label=f'Mean: {env_surprise_during.mean():.2f}')
            ax2.set_xlabel("Mean Surprise During Disturbance")
            ax2.set_ylabel("Number of Environments")
            ax2.set_title("Distribution of Surprise Across Environments")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            hist_path = os.path.join(output_dir, f"surprise_histogram_{timestamp}.png")
            plt.savefig(hist_path, dpi=150)
            print(f"[INFO]: Histogram saved to {hist_path}")
            
            # print video sync info
            if args_cli.video:
                print(f"\n[INFO]: VIDEO SYNCHRONIZATION INFO")
                print(f"       Disturbance starts at video time: {disturbance_start_time:.2f}s")
                print(f"       Disturbance ends at video time:   {disturbance_end_time:.2f}s")
                print(f"       Total video duration:             {time_axis[-1]:.2f}s")
            
        except ImportError:
            print("[WARNING]: matplotlib not available for plotting")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
