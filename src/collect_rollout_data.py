# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Script to collect rollout data (o_t, a_t, o_{t+1}) from a trained policy for surprise model training."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Collect rollout data from a trained RSL-RL agent.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Unitree-Go2-v0", help="Name of the task.")
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Path to the model checkpoint.",
)
parser.add_argument("--num_steps", type=int, default=10000, help="Number of steps to collect per environment.")
parser.add_argument("--output_dir", type=str, default=None, help="Directory to save collected data.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import numpy as np
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Collect rollout data from trained policy."""
    # override configurations
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # setup output directory
    if args_cli.output_dir is None:
        checkpoint_dir = os.path.dirname(args_cli.checkpoint)
        output_dir = os.path.join(checkpoint_dir, "rollout_data")
    else:
        output_dir = args_cli.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # set the log directory for the environment
    env_cfg.log_dir = output_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # convert to single-agent instance if required
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # load trained model
    resume_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # get observation and action dimensions
    obs_dim = env.observation_space["policy"].shape[-1]
    action_dim = env.num_actions
    num_envs = env.num_envs

    print(f"[INFO]: Observation dim: {obs_dim}, Action dim: {action_dim}, Num envs: {num_envs}")

    # allocate storage for data collection
    # we collect: o_t, a_t, o_{t+1}
    total_transitions = args_cli.num_steps * num_envs
    obs_data = torch.zeros((total_transitions, obs_dim), dtype=torch.float32, device="cpu")
    action_data = torch.zeros((total_transitions, action_dim), dtype=torch.float32, device="cpu")
    next_obs_data = torch.zeros((total_transitions, obs_dim), dtype=torch.float32, device="cpu")
    done_data = torch.zeros((total_transitions,), dtype=torch.bool, device="cpu")

    # reset environment
    obs = env.get_observations()
    obs_policy = obs["policy"]

    print(f"[INFO]: Collecting {args_cli.num_steps} steps from {num_envs} environments...")

    idx = 0
    for step in range(args_cli.num_steps):
        with torch.inference_mode():
            # get action from policy
            actions = policy(obs)

            # store current observation and action
            start_idx = idx
            end_idx = idx + num_envs

            obs_data[start_idx:end_idx] = obs_policy.cpu()
            action_data[start_idx:end_idx] = actions.cpu()

            # step environment
            obs, _, dones, _ = env.step(actions)
            obs_policy = obs["policy"]

            # store next observation and done flags
            next_obs_data[start_idx:end_idx] = obs_policy.cpu()
            done_data[start_idx:end_idx] = dones.bool().cpu()

            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)

            idx += num_envs

        if (step + 1) % 1000 == 0:
            print(f"[INFO]: Collected {step + 1}/{args_cli.num_steps} steps")

    # convert to numpy and save
    print(f"[INFO]: Saving data to {output_dir}")

    np.savez_compressed(
        os.path.join(output_dir, "rollout_data.npz"),
        obs=obs_data.numpy(),
        actions=action_data.numpy(),
        next_obs=next_obs_data.numpy(),
        dones=done_data.numpy(),
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_envs=num_envs,
        num_steps=args_cli.num_steps,
    )

    # compute and save normalization statistics (excluding transitions after reset)
    valid_mask = ~done_data.numpy()
    obs_valid = obs_data.numpy()[valid_mask]
    action_valid = action_data.numpy()[valid_mask]
    next_obs_valid = next_obs_data.numpy()[valid_mask]

    stats = {
        "obs_mean": obs_valid.mean(axis=0),
        "obs_std": obs_valid.std(axis=0) + 1e-6,
        "action_mean": action_valid.mean(axis=0),
        "action_std": action_valid.std(axis=0) + 1e-6,
        "next_obs_mean": next_obs_valid.mean(axis=0),
        "next_obs_std": next_obs_valid.std(axis=0) + 1e-6,
    }

    np.savez(os.path.join(output_dir, "normalization_stats.npz"), **stats)

    print(f"[INFO]: Data collection complete!")
    print(f"[INFO]: Total transitions collected: {total_transitions}")
    print(f"[INFO]: Valid transitions (not after reset): {valid_mask.sum()}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()


