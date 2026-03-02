# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a zero-action truncation test and report survival statistics."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Zero-action truncation test for Isaac Lab tasks.")
parser.add_argument("--task", type=str, required=True, help="Task name.")
parser.add_argument("--num_envs", type=int, default=256, help="Number of vectorized environments.")
parser.add_argument("--horizon_steps", type=int, default=200, help="Zero-action rollout horizon.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.utils import close_simulation
from isaaclab.utils.timer import Timer

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks_experimental  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

Timer.enable = False
Timer.enable_display_output = False


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    init_pos = env.unwrapped.cfg.object_cfg.init_state.pos
    first_done_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    zero_actions = torch.zeros((num_envs, env.action_space.shape[-1]), device=device)

    for step in range(args_cli.horizon_steps):
        with torch.inference_mode():
            _, _, terminated, truncated, _ = env.step(zero_actions)
        done = terminated | truncated
        newly_done = done & (first_done_step < 0)
        first_done_step[newly_done] = step + 1

    env.close()

    done_mask = first_done_step > 0
    survived_mask = first_done_step < 0
    done_count = int(done_mask.sum().item())
    survived_count = int(survived_mask.sum().item())

    print("=" * 80)
    print("Zero-Action Truncation Test")
    print(f"task={args_cli.task}")
    print(f"num_envs={num_envs}, horizon_steps={args_cli.horizon_steps}")
    print(f"object_init_pos_cfg={init_pos}")
    print(f"done_within_horizon={done_count}/{num_envs} ({done_count / num_envs:.2%})")
    print(f"survived_to_horizon={survived_count}/{num_envs} ({survived_count / num_envs:.2%})")
    if done_count > 0:
        done_steps = first_done_step[done_mask].float()
        print(
            "first_done_step_stats: "
            f"mean={done_steps.mean().item():.2f}, "
            f"p50={done_steps.median().item():.2f}, "
            f"min={done_steps.min().item():.0f}, "
            f"max={done_steps.max().item():.0f}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
    close_simulation(simulation_app)
