# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Diagnose Leap Hand contact/collider effectiveness.

Runs a zero-action rollout and records per-step object position and
contact-related diagnostics to identify whether the cube is actually
being held by the hand's colliders.
"""

import argparse
import csv
import os
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Diagnose Leap Hand contact/collider effectiveness.")
parser.add_argument("--num_steps", type=int, default=300, help="Number of environment steps to roll out.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Task name.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry point in task registry."
)
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment.")
parser.add_argument("--output_dir", type=str, default=None, help="Output directory for CSV/PNG.")
parser.add_argument("--output_name", type=str, default="contact_diag", help="Output file stem.")
parser.add_argument("--zero_action", action="store_true", default=True, help="Use zero actions (default: True).")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use a pre-trained checkpoint (ignored for zero-action mode).",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.utils import close_simulation

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks_experimental  # noqa: F401

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Disable fail-fast for diagnostics
    if hasattr(env_cfg, "nonfinite_fail_event_threshold"):
        env_cfg.nonfinite_fail_event_threshold = int(1e9)
    if hasattr(env_cfg, "fail_fast_episode_length_streak_steps"):
        env_cfg.fail_fast_episode_length_streak_steps = 0

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=None)
    uenv = env.unwrapped

    # Setup output
    output_dir = args_cli.output_dir or os.path.join("logs", "analysis")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    obs = env.get_observations()
    num_envs_actual = uenv.num_envs
    if hasattr(uenv, "action_manager"):
        action_dim = int(uenv.action_manager.total_action_dim)
    else:
        action_dim = int(env.action_space.shape[-1])

    rows: list[dict] = []
    episode_lengths = torch.zeros(num_envs_actual, dtype=torch.int32, device=uenv.device)
    total_dones = 0
    total_timeouts = 0

    print(f"\n{'='*60}")
    print(f"  Contact Diagnostic: {args_cli.task}")
    print(f"  num_envs={num_envs_actual}, num_steps={args_cli.num_steps}")
    print(f"  zero_action={args_cli.zero_action}")
    print(f"{'='*60}\n")

    # Print initial conditions
    print("[DIAG] Initial object positions per env:")
    for ei in range(min(num_envs_actual, 4)):
        obj_pos = uenv.object_pos[ei].detach().cpu().tolist()
        print(f"  env {ei}: object pos = [{obj_pos[0]:.4f}, {obj_pos[1]:.4f}, {obj_pos[2]:.4f}]")

    print("\n[DIAG] Initial hand joint positions (env 0):")
    hand_pos = uenv.hand_dof_pos[0].detach().cpu().tolist()
    for ji, jn in enumerate(uenv.hand.joint_names):
        print(f"  {jn}: {hand_pos[ji]:.4f}")

    print(f"\n[DIAG] Joint limits (lower, upper) for env 0:")
    lower = uenv.hand_dof_lower_limits[0].detach().cpu().tolist()
    upper = uenv.hand_dof_upper_limits[0].detach().cpu().tolist()
    for ji, jn in enumerate(uenv.hand.joint_names):
        print(f"  {jn}: [{lower[ji]:.4f}, {upper[ji]:.4f}]")

    print(f"\n[DIAG] Config: fall_dist={uenv.cfg.fall_dist}")
    if hasattr(uenv.cfg, "action_type"):
        print(f"[DIAG] Config: action_type={uenv.cfg.action_type}")
    if hasattr(uenv.cfg, "act_moving_average"):
        print(f"[DIAG] Config: act_moving_average={uenv.cfg.act_moving_average}")
    if hasattr(uenv.cfg, "action_limit_scale"):
        print(f"[DIAG] Config: action_limit_scale={uenv.cfg.action_limit_scale}")
    if hasattr(uenv.cfg, "action_clip"):
        print(f"[DIAG] Config: action_clip={uenv.cfg.action_clip}")

    for step_idx in range(int(args_cli.num_steps)):
        try:
            with torch.inference_mode():
                if args_cli.zero_action:
                    actions = torch.zeros(num_envs_actual, action_dim, device=uenv.device)
                else:
                    raise NotImplementedError("Policy-action mode not implemented in this script.")
                obs, rew, dones, extras = env.step(actions)
        except Exception as exc:
            print(f"[ERROR] Step {step_idx}: {exc}")
            break

        episode_lengths += 1
        done_mask = dones.bool()
        time_out_mask = extras.get("time_outs", torch.zeros_like(dones)).bool() if done_mask.any() else torch.zeros_like(dones, dtype=torch.bool)

        step_dones = int(done_mask.sum().item())
        step_timeouts = int(time_out_mask.sum().item())
        total_dones += step_dones
        total_timeouts += step_timeouts

        # Per-env object z and distance
        obj_z = uenv.object_pos[:, 2].detach().cpu()
        goal_dist = torch.norm(uenv.object_pos - uenv.in_hand_pos, p=2, dim=-1).detach().cpu()

        row = {
            "step": step_idx,
            "mean_reward": float(rew.mean().item()),
            "mean_object_z": float(obj_z.mean().item()),
            "min_object_z": float(obj_z.min().item()),
            "max_object_z": float(obj_z.max().item()),
            "mean_goal_dist": float(goal_dist.mean().item()),
            "min_goal_dist": float(goal_dist.min().item()),
            "max_goal_dist": float(goal_dist.max().item()),
            "num_dones": step_dones,
            "num_timeouts": step_timeouts,
            "mean_ep_len": float(episode_lengths.float().mean().item()),
        }

        # Track per-env 0 details
        row["env0_object_x"] = float(uenv.object_pos[0, 0].item())
        row["env0_object_y"] = float(uenv.object_pos[0, 1].item())
        row["env0_object_z"] = float(uenv.object_pos[0, 2].item())
        row["env0_goal_dist"] = float(goal_dist[0].item())

        rows.append(row)
        episode_lengths[done_mask] = 0

        if step_idx % 50 == 0 or step_idx == int(args_cli.num_steps) - 1:
            print(
                f"[DIAG] step={step_idx:4d} | "
                f"obj_z={row['mean_object_z']:.4f} ({row['min_object_z']:.4f}~{row['max_object_z']:.4f}) | "
                f"goal_dist={row['mean_goal_dist']:.4f} | "
                f"dones={step_dones} (timeouts={step_timeouts}) | "
                f"rew={row['mean_reward']:.3f}"
            )

    env.close()

    # Save CSV
    csv_path = os.path.join(output_dir, f"{args_cli.output_name}.csv")
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[INFO] Saved CSV: {csv_path}")

    # Save plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [r["step"] for r in rows]
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        fig.suptitle(f"Contact Diagnostic: {args_cli.task}", fontsize=13)

        # Object Z
        axes[0].fill_between(steps, [r["min_object_z"] for r in rows], [r["max_object_z"] for r in rows],
                             alpha=0.3, color="tab:blue")
        axes[0].plot(steps, [r["mean_object_z"] for r in rows], color="tab:blue", linewidth=1.5, label="mean object Z")
        axes[0].axhline(y=float(uenv.cfg.fall_dist if hasattr(uenv, 'cfg') else 0.35),
                        color="tab:red", linestyle="--", alpha=0.5, label="fall_dist threshold")
        axes[0].set_ylabel("Object Z (m)")
        axes[0].legend(loc="best", fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # Goal distance
        axes[1].fill_between(steps, [r["min_goal_dist"] for r in rows], [r["max_goal_dist"] for r in rows],
                             alpha=0.3, color="tab:orange")
        axes[1].plot(steps, [r["mean_goal_dist"] for r in rows], color="tab:orange", linewidth=1.5)
        axes[1].set_ylabel("Goal Dist")
        axes[1].grid(True, alpha=0.3)

        # Dones per step
        axes[2].bar(steps, [r["num_dones"] for r in rows], color="tab:red", alpha=0.6, label="dones")
        axes[2].bar(steps, [r["num_timeouts"] for r in rows], color="tab:green", alpha=0.6, label="timeouts")
        axes[2].set_ylabel("Dones/step")
        axes[2].legend(loc="best", fontsize=8)
        axes[2].grid(True, alpha=0.3)

        # Reward
        axes[3].plot(steps, [r["mean_reward"] for r in rows], color="tab:purple", linewidth=1.5)
        axes[3].set_ylabel("Mean Reward")
        axes[3].set_xlabel("Step")
        axes[3].grid(True, alpha=0.3)

        fig.tight_layout()
        png_path = os.path.join(output_dir, f"{args_cli.output_name}.png")
        fig.savefig(png_path, dpi=160)
        plt.close(fig)
        print(f"[INFO] Saved plot: {png_path}")
    except Exception as exc:
        print(f"[WARN] Failed to save plot: {exc}")

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  Total steps:    {len(rows)}")
    print(f"  Total dones:    {total_dones}")
    print(f"  Total timeouts: {total_timeouts}")
    if rows:
        mean_z = sum(r["mean_object_z"] for r in rows) / len(rows)
        mean_dist = sum(r["mean_goal_dist"] for r in rows) / len(rows)
        print(f"  Mean object Z:  {mean_z:.4f}")
        print(f"  Mean goal dist: {mean_dist:.4f}")
        # Check if cube stays in hand
        first_10_z = [r["mean_object_z"] for r in rows[:10]]
        last_10_z = [r["mean_object_z"] for r in rows[-10:]]
        z_drop = sum(first_10_z) / len(first_10_z) - sum(last_10_z) / len(last_10_z)
        print(f"  Z drop (first 10 to last 10): {z_drop:.4f}")
        if z_drop > 0.1:
            print("  [WARN] The cube is falling significantly — colliders may not be holding it.")
        else:
            print("  [OK] Cube Z is stable — colliders appear effective.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
    close_simulation(simulation_app)
