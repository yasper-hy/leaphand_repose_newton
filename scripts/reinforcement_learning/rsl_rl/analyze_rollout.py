# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a fixed-length policy rollout and export per-step diagnostics (CSV + PNG)."""

import argparse
import csv
import os
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Analyze a trained RSL-RL checkpoint rollout.")
parser.add_argument("--num_steps", type=int, default=400, help="Number of environment steps to roll out.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--env_index", type=int, default=0, help="Environment index to log.")
parser.add_argument("--task", type=str, default=None, help="Task name.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry point in task registry."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--output_dir", type=str, default=None, help="Output directory for CSV/PNG summary.")
parser.add_argument("--output_name", type=str, default="rollout_debug", help="Output file stem.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch simulator (or standalone, depending on flags)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.utils import close_simulation
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks_experimental  # noqa: F401

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # Keep diagnostics running through unstable segments to capture traces.
    if hasattr(env_cfg, "nonfinite_fail_event_threshold"):
        env_cfg.nonfinite_fail_event_threshold = int(1e9)
    if hasattr(env_cfg, "nonfinite_fail_window_steps"):
        env_cfg.nonfinite_fail_window_steps = int(1e9)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            raise RuntimeError("No pre-trained checkpoint is available for this task.")
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    output_dir = args_cli.output_dir or os.path.join(log_dir, "analysis")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    env_index = int(args_cli.env_index)
    if env_index < 0 or env_index >= env.unwrapped.num_envs:
        raise ValueError(f"--env_index={env_index} is out of range for num_envs={env.unwrapped.num_envs}")

    obs = env.get_observations()
    rows: list[dict] = []
    done_out_of_reach = 0
    done_timeout = 0
    error_message = ""
    prev_joint_pos = None

    for step_idx in range(int(args_cli.num_steps)):
        uenv = env.unwrapped
        pre_goal_dist = torch.norm(uenv.object_pos[env_index] - uenv.in_hand_pos[env_index], p=2).item()
        pre_object_z = float(uenv.object_pos[env_index, 2].item())
        pre_target_z = float(uenv.in_hand_pos[env_index, 2].item())
        pre_episode_length = float(env.episode_length_buf[env_index].item())

        try:
            with torch.inference_mode():
                actions = policy(obs)
                obs, rew, dones, extras = env.step(actions)
        except Exception as exc:
            error_message = str(exc)
            break

        uenv = env.unwrapped
        goal_dist = torch.norm(uenv.object_pos[env_index] - uenv.in_hand_pos[env_index], p=2).item()
        fall_dist = float(uenv.cfg.fall_dist)
        done = bool(dones[env_index].item())
        if "time_outs" in extras:
            time_out = bool(extras["time_outs"][env_index].item()) if done else False
        else:
            # RSL-RL wrapper omits timeout flags for finite-horizon tasks.
            time_out = bool(done and (pre_episode_length >= float(uenv.max_episode_length - 1)))
        out_of_reach = bool(done and not time_out)
        if out_of_reach:
            done_out_of_reach += 1
        if time_out:
            done_timeout += 1

        joint_pos = uenv.hand_dof_pos[env_index]
        cur_targets = uenv.cur_targets[env_index]
        joint_step_delta = 0.0
        if prev_joint_pos is not None:
            joint_step_delta = float(torch.norm(joint_pos - prev_joint_pos, p=2).item())
        prev_joint_pos = joint_pos.clone()

        rows.append(
            {
                "step": step_idx,
                "reward": float(rew[env_index].item()),
                "pre_episode_length": pre_episode_length,
                "episode_length": float(env.episode_length_buf[env_index].item()),
                "pre_goal_dist": float(pre_goal_dist),
                "goal_dist": float(goal_dist),
                "fall_dist": float(fall_dist),
                "pre_object_z": pre_object_z,
                "object_z": float(uenv.object_pos[env_index, 2].item()),
                "pre_target_z": pre_target_z,
                "target_z": float(uenv.in_hand_pos[env_index, 2].item()),
                "action_norm": float(torch.norm(actions[env_index], p=2).item()),
                "action_abs_max": float(torch.abs(actions[env_index]).max().item()),
                "joint_pos_l2": float(torch.norm(joint_pos, p=2).item()),
                "joint_pos_std": float(torch.std(joint_pos).item()),
                "joint_step_delta_l2": joint_step_delta,
                "target_pos_l2": float(torch.norm(cur_targets, p=2).item()),
                "target_to_joint_l2": float(torch.norm(cur_targets - joint_pos, p=2).item()),
                "done": int(done),
                "done_out_of_reach": int(out_of_reach),
                "done_timeout": int(time_out),
            }
        )

    env.close()

    csv_path = os.path.join(output_dir, f"{args_cli.output_name}.csv")
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "checkpoint": resume_path,
        "num_steps_collected": len(rows),
        "requested_num_steps": int(args_cli.num_steps),
        "num_dones": int(sum(r["done"] for r in rows)),
        "done_out_of_reach": int(done_out_of_reach),
        "done_timeout": int(done_timeout),
        "mean_reward": float(sum(r["reward"] for r in rows) / max(len(rows), 1)),
        "mean_goal_dist": float(sum(r["goal_dist"] for r in rows) / max(len(rows), 1)),
        "mean_action_norm": float(sum(r["action_norm"] for r in rows) / max(len(rows), 1)),
        "mean_joint_step_delta_l2": float(sum(r["joint_step_delta_l2"] for r in rows) / max(len(rows), 1)),
        "mean_target_to_joint_l2": float(sum(r["target_to_joint_l2"] for r in rows) / max(len(rows), 1)),
        "terminated_early": int(len(rows) < int(args_cli.num_steps)),
        "max_episode_length": int(env.unwrapped.max_episode_length),
        "error_message": error_message,
    }
    summary_path = os.path.join(output_dir, f"{args_cli.output_name}_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

    png_path = os.path.join(output_dir, f"{args_cli.output_name}.png")
    try:
        import matplotlib.pyplot as plt

        if not rows:
            raise RuntimeError("No rollout rows were collected; skipping plot.")

        steps = [r["step"] for r in rows]
        reward = [r["reward"] for r in rows]
        ep_len = [r["episode_length"] for r in rows]
        goal_dist = [r["goal_dist"] for r in rows]
        fall_dist_line = [r["fall_dist"] for r in rows]
        action_norm = [r["action_norm"] for r in rows]
        joint_step_delta = [r["joint_step_delta_l2"] for r in rows]
        done_steps_oor = [r["step"] for r in rows if r["done_out_of_reach"] == 1]
        done_steps_timeout = [r["step"] for r in rows if r["done_timeout"] == 1]

        fig, axes = plt.subplots(5, 1, figsize=(12, 10.5), sharex=True)
        axes[0].plot(steps, reward, color="tab:blue", linewidth=1.0)
        axes[0].set_ylabel("reward")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(steps, ep_len, color="tab:orange", linewidth=1.0)
        axes[1].set_ylabel("ep_len")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(steps, goal_dist, color="tab:red", linewidth=1.0, label="goal_dist")
        axes[2].plot(steps, fall_dist_line, color="tab:green", linewidth=1.0, linestyle="--", label="fall_dist")
        axes[2].set_ylabel("distance")
        axes[2].legend(loc="best")
        axes[2].grid(True, alpha=0.3)

        axes[3].plot(steps, action_norm, color="tab:purple", linewidth=1.0)
        axes[3].set_ylabel("action_norm")
        axes[3].grid(True, alpha=0.3)

        axes[4].plot(steps, joint_step_delta, color="tab:brown", linewidth=1.0)
        axes[4].set_ylabel("joint_dpos")
        axes[3].set_xlabel("step")
        axes[4].set_xlabel("step")
        axes[4].grid(True, alpha=0.3)

        for ax in axes:
            for s in done_steps_oor:
                ax.axvline(s, color="tab:red", alpha=0.08, linewidth=0.8)
            for s in done_steps_timeout:
                ax.axvline(s, color="tab:green", alpha=0.08, linewidth=0.8)

        fig.tight_layout()
        fig.savefig(png_path, dpi=160)
        plt.close(fig)
        print(f"[INFO] Saved rollout plot: {png_path}")
    except Exception as exc:
        print(f"[WARN] Failed to export plot: {exc}")

    print(f"[INFO] Saved rollout CSV: {csv_path}")
    print(f"[INFO] Saved summary: {summary_path}")
    print(f"[INFO] Summary stats: {summary}")


if __name__ == "__main__":
    main()
    close_simulation(simulation_app)
