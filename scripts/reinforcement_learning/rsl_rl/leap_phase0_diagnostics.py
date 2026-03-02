#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Run Phase-0 LeapHand diagnostics with single-variable 50-iter experiments."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


def run_cmd(cmd: str, env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, shell=True, env=env, stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {cmd}\nSee log: {log_path}")


def parse_train_metrics(log_path: Path) -> dict[str, float]:
    it = None
    rewards = []
    lengths = []
    pat_it = re.compile(r"Learning iteration\s+(\d+)/(\d+)")
    pat_rew = re.compile(r"Mean reward:\s+(-?\d+\.\d+)")
    pat_len = re.compile(r"Mean episode length:\s+(-?\d+\.\d+)")
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pat_it.search(line)
            if m:
                it = int(m.group(1))
                continue
            m = pat_rew.search(line)
            if m and it is not None:
                rewards.append((it, float(m.group(1))))
                continue
            m = pat_len.search(line)
            if m and it is not None:
                lengths.append((it, float(m.group(1))))
                it = None
    if not rewards or not lengths:
        raise RuntimeError(f"Failed to parse metrics from {log_path}")
    n = min(len(rewards), len(lengths))
    first = min(10, n)
    last = min(10, n)
    r_vals = [v for _, v in rewards[:n]]
    l_vals = [v for _, v in lengths[:n]]
    return {
        "iters": float(n),
        "reward_first10": sum(r_vals[:first]) / first,
        "reward_last10": sum(r_vals[-last:]) / last,
        "episode_length_first10": sum(l_vals[:first]) / first,
        "episode_length_last10": sum(l_vals[-last:]) / last,
    }


def parse_summary(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return {
        "num_steps": float(out.get("num_steps_collected", "0")),
        "num_dones": float(out.get("num_dones", "0")),
        "done_out_of_reach": float(out.get("done_out_of_reach", "0")),
        "mean_reward_rollout": float(out.get("mean_reward", "0")),
    }


def find_latest_run(log_root: Path, run_name: str) -> Path:
    candidates = sorted(log_root.glob(f"*_{run_name}"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise RuntimeError(f"Cannot find run directory for run_name={run_name}")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=Path, default=Path("/data/huangyongchao/IsaacLab_clean"))
    parser.add_argument("--output_dir", type=Path, default=Path("/data/huangyongchao/IsaacLab_clean/logs/phase0_diag"))
    parser.add_argument("--device_list", type=str, default="0,2")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=50)
    args = parser.parse_args()

    repo = args.repo_root
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_root = repo / "logs" / "rsl_rl" / "leap_hand_reorient"

    baseline_train_log = repo / "logs" / "leap_train_mgpu200_gloo_clip1.log"
    baseline_rollout_summary = repo / "logs" / "rollout_diag" / "leap_model199_diag_summary.txt"
    baseline_train = parse_train_metrics(baseline_train_log)
    baseline_rollout = parse_summary(baseline_rollout_summary)
    baseline_oor_ratio = (
        baseline_rollout["done_out_of_reach"] / baseline_rollout["num_steps"] if baseline_rollout["num_steps"] > 0 else 0.0
    )

    experiments = [
        ("term_fall030", {"LEAP_FALL_DIST": "0.30"}),
        ("spawn_y014", {"LEAP_OBJECT_INIT_POS_Y": "-0.14"}),
        ("capacity_b", {"LEAP_ACTUATOR_PROFILE": "capacity_b"}),
    ]

    results = []
    for run_name, env_overrides in experiments:
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": args.device_list,
                "RSL_RL_DISTRIBUTED_BACKEND": "gloo",
                "LEAP_RESET_POSE_PROFILE": "cradle_v1",
                "LEAP_ACTUATOR_PROFILE": "default",
                "LEAP_OBJECT_INIT_POS_Y": "-0.10",
                "LEAP_FALL_DIST": "0.24",
            }
        )
        env.update(env_overrides)

        train_log = output_dir / f"{run_name}_train.log"
        train_cmd = (
            f"cd {repo} && ./isaaclab.sh -p -m torch.distributed.run --standalone --nproc_per_node=2 "
            f"scripts/reinforcement_learning/rsl_rl/train.py "
            f"--task Isaac-Repose-Cube-Leap-Direct-v0 --run_name {run_name} "
            f"--num_envs {args.num_envs} --headless --distributed --max_iterations {args.max_iterations}"
        )
        run_cmd(train_cmd, env, train_log)

        run_dir = find_latest_run(log_root, run_name)
        ckpt = run_dir / f"model_{args.max_iterations - 1}.pt"
        if not ckpt.exists():
            raise RuntimeError(f"Checkpoint not found: {ckpt}")

        rollout_name = f"{run_name}_rollout"
        rollout_log = output_dir / f"{run_name}_rollout.log"
        rollout_cmd = (
            f"cd {repo} && ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/analyze_rollout.py "
            f"--task Isaac-Repose-Cube-Leap-Direct-v0 --checkpoint {ckpt} "
            f"--num_steps 300 --num_envs 1 --headless --device cuda:0 "
            f"--output_dir {output_dir} --output_name {rollout_name}"
        )
        run_cmd(rollout_cmd, env, rollout_log)

        train_metrics = parse_train_metrics(train_log)
        rollout_summary = parse_summary(output_dir / f"{rollout_name}_summary.txt")
        oor_ratio = rollout_summary["done_out_of_reach"] / rollout_summary["num_steps"] if rollout_summary["num_steps"] > 0 else 0.0

        results.append(
            {
                "experiment": run_name,
                "run_dir": str(run_dir),
                "train_log": str(train_log),
                "reward_first10": train_metrics["reward_first10"],
                "reward_last10": train_metrics["reward_last10"],
                "episode_length_first10": train_metrics["episode_length_first10"],
                "episode_length_last10": train_metrics["episode_length_last10"],
                "out_of_reach_ratio_rollout": oor_ratio,
            }
        )

    baseline = {
        "reward_last10": baseline_train["reward_last10"],
        "episode_length_last10": baseline_train["episode_length_last10"],
        "out_of_reach_ratio_rollout": baseline_oor_ratio,
    }

    for row in results:
        row["gate_episode_len"] = row["episode_length_last10"] >= 1.2 * baseline["episode_length_last10"]
        row["gate_reward"] = (row["reward_last10"] - baseline["reward_last10"]) >= 1.0
        row["gate_out_of_reach"] = row["out_of_reach_ratio_rollout"] <= 0.8 * baseline["out_of_reach_ratio_rollout"]
        row["passes_any_gate"] = row["gate_episode_len"] or row["gate_reward"] or row["gate_out_of_reach"]

    with (output_dir / "phase0_results.json").open("w", encoding="utf-8") as f:
        json.dump({"baseline": baseline, "results": results}, f, indent=2)

    lines = [
        "# Phase-0 Leap Diagnostics",
        "",
        f"- Baseline reward_last10: {baseline['reward_last10']:.4f}",
        f"- Baseline episode_length_last10: {baseline['episode_length_last10']:.4f}",
        f"- Baseline out_of_reach_ratio_rollout: {baseline['out_of_reach_ratio_rollout']:.4f}",
        "",
        "| experiment | reward_last10 | ep_len_last10 | oor_ratio | gate_ep_len | gate_reward | gate_oor | pass_any |",
        "|---|---:|---:|---:|:---:|:---:|:---:|:---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['experiment']} | {row['reward_last10']:.4f} | {row['episode_length_last10']:.4f} | "
            f"{row['out_of_reach_ratio_rollout']:.4f} | {row['gate_episode_len']} | {row['gate_reward']} | "
            f"{row['gate_out_of_reach']} | {row['passes_any_gate']} |"
        )
    (output_dir / "phase0_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[INFO] Wrote results to: {output_dir}")


if __name__ == "__main__":
    main()
