# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export a single-env rollout trajectory to NPZ without requiring video rendering."""

import argparse
import os

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Export rollout trajectory for offline rendering.")
parser.add_argument("--task", type=str, required=True, help="Gym task id.")
parser.add_argument("--steps", type=int, default=900, help="Number of rollout steps.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs (use 1 for replay export).")
parser.add_argument("--policy_mode", type=str, default="zero", choices=["zero", "policy"], help="Action source.")
parser.add_argument("--checkpoint", type=str, default=None, help="Policy checkpoint path for policy mode.")
parser.add_argument(
    "--fallback_to_zero_on_ckpt_mismatch",
    action="store_true",
    default=False,
    help="Fallback to zero actions if checkpoint loading fails due to architecture mismatch.",
)
parser.add_argument("--output", type=str, required=True, help="Output .npz path.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
import warp as wp

from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils import close_simulation
from isaaclab.utils.assets import retrieve_file_path

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks_experimental  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def main():
    if args_cli.policy_mode == "policy" and not args_cli.checkpoint:
        raise ValueError("--checkpoint is required when --policy_mode policy")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device if args_cli.device is not None else "cuda:0",
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    # Match training/play behavior: clip policy actions before stepping env.
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)

    policy = None
    if args_cli.policy_mode == "policy":
        agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
        ckpt = retrieve_file_path(args_cli.checkpoint)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        try:
            try:
                runner.load(ckpt)
            except RuntimeError as exc:
                # Backward-compat for checkpoints saved with "log_std" instead of "std".
                if "Unexpected key(s) in state_dict: \"log_std\"" in str(exc):
                    loaded = torch.load(ckpt, map_location=agent_cfg.device)
                    model_sd = loaded.get("model_state_dict", {})
                    if "log_std" in model_sd and "std" not in model_sd:
                        model_sd["std"] = model_sd.pop("log_std")
                        loaded["model_state_dict"] = model_sd
                        runner.alg.policy.load_state_dict(model_sd, strict=True)
                        print("[INFO] Loaded checkpoint with log_std->std compatibility mapping.")
                    else:
                        raise
                else:
                    raise
            policy = runner.get_inference_policy(device=env.unwrapped.device)
            print(f"[INFO] Loaded checkpoint: {ckpt}")
        except Exception as exc:
            if args_cli.fallback_to_zero_on_ckpt_mismatch:
                print(f"[WARN] Checkpoint load failed, fallback to zero actions. Error: {exc}")
                policy = None
                args_cli.policy_mode = "zero"
            else:
                raise

    obs = env.get_observations()
    base_env = env.unwrapped
    num_actions = base_env.cfg.action_space

    hand_joint_pos = []
    hand_joint_vel = []
    object_root_pose = []
    object_root_vel = []
    rewards = []
    dones = []
    actions_log = []
    goal_dist_log = []
    terminated_log = []
    timeout_log = []

    for _ in range(args_cli.steps):
        with torch.inference_mode():
            if args_cli.policy_mode == "policy":
                actions = policy(obs)
            else:
                actions = torch.zeros((base_env.num_envs, num_actions), device=base_env.device)
            obs, rew, done, _ = env.step(actions)

        # Store only env_0 to keep files compact.
        jp = _to_numpy(wp.to_torch(base_env.hand.data.joint_pos)[0])
        jv = _to_numpy(wp.to_torch(base_env.hand.data.joint_vel)[0])
        obj_pose = np.concatenate(
            [
                _to_numpy(wp.to_torch(base_env.object.data.root_pos_w)[0]),
                _to_numpy(wp.to_torch(base_env.object.data.root_quat_w)[0]),
            ],
            axis=0,
        )
        obj_vel = _to_numpy(wp.to_torch(base_env.object.data.root_vel_w)[0])

        hand_joint_pos.append(jp)
        hand_joint_vel.append(jv)
        object_root_pose.append(obj_pose)
        object_root_vel.append(obj_vel)
        rewards.append(_to_numpy(rew)[0])
        dones.append(bool(_to_numpy(done)[0]))
        actions_log.append(_to_numpy(actions)[0])
        # Done diagnostics: reconstruct out-of-reach margin and done type from env buffers.
        goal_dist = torch.norm(base_env.object_pos[0] - base_env.in_hand_pos[0], p=2).item()
        goal_dist_log.append(goal_dist)
        terminated_log.append(bool(base_env.reset_terminated[0].item()))
        timeout_log.append(bool(base_env.reset_time_outs[0].item()))

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.output)), exist_ok=True)
    np.savez_compressed(
        args_cli.output,
        hand_joint_pos=np.asarray(hand_joint_pos, dtype=np.float32),
        hand_joint_vel=np.asarray(hand_joint_vel, dtype=np.float32),
        object_root_pose=np.asarray(object_root_pose, dtype=np.float32),
        object_root_vel=np.asarray(object_root_vel, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.bool_),
        actions=np.asarray(actions_log, dtype=np.float32),
        goal_dist=np.asarray(goal_dist_log, dtype=np.float32),
        terminated=np.asarray(terminated_log, dtype=np.bool_),
        time_out=np.asarray(timeout_log, dtype=np.bool_),
        fall_dist=np.asarray(float(base_env.cfg.fall_dist), dtype=np.float32),
        task=np.asarray(args_cli.task),
    )
    print(f"[INFO] Trajectory saved to: {args_cli.output}")

    env.close()


if __name__ == "__main__":
    main()
    close_simulation(simulation_app)
