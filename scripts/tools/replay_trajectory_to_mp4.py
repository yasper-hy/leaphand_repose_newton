# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay an exported trajectory and encode an MP4."""

import argparse
import os
import shutil
import subprocess

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Replay trajectory NPZ and export MP4.")
parser.add_argument("--task", type=str, required=True, help="Gym task id.")
parser.add_argument("--trajectory", type=str, required=True, help="Input trajectory .npz file.")
parser.add_argument("--output_mp4", type=str, required=True, help="Output mp4 path.")
parser.add_argument("--fps", type=int, default=30, help="Video fps.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs (use 1).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing files.")
parser.add_argument(
    "--camera_eye",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Optional camera eye position.",
)
parser.add_argument(
    "--camera_lookat",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Optional camera look-at position.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

from isaaclab.utils import close_simulation

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks_experimental  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def _ffmpeg_encode(pattern: str, output_mp4: str, fps: int, overwrite: bool):
    overwrite_flag = "-y" if overwrite else "-n"
    cmd = [
        "ffmpeg",
        overwrite_flag,
        "-framerate",
        str(fps),
        "-i",
        pattern,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        output_mp4,
    ]
    subprocess.run(cmd, check=True)


def main():
    data = np.load(args_cli.trajectory)
    hand_joint_pos = data["hand_joint_pos"]
    hand_joint_vel = data["hand_joint_vel"]
    object_root_pose = data["object_root_pose"]
    object_root_vel = data["object_root_vel"]
    steps = hand_joint_pos.shape[0]

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device if args_cli.device is not None else "cuda:0",
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env.reset()

    base_env = env.unwrapped
    if args_cli.camera_eye is not None and args_cli.camera_lookat is not None:
        base_env.sim.set_camera_view(tuple(args_cli.camera_eye), tuple(args_cli.camera_lookat))
    env_ids = torch.tensor([0], device=base_env.device, dtype=torch.int64)

    out_dir = os.path.abspath(os.path.dirname(args_cli.output_mp4))
    os.makedirs(out_dir, exist_ok=True)
    frames_dir = os.path.join(out_dir, "_replay_frames")
    if os.path.exists(frames_dir):
        if args_cli.overwrite:
            shutil.rmtree(frames_dir)
        else:
            raise FileExistsError(f"Frames dir exists: {frames_dir}. Use --overwrite.")
    os.makedirs(frames_dir, exist_ok=True)

    for i in range(steps):
        jp = torch.tensor(hand_joint_pos[i : i + 1], device=base_env.device)
        jv = torch.tensor(hand_joint_vel[i : i + 1], device=base_env.device)
        op = torch.tensor(object_root_pose[i : i + 1], device=base_env.device)
        ov = torch.tensor(object_root_vel[i : i + 1], device=base_env.device)

        base_env.hand.write_joint_state_to_sim(jp, jv, env_ids=env_ids)
        base_env.hand.set_joint_position_target(jp, env_ids=env_ids)
        base_env.object.write_root_pose_to_sim(op, env_ids=env_ids)
        base_env.object.write_root_velocity_to_sim(ov, env_ids=env_ids)
        base_env.sim.forward()
        frame = env.render()
        if isinstance(frame, (list, tuple)):
            frame = frame[0]
        if frame is None:
            raise RuntimeError("render() returned None. Ensure --enable_cameras is set.")
        frame = np.asarray(frame)
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        Image.fromarray(frame.astype(np.uint8)).save(os.path.join(frames_dir, f"{i:06d}.png"))

    _ffmpeg_encode(os.path.join(frames_dir, "%06d.png"), os.path.abspath(args_cli.output_mp4), args_cli.fps, True)
    print(f"[INFO] MP4 saved: {args_cli.output_mp4}")
    print(f"[INFO] Frames kept at: {frames_dir}")

    env.close()


if __name__ == "__main__":
    main()
    close_simulation(simulation_app)
