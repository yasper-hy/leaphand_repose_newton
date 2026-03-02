# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play an RSL-RL checkpoint and export an MP4 via offline frame capture + ffmpeg."""

import argparse
import ctypes
import os
import shutil
import subprocess
import sys
import time

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint and export MP4 without RecordVideo/Replicator.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

parser.add_argument("--video_frames", type=int, default=300, help="Number of frames to capture.")
parser.add_argument("--video_fps", type=int, default=30, help="Output MP4 FPS.")
parser.add_argument("--video_name", type=str, default="policy_demo", help="Output video file stem.")
parser.add_argument("--video_dir", type=str, default=None, help="Output folder for frames and mp4.")
parser.add_argument("--crf", type=int, default=20, help="ffmpeg H.264 quality (lower is higher quality).")
parser.add_argument("--capture_timeout", type=float, default=5.0, help="Max seconds to wait for one frame capture.")
parser.add_argument("--warmup_updates", type=int, default=8, help="Kit updates before starting frame capture.")
parser.add_argument("--ffmpeg_bin", type=str, default="ffmpeg", help="ffmpeg executable path.")
parser.add_argument("--keep_frames", action="store_true", default=False, help="Keep extracted PNG frames.")
parser.add_argument("--skip_ffmpeg", action="store_true", default=False, help="Skip MP4 encode stage.")
parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing frames/video files.")
parser.add_argument(
    "--capture_backend",
    type=str,
    default="viewport",
    choices=["viewport", "rgb_array"],
    help="Frame capture backend. Use rgb_array for headless camera rendering.",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args


# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.utils import close_simulation, is_simulation_running
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab.utils.timer import Timer

Timer.enable = False
Timer.enable_display_output = False

import isaaclab_tasks_experimental  # noqa: F401

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


class OfflineFrameCapture:
    """Captures viewport LDR frames into PNG files."""

    def __init__(self, viewport_api, timeout_s: float):
        self._viewport_api = viewport_api
        self._timeout_s = timeout_s
        self._done = False
        self._error: str | None = None
        self._frame_path = ""

    def _on_viewport_captured(self, buffer, buffer_size, width, height, _fmt) -> None:
        try:
            ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.POINTER(ctypes.c_ubyte * buffer_size)
            ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
            content = ctypes.pythonapi.PyCapsule_GetPointer(buffer, None)

            rgba = np.frombuffer(content.contents, dtype=np.uint8).reshape(height, width, 4)
            rgb = rgba[:, :, :3]
            Image.fromarray(rgb, mode="RGB").save(self._frame_path)
        except Exception as exc:
            self._error = str(exc)
        finally:
            self._done = True

    def capture(self, frame_path: str) -> None:
        import omni.kit.app
        import omni.kit.viewport.utility as vp_utils

        self._frame_path = frame_path
        self._done = False
        self._error = None

        vp_utils.capture_viewport_to_buffer(self._viewport_api, self._on_viewport_captured)

        app = omni.kit.app.get_app()
        deadline = time.time() + self._timeout_s
        while not self._done and time.time() < deadline:
            app.update()

        if not self._done:
            raise RuntimeError(f"Timed out while capturing frame: {frame_path}")
        if self._error is not None:
            raise RuntimeError(f"Capture callback failed for {frame_path}: {self._error}")


def _save_rgb_array_frame(frame_data, frame_path: str) -> None:
    """Save an RGB/RGBA frame array to PNG."""
    if frame_data is None:
        raise RuntimeError("render() returned None. Try enabling cameras with --enable_cameras.")
    if isinstance(frame_data, (list, tuple)):
        if len(frame_data) == 0:
            raise RuntimeError("render() returned an empty frame list.")
        frame_data = frame_data[0]
    frame = np.asarray(frame_data)
    if frame.ndim != 3:
        raise RuntimeError(f"Unexpected frame shape from render(): {frame.shape}")
    if frame.shape[2] >= 4:
        frame = frame[:, :, :3]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    Image.fromarray(frame, mode="RGB").save(frame_path)


def _encode_mp4(frame_pattern: str, output_mp4: str) -> None:
    ffmpeg_overwrite_flag = "-y" if args_cli.overwrite else "-n"
    cmd = [
        args_cli.ffmpeg_bin,
        ffmpeg_overwrite_flag,
        "-framerate",
        str(args_cli.video_fps),
        "-i",
        frame_pattern,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(args_cli.crf),
        output_mp4,
    ]
    print(f"[INFO] Encoding mp4 with ffmpeg: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play and capture viewport frames without Gym RecordVideo."""
    use_viewport = args_cli.capture_backend == "viewport"
    if use_viewport:
        try:
            import omni.kit.app
            import omni.kit.viewport.utility as vp_utils
        except ImportError as exc:
            raise RuntimeError(
                "Viewport capture API is unavailable. "
                "Use `--capture_backend rgb_array --headless --enable_cameras` for headless capture."
            ) from exc
    else:
        import omni.kit.app

    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for loading experiments
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    # output layout
    output_dir = args_cli.video_dir or os.path.join(log_dir, "videos", "offline")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, f"{args_cli.video_name}_frames")
    video_path = os.path.join(output_dir, f"{args_cli.video_name}.mp4")
    if os.path.exists(frames_dir):
        if args_cli.overwrite:
            shutil.rmtree(frames_dir)
        else:
            raise FileExistsError(f"Frames directory already exists: {frames_dir}. Use --overwrite to replace it.")
    os.makedirs(frames_dir, exist_ok=True)

    # create environment (no RecordVideo; capture backend controlled by args)
    render_mode = None if use_viewport else "rgb_array"
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    frame_capture = None
    if use_viewport:
        viewport_api = vp_utils.get_active_viewport()
        if viewport_api is None:
            raise RuntimeError("No active viewport found. Please run with viewer enabled (not headless).")
        frame_capture = OfflineFrameCapture(viewport_api, timeout_s=args_cli.capture_timeout)

    app = omni.kit.app.get_app()
    for _ in range(args_cli.warmup_updates):
        app.update()

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    print(f"[INFO] Capturing {args_cli.video_frames} frames into: {frames_dir}")
    for frame_idx in range(args_cli.video_frames):
        if not is_simulation_running(simulation_app, env.unwrapped.sim):
            print(f"[WARN] Simulation stopped early at frame {frame_idx}.")
            break

        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        frame_path = os.path.join(frames_dir, f"frame_{frame_idx:06d}.png")
        if use_viewport:
            frame_capture.capture(frame_path)
        else:
            frame = env.render()
            _save_rgb_array_frame(frame, frame_path)

        if args_cli.real_time:
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    env.close()

    if args_cli.skip_ffmpeg:
        print(f"[INFO] Skipping ffmpeg encode. Frames available at: {frames_dir}")
        return

    frame_pattern = os.path.join(frames_dir, "frame_%06d.png")
    _encode_mp4(frame_pattern, video_path)
    print(f"[INFO] MP4 exported: {video_path}")

    if not args_cli.keep_frames:
        shutil.rmtree(frames_dir, ignore_errors=True)
        print(f"[INFO] Removed temporary frames: {frames_dir}")


if __name__ == "__main__":
    main()
    close_simulation(simulation_app)
