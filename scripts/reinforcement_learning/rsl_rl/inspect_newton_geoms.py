"""Inspect Newton model geom shapes to diagnose collision issues."""

import argparse
import sys
import os

os.environ.setdefault("OMNI_LOG_LEVEL", "WARNING")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Repose-Cube-Leap-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import warp as wp

import isaaclab_tasks  # noqa
import isaaclab_tasks_experimental  # noqa

from isaaclab.envs import DirectRLEnvCfg
from isaaclab_newton.physics.newton_manager import NewtonManager

# Resolve the env_cfg_entry_point string to an actual class
spec = gym.spec(args_cli.task)
entry_point_str = spec.kwargs["env_cfg_entry_point"]
# Parse "module.path:ClassName" format
module_path, class_name = entry_point_str.rsplit(":", 1)
import importlib
mod = importlib.import_module(module_path)
CfgClass = getattr(mod, class_name)
env_cfg = CfgClass()
env_cfg.scene.num_envs = args_cli.num_envs
env = gym.make(args_cli.task, cfg=env_cfg)

# Inspect Newton model
model = NewtonManager.get_model()
if model is None:
    print("ERROR: No Newton model found")
    sys.exit(1)

print("\n" + "=" * 80)
print("NEWTON MODEL INSPECTION")
print("=" * 80)

# Bodies
body_count = model.body_count
print(f"\nBodies: {body_count}")

# Geoms (collision shapes)
geom_count = model.geom_count
print(f"Geoms (collision shapes): {geom_count}")

# Get geom types, sizes, and body assignments
if geom_count > 0:
    geom_types = wp.to_torch(model.geom_type).cpu()
    geom_body = wp.to_torch(model.geom_body).cpu()

    # MuJoCo geom types:
    # 0=PLANE, 1=HFIELD, 2=SPHERE, 3=CAPSULE, 4=ELLIPSOID, 5=CYLINDER, 6=BOX, 7=MESH, 8=SDF
    type_names = {0: "PLANE", 1: "HFIELD", 2: "SPHERE", 3: "CAPSULE", 4: "ELLIPSOID",
                  5: "CYLINDER", 6: "BOX", 7: "MESH", 8: "SDF"}

    print(f"\nGeom type distribution:")
    for t in range(9):
        count = int((geom_types == t).sum().item())
        if count > 0:
            print(f"  {type_names.get(t, f'UNKNOWN({t})')}: {count}")

    # Print all geoms with their body assignments
    print(f"\nAll geoms (body, type):")
    for i in range(min(80, geom_count)):
        body_id = int(geom_body[i].item())
        geom_type = int(geom_types[i].item())
        type_name = type_names.get(geom_type, f"UNKNOWN({geom_type})")
        body_name = f"body_{body_id}"
        try:
            if hasattr(model, 'body_name') and body_id >= 0:
                body_name = model.body_name[body_id]
        except Exception:
            pass
        print(f"  geom[{i}]: body={body_id} ({body_name}), type={type_name}")
else:
    print("\nWARNING: No geoms found in the model!")

# Check contacts
print(f"\n--- Contact info ---")
for attr in ['ncon', 'njmax', 'nconmax']:
    if hasattr(model, attr):
        print(f"  {attr}: {getattr(model, attr)}")

# Step once and check
env.reset()

# Print cube info
print(f"\n--- Object (cube) info ---")
object_data = env.unwrapped.object.data
print(f"  root pos: {wp.to_torch(object_data.root_pos_w)[0].cpu().tolist()}")

# Print hand info
hand_data = env.unwrapped.hand.data
print(f"\n--- Hand info ---")
print(f"  root pos: {wp.to_torch(hand_data.root_pos_w)[0].cpu().tolist()}")
print(f"  root quat: {wp.to_torch(hand_data.root_quat_w)[0].cpu().tolist()}")
print(f"  joint count: {hand_data.joint_pos.shape[-1]}")
joint_pos = wp.to_torch(hand_data.joint_pos)[0].cpu()
print(f"  joint pos: {[round(x, 3) for x in joint_pos.tolist()]}")

# Step a few times and track cube Z
print(f"\n--- Cube Z tracking (15 zero-action steps) ---")
zero_action = torch.zeros(args_cli.num_envs, env.unwrapped.cfg.action_space, device=env.unwrapped.device)
for step_i in range(15):
    env.step(zero_action)
    obj_pos = wp.to_torch(env.unwrapped.object.data.root_pos_w)[0].cpu()
    obj_vel = wp.to_torch(env.unwrapped.object.data.root_vel_w)[0].cpu()
    hand_pos_z = wp.to_torch(env.unwrapped.hand.data.root_pos_w)[0, 2].cpu().item()
    ft_pos = wp.to_torch(env.unwrapped.hand.data.body_pos_w)[0].cpu()
    ft_z_vals = ft_pos[:, 2]
    print(
        f"  step {step_i+1}: cube_z={obj_pos[2].item():.4f}, "
        f"cube_vz={obj_vel[2].item():.4f}, "
        f"hand_base_z={hand_pos_z:.4f}, "
        f"fingertip_z=[{ft_z_vals.min():.4f}, {ft_z_vals.max():.4f}]"
    )

print("\n" + "=" * 80)
env.close()
simulation_app.close()
