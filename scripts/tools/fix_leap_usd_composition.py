#!/usr/bin/env python3
"""Fix Leap Hand collider composition by flattening source colliders into physics USD.

This script restores reliable collider composition for Newton by:
1) reading original fine-grained colliders from leap_hand_right_base.usd under /colliders/<body>
2) clearing fragile internal references on /leap_right/<body>/collisions in leap_hand_right_physics.usd
3) deep-copying the original collider prim subtrees into each collisions scope
4) applying PhysicsCollisionAPI on every copied geometric prim (Cube/Mesh/etc.)

Result: physics layer contains explicit per-body collider geometry and does not depend on
internal reference composition at runtime.
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from collections import defaultdict

from pxr import Sdf, Usd, UsdGeom, UsdPhysics


LEAP_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "source",
        "isaaclab_assets",
        "data",
        "Robots",
        "LeapHand",
    )
)
BASE_USD = os.path.join(LEAP_DIR, "configuration", "leap_hand_right_base.usd")
PHYSICS_USD = os.path.join(LEAP_DIR, "configuration", "leap_hand_right_physics.usd")
PHYSICS_BAK_USD = os.path.join(LEAP_DIR, "configuration", "leap_hand_right_physics.usd.bak")

BODIES = [
    "base",
    "mcp_joint",
    "pip",
    "dip",
    "fingertip",
    "thumb_temp_base",
    "thumb_pip",
    "thumb_dip",
    "thumb_fingertip",
    "mcp_joint_2",
    "pip_2",
    "dip_2",
    "fingertip_2",
    "mcp_joint_3",
    "pip_3",
    "dip_3",
    "fingertip_3",
]


def _ensure_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(path)


def _restore_from_bak(physics_path: str, bak_path: str) -> None:
    _ensure_exists(bak_path)
    shutil.copy2(bak_path, physics_path)
    print(f"[ROLLBACK] Restored physics USD from backup:\n  src={bak_path}\n  dst={physics_path}")


def _make_pre_fix_backup(physics_path: str) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = f"{physics_path}.pre_fix_{ts}.bak"
    shutil.copy2(physics_path, out)
    return out


def _copy_body_colliders(
    base_stage: Usd.Stage,
    physics_stage: Usd.Stage,
    base_layer: Sdf.Layer,
    physics_layer: Sdf.Layer,
    body_name: str,
) -> tuple[int, int]:
    """Copy one body's original collider subtree into physics collisions scope.

    Returns:
        copied_subtrees, copied_geoms
    """
    src_body_path = Sdf.Path(f"/colliders/{body_name}")
    dst_col_path = Sdf.Path(f"/leap_right/{body_name}/collisions")

    src_body = base_stage.GetPrimAtPath(src_body_path)
    dst_col = physics_stage.GetPrimAtPath(dst_col_path)
    if not src_body.IsValid():
        raise RuntimeError(f"Missing source body path: {src_body_path}")
    if not dst_col.IsValid():
        raise RuntimeError(f"Missing destination collisions path: {dst_col_path}")

    # Remove fragile internal references that can fail during composed-stage resolution.
    dst_col.GetReferences().ClearReferences()

    # Remove previous children to keep output deterministic.
    for child in list(dst_col.GetChildren()):
        physics_stage.RemovePrim(child.GetPath())

    copied_subtrees = 0
    copied_geoms = 0

    # Copy all direct children (typically Xform containers like mesh_0, mesh_1, ...),
    # preserving each subtree's local transforms and geom attributes exactly.
    for src_child in src_body.GetChildren():
        src_path = src_child.GetPath()
        dst_path = dst_col_path.AppendChild(src_child.GetName())
        ok = Sdf.CopySpec(base_layer, src_path, physics_layer, dst_path)
        if not ok:
            raise RuntimeError(f"Sdf.CopySpec failed: {src_path} -> {dst_path}")
        copied_subtrees += 1

    # Apply CollisionAPI directly on every geometric prim under /collisions subtree.
    dst_col = physics_stage.GetPrimAtPath(dst_col_path)
    for prim in Usd.PrimRange(dst_col):
        if prim == dst_col:
            continue
        if prim.IsA(UsdGeom.Gprim):
            UsdPhysics.CollisionAPI.Apply(prim)
            copied_geoms += 1

    return copied_subtrees, copied_geoms


def _verify(physics_path: str) -> dict:
    stage = Usd.Stage.Open(physics_path)
    total_geoms = 0
    total_collision_api = 0
    type_counts = defaultdict(int)
    per_body = {}

    for body_name in BODIES:
        col = stage.GetPrimAtPath(f"/leap_right/{body_name}/collisions")
        geoms = 0
        with_api = 0
        if col.IsValid():
            for prim in Usd.PrimRange(col):
                if prim == col:
                    continue
                if prim.IsA(UsdGeom.Gprim):
                    geoms += 1
                    type_counts[prim.GetTypeName()] += 1
                    if prim.HasAPI(UsdPhysics.CollisionAPI):
                        with_api += 1
        per_body[body_name] = {"geoms": geoms, "collision_api": with_api}
        total_geoms += geoms
        total_collision_api += with_api

    return {
        "total_geoms": total_geoms,
        "total_collision_api": total_collision_api,
        "type_counts": dict(type_counts),
        "per_body": per_body,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore-from-bak", action="store_true", help="Restore physics USD from .bak before fixing.")
    parser.add_argument("--dry-run", action="store_true", help="Inspect only; do not write.")
    args = parser.parse_args()

    _ensure_exists(BASE_USD)
    _ensure_exists(PHYSICS_USD)
    _ensure_exists(PHYSICS_BAK_USD)

    print("[PATHS]")
    print(f"  BASE_USD:    {BASE_USD}")
    print(f"  PHYSICS_USD: {PHYSICS_USD}")
    print(f"  BAK_USD:     {PHYSICS_BAK_USD}")

    if args.restore_from_bak:
        _restore_from_bak(PHYSICS_USD, PHYSICS_BAK_USD)

    if args.dry_run:
        v = _verify(PHYSICS_USD)
        print("[DRY-RUN VERIFY]", v)
        return

    pre_fix_backup = _make_pre_fix_backup(PHYSICS_USD)
    print(f"[BACKUP] Saved pre-fix backup: {pre_fix_backup}")

    base_stage = Usd.Stage.Open(BASE_USD)
    physics_stage = Usd.Stage.Open(PHYSICS_USD)
    if base_stage is None or physics_stage is None:
        raise RuntimeError("Failed to open one or more USD stages.")

    base_layer = base_stage.GetRootLayer()
    physics_layer = physics_stage.GetRootLayer()

    total_subtrees = 0
    total_geoms = 0

    print("\n[FIX] Flattening original colliders into physics layer...")
    for body_name in BODIES:
        copied_subtrees, copied_geoms = _copy_body_colliders(
            base_stage, physics_stage, base_layer, physics_layer, body_name
        )
        total_subtrees += copied_subtrees
        total_geoms += copied_geoms
        print(f"  {body_name:<16} subtrees={copied_subtrees:>2}  geoms={copied_geoms:>2}")

    physics_stage.GetRootLayer().Save()
    print(f"\n[SAVE] Updated: {PHYSICS_USD}")
    print(f"[SUMMARY] copied_subtrees={total_subtrees}, copied_geoms={total_geoms}")

    print("\n[VERIFY]")
    v = _verify(PHYSICS_USD)
    for body_name in BODIES:
        b = v["per_body"][body_name]
        print(f"  {body_name:<16} geoms={b['geoms']:>2}  collision_api={b['collision_api']:>2}")
    print(f"  TOTAL geoms={v['total_geoms']}, with_collision_api={v['total_collision_api']}")
    print(f"  TYPE distribution={v['type_counts']}")


if __name__ == "__main__":
    main()
