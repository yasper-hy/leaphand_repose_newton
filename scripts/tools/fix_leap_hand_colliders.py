#!/usr/bin/env python3
"""Fix Leap Hand USD collision geometry for Newton compatibility.

Problem:
  The Leap Hand physics USD uses internal references from each body's
  /collisions Xform to /colliders/<body> (defined in the base sublayer).
  These references don't compose through in the main stage, leaving Newton
  with empty Xform prims that it can't recognize as collision geometry.

Fix:
  Read the collision Cube transforms from the physics sublayer (where
  composition works), compute a single AABB bounding box per body link,
  then write one inline Cube per body into the physics USD with
  PhysicsCollisionAPI applied.  This keeps collision shape count at 17.

Usage:
  python fix_leap_hand_colliders.py [--dry-run]
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


LEAP_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "source", "isaaclab_assets", "data", "Robots", "LeapHand",
    )
)
PHYSICS_USD = os.path.join(LEAP_DIR, "configuration", "leap_hand_right_physics.usd")

BODIES = [
    "base",
    "mcp_joint", "pip", "dip", "fingertip",
    "thumb_temp_base", "thumb_pip", "thumb_dip", "thumb_fingertip",
    "mcp_joint_2", "pip_2", "dip_2", "fingertip_2",
    "mcp_joint_3", "pip_3", "dip_3", "fingertip_3",
]


def _compute_world_corners(
    translate: Gf.Vec3d,
    orient_quat: Gf.Quatd,
    scale: Gf.Vec3d,
    cube_size: float,
) -> list[Gf.Vec3d]:
    """Compute 8 world-space corners of a cube with TRS xform ops."""
    hs = cube_size / 2.0
    # Local corners of a cube of size=cube_size
    local_corners = [
        Gf.Vec3d(s0 * hs, s1 * hs, s2 * hs)
        for s0 in (-1, 1) for s1 in (-1, 1) for s2 in (-1, 1)
    ]

    rot = Gf.Rotation(orient_quat)
    rot_mat = Gf.Matrix3d(rot)

    world_corners = []
    for c in local_corners:
        # Apply scale, then rotation, then translation
        scaled = Gf.Vec3d(c[0] * scale[0], c[1] * scale[1], c[2] * scale[2])
        rotated = rot_mat * scaled
        translated = rotated + translate
        world_corners.append(translated)
    return world_corners


def _compute_bbox_for_body(stage: Usd.Stage, body_name: str) -> dict | None:
    """Compute AABB from all collision shapes of a body."""
    root = stage.GetPrimAtPath(f"/colliders/{body_name}")
    if not root or not root.IsValid():
        return None

    all_corners = []
    for prim in Usd.PrimRange(root):
        if prim == root:
            continue
        tname = prim.GetTypeName()

        if tname == "Cube":
            cube_size = UsdGeom.Cube(prim).GetSizeAttr().Get() or 2.0
            # Get parent Xform ops
            parent = prim.GetParent()
            xformable = UsdGeom.Xformable(parent)
            ops = xformable.GetOrderedXformOps()

            translate = Gf.Vec3d(0, 0, 0)
            orient = Gf.Quatd(1, 0, 0, 0)
            scale = Gf.Vec3d(1, 1, 1)

            for op in ops:
                name = op.GetName()
                if "translate" in name:
                    v = op.Get()
                    translate = Gf.Vec3d(float(v[0]), float(v[1]), float(v[2]))
                elif "orient" in name:
                    v = op.Get()
                    # USD may return quaternion either as tuple-like or Gf.Quat* object.
                    if isinstance(v, (Gf.Quatd, Gf.Quatf, Gf.Quath)):
                        imag = v.GetImaginary()
                        orient = Gf.Quatd(float(v.GetReal()), float(imag[0]), float(imag[1]), float(imag[2]))
                    else:
                        orient = Gf.Quatd(float(v[0]), float(v[1]), float(v[2]), float(v[3]))
                elif "scale" in name:
                    v = op.Get()
                    scale = Gf.Vec3d(float(v[0]), float(v[1]), float(v[2]))

            corners = _compute_world_corners(translate, orient, scale, cube_size)
            all_corners.extend(corners)

        elif tname == "Mesh" or prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            pts = mesh.GetPointsAttr().Get()
            parent = prim.GetParent()
            xformable = UsdGeom.Xformable(parent)
            local_to_world = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            if pts:
                for pt in pts:
                    p = local_to_world.Transform(Gf.Vec3d(float(pt[0]), float(pt[1]), float(pt[2])))
                    all_corners.append(p)

    if not all_corners:
        return None

    mn = Gf.Vec3d(
        min(c[0] for c in all_corners),
        min(c[1] for c in all_corners),
        min(c[2] for c in all_corners),
    )
    mx = Gf.Vec3d(
        max(c[0] for c in all_corners),
        max(c[1] for c in all_corners),
        max(c[2] for c in all_corners),
    )

    center = (mn + mx) * 0.5
    half_extents = (mx - mn) * 0.5

    return {"center": center, "half_extents": half_extents}


def _write_simplified_colliders(layer: Sdf.Layer, bbox_data: dict) -> int:
    """Write one Cube per body link, removing broken internal references."""
    total = 0
    for body_name, bbox in bbox_data.items():
        if bbox is None:
            print(f"  [SKIP] {body_name}")
            continue

        collision_path = Sdf.Path(f"/leap_right/{body_name}/collisions")
        collision_spec = layer.GetPrimAtPath(collision_path)
        if not collision_spec:
            print(f"  [WARN] {collision_path} not found")
            continue

        # Remove broken internal reference
        if collision_spec.referenceList.prependedItems:
            collision_spec.referenceList.prependedItems.clear()

        # Ensure CollisionAPI is not applied to the Xform itself.
        # Newton expects collision schemas on concrete gprims (e.g. Cube), not Xform containers.
        collision_spec.SetInfo("apiSchemas", Sdf.TokenListOp.CreateExplicit([]))

        center = bbox["center"]
        he = bbox["half_extents"]

        # Create Cube directly under collisions — Newton recognizes UsdGeom.Cube
        # with PhysicsCollisionAPI
        cube_path = collision_path.AppendChild("collision_box")
        existing_cube = layer.GetPrimAtPath(cube_path)
        if existing_cube:
            # Keep idempotent: if simplified cube already exists, don't recreate attrs.
            existing_cube.SetInfo("apiSchemas", Sdf.TokenListOp.CreateExplicit(["PhysicsCollisionAPI"]))
            total += 1
            print(f"  ✓ {body_name}: reuse existing collision_box")
            continue

        cube_spec = Sdf.CreatePrimInLayer(layer, cube_path)
        cube_spec.typeName = "Cube"
        cube_spec.specifier = Sdf.SpecifierDef

        # Cube size=1.0 means half-extent of 0.5 in each axis
        # Use translate + scale to position and size it
        size_attr = Sdf.AttributeSpec(cube_spec, "size", Sdf.ValueTypeNames.Double)
        size_attr.default = 1.0

        # Translate to center
        trans_attr = Sdf.AttributeSpec(
            cube_spec, "xformOp:translate", Sdf.ValueTypeNames.Double3
        )
        trans_attr.default = Gf.Vec3d(center[0], center[1], center[2])

        # Scale by 2*half_extents (since size=1.0 means ±0.5)
        scale_attr = Sdf.AttributeSpec(
            cube_spec, "xformOp:scale", Sdf.ValueTypeNames.Double3
        )
        scale_attr.default = Gf.Vec3d(he[0] * 2.0, he[1] * 2.0, he[2] * 2.0)

        ops_attr = Sdf.AttributeSpec(
            cube_spec, "xformOpOrder", Sdf.ValueTypeNames.TokenArray
        )
        ops_attr.default = ["xformOp:translate", "xformOp:scale"]

        # Apply PhysicsCollisionAPI
        cube_spec.SetInfo("apiSchemas", Sdf.TokenListOp.CreateExplicit(
            ["PhysicsCollisionAPI"]
        ))

        total += 1
        print(f"  ✓ {body_name}: center=({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}), "
              f"he=({he[0]:.4f}, {he[1]:.4f}, {he[2]:.4f})")

    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"Physics USD: {PHYSICS_USD}")

    # Read collision data
    print("\n--- Reading collision data ---")
    physics_stage = Usd.Stage.Open(PHYSICS_USD)
    bbox_data: dict = {}
    for body_name in BODIES:
        bbox = _compute_bbox_for_body(physics_stage, body_name)
        if bbox:
            he = bbox["half_extents"]
            print(f"  {body_name}: half_extents=({he[0]:.4f}, {he[1]:.4f}, {he[2]:.4f})")
        else:
            print(f"  {body_name}: no data")
        bbox_data[body_name] = bbox
    del physics_stage

    n_valid = sum(1 for v in bbox_data.values() if v is not None)
    print(f"\nBodies with data: {n_valid}/{len(BODIES)}")

    if args.dry_run:
        print("\n[DRY RUN] Would modify physics USD.")
        return

    # Backup
    backup = PHYSICS_USD + ".bak"
    if not os.path.exists(backup):
        shutil.copy2(PHYSICS_USD, backup)
        print(f"\nBacked up: {backup}")

    # Write
    print("\n--- Writing simplified colliders ---")
    layer = Sdf.Layer.FindOrOpen(PHYSICS_USD)
    written = _write_simplified_colliders(layer, bbox_data)
    layer.Save()
    print(f"\nWrote {written} collision shapes")

    # Verify
    print("\n--- Verifying ---")
    main_stage = Usd.Stage.Open(os.path.join(LEAP_DIR, "leap_hand_right.usd"))
    ok = True
    for body_name in BODIES:
        col = main_stage.GetPrimAtPath(f"/leap_right/{body_name}/collisions")
        cubes = sum(1 for p in Usd.PrimRange(col) if p.GetTypeName() == "Cube")
        has_api = any(
            p.HasAPI(UsdPhysics.CollisionAPI)
            for p in Usd.PrimRange(col) if p.GetTypeName() == "Cube"
        )
        s = "✓" if cubes > 0 and has_api else "✗"
        if not (cubes > 0 and has_api):
            ok = False
        print(f"  {s} {body_name}: {cubes} cube(s), CollisionAPI={has_api}")

    print(f"\n{'✓ All bodies OK!' if ok else '✗ Some bodies failed'}")


if __name__ == "__main__":
    main()
