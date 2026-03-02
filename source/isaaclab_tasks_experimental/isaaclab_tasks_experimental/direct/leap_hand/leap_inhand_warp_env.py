# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab_assets.robots.leap_hand import LEAP_HAND_JOINT_LIMITS_RAD
from isaaclab_tasks_experimental.direct.inhand_manipulation.inhand_manipulation_warp_env import InHandManipulationWarpEnv

if TYPE_CHECKING:
    from .leap_hand_env_cfg import LeapHandEnvCfg


class LeapInHandManipulationWarpEnv(InHandManipulationWarpEnv):
    """Leap-specific wrapper around the Warp in-hand environment."""

    cfg: LeapHandEnvCfg

    def __init__(self, cfg: LeapHandEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        self._repair_joint_limits_from_leap_defaults()

    def _repair_joint_limits_from_leap_defaults(self) -> None:
        """Write explicit LEAP limits into simulation by joint name."""
        lower = wp.to_torch(self.hand.data.joint_pos_limits_lower).to(self.device).clone()
        upper = wp.to_torch(self.hand.data.joint_pos_limits_upper).to(self.device).clone()

        pre_invalid = (~torch.isfinite(lower)) | (~torch.isfinite(upper)) | ((upper - lower) <= 1.0e-6)
        pre_invalid_joint_ids = torch.where(pre_invalid.any(dim=0))[0].tolist()

        written_joint_names: list[str] = []
        missing_joint_names: list[str] = []
        for joint_name, (joint_lower, joint_upper) in LEAP_HAND_JOINT_LIMITS_RAD.items():
            if joint_name not in self.hand.joint_names:
                missing_joint_names.append(joint_name)
                continue
            joint_id = self.hand.joint_names.index(joint_name)
            lower[:, joint_id] = joint_lower
            upper[:, joint_id] = joint_upper
            written_joint_names.append(joint_name)

        # If there are joints that were invalid and not covered by fallback map, fail fast.
        unresolved_invalid_names: list[str] = []
        for joint_id in pre_invalid_joint_ids:
            joint_name = self.hand.joint_names[joint_id]
            if joint_name not in LEAP_HAND_JOINT_LIMITS_RAD:
                unresolved_invalid_names.append(joint_name)

        if missing_joint_names or unresolved_invalid_names:
            missing_all = sorted(set(missing_joint_names + unresolved_invalid_names))
            raise RuntimeError("Joint-limit fallback map is incomplete. Missing joints: " + ", ".join(missing_all))

        self.hand.write_joint_position_limit_to_sim(lower, upper)

        # Keep local references aligned with simulation limits used by action scaling kernels.
        self.hand_dof_lower_limits = self.hand.data.joint_pos_limits_lower
        self.hand_dof_upper_limits = self.hand.data.joint_pos_limits_upper

        post_lower = wp.to_torch(self.hand.data.joint_pos_limits_lower).to(self.device)
        post_upper = wp.to_torch(self.hand.data.joint_pos_limits_upper).to(self.device)
        post_width = post_upper - post_lower
        post_invalid = (~torch.isfinite(post_lower)) | (~torch.isfinite(post_upper)) | (post_width <= 1e-6)
        if torch.any(post_invalid):
            raise RuntimeError("LEAP joint-limit fallback applied but invalid limits still detected in simulation.")

        print(
            f"[LEAP] Applied explicit joint limits for {len(written_joint_names)} joints; "
            f"pre_invalid_joints={len(pre_invalid_joint_ids)}, "
            f"post_width_min={post_width.min().item():.4f}, post_width_max={post_width.max().item():.4f}"
        )
