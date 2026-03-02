# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

import os

from isaaclab_assets.robots.leap_hand import (
    LEAP_HAND_ACTUATOR_PROFILES,
    LEAP_HAND_CFG,
    LEAP_HAND_RESET_POSE_PROFILES,
    make_leap_hand_cfg,
)

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

@configclass
class LeapHandEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0
    action_space = 16
    observation_space = 124
    state_space = 0
    asymmetric_obs = False
    obs_type = "full"
    viewer: ViewerCfg = ViewerCfg(
        eye=(0.42, 0.10, 0.72),
        lookat=(0.00, -0.10, 0.56),
        origin_type="world",
        env_index=0,
    )

    solver_cfg = MJWarpSolverCfg(
        solver="newton",
        integrator="implicitfast",
        # LeapHand contact density is higher than Allegro under current colliders.
        # Keep solver buffers explicit to avoid narrowphase/nefc overflow cascades.
        njmax=2000,
        nconmax=2000,
        impratio=10.0,
        cone="elliptic",
        update_data_interval=2,
        iterations=100,
        ls_iterations=15,
        ls_parallel=True,
    )

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.5,
            dynamic_friction=1.5,
            restitution=0.0,
            friction_combine_mode="max",
            restitution_combine_mode="min",
        ),
        physics=NewtonCfg(
            solver_cfg=solver_cfg,
            num_substeps=2,
            debug_mode=False,
            use_cuda_graph=False,
        ),
    )

    # robot
    reset_pose_profile: str = "cradle_v1"
    actuator_profile: str = "default"
    object_init_pos_y: float = -0.10

    robot_cfg: ArticulationCfg = LEAP_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    actuated_joint_names = [
        "a_0",
        "a_1",
        "a_2",
        "a_3",
        "a_4",
        "a_5",
        "a_6",
        "a_7",
        "a_8",
        "a_9",
        "a_10",
        "a_11",
        "a_12",
        "a_13",
        "a_14",
        "a_15",
    ]
    fingertip_body_names = [
        "fingertip",
        "thumb_fingertip",
        "fingertip_2",
        "fingertip_3",
    ]

    # in-hand object
    object_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            mass_props=sim_utils.MassPropertiesCfg(density=400.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=0.2,
            ),
            scale=(1.2, 1.2, 1.2),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, -0.10, 0.56), rot=(0.0, 0.0, 0.0, 1.0)),
        actuators={},
        articulation_root_prim_path="",
    )
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(1.2, 1.2, 1.2),
            )
        },
    )
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8192, env_spacing=0.75, replicate_physics=True, clone_in_fabric=True
    )
    # reset
    reset_position_noise = 0.01
    reset_object_rot_noise_scale_x = 1.0
    reset_object_rot_noise_scale_y = 1.0
    reset_dof_pos_noise = 0.05
    reset_dof_vel_noise = 0.0
    # reward scales
    dist_reward_scale = -10.0
    rot_reward_scale = 1.0
    rot_eps = 0.1
    action_penalty_scale = -0.0002
    reach_goal_bonus = 250
    fall_penalty = 0
    fall_dist = 0.30
    vel_obs_scale = 0.2
    success_tolerance = 0.2
    max_consecutive_success = 0
    av_factor = 0.1
    act_moving_average = 1.0
    force_torque_obs_scale = 10.0

    def __post_init__(self):
        # Optional env-var overrides for experiment automation.
        self.reset_pose_profile = os.getenv("LEAP_RESET_POSE_PROFILE", self.reset_pose_profile)
        self.actuator_profile = os.getenv("LEAP_ACTUATOR_PROFILE", self.actuator_profile)
        self.object_init_pos_y = float(os.getenv("LEAP_OBJECT_INIT_POS_Y", str(self.object_init_pos_y)))
        if os.getenv("LEAP_FALL_DIST") is not None:
            self.fall_dist = float(os.getenv("LEAP_FALL_DIST"))

        if self.reset_pose_profile not in LEAP_HAND_RESET_POSE_PROFILES:
            available = ", ".join(sorted(LEAP_HAND_RESET_POSE_PROFILES.keys()))
            raise ValueError(f"Unknown reset_pose_profile '{self.reset_pose_profile}'. Available: {available}")
        if self.actuator_profile not in LEAP_HAND_ACTUATOR_PROFILES:
            available = ", ".join(sorted(LEAP_HAND_ACTUATOR_PROFILES.keys()))
            raise ValueError(f"Unknown actuator_profile '{self.actuator_profile}'. Available: {available}")

        self.robot_cfg = make_leap_hand_cfg(
            reset_pose_profile=self.reset_pose_profile,
            actuator_profile=self.actuator_profile,
        ).replace(prim_path="/World/envs/env_.*/Robot")

        self.object_cfg.init_state.pos = (0.0, self.object_init_pos_y, 0.56)
