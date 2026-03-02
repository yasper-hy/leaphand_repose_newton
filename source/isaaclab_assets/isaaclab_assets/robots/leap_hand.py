# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the LEAP Hand robot."""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

_LEAP_HAND_USD = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "Robots", "LeapHand", "leap_hand_right.usd")
)

LEAP_HAND_JOINT_LIMITS_RAD = {
    "a_0": (-1.0470006468, 1.0470006468),
    "a_1": (-0.3139989704, 2.2299941999),
    "a_2": (-0.5059992815, 1.8857014179),
    "a_3": (-0.3660002759, 2.0420009390),
    "a_4": (-1.0470006468, 1.0470006468),
    "a_5": (-0.3139989704, 2.2299941999),
    "a_6": (-0.5059992815, 1.8857014179),
    "a_7": (-0.3660002759, 2.0420009390),
    "a_8": (-1.0470006468, 1.0470006468),
    "a_9": (-0.3139989704, 2.2299941999),
    "a_10": (-0.5059992815, 1.8857014179),
    "a_11": (-0.3660002759, 2.0420009390),
    "a_12": (-0.3489990258, 2.0939995592),
    "a_13": (-0.4699997217, 2.4429878271),
    "a_14": (-1.1999991207, 1.8999996683),
    "a_15": (-1.3400001721, 1.8799985279),
}
"""Explicit LEAP Hand joint limits (radians), used when Newton fails to parse USD limits."""

LEAP_HAND_RESET_POSE_PROFILES = {
    # Baseline cradle pose used in current training runs.
    "cradle_v1": {
        "a_0": 0.0,
        "a_1": 0.8,
        "a_2": 0.8,
        "a_3": 0.8,
        "a_4": 0.0,
        "a_5": 0.8,
        "a_6": 0.8,
        "a_7": 0.8,
        "a_8": 0.0,
        "a_9": 0.8,
        "a_10": 0.8,
        "a_11": 0.8,
        "a_12": 0.8,
        "a_13": 1.0,
        "a_14": 0.4,
        "a_15": 0.2,
    },
    # Slightly more open pose to reduce immediate contact spikes.
    "cradle_open": {
        "a_0": 0.0,
        "a_1": 0.65,
        "a_2": 0.65,
        "a_3": 0.65,
        "a_4": 0.0,
        "a_5": 0.65,
        "a_6": 0.65,
        "a_7": 0.65,
        "a_8": 0.0,
        "a_9": 0.65,
        "a_10": 0.65,
        "a_11": 0.65,
        "a_12": 0.7,
        "a_13": 0.9,
        "a_14": 0.35,
        "a_15": 0.15,
    },
    # Pre-grasp cup pose: slightly more curled to cradle cube from the first physics step.
    "pregrasp_cup": {
        "a_0": 0.0,
        "a_1": 0.95,
        "a_2": 1.05,
        "a_3": 0.95,
        "a_4": 0.0,
        "a_5": 0.95,
        "a_6": 1.05,
        "a_7": 0.95,
        "a_8": 0.0,
        "a_9": 0.95,
        "a_10": 1.05,
        "a_11": 0.95,
        "a_12": 1.05,
        "a_13": 1.15,
        "a_14": 0.55,
        "a_15": 0.30,
    },
}

LEAP_HAND_ACTUATOR_PROFILES = {
    # Current baseline gains/limits.
    "default": {
        "stiffness": 3.0,
        "damping": 0.1,
        "effort_limit_sim": 0.5,
        "friction": 0.01,
    },
    # Capacity B: slightly stronger and better damped for contact-rich in-hand control.
    "capacity_b": {
        "stiffness": 5.0,
        "damping": 0.2,
        "effort_limit_sim": 0.8,
        "friction": 0.02,
    },
}

def make_leap_hand_cfg(
    reset_pose_profile: str = "cradle_v1",
    actuator_profile: str = "default",
) -> ArticulationCfg:
    """Build LEAP Hand articulation cfg with explicit profile selection."""
    if reset_pose_profile not in LEAP_HAND_RESET_POSE_PROFILES:
        available = ", ".join(sorted(LEAP_HAND_RESET_POSE_PROFILES.keys()))
        raise ValueError(f"Unknown LEAP reset pose profile '{reset_pose_profile}'. Available: {available}")
    if actuator_profile not in LEAP_HAND_ACTUATOR_PROFILES:
        available = ", ".join(sorted(LEAP_HAND_ACTUATOR_PROFILES.keys()))
        raise ValueError(f"Unknown LEAP actuator profile '{actuator_profile}'. Available: {available}")

    reset_joint_pos = LEAP_HAND_RESET_POSE_PROFILES[reset_pose_profile]
    actuator = LEAP_HAND_ACTUATOR_PROFILES[actuator_profile]

    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=_LEAP_HAND_USD,
            # Keep hand base dynamics explicit for Newton path (avoid relying on USD defaults).
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                angular_damping=0.01,
                max_linear_velocity=1000.0,
                max_angular_velocity=64 / 3.141592653589793 * 180.0,
                max_depenetration_velocity=1.0,
                max_contact_impulse=1e32,
                enable_gyroscopic_forces=False,
                retain_accelerations=False,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0005,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(0.5, 0.5, -0.5, 0.5),
            joint_pos=reset_joint_pos,
        ),
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                # Explicitly set gains in config (do not rely on USD defaults in Newton path).
                stiffness=actuator["stiffness"],
                damping=actuator["damping"],
                effort_limit_sim=actuator["effort_limit_sim"],
                friction=actuator["friction"],
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )


LEAP_HAND_CFG = make_leap_hand_cfg()
"""Configuration of LEAP Hand robot."""
