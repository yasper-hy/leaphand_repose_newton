# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility view aliases used by simulation data providers.

This module is intentionally lightweight so importing ``isaaclab.sim`` before
Omniverse app initialization does not hard-fail in environments that only use
Newton/MJWarp backends.
"""

from __future__ import annotations

__all__ = [
    "XformPrimView",
]


class _UnavailableView:
    """Placeholder class used when the Omniverse view API is unavailable."""

    def __init__(self, *args, **kwargs):
        raise ImportError(
            "XformPrimView is unavailable in this runtime. "
            "Instantiate SimulationApp before importing view classes."
        )


try:
    # Import lazily available alias used in PhysX scene providers.
    from isaacsim.core.prims import XFormPrimView as XformPrimView
except Exception:  # pragma: no cover - runtime dependent import
    XformPrimView = _UnavailableView

