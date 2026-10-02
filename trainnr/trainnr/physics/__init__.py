"""Physics: CPU MuJoCo is the metrology instrument (`mujoco.sysid` is
CPU MuJoCo, MJCF stays the canonical robot format); MJX-Warp is the
throughput instrument (`MJXWarpBackend`, docs/e2e-research/49) — same
compiled mjModel, batched on the device, float32, admitted through the
acceptance gauntlet in tests/test_mjx_backend.py. Engines surface to
policies as gymnasium envs over the same tasks (trainnr.envs);
that seam keeps every stage independent of the engine underneath.
Engines register by name (`registry.py`, the `trainnr.engines`
entry-point group): `resolve("mujoco").build()`.
"""

from typing import TYPE_CHECKING, Any

from trainnr.physics.backend import ModelCounts
from trainnr.physics.mujoco_backend import MuJoCoBackend, Stepper
from trainnr.physics.registry import engines, resolve

if TYPE_CHECKING:
    # The real import stays lazy (__getattr__ below): the mjx extra is
    # optional. This one is for the checker, which cannot see __getattr__.
    from trainnr.physics.mjx_backend import MJXWarpBackend

__all__ = [
    "MJXWarpBackend",
    "ModelCounts",
    "MuJoCoBackend",
    "Stepper",
    "engines",
    "resolve",
]


def __getattr__(name: str) -> Any:
    # Lazy: the mjx extra is optional, and importing it eagerly would
    # break every sim-only environment.
    if name == "MJXWarpBackend":
        from trainnr.physics.mjx_backend import MJXWarpBackend  # noqa: PLC0415

        return MJXWarpBackend
    raise AttributeError(name)
