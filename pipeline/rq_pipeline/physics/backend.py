"""What a loaded model reports about itself, for fail-loudly gates.

Until 2026-08-26 this module also declared a `PhysicsBackend` Protocol
with one implementation ever; docs/32 step 5 removed it — the gymnasium
env (rq_pipeline.envs) is the backend-agnostic seam now, and a second
engine would arrive as another env, not another Protocol.
"""

from __future__ import annotations


class ModelCounts:
    """What the loaded model actually contains, for fail-loudly gates."""

    __slots__ = ("actuators", "cameras", "geoms", "sensors")

    def __init__(
        self, actuators: int, sensors: int, geoms: int, cameras: int = 0
    ) -> None:
        self.actuators = actuators
        self.sensors = sensors
        self.geoms = geoms
        # Cameras are counted because a vision policy on a camera-less
        # model fails silently (black frames score 0% with no error);
        # the bundle contract pins how many the rig carries.
        self.cameras = cameras


# platform.machine() spells the same silicon differently per OS; the stamp
# must not.
_ARCH_ALIASES = {"AMD64": "x86_64", "aarch64": "arm64"}


def instrument_stamp(name: str, version: str, *qualifiers: str) -> str:
    """The instrument's name: engine and version, any qualifiers (a
    device, a companion library), and ALWAYS the CPU architecture.

    Measured 2026-08-27 (docs/e2e-research/47 §7.1): the same MuJoCo
    3.11.0, scene, expert and trial gives the PGS solver a different
    referee verdict on arm64 (fail) and x86_64 (pass), and every
    penetration figure differs — a version alone does not name the
    instrument a certificate was produced on.
    """
    import platform  # noqa: PLC0415

    machine = platform.machine()
    return "+".join(
        [f"{name}-{version}", *qualifiers, _ARCH_ALIASES.get(machine, machine)]
    )
