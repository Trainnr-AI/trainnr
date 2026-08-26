"""What every engine shares: the census a loaded model reports (the
fail-loudly gate), the rollout contract, the FULLPHYSICS state-row
layout, and the one rule for naming an instrument.

The harness declares what it asks of an engine (`evaluate/harness.py::
Engine`, consumer-side); the two engines here — CPU MuJoCo, the
metrology instrument, and MJX-Warp, the throughput instrument — meet
it over the same compiled model. Nothing in this module imports an
engine.
"""

from __future__ import annotations

from typing import Any


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


ROLLOUT_STATE_RANK = 2  # (nbatch, nstate)
ROLLOUT_CONTROL_RANK = 3  # (nbatch, nstep, nu)


def check_rollout_shapes(
    initial_states: Any, controls: Any, *, state_width: int | None = None
) -> tuple[Any, Any]:
    """The rollout contract every engine shares — (nbatch, nstate) x
    (nbatch, nstep, nu) -> (nbatch, nstep, nstate) — checked once, here.
    Returns the two arrays as float; refuses the wrong rank, a batch
    mismatch and, when `state_width` is given, rows that are not that
    wide (FULLPHYSICS). Both backends call this so a refusal reads the
    same whichever instrument produced it."""
    import numpy as np  # noqa: PLC0415

    initial = np.asarray(initial_states, dtype=float)
    control = np.asarray(controls, dtype=float)
    if initial.ndim != ROLLOUT_STATE_RANK:
        raise ValueError(
            f"initial_states must be (nbatch, nstate), got {initial.shape}"
        )
    if control.ndim != ROLLOUT_CONTROL_RANK or control.shape[0] != initial.shape[0]:
        raise ValueError(
            "controls must be (nbatch, nstep, nu) with the same nbatch "
            f"as initial_states, got {control.shape}"
        )
    if state_width is not None and initial.shape[1] != state_width:
        raise ValueError(
            f"initial_states rows must be {state_width} wide (FULLPHYSICS), "
            f"got {initial.shape[1]}"
        )
    return initial, control


class FullPhysicsLayout:
    """MuJoCo's FULLPHYSICS state row, `[time, qpos, qvel, act]`, as
    slices of one compiled model — the one place the layout is spelled
    in library code (task modules that slice by offset are on docs/32
    §10's list to adopt it)."""

    TIME = 0

    def __init__(self, model: Any) -> None:
        self.nq, self.nv, self.na = int(model.nq), int(model.nv), int(model.na)

    @property
    def qpos(self) -> slice:
        return slice(1, 1 + self.nq)

    @property
    def qvel(self) -> slice:
        return slice(1 + self.nq, 1 + self.nq + self.nv)

    @property
    def act(self) -> slice:
        return slice(1 + self.nq + self.nv, self.width)

    @property
    def width(self) -> int:
        return 1 + self.nq + self.nv + self.na


LOAD_DOORS = ("load_spec", "load_model", "load_mjcf_string")


def no_model_message() -> str:
    return f"no model loaded — call one of {', '.join(LOAD_DOORS)} first"
