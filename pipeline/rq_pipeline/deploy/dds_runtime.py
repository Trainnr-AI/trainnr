"""The gate's runtime for Unitree's own stack: their simulator publishes
the robot's state on DDS, their controller drives the robot from our
ONNX and a virtual gamepad, and this runtime only waits for state and
moves the sticks. The trial loop, the judge and the record are the
gate's, unchanged (`gate(open=open_dds_runtime)`).

Their controller's state machine starts passive; `reset` walks it to
standing (LT + up) and into RL (RT + A) through the pad, as a person
would. Velocity comes from their SportModeState message (the simulator
fills it from MuJoCo's frame sensors, world frame) rotated into the
body by the IMU quaternion; fell-over is the same rule our MuJoCo
runtime applies. The pad's sticks reach 1.0, so the commands this
runtime can ask for are bounded at 1 m/s and 1 rad/s whatever the
manifest's ranges say (`command_limit`).
"""

from __future__ import annotations

import time
from typing import Any, Protocol

import numpy as np

from rq_pipeline.deploy.gamepad import COMMAND_LIMIT, VirtualPad, sticks_for_command
from rq_pipeline.deploy.manifest import Manifest
from rq_pipeline.deploy.runtime import _rotate_inverse

TOPIC_STATE = "rt/sportmodestate"
TOPIC_LOW = "rt/lowstate"
NETWORK = "lo"
FSM_SETTLE_S = 2.0  # the fixed stand takes about this long to reach
CHORD_ATTEMPTS = 2
STATE_TIMEOUT_MS = 1000


class Bus(Protocol):
    """What the runtime reads: the latest base state their stack publishes."""

    def latest(self, timeout_ms: int) -> tuple[np.ndarray, np.ndarray] | None:
        """(quaternion wxyz, world-frame velocity) or None on timeout."""


class SdkBus:
    """Unitree's Python SDK over CycloneDDS: one subscriber, the latest message."""

    def __init__(self, network: str = NETWORK) -> None:
        from unitree_sdk2py.core.channel import (  # noqa: PLC0415
            ChannelFactoryInitialize,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import (  # noqa: PLC0415
            SportModeState_,
        )

        ChannelFactoryInitialize(0, network)
        self._sub = ChannelSubscriber(TOPIC_STATE, SportModeState_)
        self._sub.Init()

    def latest(self, timeout_ms: int) -> tuple[np.ndarray, np.ndarray] | None:
        msg = self._sub.Read(timeout_ms)
        if msg is None:
            return None
        quat = np.asarray(msg.imu_state.quaternion, dtype=np.float64)  # w x y z
        velocity = np.asarray(msg.velocity, dtype=np.float64)
        return quat, velocity


class DdsRuntime:
    """The seven calls the gate makes, answered by their stack."""

    command_limit = COMMAND_LIMIT
    instrument = "unitree_mujoco + go2_ctrl over DDS (unitree_rl_mjlab)"
    record_file = "gate-dds.json"  # beside the MuJoCo gate's gate.json, never over it

    def __init__(
        self, manifest: Manifest, *, bus: Bus, pad: Any, sleep: Any = time.sleep
    ) -> None:
        self.manifest = manifest
        self.bus = bus
        self.pad = pad
        self._sleep = sleep
        self.command = np.zeros(3, dtype=np.float32)
        self._quat = np.array([1.0, 0.0, 0.0, 0.0])
        self._velocity_w = np.zeros(3)
        self.step_dt = 1.0 / float(manifest.control["control_hz"])
        self.ticks = 0
        self._next_tick: float | None = None

    def reset(self, *, keyframe: int = 0) -> None:
        """Sticks to zero, their state machine to RL: passive -> fixed
        stand (LT + up), settle, -> velocity (RT + A), settle."""
        self.pad.sticks(lx=0.0, ly=0.0, rx=0.0, ry=0.0)
        # Each chord twice: a repeat is a no-op in the state it leads to,
        # and a single press was missed once (2026-09-11).
        for trigger, button in (("LT", "up"), ("RT", "A")):
            for _ in range(CHORD_ATTEMPTS):
                self.pad.chord(trigger, button)
                self._sleep(FSM_SETTLE_S)
        self.ticks = 0
        self._next_tick = None
        self._read()

    def _read(self) -> None:
        latest = self.bus.latest(STATE_TIMEOUT_MS)
        if latest is None:
            raise TimeoutError(
                f"no {TOPIC_STATE} within {STATE_TIMEOUT_MS} ms: is their simulator up?"
            )
        self._quat, self._velocity_w = latest

    def observe(self) -> np.ndarray:
        """One control period at their rate, then the latest state. Their
        controller observes for itself; the gate needs nothing here."""
        now = time.monotonic()
        if self._next_tick is None:
            self._next_tick = now
        self._next_tick += self.step_dt
        if self._next_tick > now:
            self._sleep(self._next_tick - now)
        self._read()
        return np.zeros(0, dtype=np.float32)

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros(0, dtype=np.float32)

    def apply(self, action: np.ndarray) -> None:
        """The held command as stick positions; their controller does the rest."""
        self.pad.sticks(**sticks_for_command(self.command))
        self.ticks += 1

    def base_velocity_b(self) -> np.ndarray:
        return _rotate_inverse(self._quat, self._velocity_w)

    def fell_over(self) -> bool:
        limit = self.manifest.raw.get("termination", {}).get("fell_over_deg")
        if limit is None:
            return False
        gravity_b = _rotate_inverse(self._quat, np.array([0.0, 0.0, -1.0]))
        cos = float(np.clip(-gravity_b[2], -1.0, 1.0))
        return np.degrees(np.arccos(cos)) > float(limit)

    def close(self) -> None:
        self.pad.sticks(lx=0.0, ly=0.0, rx=0.0, ry=0.0)


def open_dds_runtime(manifest: Manifest, *, assets_dir: Any = None) -> DdsRuntime:
    """The live one: the SDK bus on loopback and a virtual pad."""
    return DdsRuntime(manifest, bus=SdkBus(), pad=VirtualPad())
