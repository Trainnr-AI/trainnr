"""The gate's runtime for Unitree's own stack: their simulator publishes
the robot's state on DDS, their controller drives the robot from our
ONNX and a virtual gamepad, and this runtime only waits for state and
moves the sticks. The trial loop, the judge and the record are the
gate's, unchanged (`gate(runtime="dds")`).

Their controller's state machine starts passive; `reset` walks it to
standing (LT + up) and into RL (RT + A) through the pad, as a person
would. Velocity comes from their SportModeState message (the simulator
fills it from MuJoCo's frame sensors, world frame) rotated into the
body by the IMU quaternion; fell-over is the rule our MuJoCo runtime
applies (`runtime.fell_over`). The pad's sticks reach 1.0, so the
commands this runtime can ask for are bounded at 1 m/s and 1 rad/s
whatever the manifest's ranges say (`command_limit`). Linux only: the
registry refuses it by name elsewhere.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol

import numpy as np

from rq_pipeline.deploy.gamepad import COMMAND_LIMIT, VirtualPad, sticks_for_command
from rq_pipeline.deploy.manifest import Manifest
from rq_pipeline.deploy.runtime import fell_over, rotate_inverse
from rq_pipeline.deploy.runtimes import RUNTIMES, require_platform

TOPIC_STATE = "rt/sportmodestate"
TOPIC_LOW = "rt/lowstate"
NETWORK = "lo"  # their simulator and controller meet on loopback
DOMAIN_ID = 0
FSM_SETTLE_S = 2.0  # the fixed stand takes about this long to reach
CHORD_ATTEMPTS = 2
STATE_TIMEOUT_MS = 1000
# The two chords their state machine takes from passive to RL (their
# `config.yaml` FSM): fixed stand, then velocity mode.
FSM_CHORDS = (("LT", "up"), ("RT", "A"))
STICKS_CENTERED = {"lx": 0.0, "ly": 0.0, "rx": 0.0, "ry": 0.0}
Clock = Callable[[], float]
Sleep = Callable[[float], None]


class Bus(Protocol):
    """What the runtime reads: the latest base state their stack publishes."""

    def latest(self, timeout_ms: int) -> tuple[np.ndarray, np.ndarray] | None:
        """(quaternion wxyz, world-frame velocity) or None on timeout."""


class Pad(Protocol):
    """What the runtime moves: sticks and the chords of a gamepad."""

    def sticks(self, **positions: float) -> None: ...
    def chord(self, trigger: str, button: str) -> None: ...
    def close(self) -> None: ...


class SdkBus:
    """Unitree's Python SDK over CycloneDDS: one subscriber, the latest message."""

    def __init__(self, network: str = NETWORK, domain_id: int = DOMAIN_ID) -> None:
        require_platform(RUNTIMES["dds"])
        from unitree_sdk2py.core.channel import (  # noqa: PLC0415
            ChannelFactoryInitialize,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import (  # noqa: PLC0415
            SportModeState_,
        )

        ChannelFactoryInitialize(domain_id, network)
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

    command_limit: float | None = COMMAND_LIMIT

    def __init__(
        self,
        manifest: Manifest,
        *,
        bus: Bus,
        pad: Pad,
        sleep: Sleep = time.sleep,
        clock: Clock = time.monotonic,
    ) -> None:
        self.manifest = manifest
        self.bus = bus
        self.pad = pad
        self._sleep = sleep
        self._clock = clock
        self.command = np.zeros(3, dtype=np.float32)
        self._quat = np.array([1.0, 0.0, 0.0, 0.0])
        self._velocity_w = np.zeros(3)
        self.step_dt = manifest.control.step_dt
        self.ticks = 0
        self._next_tick: float | None = None

    @property
    def instrument(self) -> str:
        """Their simulator and the controller the manifest names, over DDS."""
        stack = self.manifest.unitree
        controller = stack.controller if stack else "unrecorded"
        return f"unitree_mujoco + {controller} over DDS (unitree_rl_mjlab)"

    def reset(self, *, keyframe: int = 0) -> None:
        """Sticks to zero, their state machine to RL: passive -> fixed
        stand (LT + up), settle, -> velocity (RT + A), settle."""
        self.pad.sticks(**STICKS_CENTERED)
        # Each chord twice: a repeat is a no-op in the state it leads to,
        # and a single press was missed once (2026-09-11).
        for trigger, button in FSM_CHORDS:
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
        now = self._clock()
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
        return rotate_inverse(self._quat, self._velocity_w)

    def fell_over(self) -> bool:
        return fell_over(self._quat, self.manifest.termination.fell_over_deg)

    def close(self) -> None:
        self.pad.sticks(**STICKS_CENTERED)


def open_dds_runtime(manifest: Manifest, *, assets_dir: object = None) -> DdsRuntime:
    """The live one: the SDK bus on loopback and a virtual pad."""
    return DdsRuntime(manifest, bus=SdkBus(), pad=VirtualPad())
