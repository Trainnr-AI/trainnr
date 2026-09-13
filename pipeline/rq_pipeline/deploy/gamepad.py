"""A virtual Xbox gamepad through the kernel's uinput: what a person's
thumbs do to Unitree's simulator and controller, from Python.

Their simulator reads `/dev/input/js0` and packs it into the robot's
state as the wireless remote; their controller's state machine moves
on button chords ("LT + up", "RT + A") and its RL mode takes the
velocity command from the sticks (forward = left stick y, sideways =
minus left stick x, turn = minus right stick x, each in [-1, 1] and
clamped to the policy's ranges - deploy/include, `velocity_commands`).
The layout below is theirs (`simulate/src/physics_joystick.h`, xbox):
the joystick driver numbers axes and buttons in the order they are
registered, so the order here IS the mapping. Linux only (uinput): the
registry refuses it by name elsewhere.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from rq_pipeline.deploy.runtimes import RUNTIMES, require_platform

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Their xbox layout: js axis index -> evdev absolute axis, in order.
AXES = (
    "ABS_X",
    "ABS_Y",
    "ABS_Z",
    "ABS_RX",
    "ABS_RY",
    "ABS_RZ",
    "ABS_HAT0X",
    "ABS_HAT0Y",
)
STICK_AXIS = {"lx": "ABS_X", "ly": "ABS_Y", "rx": "ABS_RX", "ry": "ABS_RY"}
TRIGGER_AXIS = {"LT": "ABS_Z", "RT": "ABS_RZ"}
DPAD = {
    "up": ("ABS_HAT0Y", -1),
    "down": ("ABS_HAT0Y", 1),
    "left": ("ABS_HAT0X", -1),
    "right": ("ABS_HAT0X", 1),
}
# js button index -> evdev key, in order: A B X Y LB RB back start.
BUTTONS = (
    "BTN_A",
    "BTN_B",
    "BTN_X",
    "BTN_Y",
    "BTN_TL",
    "BTN_TR",
    "BTN_SELECT",
    "BTN_START",
)
# their names for those buttons (the state machine's "RT + A")
BUTTON_KEY = dict(
    zip(("A", "B", "X", "Y", "LB", "RB", "back", "start"), BUTTONS, strict=True)
)
JOYSTICK_BITS = 16  # their `joystick_bits`: the sticks' resolution on the wire
AXIS_MAX = 2 ** (JOYSTICK_BITS - 1) - 1  # a signed reading of that width
INVERTED = ("ly", "ry")  # their reading negates the y sticks
# How long a press or a chord is held; 0.15 s was missed by their 1 kHz machine.
PRESS_S = 0.3
COMMAND_LIMIT = 1.0  # a stick reaches 1.0 at most: the pad's command envelope
# What the device announces itself as: Microsoft's Xbox 360 controller ids,
# which their joystick reader treats as an xbox layout.
XBOX_VENDOR = 0x045E
XBOX_PRODUCT = 0x028E
PAD_NAME = "trainnr virtual xbox pad"


def axis_value(x: float, *, inverted: bool = False) -> int:
    """A stick value in [-1, 1] as the 16-bit axis their reader divides
    by the maximum (and negates for the y sticks)."""
    x = max(-1.0, min(1.0, float(x)))
    return round((-x if inverted else x) * AXIS_MAX)


def sticks_for_command(command: Sequence[float] | NDArray[Any]) -> dict[str, float]:
    """The stick positions that make their controller command
    (forward, sideways, turn): ly = forward, lx = -sideways, rx = -turn."""
    vx, vy, wz = (float(c) for c in command[:3])
    return {"ly": vx, "lx": -vy, "rx": -wz, "ry": 0.0}


class VirtualPad:
    """The device. `evdev.UInput` creates it; the joystick driver exposes
    it as `/dev/input/js<n>` for their simulator."""

    def __init__(
        self, name: str = PAD_NAME, sleep: Callable[[float], None] = time.sleep
    ) -> None:
        require_platform(RUNTIMES["dds"])
        from evdev import AbsInfo, UInput, ecodes  # noqa: PLC0415

        self._e = ecodes
        self._sleep = sleep
        info = AbsInfo(
            value=0, min=-AXIS_MAX, max=AXIS_MAX, fuzz=0, flat=0, resolution=0
        )
        hat = AbsInfo(value=0, min=-1, max=1, fuzz=0, flat=0, resolution=0)
        # Any: evdev annotates capabilities as codes only, but its docs
        # (and the kernel) take (code, AbsInfo) pairs for EV_ABS.
        capabilities: dict[int, Any] = {
            ecodes.EV_KEY: [getattr(ecodes, b) for b in BUTTONS],
            ecodes.EV_ABS: [
                (getattr(ecodes, a), hat if a.startswith("ABS_HAT") else info)
                for a in AXES
            ],
        }
        self._ui = UInput(
            capabilities, name=name, vendor=XBOX_VENDOR, product=XBOX_PRODUCT
        )

    def _abs(self, axis: str, value: int) -> None:
        self._ui.write(self._e.EV_ABS, getattr(self._e, axis), value)

    def _key(self, button: str, down: bool) -> None:
        self._ui.write(self._e.EV_KEY, getattr(self._e, button), 1 if down else 0)

    def sticks(self, **positions: float) -> None:
        """Set any of lx, ly, rx, ry (in [-1, 1]); unnamed sticks stay."""
        for name, value in positions.items():
            self._abs(STICK_AXIS[name], axis_value(value, inverted=name in INVERTED))
        self._ui.syn()

    def chord(self, trigger: str, button: str, hold_s: float = PRESS_S) -> None:
        """Hold a trigger (LT/RT) and press a button or a d-pad direction:
        their state machine's transitions ("LT + up", "RT + A")."""
        self._abs(TRIGGER_AXIS[trigger], AXIS_MAX)
        self._ui.syn()
        self._sleep(hold_s)
        if button in DPAD:
            axis, direction = DPAD[button]
            self._abs(axis, direction)
            self._ui.syn()
            self._sleep(hold_s)
            self._abs(axis, 0)
        else:
            key = BUTTON_KEY[button]
            self._key(key, True)
            self._ui.syn()
            self._sleep(hold_s)
            self._key(key, False)
        self._ui.syn()
        self._sleep(hold_s)
        self._abs(TRIGGER_AXIS[trigger], 0)
        self._ui.syn()

    def close(self) -> None:
        self._ui.close()
