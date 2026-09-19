"""The commanded twist a locomotion policy is judged at, in words.

A velocity-tracking evaluation and a deployment both carry the ranges
the held command was drawn from (`lin_vel_x`, `lin_vel_y`, `ang_vel_z`,
each `[low, high]`); the Studio's cards and drawers say them the same
way everywhere: "forward -1.5 to 2.0 m/s, turn ±0.7 rad/s". The keys
are mjlab's velocity-command names, kept as they are.
"""

from __future__ import annotations

from typing import Any

FORWARD, SIDEWAYS, TURN = "lin_vel_x", "lin_vel_y", "ang_vel_z"
TWIST_KEYS = (FORWARD, SIDEWAYS, TURN)
RANGE_ENDS = 2  # a range is two numbers
# The column that names a held command in a trials table.
TWIST_LABEL = "command (vx, vy, wz)"


def _range(value: Any) -> tuple[float, float] | None:
    if isinstance(value, (list, tuple)) and len(value) == RANGE_ENDS:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None
    return None


def describe_twist(ranges: Any) -> str | None:
    """The envelope in a few words, or None when `ranges` holds none of
    the twist keys: forward and turn are what tell two judgments of one
    checkpoint apart on a card; sideways is said when it is non-zero."""
    if not isinstance(ranges, dict):
        return None
    parts = []
    forward = _range(ranges.get(FORWARD))
    if forward is not None:
        parts.append(f"forward {forward[0]:g} to {forward[1]:g} m/s")
    sideways = _range(ranges.get(SIDEWAYS))
    if sideways is not None and sideways != (0.0, 0.0):
        parts.append(f"sideways {sideways[0]:g} to {sideways[1]:g} m/s")
    turn = _range(ranges.get(TURN))
    if turn is not None:
        parts.append(
            f"turn ±{turn[1]:g} rad/s"
            if turn[0] == -turn[1]
            else f"turn {turn[0]:g} to {turn[1]:g} rad/s"
        )
    return ", ".join(parts) or None
