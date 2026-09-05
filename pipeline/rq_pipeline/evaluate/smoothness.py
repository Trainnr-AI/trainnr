"""Smoothness of an executed action stream, as certificate columns.

Astribot's SmoothRL reports RMS acceleration and jerk of one rollout as
its smoothness evidence (docs/e2e-research/71 §2). Here the same three
derivatives of whatever a policy actually issued - velocity,
acceleration, jerk, RMS over an episode, per actuator-vector norm -
become per-trial numbers any certificate can carry with an interval.
Pure numpy over a `(T, nu)` array, or streamed one tick at a time for
loops that never hold the trajectory (a batched GPU env, a bridge);
the two agree exactly (tested). Units: the action's per second^k when
`dt` is given, per tick otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

ORDERS = ("velocity", "acceleration", "jerk")


@dataclass(frozen=True)
class Smoothness:
    """RMS of the k-th difference of the action stream, k = 1, 2, 3."""

    velocity: float
    acceleration: float
    jerk: float

    def columns(self, prefix: str = "rms_") -> dict[str, float]:
        """The row's spelling, rounded like the other variations."""
        return {f"{prefix}{k}": round(float(getattr(self, k)), 5) for k in ORDERS}

    @classmethod
    def of(cls, actions: Any, dt: float = 1.0) -> Smoothness:
        """Over a whole `(T, nu)` trajectory."""
        rows = np.asarray(actions, dtype=float)
        if rows.ndim != 2:  # noqa: PLR2004 - (T, nu)
            raise ValueError(f"actions must be (T, nu), got shape {rows.shape}")
        values = []
        diff = rows
        for k in range(1, 4):
            diff = np.diff(diff, axis=0)
            per_tick = np.sqrt(np.mean(np.sum(diff**2, axis=1))) if len(diff) else 0.0
            values.append(per_tick / dt**k)
        return cls(*values)


def control_period(stepper: Any, control_interval: int) -> float | None:
    """The seconds one control tick lasts, from the stepper's physics
    timestep when it has one (MuJoCo and MJX steppers carry `model.opt.
    timestep`); None when the engine does not say - then no smoothness
    column is written, rather than one in unstated units."""
    model = getattr(stepper, "model", None)
    opt = getattr(model, "opt", None)
    timestep = getattr(opt, "timestep", None)
    return None if timestep is None else float(timestep) * control_interval


class SmoothnessMeter:
    """The same numbers, one tick at a time, for `worlds` parallel
    streams: keeps the last three differences per world and the running
    sums of squares. `observe(actions)` takes `(worlds, nu)` each tick;
    `result(world)` gives that stream's `Smoothness`."""

    def __init__(self, worlds: int, dt: float = 1.0) -> None:
        self.dt = dt
        self._last: list[np.ndarray | None] = [None, None, None]  # a, d1, d2
        self._sums = np.zeros((3, worlds))
        self._counts = np.zeros((3, worlds))

    def observe(self, actions: Any, active: Any = None) -> None:
        """One tick of `(worlds, nu)` actions; `active` (a bool mask per
        world) keeps a finished world's stale actions out of its numbers."""
        current: np.ndarray | None = np.asarray(actions, dtype=float)
        live = (
            np.ones(self._sums.shape[1], dtype=bool)
            if active is None
            else np.asarray(active, dtype=bool)
        )
        for k in range(3):
            previous = self._last[k]
            self._last[k] = current
            if previous is None or current is None:
                current = None
                continue
            current = current - previous
            self._sums[k] += np.sum(current**2, axis=1) * live
            self._counts[k] += live

    def result(self, world: int) -> Smoothness:
        values = []
        for k in range(3):
            n = self._counts[k, world]
            rms = float(np.sqrt(self._sums[k, world] / n)) if n else 0.0
            values.append(rms / self.dt ** (k + 1))
        return Smoothness(*values)
