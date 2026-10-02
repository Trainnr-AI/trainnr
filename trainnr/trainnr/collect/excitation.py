"""Turn a wire recording into identification data for the drivetrain.

The bridge between stage ④ (what the rig recorded) and stage ②'s
`identify()` — the last software link in Paper 0's chain.

Assumptions, stated because they are load-bearing:

- **The sweep, not the chase.** The wire status carries ONE scalar duty,
  so a chase's per-wheel split during turns is unobserved. The
  calibration sweep drives both wheels with the same known duty, which
  makes the two encoder streams two independent single-wheel
  experiments. This adapter therefore replicates the scalar duty across
  both actuators and is only valid for sweep-style recordings.
- **The run starts at rest at tick zero.** Times are rebased to the
  first status, tick origins are subtracted, and the model's home state
  (zero position, zero velocity) is the initial condition `identify()`
  uses. A recording that starts mid-motion violates this and must be
  trimmed first.
- **Column order is (left, right)** — a contract with
  `robots/rig-drivetrain/model.xml`'s sensor order.
- The tick scale comes from the bundle's `RobotProfile`, never fitted:
  `ticks_per_revolution` is degenerate with the motor gain (both scale
  the output), so the profile carries it with a provenance string —
  encoder datasheet or hand-count — and the fit record inherits that.
"""

from __future__ import annotations

from math import tau

from trainnr.bundles.profile import RobotProfile
from trainnr.collect.frames import STATUS_HZ
from trainnr.collect.wire import Recording
from trainnr.robot.identify import ExcitationData

# Two frames is the bare minimum for any dynamics to be visible at all.
_MINIMUM_STATUS_FRAMES = 2


def drivetrain_excitation(
    recording: Recording,
    profile: RobotProfile,
) -> ExcitationData:
    """Statuses to (times, duty commands x2, wheel angles in radians x2)."""
    import numpy as np  # noqa: PLC0415 - keeps the fast import path light

    statuses = recording.statuses
    if len(statuses) < _MINIMUM_STATUS_FRAMES:
        raise ValueError(
            f"need at least {_MINIMUM_STATUS_FRAMES} status frames, got {len(statuses)}"
        )

    first_seq = statuses[0].seq
    times = np.array([(status.seq - first_seq) / STATUS_HZ for status in statuses])
    duty = np.array([float(status.duty_percent) for status in statuses])
    # Same commanded duty on both wheels — the sweep assumption above.
    controls = np.column_stack([duty, duty])

    radians_per_tick = tau / profile.ticks_per_revolution
    left = np.array([status.ticks_left for status in statuses], dtype=float)
    right = np.array([status.ticks_right for status in statuses], dtype=float)
    measurements = np.column_stack(
        [
            (left - left[0]) * radians_per_tick,
            (right - right[0]) * radians_per_tick,
        ]
    )
    return ExcitationData(times=times, controls=controls, measurements=measurements)
