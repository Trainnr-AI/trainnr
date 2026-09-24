"""What a recording's clock and joints actually did — measured, so a
card can say "499.8 Hz, 0.2 % dropouts, hips swept ±0.38 rad" instead
of repeating the rate a vendor's datasheet promised.

Every number here is read off the samples: the rate is the count over
the span, the jitter the spread of the intervals, a dropout an interval
longer than DROPOUT_FACTOR times the median, the joint range the min
and max the recording reached. An adapter attaches the result to its
census under QUALITY_KEY; the drawer shows it; a fit refuses a channel
whose range never left its noise. Shared by every adapter, so a wire
file and a ROS bag are described in the same words.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rq_pipeline.robots.recording import (
    JOINT_POSITION,
    JOINT_VELOCITY,
    Channel,
    Recording,
)

QUALITY_KEY = "quality"
DROPOUT_FACTOR = 3.0  # an interval this many medians long is a dropout
MOVING_RAD_S = 0.05  # a joint faster than this counts as moving
PERCENTILES = (50, 99)


@dataclass(frozen=True)
class ClockQuality:
    """A channel's clock, measured."""

    samples: int
    duration_s: float
    rate_hz: float | None
    interval_p50_ms: float | None
    interval_p99_ms: float | None
    dropouts: int
    longest_gap_ms: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "duration_s": round(self.duration_s, 3),
            "rate_hz": _r(self.rate_hz, 1),
            "interval_p50_ms": _r(self.interval_p50_ms, 3),
            "interval_p99_ms": _r(self.interval_p99_ms, 3),
            "dropouts": self.dropouts,
            "longest_gap_ms": _r(self.longest_gap_ms, 1),
        }


def clock_quality(times: np.ndarray) -> ClockQuality:
    """The clock of one channel from its sample times (seconds)."""
    n = len(times)
    if n < 2:  # noqa: PLR2004 - an interval needs two samples
        return ClockQuality(n, 0.0, None, None, None, 0, None)
    gaps = np.diff(np.asarray(times, dtype=np.float64))
    span = float(times[-1] - times[0])
    p50, p99 = (float(v) for v in np.percentile(gaps, PERCENTILES))
    dropouts = int(np.count_nonzero(gaps > DROPOUT_FACTOR * p50)) if p50 > 0 else 0
    return ClockQuality(
        samples=n,
        duration_s=span,
        rate_hz=(n - 1) / span if span > 0 else None,
        interval_p50_ms=p50 * 1e3,
        interval_p99_ms=p99 * 1e3,
        dropouts=dropouts,
        longest_gap_ms=float(gaps.max()) * 1e3,
    )


def joint_range(channel: Channel) -> dict[str, list[float]]:
    """Per component, the [min, max] the recording reached."""
    values = channel.values.reshape(len(channel.times), -1)
    names = channel.components or tuple(str(i) for i in range(channel.width))
    out: dict[str, list[float]] = {}
    for col, name in enumerate(names):
        column = values[:, col]
        finite = column[np.isfinite(column)]
        if finite.size:
            out[name] = [round(float(finite.min()), 4), round(float(finite.max()), 4)]
    return out


def moving_fraction(velocity: Channel, threshold: float = MOVING_RAD_S) -> float:
    """The share of samples where any joint moved faster than `threshold`."""
    values = np.abs(velocity.values.reshape(len(velocity.times), -1))
    if not len(values):
        return 0.0
    return round(float(np.mean(np.any(values > threshold, axis=1))), 4)


def describe(recording: Recording) -> dict[str, Any]:
    """The quality block for a recording's census: every channel's clock,
    the joint ranges covered, the fraction of time moving."""
    out: dict[str, Any] = {
        "clock": {
            name: clock_quality(c.times).as_dict()
            for name, c in recording.channels.items()
        }
    }
    position = recording.channels.get(JOINT_POSITION)
    if position is not None:
        out["joint_range"] = joint_range(position)
    velocity = recording.channels.get(JOINT_VELOCITY)
    if velocity is not None:
        out["moving_fraction"] = moving_fraction(velocity)
        out["moving_threshold"] = f"any joint over {MOVING_RAD_S} rad/s"
    return out


def _r(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)
