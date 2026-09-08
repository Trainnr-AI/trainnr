"""This repo's own rig: the `.wire` telemetry the Pico firmware emits.

Wraps `collect/wire.parse_recording` — the reader the Rust gate mirrors —
into channels. The wire format has no timestamps; the status stream is
the 50 Hz clock (`collect/frames.STATUS_HZ`), so time is the sequence
number over that rate, and the manifest says so.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rq_pipeline.collect.frames import STATUS_HZ
from rq_pipeline.collect.wire import parse_recording
from rq_pipeline.robots.adapter import adapter
from rq_pipeline.robots.recording import (
    COLLECTION_ROBOT_OP,
    JOINT_COMMAND,
    Channel,
    Recording,
)

NAME = "wire"
SUFFIX = ".wire"
NOTE_CLOCK = (
    f"time is the status sequence number over {STATUS_HZ:g} Hz; the wire "
    "format carries no timestamps"
)


@adapter(NAME, doc="This repo's Pico rig: .wire telemetry, 50 Hz status frames")
class WireAdapter:
    """This repo's Pico rig: .wire telemetry, 50 Hz status frames."""

    name = NAME

    def accepts(self, source: Path) -> bool:
        return Path(source).is_file() and Path(source).suffix == SUFFIX

    def read(self, source: Path) -> Recording:
        source = Path(source)
        if not self.accepts(source):
            raise ValueError(f"{source} is not a {SUFFIX} recording")
        parsed = parse_recording(source)
        statuses = parsed.statuses
        if not statuses and not parsed.servo_pulses:
            raise ValueError(
                f"{source}: no status frames and no servo pulses — nothing this "
                "adapter reads. (The arm's `J` joint stream is parsed by the Rust "
                "`hil-host` reader only; a Python reader for it is not built.)"
            )
        # Sequence numbers may repeat across a reconnect; keep strictly
        # increasing time by using the frame's position when they do.
        channels: dict[str, Channel] = {}
        if parsed.servo_pulses:
            # Servo notes ride the same 50 Hz wire between status frames;
            # with no timestamps their clock is their own count over it.
            pulse_times = (
                np.arange(len(parsed.servo_pulses), dtype=np.float64) / STATUS_HZ
            )
            channels["servo.pulse_us"] = Channel(
                "servo.pulse_us",
                pulse_times,
                np.array(parsed.servo_pulses, dtype=np.float64),
                unit="us",
                components=("ch0", "ch1", "ch2"),
            )
        if not statuses:
            return Recording(
                source=source.name,
                adapter=NAME,
                collection=COLLECTION_ROBOT_OP,
                channels=channels,
                census=_census(parsed),
                notes=[NOTE_CLOCK],
            )
        seq = np.array([s.seq for s in statuses], dtype=np.float64)
        if not np.all(np.diff(seq) > 0):
            seq = np.arange(len(statuses), dtype=np.float64)
        times = seq / STATUS_HZ
        channels |= {
            "wheel.ticks": Channel(
                "wheel.ticks",
                times,
                np.array(
                    [[s.ticks_left, s.ticks_right] for s in statuses], dtype=np.float64
                ),
                unit="ticks",
                components=("left", "right"),
            ),
            "wheel.errors": Channel(
                "wheel.errors",
                times,
                np.array(
                    [[s.errors_left, s.errors_right] for s in statuses],
                    dtype=np.float64,
                ),
                unit="count",
                components=("left", "right"),
            ),
            JOINT_COMMAND: Channel(
                JOINT_COMMAND,
                times,
                np.array([[s.duty_percent] for s in statuses], dtype=np.float64),
                unit="percent",
                components=("duty",),
            ),
            "pose": Channel(
                "pose",
                times,
                np.array([[s.x, s.y, s.heading] for s in statuses], dtype=np.float64),
                unit="m,m,rad",
                components=("x", "y", "heading"),
            ),
            "stalled": Channel(
                "stalled",
                times,
                np.array([[1.0 if s.stalled else 0.0] for s in statuses]),
                unit="flag",
                components=("stalled",),
            ),
        }
        return Recording(
            source=source.name,
            adapter=NAME,
            collection=COLLECTION_ROBOT_OP,
            channels=channels,
            census=_census(parsed),
            notes=[NOTE_CLOCK],
        )


def _census(parsed) -> dict:  # type: ignore[no-untyped-def]
    return {
        "status_frames": len(parsed.statuses),
        "images": len(parsed.images),
        "torn_images": parsed.torn_images,
        "servo_pulses": len(parsed.servo_pulses),
        "unparsable": parsed.unparsable,
        "stalled_frames": parsed.stalled_count,
    }
