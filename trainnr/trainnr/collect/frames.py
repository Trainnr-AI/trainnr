"""Align a parsed recording into (observation, action) training frames.

Two constraints the wire format imposes, both handled here rather than
hidden:

- **There are no timestamps.** The recorder's only clock is the status
  sequence counter, which the firmware increments at exactly 50 Hz — so
  a frame's time is `paired_status.seq / 50.0`, and that provenance is
  worth a docstring because docs/e2e-research/27 §6 lists the missing
  timestamps as a known gap of the format.
- **The camera is the slow sensor.** Images arrive at a few Hz against
  50 Hz statuses, so the dataset's frame rate is the image rate: each
  completed image opens a frame, filled with the FIRST status and FIRST
  servo command that follow it in line order — the state the controller
  acted from and the action it then took. A frame that never sees both
  before the file ends is dropped and counted, never half-filled.
"""

from __future__ import annotations

from dataclasses import dataclass

from trainnr.bundles.hashing import require_stamp
from trainnr.collect.wire import Recording, StatusFrame

STATUS_HZ = 50.0

# RGB565 field widths, for bit-replicating expansion to RGB888 — the same
# expansion crates/blob uses, so both languages render identical pixels.
_FIVE_BIT_MAX = 0b11111
_SIX_BIT_MAX = 0b111111


def rgb565_to_rgb888(pixel: int) -> tuple[int, int, int]:
    """Bit-replicating expansion: full-scale white stays full-scale white."""
    red_5 = (pixel >> 11) & _FIVE_BIT_MAX
    green_6 = (pixel >> 5) & _SIX_BIT_MAX
    blue_5 = pixel & _FIVE_BIT_MAX
    return (
        (red_5 << 3) | (red_5 >> 2),
        (green_6 << 2) | (green_6 >> 4),
        (blue_5 << 3) | (blue_5 >> 2),
    )


@dataclass(frozen=True)
class Frame:
    """One aligned training frame: what the robot saw, knew, and did."""

    index: int
    timestamp_s: float
    width: int
    height: int
    rgb888: bytes  # width * height * 3, row-major
    state: StatusFrame
    servo_pulses_us: tuple[int, int, int]

    @property
    def action(self) -> tuple[float, float, float, float]:
        """Pan, tilt, grip pulse widths plus wheel duty — the commanded outputs."""
        pan, tilt, grip = self.servo_pulses_us
        return (
            float(pan),
            float(tilt),
            float(grip),
            float(self.state.duty_percent),
        )


@dataclass(frozen=True)
class AlignedEpisode:
    source: str  # recording identity, name@hash — provenance is not optional
    frames: tuple[Frame, ...]
    dropped_incomplete: int


def _expand_image(rows: tuple[tuple[int, ...], ...]) -> bytes:
    return bytes(
        channel for row in rows for pixel in row for channel in rgb565_to_rgb888(pixel)
    )


def align(recording: Recording, source: str) -> AlignedEpisode:
    """Pair each image with the first status and servo command that follow it."""
    require_stamp(source, "source identity")
    frames: list[Frame] = []
    dropped = 0
    events = recording.events
    for position, (kind, image_index) in enumerate(events):
        if kind != "image":
            continue
        state: StatusFrame | None = None
        pulses: tuple[int, int, int] | None = None
        for later_kind, later_index in events[position + 1 :]:
            if later_kind == "status" and state is None:
                state = recording.statuses[later_index]
            elif later_kind == "servo" and pulses is None:
                pulses = recording.servo_pulses[later_index]
            if state is not None and pulses is not None:
                break
        if state is None or pulses is None:
            dropped += 1
            continue
        image = recording.images[image_index]
        frames.append(
            Frame(
                index=len(frames),
                timestamp_s=state.seq / STATUS_HZ,
                width=image.width,
                height=image.height,
                rgb888=_expand_image(image.rows),
                state=state,
                servo_pulses_us=pulses,
            )
        )
    return AlignedEpisode(
        source=source, frames=tuple(frames), dropped_incomplete=dropped
    )
