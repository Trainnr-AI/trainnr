"""The press's Studio feed — docs/66 §0: everything that happens
streams to the Studio.

`press.PressFeed` is the loop's seam; this module is the Studio
implementation: keep counters as series on an `attempt` timeline, every
attempt's verdict as a text log, and the latest kept episode's frames
as images — one entity per camera. Connection is fire-and-forget
through the Studio's one ingest address (`viz.STUDIO_ADDRESS`): the
Studio may not be up, Rerun buffers and drops, and pressing never
blocks on the viewer. A venv without rerun-sdk gets one loud line and
an unwatched press, not a crash — data generation does not die of a
missing viewer (the `viz` extra adds it).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from trainnr.viz import STUDIO_ADDRESS

if TYPE_CHECKING:
    from trainnr.collect.press import DemoBatch, PressResult, Say

APP_ID = "trainnr-data"
ATTEMPT_TIMELINE = "attempt"
TICK_TIMELINE = "tick"  # a kept episode's frames, by physics step
ENTITY_ROOT = "press"
FRAMES_PER_KEEPER = 250  # at most this many frames per camera per keeper
JPEG_QUALITY = 80


def thinned(frames: list[tuple[int, Any]], limit: int) -> list[tuple[int, Any]]:
    """At most `limit` frames, evenly spaced, the last one always kept."""
    if len(frames) <= limit:
        return list(frames)
    step = len(frames) / limit
    picked = [frames[int(i * step)] for i in range(limit)]
    if picked[-1] is not frames[-1]:
        picked[-1] = frames[-1]
    return picked


# The saved stream of a data-generation batch, inside its folder (docs/76 §10.5).
PRESS_STREAM = "press"


class StudioPressFeed:
    """Streams one press run to the Studio (a `press.PressFeed`)."""

    def __init__(self, run_name: str, file: Path | None = None) -> None:
        from trainnr.viz import open_stream  # noqa: PLC0415 - viz extra

        self._rr: Any = open_stream(
            f"{APP_ID}-{run_name}", address=STUDIO_ADDRESS, file=file
        )

    @classmethod
    def connect(
        cls, run_name: str, say: Say = print, file: Path | None = None
    ) -> StudioPressFeed | None:
        """The feed, or None with ONE loud line when rerun-sdk is not in
        this venv — pressing continues unwatched (docs/66 §0)."""
        try:
            return cls(run_name, file)
        except ImportError:
            say(
                "no rerun-sdk in this venv — pressing continues UNWATCHED "
                "(docs/66 §0; `uv sync --extra viz` to stream to the Studio)"
            )
            return None

    def attempt(
        self, attempts: int, kept: int, wanted: int, result: PressResult
    ) -> None:
        rr = self._rr
        rr.set_time(ATTEMPT_TIMELINE, sequence=attempts)
        rr.log(f"{ENTITY_ROOT}/kept", rr.Scalars(float(kept)))
        rr.log(f"{ENTITY_ROOT}/wanted", rr.Scalars(float(wanted)))
        rr.log(f"{ENTITY_ROOT}/keep_rate_bound", rr.Scalars(kept / attempts))
        verdict = "KEEP" if result.succeeded else "discard"
        note = f" ({result.note})" if result.note else ""
        rr.log(f"{ENTITY_ROOT}/log", rr.TextLog(f"attempt {attempts}: {verdict}{note}"))
        if not result.succeeded:
            return
        # Every frame of the keeper on its own `tick` timeline, so the
        # operator can scrub the episode; the attempt timeline stays
        # set, so scrubbing attempts shows each keeper's last frame.
        # JPEG-compressed: raw RGB at 50 Hz over three cameras would
        # sit on the viewer's throat (docs/07 2026-09-03).
        cameras = dict(result.camera_frames or {})
        if result.frames:  # the single-camera path (kitting's layout)
            cameras.setdefault("camera", result.frames)
        for camera, frames in cameras.items():
            for tick, image in thinned(frames, FRAMES_PER_KEEPER):
                rr.set_time(TICK_TIMELINE, sequence=int(tick))
                rr.log(
                    f"{ENTITY_ROOT}/{camera}",
                    rr.Image(image).compress(jpeg_quality=JPEG_QUALITY),
                )
        rr.disable_timeline(TICK_TIMELINE)

    def done(self, batch: DemoBatch) -> None:
        self._rr.log(
            f"{ENTITY_ROOT}/log",
            self._rr.TextLog(
                f"kept {batch.kept}/{batch.attempts} attempts -> {batch.out}"
            ),
        )
