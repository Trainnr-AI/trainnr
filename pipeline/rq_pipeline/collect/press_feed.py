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

from typing import TYPE_CHECKING, Any

from rq_pipeline.viz import STUDIO_ADDRESS

if TYPE_CHECKING:
    from rq_pipeline.collect.press import DemoBatch, PressResult, Say

APP_ID = "rq-press"
ATTEMPT_TIMELINE = "attempt"
ENTITY_ROOT = "press"


class StudioPressFeed:
    """Streams one press run to the Studio (a `press.PressFeed`)."""

    def __init__(self, run_name: str) -> None:
        import rerun as rr  # noqa: PLC0415 - viz extra

        self._rr: Any = rr
        rr.init(f"{APP_ID}-{run_name}")
        rr.connect_grpc(STUDIO_ADDRESS)

    @classmethod
    def connect(cls, run_name: str, say: Say = print) -> StudioPressFeed | None:
        """The feed, or None with ONE loud line when rerun-sdk is not in
        this venv — pressing continues unwatched (docs/66 §0)."""
        try:
            return cls(run_name)
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
        for camera, frames in (result.camera_frames or {}).items():
            if frames:
                rr.log(f"{ENTITY_ROOT}/{camera}", rr.Image(frames[-1][1]))
        if result.frames:  # the single-camera path (kitting's layout)
            rr.log(f"{ENTITY_ROOT}/camera", rr.Image(result.frames[-1][1]))

    def done(self, batch: DemoBatch) -> None:
        self._rr.log(
            f"{ENTITY_ROOT}/log",
            self._rr.TextLog(
                f"kept {batch.kept}/{batch.attempts} attempts -> {batch.out}"
            ),
        )
