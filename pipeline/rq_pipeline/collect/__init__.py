"""Stage ④ — collection: recordings become datasets, stamped with provenance."""

from rq_pipeline.collect.wire import (
    CameraNote,
    Recording,
    StatusFrame,
    parse_recording,
)

__all__ = ["CameraNote", "Recording", "StatusFrame", "parse_recording"]
