"""Stage ④ — collection: recordings become datasets, stamped with provenance."""

from trainnr.collect.frames import AlignedEpisode, Frame, align
from trainnr.collect.wire import (
    CameraNote,
    Recording,
    StatusFrame,
    parse_recording,
)

__all__ = [
    "AlignedEpisode",
    "CameraNote",
    "Frame",
    "Recording",
    "StatusFrame",
    "align",
    "parse_recording",
]
