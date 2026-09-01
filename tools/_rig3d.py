"""Compatibility shim: the mirror moved into the package so the mjlab
recorder can use it too (2026-09-01). Tools keep this import path."""

from rq_pipeline.viz import RIG_PATH, RigMirror, mat_to_xyzw

__all__ = ["RIG_PATH", "RigMirror", "mat_to_xyzw"]
