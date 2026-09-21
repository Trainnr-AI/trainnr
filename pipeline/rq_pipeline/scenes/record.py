"""The scene artifact's record (docs/78 §3): how the scene was made, what
the eye sees, what the solver touches, how far apart those are, and
which physics were measured against which were declared.

Nothing here is invented: a field the capture did not record reads
`unrecorded`; a physics value carries its basis word beside it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rq_pipeline.bundles.json_record import JsonRecord

SCENE_FILE = "scene.json"
SCENE_SCHEMA = "trainnr-scene/1"
SPLAT_FILE = "splat.ply"  # the scene in the world frame, 3DGS PLY
PROXY_FILE = "proxy.obj"  # the collision proxy in the world frame
PROXY_MJCF = "proxy.xml"  # the proxy as MuJoCo assets and geoms, included by a task
UNRECORDED = "unrecorded"

# A physics parameter's basis: measured by a robot with an interval, or
# declared by someone with a span (docs/78 §3, §4 E5).
MEASURED = "measured"
DECLARED = "declared"
BASES = (MEASURED, DECLARED)


@dataclass(frozen=True)
class Capture:
    """The capture's provenance as the source recorded it."""

    device: str = UNRECORDED
    app: str = UNRECORDED
    frames: int | str = UNRECORDED
    resolution: str = UNRECORDED
    duration_s: float | str = UNRECORDED
    lighting: str = UNRECORDED


@dataclass(frozen=True)
class Tool:
    """A tool in the chain, with its licence: the chain's honesty."""

    name: str
    version: str = UNRECORDED
    license: str = UNRECORDED
    role: str = ""


@dataclass(frozen=True)
class Alignment:
    """The similarity that took the capture frame to the world frame."""

    scale: float
    rotation_euler_xyz: tuple[float, float, float]
    translation: tuple[float, float, float]
    source: str  # who fixed it: the importer's file, a fitted floor, unrecorded


@dataclass(frozen=True)
class Gap:
    """The visible surface against the collision proxy (docs/78 §3).
    Distances in metres; `None` where the audit could not run."""

    chamfer_m: float | None
    p95_m: float | None
    beyond_tolerance_fraction: float | None
    hidden_fraction: (
        float | None
    )  # proxy surface farther than the tolerance from any visible gaussian
    tolerance_m: float
    visible_samples: int  # visible gaussians inside the proxy's footprint, judged
    proxy_samples: int
    method: str
    # The share of ALL visible gaussians inside the proxy's footprint:
    # what the solver covers of what the eye sees. None when unmeasured.
    footprint_fraction: float | None = None
    note: str = ""


@dataclass(frozen=True)
class Physics:
    """One physics parameter with its basis."""

    name: str
    value: float | list[float]
    basis: str  # MEASURED or DECLARED
    span: float | None = None  # declared: the randomization half-width, relative
    interval: tuple[float, float] | None = None  # measured: the confidence interval
    cites: str = ""  # the fit record or the declaring document
    unit: str = ""

    def __post_init__(self) -> None:
        if self.basis not in BASES:
            raise ValueError(
                f"{self.name}: basis is one of {BASES}, not {self.basis!r}"
            )
        if self.basis == MEASURED and self.interval is None:
            raise ValueError(f"{self.name}: a measured value carries its interval")
        if self.basis == DECLARED and self.span is None:
            raise ValueError(f"{self.name}: a declared value carries its span")


@dataclass(frozen=True)
class SceneRecord(JsonRecord):
    """The artifact: everything §3 asks for."""

    name: str
    source: str  # where the scene came from: a benchmark's id, a capture's path
    capture: Capture
    tools: tuple[Tool, ...]
    splat: dict[str, Any]  # `splat.describe`'s facts, plus the file's name
    proxy: dict[str, Any]  # vertices, faces, watertight, the file's name
    alignment: Alignment
    gap: Gap
    physics: tuple[Physics, ...]
    created_utc: str
    code: str
    schema: str = SCENE_SCHEMA
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def declared(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.physics if p.basis == DECLARED)


def load_scene_record(path: Path) -> SceneRecord:
    """A record back from disk, its parts typed; refuses another schema."""
    import json  # noqa: PLC0415

    raw: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = raw.get("schema")
    if schema != SCENE_SCHEMA:
        raise ValueError(
            f"{path}: schema {schema!r}, this reader speaks {SCENE_SCHEMA!r}"
        )
    raw["capture"] = Capture(**raw.get("capture", {}))
    raw["tools"] = tuple(Tool(**t) for t in raw.get("tools", []))
    align = raw["alignment"]
    raw["alignment"] = Alignment(
        scale=float(align["scale"]),
        rotation_euler_xyz=tuple(align["rotation_euler_xyz"]),
        translation=tuple(align["translation"]),
        source=align.get("source", UNRECORDED),
    )
    raw["gap"] = Gap(**raw["gap"])
    physics = []
    for p in raw.get("physics", []):
        if p.get("interval") is not None:
            p["interval"] = tuple(p["interval"])
        physics.append(Physics(**p))
    raw["physics"] = tuple(physics)
    raw["notes"] = tuple(raw.get("notes", ()))
    known = set(SceneRecord.__dataclass_fields__)
    return SceneRecord(**{k: v for k, v in raw.items() if k in known})
