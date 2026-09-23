"""The scene artifact's record (docs/78 §3): how the scene was made, what
the eye sees, what the solver touches, how far apart those are, and
which physics were measured against which were declared.

Nothing here is invented: a field the capture did not record reads
`unrecorded`; a physics value carries its basis word beside it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rq_pipeline.bundles.json_record import JsonRecord

SCENE_FILE = "scene.json"
SCENE_SCHEMA = "trainnr-scene/1"
SPLAT_FILE = "splat.ply"  # the scene in the world frame, 3DGS PLY
PROXY_FILE = "proxy.obj"  # the collision proxy in the world frame
# The proxy's two halves (`scenes.volume`): the ground the walker stands
# on (a top surface below the clearance) and what stands above it (an
# occupancy volume: a table top with air beneath, a wall, a bush).
GROUND_FILE = "ground.obj"
OVERHANG_FILE = "overhang.obj"
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
    # The course the scene's author laid out, when the scene names one:
    # ordered waypoints in the world frame and where they came from. A
    # task's start is derived from it (`scenes.stage`), never guessed.
    course: dict[str, Any] = field(default_factory=dict)

    @property
    def declared(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.physics if p.basis == DECLARED)


def load_scene_record(path: Path) -> SceneRecord:
    """A record back from disk, its parts typed; refuses another schema."""
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


COLLISION_GROUP = 3  # the group the stage and the viewer treat as colliders
PROXY_RGBA = (0.4, 0.4, 0.4, 0.3)  # the proxy's look in any viewer: grey, faint
# What a proxy built from the splat itself says about its origin (the
# capture writes it; the card reads it: the gap is then self-referential,
# docs/78 §8.6).
PROXY_FROM_SPLAT = "the visible gaussian centres (opacity >= {opacity})"


def proxy_from_splat(record: dict[str, Any]) -> bool:
    """Whether a scene record's proxy came from its own splat."""
    origin = str((record.get("proxy") or {}).get("from", ""))
    return origin.startswith(PROXY_FROM_SPLAT.split("{", 1)[0])


# The capture chain's log, written stage by stage into the scene's folder
# before the record exists (`scenes.capture`); its presence without the
# record is a scene in progress, and its last lines say which stage.
CAPTURE_LOG_FILE = "capture.log"
CAPTURE_STAGE_PREFIX = "[capture] "  # the chain's own lines
CAPTURE_COMMAND_PREFIX = "$ "  # a tool's command line
CAPTURE_FAILED_WORD = "failed: "  # after the stage prefix: the chain's last word


def capture_in_progress(folder: Path) -> bool:
    """A scene folder the chain is still working in: its log without its record."""
    return (folder / CAPTURE_LOG_FILE).is_file() and not (folder / SCENE_FILE).is_file()


def capture_failed(folder: Path) -> str | None:
    """Why a capture in this folder died, when its log's last note says
    so; None while it runs or when it never wrote one (then it is in
    progress until the job's own record says otherwise)."""
    stage = capture_stage(folder)
    if not stage.startswith(CAPTURE_FAILED_WORD):
        return None
    return stage[len(CAPTURE_FAILED_WORD) :]


def capture_stage(folder: Path) -> str:
    """The chain's latest word from its log: the tool it last started,
    or its own last note; `starting` before either."""
    try:
        lines = (folder / CAPTURE_LOG_FILE).read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        return "starting"
    for line in reversed(lines.splitlines()):
        if line.startswith(CAPTURE_COMMAND_PREFIX):
            argv = line[len(CAPTURE_COMMAND_PREFIX) :].split()
            # the tool's bare name whichever OS wrote the line: either
            # separator, and `.exe` off
            tool = Path(re.split(r"[\\/]", argv[0])[-1]).stem if argv else "a tool"
            after = argv[1] if len(argv) > 1 else ""
            if after == "-m" and argv[2:]:  # an interpreter running a module
                return argv[2].rsplit(".", 1)[-1]
            # COLMAP's subcommand is a bare word; Brush's first argument is a path
            verb = after if after.isidentifier() else ""
            return f"{tool} {verb}".strip()
        if line.startswith(CAPTURE_STAGE_PREFIX):
            return line[len(CAPTURE_STAGE_PREFIX) :].strip()
    return "starting"


# The record's declared physics, by name (docs/78 §3): the floor's friction
# as a scene's author or importer declares it, with this project's span.
FLOOR_FRICTION = "floor_friction"
DECLARED_FRICTION_SPAN = 0.2
# MuJoCo's own geom friction (sliding, torsional, rolling): what a
# declaration shorter than three numbers is padded with, so the proxy's
# MJCF and a stage's terrain give the solver the same triple.
MUJOCO_FRICTION = (1.0, 0.005, 0.0001)


def friction_triple(friction: Sequence[float]) -> tuple[float, float, float]:
    """A declared friction as the solver takes it: the numbers given,
    the rest MuJoCo's defaults."""
    values = [float(v) for v in friction][:3]
    return tuple(values + list(MUJOCO_FRICTION[len(values) :]))  # type: ignore[return-value]


def floor_friction(record: SceneRecord) -> list[float] | None:
    """The floor's declared friction from a record, as a list; None when
    the scene declares none."""
    declared = next((p for p in record.physics if p.name == FLOOR_FRICTION), None)
    if declared is None:
        return None
    return [float(v) for v in np.atleast_1d(declared.value)]


def proxy_mjcf(mesh_file: str, friction: list[float] | None) -> str:
    """The proxy as one mesh geom in the collision group: what every
    importer writes beside the OBJ, so a plain MuJoCo load sees it."""
    friction_attr = (
        f' friction="{" ".join(f"{v:g}" for v in friction_triple(friction))}"'
        if friction
        else ""
    )
    rgba = " ".join(f"{v:g}" for v in PROXY_RGBA)
    return (
        "<mujoco>\n"
        "  <asset>\n"
        f'    <mesh name="scene_proxy" file="{mesh_file}"/>\n'
        "  </asset>\n"
        "  <worldbody>\n"
        f'    <geom name="scene_proxy" type="mesh" mesh="scene_proxy" '
        f'group="{COLLISION_GROUP}"{friction_attr} rgba="{rgba}"/>\n'
        "  </worldbody>\n"
        "</mujoco>\n"
    )
