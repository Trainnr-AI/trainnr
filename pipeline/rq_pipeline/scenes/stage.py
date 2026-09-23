"""A scene as a task's stage (docs/78 §4 E2): the deployment's trained
scene with its plane floor replaced by the scene's terrain - its top
surface as a heightfield by default, its convex parts when asked
(`scenes.terrain`) - the robot started on the course the scene's author
laid out, and the two cameras the splat is rendered for. Composed with
MjSpec; the terrain's data is embedded in the XML, so a stage is one
self-contained file with no paths in it and a staged deployment moves
between machines like any artifact.

The perturbation assay (docs/78 §4.1) is the same composition with the
terrain moved: a whole-terrain offset in metres and a yaw about the
robot's start, the robot left where the nominal stage put it. The
field's number is ±20 mm and ±5° (2608.21416); a stage records which
perturbation it is.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rq_pipeline.bundles.hashing import stamp
from rq_pipeline.deploy.manifest import MANIFEST_FILE, Key, load_manifest
from rq_pipeline.scenes.heads import HEAD_CAMERA, HeadMount, head_mount, look_quat
from rq_pipeline.scenes.record import (
    SCENE_FILE,
    SceneRecord,
    floor_friction,
    load_scene_record,
)
from rq_pipeline.scenes.terrain import (
    DEFAULT_TERRAIN,
    RAY_FROM_M,
    TerrainFacts,
    terrain_builder,
)

STAGE_FILE = "stage.xml"
STAGE_TERRAIN_BODY = "scene_terrain"  # the stage's own body for the terrain
FLOOR_GEOM = "floor"  # the trained scene's plane, as the walk export names it
# The robot starts this far before the first waypoint, along the course.
START_BEHIND_M = 1.0
# Cameras: the robot's head camera (`scenes.heads`, by robot family) and a
# course camera behind and above the start looking along the course.
COURSE_CAMERA = "course"
COURSE_BEHIND_M = 2.5
COURSE_ABOVE_M = 1.5
COURSE_AHEAD_M = 3.0  # what the course camera looks at, along the heading
COURSE_FOVY = 60.0
SCENE_TERRAIN_WORD = "scene"  # the manifest's terrain word for a stage
# The head a stage puts on a scene with no registered robot (a test's
# tiny robot): a metre-free look along the base's x, on the floating base.
FREE_BASE_HEAD = HeadMount(pos=(0.0, 0.0, 0.0), fovy=90.0)


@dataclass(frozen=True)
class Perturbation:
    """The terrain moved under a policy that believes it did not."""

    label: str
    offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw_deg: float = 0.0


NOMINAL = Perturbation("nominal")
SHIFT_M = 0.02
TURN_DEG = 5.0


def _shift_label(axis: str, sign: str) -> str:
    return f"{axis}{sign}{SHIFT_M * 1000:g}mm"


def _turn_label(sign: str) -> str:
    return f"yaw{sign}{TURN_DEG:g}deg"


# The assay's set: the field's ±20 mm on each axis and ±5° of yaw.
ASSAY = (
    NOMINAL,
    Perturbation(_shift_label("x", "+"), (SHIFT_M, 0.0, 0.0)),
    Perturbation(_shift_label("x", "-"), (-SHIFT_M, 0.0, 0.0)),
    Perturbation(_shift_label("y", "+"), (0.0, SHIFT_M, 0.0)),
    Perturbation(_shift_label("y", "-"), (0.0, -SHIFT_M, 0.0)),
    Perturbation(_shift_label("z", "+"), (0.0, 0.0, SHIFT_M)),
    Perturbation(_shift_label("z", "-"), (0.0, 0.0, -SHIFT_M)),
    Perturbation(_turn_label("+"), yaw_deg=TURN_DEG),
    Perturbation(_turn_label("-"), yaw_deg=-TURN_DEG),
)


@dataclass(frozen=True)
class Stage:
    """One composed stage and the facts its manifest records."""

    xml: str
    scene: str  # the scene's stamp
    terrain: TerrainFacts
    start: tuple[float, float, float]  # the base's position at the keyframe
    heading_deg: float
    surface_z: float  # the nominal terrain's height under the start
    friction: list[float] | None
    cameras: tuple[str, ...]
    perturbation: Perturbation
    floor_removed: bool
    # the scene's course, verbatim from its record: what a course gate
    # walks (`deploy.course`); empty for a scene that lays out none
    course: dict[str, Any]

    def facts(self) -> dict[str, Any]:
        """The manifest's scene block for this stage."""
        return {
            "file": STAGE_FILE,
            "terrain": f"{SCENE_TERRAIN_WORD} {self.scene}",
            "terrain_kind": self.terrain.kind,
            "terrain_geoms": self.terrain.geoms,
            "terrain_gap": self.terrain.gap,
            "terrain_note": self.terrain.note,
            "start": [round(v, 4) for v in self.start],
            "heading_deg": round(self.heading_deg, 3),
            "surface_z": round(self.surface_z, 4),
            "floor": {"friction": self.friction}
            if self.friction is not None
            else "the model's default: the scene declares none",
            "cameras": list(self.cameras),
            "perturbation": asdict(self.perturbation),
            "trained_floor_removed": self.floor_removed,
            "course": self.course,
        }


def course_start(record: SceneRecord) -> tuple[tuple[float, float], float]:
    """Where the course begins: `START_BEHIND_M` before the first waypoint
    along the line to the second, heading along it; refuses a scene
    that lays out no course (a start is given, never guessed)."""
    points = [
        np.asarray(p, dtype=np.float64) for p in record.course.get("waypoints", [])
    ]
    if not points:
        raise ValueError(
            f"scene {record.name} lays out no course: give the start explicitly"
        )
    first = points[0][:2]
    if len(points) > 1 and np.linalg.norm(points[1][:2] - first) > 0:
        direction = points[1][:2] - first
        direction /= np.linalg.norm(direction)
    else:
        direction = np.array([1.0, 0.0])
    start = first - START_BEHIND_M * direction
    return (float(start[0]), float(start[1])), math.degrees(
        math.atan2(direction[1], direction[0])
    )


def _yaw_quat(yaw_deg: float) -> np.ndarray:
    half = math.radians(yaw_deg) / 2
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)])


def _free_body(spec: Any) -> Any:
    import mujoco  # noqa: PLC0415

    for joint in spec.joints:
        if joint.type == mujoco.mjtJoint.mjJNT_FREE:
            return joint.parent
    raise ValueError("the scene has no floating base (no free joint)")


def _surface_z(model: Any, x: float, y: float) -> float:
    """The terrain's height under (x, y): a ray down from `RAY_FROM_M`
    against the terrain body's geoms only (the robot's colliders share
    the group and must not answer); refuses thin air."""
    import mujoco  # noqa: PLC0415

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    terrain = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, STAGE_TERRAIN_BODY)
    origin = np.array([x, y, RAY_FROM_M])
    down = np.array([0.0, 0.0, -1.0])
    rays = {
        mujoco.mjtGeom.mjGEOM_MESH: mujoco.mj_rayMesh,
        mujoco.mjtGeom.mjGEOM_HFIELD: mujoco.mj_rayHfield,
    }
    hits = [
        rays[model.geom_type[g]](model, data, g, origin, down)
        for g in range(model.ngeom)
        if model.geom_bodyid[g] == terrain and model.geom_type[g] in rays
    ]
    ahead = [d for d in hits if d >= 0]
    if not ahead:
        raise ValueError(f"no terrain under the start ({x:.2f}, {y:.2f})")
    return RAY_FROM_M - float(min(ahead))


def compose(  # noqa: PLR0913 - the stage's own knobs, each named
    scene_xml: str,
    assets: dict[str, bytes],
    scene_dir: Path,
    *,
    scene_stamp: str,
    start_xy: tuple[float, float] | None = None,
    heading_deg: float | None = None,
    perturbation: Perturbation = NOMINAL,
    terrain: str = DEFAULT_TERRAIN,
    floor_geom: str = FLOOR_GEOM,
    keyframe: int = 0,
    mount: HeadMount = FREE_BASE_HEAD,
) -> Stage:
    """The trained scene on the captured one. `scene_xml` and `assets`
    are the deployment's (`deploy.runtime.load_scene` reads the same);
    the start comes from the scene's course unless given; `terrain` is
    one of `scenes.terrain.TERRAINS`; `mount` is where the robot's head
    camera rides (`scenes.heads`; the default sits it on the floating
    base a test scene has)."""
    import mujoco  # noqa: PLC0415

    scene_dir = Path(scene_dir)
    record = load_scene_record(scene_dir / SCENE_FILE)
    build = terrain_builder(terrain)
    if start_xy is None or heading_deg is None:
        course_xy, course_heading = course_start(record)
        start_xy = start_xy if start_xy is not None else course_xy
        heading_deg = heading_deg if heading_deg is not None else course_heading
    friction = floor_friction(record)
    spec = mujoco.MjSpec.from_string(scene_xml, assets=assets)
    floor = next((g for g in spec.geoms if g.name == floor_geom), None)
    if floor is not None:
        spec.delete(floor)
    ground = spec.worldbody.add_body(name=STAGE_TERRAIN_BODY)
    facts = build(spec, ground, scene_dir, friction)
    base = spec.body(mount.body) if mount.body else _free_body(spec)
    head = base.add_camera(name=HEAD_CAMERA, fovy=mount.fovy)
    head.pos[:] = mount.pos
    head.quat[:] = mount.quat
    # the nominal terrain's height under the start, before any perturbation
    surface_z = _surface_z(spec.compile(), *start_xy)
    key = spec.keys[keyframe]
    standing = float(key.qpos[2])  # the keyframe's height over the trained floor at z=0
    start = (start_xy[0], start_xy[1], surface_z + standing)
    for i, value in enumerate([*start, *_yaw_quat(heading_deg)]):
        key.qpos[i] = float(value)  # a keyframe's qpos is a MuJoCo vector: per element
    heading = np.array(
        [math.cos(math.radians(heading_deg)), math.sin(math.radians(heading_deg)), 0.0]
    )
    eye = np.array(start) - COURSE_BEHIND_M * heading + np.array([0, 0, COURSE_ABOVE_M])
    course = spec.worldbody.add_camera(name=COURSE_CAMERA, fovy=COURSE_FOVY)
    course.pos[:] = eye
    course.quat[:] = look_quat(
        np.array(start) + COURSE_AHEAD_M * heading - eye, np.array([0.0, 0.0, 1.0])
    )
    # the perturbation: the terrain turned about the start and shifted
    pivot = np.array([start_xy[0], start_xy[1], 0.0])
    yaw = _yaw_quat(perturbation.yaw_deg)
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, yaw)
    ground.quat[:] = yaw
    ground.pos[:] = (
        pivot - rotation.reshape(3, 3) @ pivot + np.array(perturbation.offset_m)
    )
    spec.compile()
    return Stage(
        xml=spec.to_xml(),
        scene=scene_stamp,
        terrain=facts,
        start=start,
        heading_deg=float(heading_deg),
        surface_z=surface_z,
        friction=friction,
        cameras=(HEAD_CAMERA, COURSE_CAMERA),
        perturbation=perturbation,
        floor_removed=floor is not None,
        course=dict(record.course),
    )


def stage_deployment(  # noqa: PLR0913 - the staging's own knobs, each named
    deployment_dir: Path,
    scene_dir: Path,
    out_dir: Path,
    *,
    assets_dir: Path,
    start_xy: tuple[float, float] | None = None,
    heading_deg: float | None = None,
    perturbation: Perturbation = NOMINAL,
    terrain: str = DEFAULT_TERRAIN,
) -> Stage:
    """A deployment on a scene, as a new deployment folder: the manifest
    with the stage as its scene, the policy copied, nothing else (a
    fresh deployment has no gate). Refuses an `out_dir` that exists."""
    deployment_dir, scene_dir, out_dir = (
        Path(deployment_dir),
        Path(scene_dir),
        Path(out_dir),
    )
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; a staged deployment is written once")
    manifest = load_manifest(deployment_dir)
    assets = {p.name: p.read_bytes() for p in Path(assets_dir).iterdir() if p.is_file()}
    stage = compose(
        manifest.scene_path.read_text(encoding="utf-8"),
        assets,
        scene_dir,
        scene_stamp=stamp(scene_dir.name, scene_dir),
        mount=head_mount(str(manifest.raw[Key.ROBOT])),
        start_xy=start_xy,
        heading_deg=heading_deg,
        perturbation=perturbation,
        terrain=terrain,
    )
    out_dir.mkdir(parents=True)
    (out_dir / STAGE_FILE).write_text(stage.xml, encoding="utf-8")
    shutil.copy2(manifest.policy_path, out_dir / manifest.policy_path.name)
    raw = dict(manifest.raw)
    raw[Key.SCENE] = stage.facts() | {
        "staged_from": {
            "deployment": deployment_dir.name,
            "scene_file": manifest.raw[Key.SCENE].get("file"),
            "terrain": manifest.raw[Key.SCENE].get("terrain"),
        }
    }
    raw[Key.STAMP_OF] = f"{manifest.raw.get(Key.STAMP_OF)} on {scene_dir.name}"
    (out_dir / MANIFEST_FILE).write_text(
        json.dumps(raw, indent=1) + "\n", encoding="utf-8"
    )
    return stage


def scene_name_of(manifest_raw: dict[str, Any]) -> str | None:
    """The scene a staged deployment stands on, by name, from its
    manifest's terrain word (`scene <name>@<hash>`); None for a plane."""
    terrain = str((manifest_raw.get(Key.SCENE) or {}).get("terrain") or "")
    word, _, rest = terrain.partition(" ")
    if word != SCENE_TERRAIN_WORD or not rest:
        return None
    return rest.split("@", 1)[0]
