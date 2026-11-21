"""The walk families: locomotion environments a project declares by a
spec and trainnr_mjlab builds in its own venv (docs/77 §2).

A walk is not a `Task` (no scripted expert, no manipulation protocol);
it is the knobs of mjlab's velocity task the loop cares about, named by
their content: the actuator randomization span around the identified
or declared basis, the terrain, the episode length, the paired trials
an evaluation runs. `build_*` returns a `WalkTask` — the spec, its
stamp, the robot's bundle when the project holds one — and compiles the
bundle's model as the honesty check that the robot exists. The
learnability smoke (two environments, two iterations, in trainnr_mjlab) is
the walk's acceptance, run by `tools/accept-task.py`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco

from trainnr.bundles.bundle import model_file_of
from trainnr.bundles.fetch import missing_files
from trainnr.bundles.hashing import STAMP_SEPARATOR, content_stamp, stamp
from trainnr.bundles.locate import find_bundle
from trainnr.tasks.registry import register, resolve, walk_entries

TERRAINS = ("flat",)  # rough terrain arrives with its own sensors (docs/77)
DEFAULT_SPAN = 0.10  # the walk study's middle arm
DEFAULT_EPISODE_S = 20.0  # mjlab's velocity task
DEFAULT_TRIALS = 40  # the walk verdict's paired trials


@dataclass(frozen=True)
class WalkSpec:
    """A walk's knobs. `dr_span` is the declared randomization span
    around the actuator basis (0 = the basis exactly); `terrain` is
    flat until the rough variant's sensors are wired; `episode_s` the
    episode length; `trials` the paired trials an evaluation runs."""

    dr_span: float = DEFAULT_SPAN
    terrain: str = "flat"
    episode_s: float = DEFAULT_EPISODE_S
    trials: int = DEFAULT_TRIALS

    def __post_init__(self) -> None:
        if self.terrain not in TERRAINS:
            raise ValueError(
                f"terrain is one of {', '.join(TERRAINS)}, got {self.terrain!r}"
            )
        if not 0.0 <= self.dr_span < 1.0:
            raise ValueError(f"dr_span is a fraction in [0, 1), got {self.dr_span}")
        if self.trials < 1 or self.episode_s <= 0:
            raise ValueError("trials must be >= 1 and episode_s > 0")


DEFAULT_WALK = WalkSpec()


@dataclass(frozen=True)
class WalkTask:
    """What a declared walk is, for the doors and the readers: its
    name, the robot it walks (the walk registry's name and the bundle
    when the project holds one), its spec and stamp, the compiled
    model's spec for the scene card."""

    name: str
    robot: str
    task_spec: WalkSpec
    bundle_dir: Path | None
    spec: Any  # an MjSpec of the bundle's model, or None for a simulator asset
    instruction: str = "track the commanded velocity without falling"
    # mjlab's velocity task: physics at 0.005 s, one action per 4 steps.
    control_hz: int = 50

    @property
    def stamp(self) -> str:
        # The robot is part of the content: with the spec's knobs alone,
        # go1-walk and go2-walk at their defaults shared one hash
        # (fresh-install audit, 2026-10-08). A task already declared
        # keeps the stamp its task.json recorded.
        return content_stamp(self.name, {"robot": self.robot, **asdict(self.task_spec)})


def _walk(name: str, robot: str, spec: WalkSpec, *, bundle: str | None) -> WalkTask:
    bundle_dir = find_bundle(bundle) if bundle else None
    mjspec = None
    if bundle:
        if bundle_dir is None:
            raise FileNotFoundError(
                f"no bundle {bundle!r} in the project or the library: onboard the "
                f"robot (onboard_robot) before declaring its walk"
            )
        model_file = model_file_of(bundle_dir)
        if model_file is None:
            raise FileNotFoundError(f"{bundle_dir}: no model file recorded to compile")
        if not missing_files(bundle_dir):  # fetched by training, not by a describe
            mjspec = mujoco.MjSpec.from_file(str(model_file))
            mjspec.compile()  # the honesty check: the robot exists and builds
    return WalkTask(
        name=name, robot=robot, task_spec=spec, bundle_dir=bundle_dir, spec=mjspec
    )


@register("go2-walk", rig="go2", walk=True)
def build_go2_walk(*, spec: WalkSpec = DEFAULT_WALK) -> WalkTask:
    """The Unitree Go2 walk on the project's onboarded bundle."""
    return _walk("go2-walk", "go2", spec, bundle="go2")


@register("microduck-walk", rig="microduck", walk=True)
def build_microduck_walk(*, spec: WalkSpec = DEFAULT_WALK) -> WalkTask:
    """The microduck walk on its certified bundle."""
    return _walk("microduck-walk", "microduck", spec, bundle="microduck")


@register("go1-walk", rig="go1", walk=True)
def build_go1_walk(*, spec: WalkSpec = DEFAULT_WALK) -> WalkTask:
    """mjlab's own Go1 walk (the asset ships with the simulator)."""
    return _walk("go1-walk", "go1", spec, bundle=None)


@dataclass(frozen=True)
class WalkShape:
    """What the trainer reads off a walk robot BY NAME. The source is the
    mjlab task in trainnr-mjlab (`trainnr_mjlab/go2_walk.py`: FOOT_GEOMS,
    FOOT_SITES, the `imu` site, TRUNK); it is spelled here, on the trainnr
    side of the layer rule, so acceptance can refuse a model in the wrong
    shape in one sentence before the learnability smoke, instead of
    mjlab's "Not all regular expressions are matched" traceback in the
    log (a Menagerie Go2, stranger test 2026-10-03)."""

    foot_geoms: tuple[str, ...]
    foot_sites: tuple[str, ...]
    imu_site: str
    trunk: str
    reference: str  # where a model in this shape comes from


GO2_LEGS = ("FR", "FL", "RR", "RL")
WALK_SHAPES: dict[str, WalkShape] = {
    "go2": WalkShape(
        foot_geoms=tuple(f"{leg}_foot_collision" for leg in GO2_LEGS),
        foot_sites=GO2_LEGS,
        imu_site="imu",
        trunk="base_link",
        reference=(
            "Unitree's own go2.xml (unitree_rl_mjlab, "
            "src/assets/robots/unitree_go2/xmls/); Menagerie's go2.xml is the "
            "viewer's twin: unnamed collision geoms, no foot sites, trunk `base`"
        ),
    ),
}


def walk_shape_missing(robot: str, model: mujoco.MjModel) -> list[str]:
    """What `model` lacks of the shape the walk family of `robot` reads by
    name — each as `<kind> <name>` — or [] when it has it all, or when no
    shape is registered for that robot (microduck, go1: the trainer's own
    assets)."""
    shape = WALK_SHAPES.get(robot)
    if shape is None:
        return []
    wanted = (
        *((mujoco.mjtObj.mjOBJ_GEOM, "geom", g) for g in shape.foot_geoms),
        *((mujoco.mjtObj.mjOBJ_SITE, "site", s) for s in shape.foot_sites),
        (mujoco.mjtObj.mjOBJ_SITE, "site", shape.imu_site),
        (mujoco.mjtObj.mjOBJ_BODY, "body", shape.trunk),
    )
    return [
        f"{kind} {name}"
        for obj, kind, name in wanted
        if mujoco.mj_name2id(model, obj, name) < 0
    ]


def walk_shape_sentence(robot: str, missing: list[str]) -> str:
    """The refusal, in one sentence: what is missing and where the right
    model is."""
    shape = WALK_SHAPES[robot]
    return (
        f"the robot bundle is not in the shape the {robot} walk trains on: "
        f"missing {', '.join(missing)}; use {shape.reference}"
    )


def bundle_walk_shape_missing(robot: str) -> list[str]:
    """`walk_shape_missing` for the bundle the project (or the library)
    holds under `robot`, compiled; [] when there is no bundle to check
    (the walk door will say so itself)."""
    bundle_dir = find_bundle(robot)
    model_file = model_file_of(bundle_dir) if bundle_dir else None
    if model_file is None or (bundle_dir and missing_files(bundle_dir)):
        return []
    return walk_shape_missing(robot, mujoco.MjModel.from_xml_path(str(model_file)))


def walk_robot(task_id: str) -> str | None:
    """The walk robot of a walk family (its registered rig), else None
    for a family that is not a walk."""
    entry = resolve(task_id)
    return entry.rig if entry.walk else None


def walk_robots() -> tuple[str, ...]:
    """Every robot a walk family is registered for, sorted: what the
    walk doors accept by name (one truth: the task registry)."""
    return tuple(sorted({e.rig for e in walk_entries().values()}))


# -- a walk for any onboarded legged robot (#100) -------------------------

LEGGED_FAMILY = "legged-walk"
LEGGED_RIG = "legged"  # the trainer's walk for a declared spec (trainnr_mjlab)
MODEL_ACTUATORS = "model"  # the model's own actuators, declared, not identified
# The keyframe a robot stands in, by name (any case), before the first.
STAND_KEYFRAMES = ("home", "stand")
# The sensors mjlab's velocity task reads off the robot (the IMU's).
IMU_SENSORS = ("imu_lin_vel", "imu_ang_vel")
FOOT_WORD = "foot"
PAIR = 2  # a [low, high] range


@dataclass(frozen=True)
class LeggedWalkSpec:
    """A walk on any onboarded legged robot. `robot` (a bundle onboarded
    into the project, by name) and `actuator` are required; the rest is
    read off the model when left empty: the trunk is the body of its free
    joint, the feet its sites and collision geoms named `foot`, the stand
    pose its `home` or `stand` keyframe (else the first). `actuator` is an
    actuator bundle (a certified one from the library by name, e.g.
    `xl330.m6`, or a file in the project, e.g. one `identify_system`
    wrote), or `model` for the model's own actuators, declared and so
    recorded. `pinned_joints` (names or patterns) are held at the stand
    pose and left out of the action; `reset_z` is the trunk height's
    spread above the stand pose at reset; the velocity commands default
    to microduck's (`lin_vel_x`, `lin_vel_y`, `ang_vel_z`). `limits` says
    what breaking the robot's limits over several joints does in training,
    when its bundle declares them (`limits.json`, #121): `terminate` (the
    default) ends the episode, `cost` pays for it, `off` ignores them."""

    robot: str = ""
    actuator: str = ""
    stand_keyframe: str = ""
    trunk: str = ""
    foot_sites: tuple[str, ...] = ()
    foot_geoms: tuple[str, ...] = ()
    pinned_joints: tuple[str, ...] = ()
    reset_z: tuple[float, ...] = ()
    lin_vel_x: tuple[float, ...] = ()
    lin_vel_y: tuple[float, ...] = ()
    ang_vel_z: tuple[float, ...] = ()
    limits: str = ""
    dr_span: float = DEFAULT_SPAN
    terrain: str = "flat"
    episode_s: float = DEFAULT_EPISODE_S
    trials: int = DEFAULT_TRIALS

    def __post_init__(self) -> None:
        WalkSpec(self.dr_span, self.terrain, self.episode_s, self.trials)
        for name in ("reset_z", "lin_vel_x", "lin_vel_y", "ang_vel_z"):
            value = getattr(self, name)
            if value and (len(value) != PAIR or value[0] > value[1]):
                raise ValueError(f"{name} is a [low, high] pair, got {list(value)}")


DEFAULT_LEGGED = LeggedWalkSpec()


@dataclass(frozen=True)
class LeggedWalk:
    """A declared legged walk, its spec and what the model gave it."""

    name: str
    robot: str
    task_spec: LeggedWalkSpec
    bundle_dir: Path | None
    spec: Any
    resolved: dict[str, Any]
    instruction: str = "track the commanded velocity without falling"
    control_hz: int = 50

    @property
    def stamp(self) -> str:
        # the robot's version is content: a revision of it is another walk
        return content_stamp(self.name, {**asdict(self.task_spec), "robot": self.robot})


def _name(model: mujoco.MjModel, obj: Any, i: int) -> str:
    return mujoco.mj_id2name(model, obj, i) or ""


def resolve_legged(spec: LeggedWalkSpec) -> tuple[dict[str, Any], Path, Any]:
    """The spec with everything the model can say filled in, the bundle's
    folder and the compiled model; refuses by name a robot that is not
    onboarded, an actuator that is not there, and any part of the walk's
    shape the model lacks (a field to declare, or a model to fix)."""
    if not spec.robot:
        raise ValueError(
            f"{LEGGED_FAMILY} needs a robot: the name of a bundle onboarded into "
            "the project (onboard_robot)"
        )
    if not spec.actuator:
        raise ValueError(
            f"{LEGGED_FAMILY} needs an actuator: an actuator bundle (a library one "
            "by name, e.g. xl330.m6, or a file in the project) or 'model' for the "
            "model's own actuators, declared and not identified"
        )
    name = spec.robot.split(STAMP_SEPARATOR, 1)[0]
    bundle_dir = find_bundle(name)
    if bundle_dir is None:
        raise FileNotFoundError(
            f"no robot {name!r} in the project or the library: onboard it first "
            "(onboard_robot)"
        )
    model_file = model_file_of(bundle_dir)
    if model_file is None:
        raise FileNotFoundError(f"{bundle_dir}: no model file recorded to compile")
    model = mujoco.MjModel.from_xml_path(str(model_file))
    obj = mujoco.mjtObj
    free = [
        j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
    ]
    if len(free) != 1:
        raise ValueError(
            f"{name} has no floating base (one free joint): a walk needs one"
        )
    trunk = spec.trunk or _name(model, obj.mjOBJ_BODY, int(model.jnt_bodyid[free[0]]))
    sites = [_name(model, obj.mjOBJ_SITE, i) for i in range(model.nsite)]
    geoms = [
        _name(model, obj.mjOBJ_GEOM, i)
        for i in range(model.ngeom)
        if model.geom_contype[i] or model.geom_conaffinity[i]
    ]
    foot_sites = list(spec.foot_sites) or [s for s in sites if FOOT_WORD in s.lower()]
    foot_geoms = list(spec.foot_geoms) or [g for g in geoms if FOOT_WORD in g.lower()]
    keys = [_name(model, obj.mjOBJ_KEY, k) for k in range(model.nkey)]
    stand = spec.stand_keyframe or next(
        (k for want in STAND_KEYFRAMES for k in keys if k.lower() == want),
        keys[0] if keys else "",
    )
    missing = [f"site {s}" for s in foot_sites if s not in sites]
    missing += [
        f"geom {g}"
        for g in foot_geoms
        if mujoco.mj_name2id(model, obj.mjOBJ_GEOM, g) < 0
    ]
    missing += [
        f"sensor {s}"
        for s in IMU_SENSORS
        if mujoco.mj_name2id(model, obj.mjOBJ_SENSOR, s) < 0
    ]
    if mujoco.mj_name2id(model, obj.mjOBJ_BODY, trunk) < 0:
        missing.append(f"body {trunk}")
    if stand and stand not in keys:
        missing.append(f"keyframe {stand}")
    if not foot_sites:
        missing.append("foot sites (none named 'foot'; declare foot_sites)")
    if not foot_geoms:
        missing.append("foot collision geoms (none named 'foot'; declare foot_geoms)")
    if missing:
        imu = any(m.startswith("sensor ") for m in missing)
        hint = (
            ". The IMU sensors are a velocimeter and a gyro on the trunk's IMU "
            f"site, named {' and '.join(IMU_SENSORS)}"
            if imu
            else ""
        )
        raise ValueError(
            f"{name} is not in a walk's shape: missing {', '.join(missing)}{hint}"
        )
    if stand:
        _require_stand_in_range(model, keys.index(stand), name, stand)
    actuator = _actuator_of(spec.actuator, bundle_dir, model, name)
    limits = _limits_of(spec.limits, bundle_dir, model, stand, name)
    resolved = {
        "robot": stamp(name, bundle_dir),
        "bundle_dir": str(bundle_dir),
        "model_file": model_file.name,
        "actuator": actuator,
        "stand_keyframe": stand or None,
        "trunk": trunk,
        "foot_sites": foot_sites,
        "foot_geoms": foot_geoms,
        "pinned_joints": list(spec.pinned_joints),
        "reset_z": list(spec.reset_z) or None,
        "command": {
            k: list(v)
            for k in ("lin_vel_x", "lin_vel_y", "ang_vel_z")
            if (v := getattr(spec, k))
        }
        or None,
        "dr_span": spec.dr_span,
        "limits": limits,
    }
    return resolved, bundle_dir, model


LIMITS_MODES = ("terminate", "cost", "off")


def _limits_of(
    mode: str, bundle_dir: Path, model: mujoco.MjModel, stand: str, robot: str
) -> dict[str, str] | None:
    """The robot's declared limits over several joints and what breaking
    one does in training; None when it declares none or the walk turns
    them off. Refused by name: a mode it does not know, limits on joints
    the model lacks, a stand pose that breaks one."""
    from trainnr.robot.joint_limits import (  # noqa: PLC0415
        LIMITS_FILE,
        JointLimits,
        check_keyframes,
    )

    given, mode = mode, mode or LIMITS_MODES[0]
    if mode not in LIMITS_MODES:
        raise ValueError(f"limits is one of {', '.join(LIMITS_MODES)}, got {mode!r}")
    path = bundle_dir / LIMITS_FILE
    if not path.is_file():
        if given and given != "off":  # asked for, so never ignored
            raise ValueError(
                f"{robot} declares no limits ({LIMITS_FILE} in its bundle)"
            )
        return None
    if mode == "off":
        return None
    limits = JointLimits.load(path)
    try:
        broken = check_keyframes(limits, model)
    except KeyError as why:
        raise ValueError(f"{robot}'s {LIMITS_FILE}: {str(why).strip(chr(39))}") from why
    if stand and stand in broken:
        raise ValueError(
            f"{robot}'s keyframe {stand!r} breaks its limits: "
            f"{', '.join(broken[stand])}; fix the keyframe or name another"
        )
    return {"file": str(path), "mode": mode}


def _actuator_of(
    actuator: str, bundle_dir: Path, model: mujoco.MjModel, robot: str
) -> str:
    """The actuator a legged walk names, resolved: MODEL_ACTUATORS, or the
    verified bundle file (the library's by name, else one in the project)."""
    from trainnr.robot.actuator_bundle import BUNDLE_STORE, read_bundle  # noqa: PLC0415

    if actuator == MODEL_ACTUATORS:
        if model.nu == 0:
            raise ValueError(f"{robot}'s model has no actuators, so 'model' has none")
        return actuator
    library = BUNDLE_STORE / f"{actuator}.bundle.json"
    found = library if library.is_file() else _beside_robot(bundle_dir, actuator)
    if found is None:
        known = sorted(
            f.name.removesuffix(".bundle.json")
            for f in BUNDLE_STORE.glob("*.bundle.json")
        )
        raise FileNotFoundError(
            f"no actuator bundle {actuator!r}: a library one ({', '.join(known[:6])}, "
            "...), a file in the project, or 'model'"
        )
    read_bundle(found)  # verified: a tampered or foreign file is refused
    return str(found)


def _require_stand_in_range(
    model: mujoco.MjModel, k: int, robot: str, key: str
) -> None:
    """A stand pose outside a joint's range is refused by name: the
    policy's default pose and every reset would start out of bounds."""
    outside = []
    for j in range(model.njnt):
        if not model.jnt_limited[j] or model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        value = float(model.key_qpos[k][model.jnt_qposadr[j]])
        lo, hi = model.jnt_range[j]
        if not lo - 1e-6 <= value <= hi + 1e-6:
            outside.append(f"{_name(model, mujoco.mjtObj.mjOBJ_JOINT, j)} {value:.3f}")
    if outside:
        raise ValueError(
            f"{robot}'s keyframe {key!r} is outside its joints' ranges: "
            f"{', '.join(outside)}; fix the keyframe or name another (stand_keyframe)"
        )


def _beside_robot(bundle_dir: Path, path: str) -> Path | None:
    """A file named relative to the project that holds the robot (its
    bundle is `<project>/robots/<name>`), inside that project."""
    root = bundle_dir.resolve().parent.parent
    found = (root / path).resolve()
    return found if found.is_file() and found.is_relative_to(root) else None


@register(LEGGED_FAMILY, rig=LEGGED_RIG, walk=True)
def build_legged_walk(*, spec: LeggedWalkSpec = DEFAULT_LEGGED) -> LeggedWalk:
    """A walk on any legged robot onboarded into the project: its spec
    names the robot and the actuator, and the rest is read off its model."""
    resolved, bundle_dir, _ = resolve_legged(spec)
    mjspec = mujoco.MjSpec.from_file(str(bundle_dir / resolved["model_file"]))
    mjspec.compile()  # the honesty check: the robot exists and builds
    return LeggedWalk(
        name=LEGGED_FAMILY,
        robot=resolved["robot"],
        task_spec=spec,
        bundle_dir=bundle_dir,
        spec=mjspec,
        resolved=resolved,
    )


def is_legged(task_id: str) -> bool:
    return task_id.split("/", 1)[-1] == LEGGED_FAMILY


LEGGED_SPEC_FILE = "legged-spec.json"


def write_legged_spec(
    task_folder: Path, overlay: dict[str, Any], task_stamp: str
) -> Path:
    """The declared legged walk resolved against the robot as it is now
    (its current revision, the model's own names), written beside the
    task for the trainer (`trainnr_mjlab.walks.LEGGED`)."""
    import json  # noqa: PLC0415

    from trainnr.tasks.overlay import build_variant  # noqa: PLC0415

    task, _ = build_variant(LEGGED_FAMILY, overlay)
    out = Path(task_folder) / LEGGED_SPEC_FILE
    out.write_text(json.dumps({"name": task_stamp, **task.resolved}, indent=1) + "\n")
    return out
