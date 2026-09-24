"""A deployment in the Studio's own MuJoCo viewport (the Simulator page's
native picture, not the embedded Rerun viewer): the operator's ask of
2026-09-24, after the gate and the pre-flight ran on Unitree's stack,
was to watch the deployed dog "not in the 3d rerun viewer but our
mujoco viewer".

A viewport scene names a deployment and what to show of it:

    deploy:<name>                       live: the exported policy driven by
                                        the plain runtime, the command from
                                        the Commands tab and WASD
    deploy:<name>:gate:<runtime>:<i>    trial i of that runtime's gate
    deploy:<name>:preflight:<i>         segment i of the pre-flight's poses

One truth for every picture: live and a re-run trial go through the
gate's own code (`deploy.gate.run_trial`, `deploy.course.run_course_trial`,
`deploy.ticks.Ticks` over `deploy.runtime.Runtime`), the physics loop's
pump standing where the gate's Studio mirror stands. A trial whose
runtime is another process's simulator (Unitree's, over DDS) cannot run
again in ours: its recorded poses are replayed (`deploy.poses`), no
physics. A mode is a registry entry (`MODES`), never a branch.

`tools/studio-render-stream.py` builds the model from `open_scene` in
both of its processes and hands the physics one's pump to `run`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from rq_pipeline.deploy.manifest import (
    MANIFEST_FILE,
    TWIST_SHORT,
    Manifest,
    load_manifest,
    read_gates,
)
from rq_pipeline.deploy.poses import PoseFile, poses_file, trial_segment
from rq_pipeline.deploy.runtimes import RUNTIMES, runtime_spec

# The viewport's word for a deployment scene: the Studio's picker and its
# control surface spell it the same (`viewport.rs::DEPLOY_PREFIX`, pinned
# by tests/test_studio_mirrors.py).
DEPLOY_PREFIX = "deploy:"
SEPARATOR = ":"
LIVE = "live"
GATE = "gate"
PREFLIGHT = "preflight"
# The index's summary key listing a deployment's viewport scenes as
# [label, scene] pairs (hidden on the card; the drawer reads it).
VIEWPORT_KEY = "viewport"

# Pauses the picture holds before it starts over, so the end of a trial
# can be seen: after a fall in the live scene, after a replayed segment.
FALL_HOLD_S = 1.0
END_HOLD_S = 1.0
# The camera: a Go2-sized robot seen from its front quarter, followed.
CAMERA = {
    "azimuth": 135.0,
    "elevation": -20.0,
    "distance": 1.6,
    "lookat": (0.0, 0.0, 0.3),
    "follow": 1.6,
}

OUTCOME_TRACKED = "tracked"
OUTCOME_FELL = "fell"
OUTCOME_UNTRACKED = "survived, did not track"
RERUN_WORD = "re-run in plain MuJoCo, the gate's own trial"
REPLAY_WORD = "replayed from its recorded poses (no physics)"


class Loop(Protocol):
    """What drives the picture: the render stream's physics pump. `tick`
    publishes the state and paces to real time; `on_command` receives the
    mailbox's words the pump does not know (the twist)."""

    on_command: Callable[[str, int, float], None] | None

    def tick(self, data: Any, pace_seconds: float) -> None: ...


@dataclass(frozen=True)
class DeployScene:
    """A parsed viewport scene: the deployment, the mode, its arguments."""

    name: str
    mode: str
    args: tuple[str, ...] = ()

    def text(self) -> str:
        return DEPLOY_PREFIX + SEPARATOR.join((self.name, *self._tail()))

    def _tail(self) -> tuple[str, ...]:
        return () if self.mode == LIVE else (self.mode, *self.args)


def scene_text(name: str, mode: str = LIVE, *args: object) -> str:
    return DeployScene(name, mode, tuple(str(a) for a in args)).text()


def is_deploy_scene(text: str) -> bool:
    return text.startswith(DEPLOY_PREFIX)


def parse_scene(text: str) -> DeployScene:
    """`deploy:<name>[:<mode>:<args>...]`, refused by name when malformed."""
    if not is_deploy_scene(text):
        raise ValueError(f"not a deployment scene: {text!r} lacks {DEPLOY_PREFIX!r}")
    parts = text[len(DEPLOY_PREFIX) :].split(SEPARATOR)
    name, rest = parts[0], parts[1:]
    if not name:
        raise ValueError(f"a deployment scene names its deployment: {text!r}")
    mode = rest[0] if rest else LIVE
    if mode not in MODES:
        raise ValueError(
            f"{text!r}: no viewport mode {mode!r}; one of {', '.join(MODES)}"
        )
    args = tuple(rest[1:])
    if len(args) != MODES[mode].arity:
        raise ValueError(
            f"{text!r}: mode {mode!r} takes {MODES[mode].arity} argument(s) "
            f"({MODES[mode].shape}), got {len(args)}"
        )
    return DeployScene(name, mode, args)


def _index(text: str, what: str) -> int:
    if not text.isdigit():
        raise ValueError(f"{what} is an index, not {text!r}")
    return int(text)


def clip_twist(manifest: Manifest, twist: np.ndarray) -> np.ndarray:
    """The commanded twist inside the manifest's trained ranges (vx, vy, wz):
    the policy is never asked for what it was never trained on."""
    ranges = twist_ranges(manifest)
    lo = np.array([r[0] for r in ranges])
    hi = np.array([r[1] for r in ranges])
    return np.clip(np.asarray(twist, dtype=np.float64), lo, hi).astype(np.float32)


def twist_ranges(manifest: Manifest) -> list[list[float]]:
    c = manifest.commands
    return [list(c.lin_vel_x), list(c.lin_vel_y), list(c.ang_vel_z)]


def _twist_words(command: Any) -> str:
    return ", ".join(
        f"{axis} {float(v):+.2f}" for axis, v in zip(TWIST_SHORT, command, strict=True)
    )


def outcome_of(row: dict[str, Any]) -> str:
    """A gate trial's outcome in the words the viewport shows."""
    if row.get("success"):
        return OUTCOME_TRACKED
    return OUTCOME_FELL if row.get("fell") else OUTCOME_UNTRACKED


# -- the plan: what a mode shows, and how it runs ---------------------------------


@dataclass
class Plan:
    """A mode's picture: a caption for the viewport's bar, the command
    ranges the Commands tab offers (None: nothing to command), and the
    loop that drives it in the physics process."""

    caption: str
    run: Callable[[Any, Loop, Context], None]
    twist_ranges: list[list[float]] | None = None


@dataclass(frozen=True)
class Context:
    """Everything a mode reads: the scene, the deployment's folder and
    manifest, and the wire words of the stream that hosts it."""

    scene: DeployScene
    folder: Path
    manifest: Manifest
    twist_word: str = "twist"  # the stream's MAILBOX_TWIST
    twist_axes: int = 3  # the stream's TWIST_AXES


@dataclass(frozen=True)
class Mode:
    name: str
    arity: int
    shape: str  # its arguments, as a refusal words them
    plan: Callable[[Context], Plan]


# -- live ------------------------------------------------------------------------------


class _Pumped:
    """The gate's Studio mirror, stood in for by the physics pump: every
    control tick the gate's code would draw becomes one published,
    paced state of the viewport. `trial`, `note` and `course` are the
    mirror's words; the viewport's caption already says them."""

    def __init__(self, runtime: Any, loop: Loop) -> None:
        self.runtime = runtime
        self.loop = loop

    def tick(self, dt: float, _pose: Any, _command: Any, _velocity: Any) -> None:
        self.loop.tick(self.runtime.data, dt)

    def trial(self, _index: int, _command: Any) -> None:
        return None

    def note(self, _text: str) -> None:
        return None

    def course(self, _path: Any) -> None:
        return None


def _hold(runtime: Any, loop: Loop, seconds: float) -> None:
    """The picture held still for `seconds`, still published and paced."""
    for _ in range(max(1, round(seconds / runtime.step_dt))):
        loop.tick(runtime.data, runtime.step_dt)


def _open_runtime(model: Any, manifest: Manifest, *, policy: bool = True) -> Any:
    """The plain runtime on the viewport's own model (the ring was built
    from it): the ONNX session when the policy drives, none to replay."""
    import mujoco  # noqa: PLC0415 - the sim extra

    from rq_pipeline.deploy.runtime import ONNX_PROVIDERS, Runtime  # noqa: PLC0415

    session = None
    if policy:
        import onnxruntime as ort  # noqa: PLC0415 - the deploy extra

        session = ort.InferenceSession(
            str(manifest.policy_path), providers=list(ONNX_PROVIDERS)
        )
    runtime = Runtime(
        manifest=manifest, model=model, data=mujoco.MjData(model), session=session
    )
    runtime.reset()
    return runtime


def _plan_live(ctx: Context) -> Plan:
    return Plan(
        caption=f"{ctx.scene.name} · live: drive it with WASD or the Commands "
        "tab, inside the manifest's trained ranges",
        run=_run_live,
        twist_ranges=twist_ranges(ctx.manifest),
    )


class TwistInput:
    """The human's twist off the stream's mailbox: one slot per axis,
    `arg_i = world * axes + axis` (the walk scene's encoding); world 0 is
    the one robot, another world is not this scene's; -1 hands the
    commands back, and the robot stands still."""

    def __init__(self, word: str, axes: int) -> None:
        self.word = word
        self.axes = axes
        self.value = np.zeros(axes, dtype=np.float32)

    def on_command(self, name: str, arg_i: int, arg_f: float) -> None:
        if name != self.word:
            return
        if arg_i < 0:
            self.value[:] = 0.0
            return
        world, axis = divmod(arg_i, self.axes)
        if world == 0:
            self.value[axis] = arg_f


def _run_live(model: Any, loop: Loop, ctx: Context) -> None:
    """The exported policy through the gate's own tick, forever; the
    command is the human's twist, clipped; a fall holds, then resets."""
    from rq_pipeline.deploy.ticks import Ticks  # noqa: PLC0415

    runtime = _open_runtime(model, ctx.manifest)
    twist = TwistInput(ctx.twist_word, ctx.twist_axes)
    loop.on_command = twist.on_command
    mirror: Any = _Pumped(runtime, loop)
    while True:  # an episode per fall: the gate's own tick, a fresh meter
        runtime.reset()
        meter = Ticks(runtime.step_dt, mirror=mirror)
        while not meter.tick(runtime, clip_twist(ctx.manifest, twist.value)):
            pass
        _hold(runtime, loop, FALL_HOLD_S)


# -- a gate trial ------------------------------------------------------------------


def _gate_trial(ctx: Context) -> tuple[str, int, dict[str, Any], dict[str, Any]]:
    """(runtime, trial index, the record, the trial's row), refused by name."""
    runtime_name, index_text = ctx.scene.args
    runtime_spec(runtime_name)  # an unknown runtime, refused by name
    record = read_gates(ctx.folder).get(runtime_name)
    if record is None:
        raise FileNotFoundError(
            f"{ctx.scene.name}: no {runtime_name} gate has run; gate_deployment first"
        )
    index = _index(index_text, "a trial")
    rows = record.get("records") or []
    if not 0 <= index < len(rows):
        raise ValueError(f"no trial {index}: the {runtime_name} gate ran {len(rows)}")
    return runtime_name, index, record, rows[index]


def _plan_gate(ctx: Context) -> Plan:
    runtime_name, index, record, row = _gate_trial(ctx)
    spec = RUNTIMES[runtime_name]
    how = RERUN_WORD if spec.rerun_in_plain else REPLAY_WORD
    what = (
        _twist_words(row["command"])
        if "command" in row
        else f"along the course at {float(row.get('speed', 0.0)):.2f} m/s"
    )
    caption = (
        f"{ctx.scene.name} · {spec.instrument} gate · trial {index} of "
        f"{len(record.get('records') or [])} · {what} · {outcome_of(row)} · {how}"
    )
    if spec.rerun_in_plain:
        return Plan(caption=caption, run=_run_gate_rerun)
    _poses_or_refuse(ctx, _gate_poses_path(ctx.folder, runtime_name))
    return Plan(caption=caption, run=_run_gate_poses)


def _gate_poses_path(folder: Path, runtime_name: str) -> Path:
    from rq_pipeline.deploy.gate import gate_stream_name  # noqa: PLC0415
    from rq_pipeline.viz import viewer_file  # noqa: PLC0415

    return poses_file(viewer_file(folder, gate_stream_name(runtime_name)))


def _run_gate_rerun(model: Any, loop: Loop, ctx: Context) -> None:
    """The trial again, through the gate's own trial code, forever: the
    held command of the record's row (or the course at its speed)."""
    from rq_pipeline.deploy.course import (  # noqa: PLC0415
        Course,
        Steering,
        run_course_trial,
    )
    from rq_pipeline.deploy.gate import run_trial  # noqa: PLC0415

    _runtime_name, index, _record, row = _gate_trial(ctx)
    runtime = _open_runtime(model, ctx.manifest)
    mirror: Any = _Pumped(runtime, loop)
    course = Course.of_manifest(ctx.manifest)
    while True:
        if "command" in row:
            run_trial(
                ctx.manifest,
                runtime,
                np.asarray(row["command"], dtype=np.float64),
                mirror=mirror,
                index=index,
            )
        elif course is not None:
            run_course_trial(
                ctx.manifest,
                runtime,
                course,
                Steering.of_manifest(ctx.manifest),
                float(row["speed"]),
                mirror=mirror,
                index=index,
            )
        else:
            raise ValueError(
                f"trial {index} names neither a command nor a course speed"
            )
        _hold(runtime, loop, END_HOLD_S)


def _run_gate_poses(model: Any, loop: Loop, ctx: Context) -> None:
    runtime_name, index, _record, _row = _gate_trial(ctx)
    poses = _poses_or_refuse(ctx, _gate_poses_path(ctx.folder, runtime_name))
    segment = poses.segments.index(trial_segment(index))
    replay_forever(model, loop, ctx.manifest, poses, segment)


# -- a pre-flight segment ----------------------------------------------------------


def _preflight_poses_path(folder: Path) -> Path:
    from rq_pipeline.deploy.preflight import STREAM  # noqa: PLC0415
    from rq_pipeline.viz import viewer_file  # noqa: PLC0415

    return poses_file(viewer_file(folder, STREAM))


def _plan_preflight(ctx: Context) -> Plan:
    poses = _poses_or_refuse(ctx, _preflight_poses_path(ctx.folder))
    index = _index(ctx.scene.args[0], "a pre-flight segment")
    if not 0 <= index < len(poses.segments):
        raise ValueError(
            f"no pre-flight segment {index}: its poses hold {len(poses.segments)}"
        )
    return Plan(
        caption=f"{ctx.scene.name} · pre-flight · {poses.segments[index]} · "
        f"{REPLAY_WORD}",
        run=_run_preflight,
    )


def _run_preflight(model: Any, loop: Loop, ctx: Context) -> None:
    poses = _poses_or_refuse(ctx, _preflight_poses_path(ctx.folder))
    replay_forever(model, loop, ctx.manifest, poses, int(ctx.scene.args[0]))


# -- the replay ---------------------------------------------------------------------


def _poses_or_refuse(ctx: Context, path: Path) -> PoseFile:
    poses = PoseFile.load(path)
    order = tuple(ctx.manifest.joints.policy_order)
    if poses.joint_names != order:
        raise ValueError(
            f"{path.name}: its joints are {list(poses.joint_names)}, the manifest's "
            f"policy order is {list(order)}"
        )
    return poses


def replay_forever(
    model: Any, loop: Loop, manifest: Manifest, poses: PoseFile, segment: int
) -> None:
    """A recorded segment, frame by frame at its own dt, set into the
    viewport's model with a forward pass (no physics), then held, again."""
    import mujoco  # noqa: PLC0415

    runtime = _open_runtime(model, manifest, policy=False)
    positions, quats, joints = poses.frames(segment)
    if not len(positions):
        raise ValueError(f"segment {poses.segments[segment]!r} holds no frames")
    while True:
        # the recording's own clock: no physics steps, so time is set here
        for frame, (position, quat, joint) in enumerate(
            zip(positions, quats, joints, strict=True)
        ):
            set_pose(runtime, position, quat, joint)
            runtime.data.time = frame * poses.dt
            mujoco.mj_forward(runtime.model, runtime.data)
            loop.tick(runtime.data, poses.dt)
        for _ in range(max(1, round(END_HOLD_S / poses.dt))):
            loop.tick(runtime.data, poses.dt)


def set_pose(
    runtime: Any, position: np.ndarray, quat: np.ndarray, joints: np.ndarray
) -> None:
    """A recorded pose into the runtime's qpos: base, orientation, joints
    in the policy order (the runtime's own addresses)."""
    base = runtime.base_qpos
    runtime.data.qpos[base : base + 3] = position
    runtime.data.qpos[base + 3 : base + 7] = quat
    runtime.data.qpos[runtime.joint_qpos] = joints
    runtime.data.qvel[:] = 0.0


MODES: dict[str, Mode] = {
    LIVE: Mode(LIVE, 0, "none", _plan_live),
    GATE: Mode(GATE, 2, "<runtime>:<trial>", _plan_gate),
    PREFLIGHT: Mode(PREFLIGHT, 1, "<segment>", _plan_preflight),
}


# -- opening a scene, listing a deployment's scenes ----------------------------------


@dataclass
class ViewportScene:
    """What both of the stream's processes build from a scene's text."""

    context: Context
    model: Any
    plan: Plan

    def run(self, loop: Loop) -> None:
        self.plan.run(self.model, loop, self.context)

    def free_body(self) -> int:
        """The floating base's body: the one robot the camera follows."""
        import mujoco  # noqa: PLC0415

        free = np.flatnonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if free.size == 0:
            raise ValueError("the deployment's scene has no floating base")
        return int(self.model.jnt_bodyid[free[0]])


def deployment_folder(name: str, project_root: Path | None = None) -> Path:
    """The current project's (or `project_root`'s, made current: the
    bundle's meshes are found project-first) deployment folder by name,
    refused by name when it has no manifest."""
    from rq_pipeline.project import current_project  # noqa: PLC0415
    from rq_pipeline.project.locate import DEPLOY_FOLDER, Project  # noqa: PLC0415

    project = Project(Path(project_root)).use() if project_root else current_project()
    folder = project.folder(DEPLOY_FOLDER) / name
    if not (folder / MANIFEST_FILE).is_file():
        raise FileNotFoundError(
            f"no deployment {name!r} with a manifest in {project.root}"
        )
    return folder


def open_scene(
    text: str,
    *,
    project_root: Path | None = None,
    offscreen_side: int | None = None,
    twist_word: str = "twist",
    twist_axes: int = 3,
) -> ViewportScene:
    """The scene's model (the deployment's own scene, the bundle's meshes)
    and its plan; every refusal is raised here, before a frame."""
    from rq_pipeline.deploy.runtime import assets_dir_of, load_scene  # noqa: PLC0415

    scene = parse_scene(text)
    folder = deployment_folder(scene.name, project_root)
    manifest = load_manifest(folder)
    model = load_scene(manifest, assets_dir=assets_dir_of(manifest))
    if offscreen_side is not None:  # raise, never lower, the offscreen budget
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, offscreen_side)
        model.vis.global_.offheight = max(model.vis.global_.offheight, offscreen_side)
    context = Context(
        scene=scene,
        folder=folder,
        manifest=manifest,
        twist_word=twist_word,
        twist_axes=twist_axes,
    )
    return ViewportScene(context, model, MODES[scene.mode].plan(context))


def scenes_of(folder: Path) -> list[list[str]]:
    """Every scene the viewport can show of a deployment, as [label,
    scene]: live first, then each gate trial it can show (re-run, or
    replayed where poses were saved), then each pre-flight segment. What
    the drawer offers; nothing it would refuse."""
    folder = Path(folder)
    name = folder.name
    out = [["▶ live, driven from the Studio", scene_text(name)]]
    for runtime_name, record in read_gates(folder).items():
        spec = RUNTIMES[runtime_name]
        rows = record.get("records") or []
        if not spec.rerun_in_plain:
            try:
                segments = set(
                    PoseFile.load(_gate_poses_path(folder, runtime_name)).segments
                )
            except (FileNotFoundError, ValueError):
                continue  # no poses saved: nothing to replay
        for i, row in enumerate(rows):
            if not spec.rerun_in_plain and trial_segment(i) not in segments:
                continue
            out.append(
                [
                    f"{spec.instrument} gate · trial {i} · {outcome_of(row)}",
                    scene_text(name, GATE, runtime_name, i),
                ]
            )
    try:
        poses = PoseFile.load(_preflight_poses_path(folder))
    except (FileNotFoundError, ValueError):
        return out
    out.extend(
        [f"pre-flight · {segment}", scene_text(name, PREFLIGHT, i)]
        for i, segment in enumerate(poses.segments)
    )
    return out
