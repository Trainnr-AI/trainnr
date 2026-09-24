"""A manifest-driven runtime in plain MuJoCo: load the deployment's
scene with the bundle's meshes, assemble the observation vector term by
term from the manifest, run the ONNX policy, map its actions to
actuator controls, step at the manifest's control rate. Nothing from
the training stack is imported — that is the point: whatever this
runtime cannot get from the manifest, a robot's runtime could not
either.

The observation terms are the velocity task's, computed the way mjlab
computes them (`mjlab/envs/mdp/observations.py`, read 2026-09-11):
the IMU's velocimeter and gyro from the model's sensors, gravity
projected into the base frame, joint positions relative to the home
pose, joint velocities, the last action, the commanded twist, and the
reference's gait clock.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from rq_pipeline.deploy.manifest import (
    SOURCE_COMMAND_TWIST,
    SOURCE_GAIT_PHASE,
    SOURCE_JOINT_POS_REL,
    SOURCE_JOINT_VEL_REL,
    SOURCE_LAST_ACTION,
    SOURCE_PROJECTED_GRAVITY,
    Manifest,
    Observation,
)

if TYPE_CHECKING:
    import onnxruntime as ort

FREE_JOINT_QPOS = 7
FREE_JOINT_QVEL = 6
TIMESTEP_TOLERANCE_S = 1e-9
GRAVITY_DOWN = np.array([0.0, 0.0, -1.0])
STANDING_COMMAND = 0.1  # the reference's threshold: no gait clock below it
SENSOR_PREFIX = "sensor "
ONNX_PROVIDERS = ("CPUExecutionProvider",)


@dataclass
class Runtime:
    """One robot, one policy, driven by the manifest."""

    manifest: Manifest
    model: mujoco.MjModel
    data: mujoco.MjData
    session: ort.InferenceSession
    command: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    command_limit: float | None = None  # the manifest's ranges are the only bound
    last_action: np.ndarray = field(init=False)
    default_pos: np.ndarray = field(init=False)
    scale: np.ndarray = field(init=False)
    action_to_ctrl: tuple[int, ...] = field(init=False)
    joint_qpos: np.ndarray = field(init=False)
    joint_qvel: np.ndarray = field(init=False)
    sensors: dict[str, tuple[int, int]] = field(init=False)
    ticks: int = field(init=False, default=0)  # control steps since reset

    def __post_init__(self) -> None:
        joints = self.manifest.joints
        n = len(joints.policy_order)
        self.last_action = np.zeros(n, dtype=np.float32)
        self.default_pos = np.asarray(joints.default_pos, dtype=np.float64)
        self.scale = np.asarray(self.manifest.action.scale, dtype=np.float64)
        self.action_to_ctrl = joints.action_to_ctrl
        # Where each policy-order joint lives in qpos/qvel (after the free joint).
        self.joint_qpos = np.array(
            [
                self.model.jnt_qposadr[self.model.joint(j).id]
                for j in joints.policy_order
            ]
        )
        self.joint_qvel = np.array(
            [self.model.jnt_dofadr[self.model.joint(j).id] for j in joints.policy_order]
        )
        self.sensors = {}
        for i in range(self.model.nsensor):
            adr, dim = int(self.model.sensor_adr[i]), int(self.model.sensor_dim[i])
            self.sensors[self.model.sensor(i).name] = (adr, dim)

    @property
    def instrument(self) -> str:
        return f"mujoco-{mujoco.__version__}"

    @property
    def step_dt(self) -> float:
        return self.manifest.control.step_dt

    @property
    def decimation(self) -> int:
        return self.manifest.control.decimation

    @property
    def quat(self) -> np.ndarray:
        """The base's orientation, w x y z of the free joint."""
        return self.data.qpos[self.base_qpos + 3 : self.base_qpos + FREE_JOINT_QPOS]

    def pose(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Where the robot is, for the Studio's mirror: base position,
        base quaternion (w x y z), joints in the policy order."""
        return (
            self.base_position(),
            self.quat.copy(),
            self.data.qpos[self.joint_qpos].copy(),
        )

    def reset(self, *, keyframe: int = 0) -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.data, keyframe)
        mujoco.mj_forward(self.model, self.data)
        self.last_action[:] = 0.0
        self.ticks = 0

    def observe(self) -> np.ndarray:
        parts = [self._term(t) for t in self.manifest.observations]
        return np.concatenate(parts).astype(np.float32)

    def _term(self, term: Observation) -> np.ndarray:
        source = term.source
        if source.startswith(SENSOR_PREFIX):
            adr, dim = self.sensors[source[len(SENSOR_PREFIX) :].split("/", 1)[-1]]
            value = self.data.sensordata[adr : adr + dim].copy()
        else:
            try:
                compute = SOURCES[source]
            except KeyError as unknown:  # the manifest loader refuses these first
                raise ValueError(
                    f"cannot compute observation source {source!r}"
                ) from unknown
            value = compute(self, term)
        value = np.asarray(value, dtype=np.float64) * np.asarray(
            term.scale, dtype=np.float64
        )
        if term.clip is not None:
            value = np.clip(value, term.clip[0], term.clip[1])
        if int(value.size) != term.width:
            raise ValueError(
                f"{term.name}: width {value.size}, manifest says {term.width}"
            )
        return value

    def act(self, obs: np.ndarray) -> np.ndarray:
        name = self.session.get_inputs()[0].name
        out = self.session.run(None, {name: obs[None, :]})[0][0]
        action = np.asarray(out, dtype=np.float32)
        clip = self.manifest.action.clip
        return action if clip is None else np.clip(action, clip[0], clip[1])

    def target_of(self, action: np.ndarray) -> np.ndarray:
        """The joint targets an action asks for: the manifest's home pose
        plus its scale times the action (one spelling; the guards and the
        pre-flight's recorder read it)."""
        return self.default_pos + self.scale * np.asarray(action, dtype=np.float64)

    def step(self, target: np.ndarray, action: np.ndarray) -> None:
        """One control tick at `target`: the controls written, `decimation`
        physics steps, the action remembered as the policy's last."""
        for i, ctrl in enumerate(self.action_to_ctrl):
            self.data.ctrl[ctrl] = target[i]
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)
        self.last_action = np.asarray(action, dtype=np.float32)
        self.ticks += 1

    def apply(self, action: np.ndarray) -> None:
        """One control tick: targets from the action, `decimation` physics steps."""
        self.step(self.target_of(action), action)

    @cached_property
    def base_qpos(self) -> int:
        """Where the floating base starts in qpos (its free joint's
        address; the free joint need not come first)."""
        return int(self.model.jnt_qposadr[self._free_joint()])

    @cached_property
    def base_qvel(self) -> int:
        return int(self.model.jnt_dofadr[self._free_joint()])

    def _free_joint(self) -> int:
        free = np.flatnonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if free.size == 0:
            raise ValueError("the scene has no floating base (no free joint)")
        return int(free[0])

    def base_position(self) -> np.ndarray:
        return self.data.qpos[self.base_qpos : self.base_qpos + 3].copy()

    def set_base_height(self, z: float) -> None:
        self.data.qpos[self.base_qpos + 2] = float(z)

    def base_linear_velocity_w(self) -> np.ndarray:
        return self.data.qvel[self.base_qvel : self.base_qvel + 3].copy()

    def base_angular_velocity(self) -> np.ndarray:
        """The free joint's angular velocity (its body frame, MuJoCo's own)."""
        return self.data.qvel[self.base_qvel + 3 : self.base_qvel + 6].copy()

    def base_velocity_b(self) -> np.ndarray:
        """The base's linear velocity in its own frame (what the verdict
        compares to the command)."""
        return rotate_inverse(self.quat, self.base_linear_velocity_w())

    def contact_points(self) -> np.ndarray | None:
        """The active contacts between the robot's bodies and the rest
        of the world, as positions; the robot is the floating base's
        kinematic tree."""
        model, data = self.model, self.data
        # the robot's tree is the free joint's, wherever the spec put it
        free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if free.size == 0:
            raise ValueError("the scene has no floating base (no free joint)")
        root = int(model.body_rootid[model.jnt_bodyid[free[0]]])
        n = data.ncon
        roots = model.body_rootid[model.geom_bodyid]
        one = roots[data.contact.geom1[:n]] == root
        two = roots[data.contact.geom2[:n]] == root
        return np.asarray(data.contact.pos[:n], dtype=np.float64)[one != two].reshape(
            -1, 3
        )

    def fell_over(self) -> bool:
        return fell_over(self.quat, self.manifest.termination.fell_over_deg)


# How the runtime computes each named observation source: one table the
# manifest's loader mirrors (`manifest.KNOWN_SOURCES`, pinned by a test),
# extended by registration, never by a longer chain.
Source = Callable[[Runtime, Observation], np.ndarray]
SOURCES: dict[str, Source] = {
    SOURCE_PROJECTED_GRAVITY: lambda rt, _t: rotate_inverse(rt.quat, GRAVITY_DOWN),
    SOURCE_JOINT_POS_REL: lambda rt, _t: rt.data.qpos[rt.joint_qpos] - rt.default_pos,
    SOURCE_JOINT_VEL_REL: lambda rt, _t: rt.data.qvel[rt.joint_qvel],
    SOURCE_LAST_ACTION: lambda rt, _t: rt.last_action.astype(np.float64),
    SOURCE_COMMAND_TWIST: lambda rt, _t: rt.command.astype(np.float64),
    SOURCE_GAIT_PHASE: lambda rt, t: gait_phase(
        rt.ticks, rt.step_dt, float(t.params["period"]), rt.command
    ),
}


def register_source(name: str, compute: Source) -> None:
    """A third party's observation source, once; the manifest loader must
    know the name too (`manifest.KNOWN_SOURCES`)."""
    if name in SOURCES:
        raise ValueError(f"observation source {name!r} is already registered")
    SOURCES[name] = compute


def gait_phase(
    ticks: int, step_dt: float, period: float, command: np.ndarray
) -> np.ndarray:
    """The reference's gait clock (unitree_rl_mjlab `mdp.phase`, and
    `gait_phase` in their deploy runtime): sine and cosine of the time
    since reset modulo `period`, zero while the command is below the
    standing threshold. `ticks` counts control steps since reset."""
    if float(np.linalg.norm(command)) < STANDING_COMMAND:
        return np.zeros(2)
    phase = (ticks * step_dt) % period / period
    return np.array([np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)])


def rotate_inverse(quat_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a world vector into the frame of the quaternion (w, x, y, z)."""
    out = np.zeros(3)
    conj = np.array([quat_wxyz[0], -quat_wxyz[1], -quat_wxyz[2], -quat_wxyz[3]])
    mujoco.mju_rotVecQuat(out, np.asarray(v, dtype=np.float64), conj)
    return out


def fell_over(quat_wxyz: np.ndarray, limit_deg: float | None) -> bool:
    """The evaluation's fall rule: the angle between the body's down and
    gravity (acos(-g_b·z)) past the limit; never, when none is declared."""
    if limit_deg is None:
        return False
    gravity_b = rotate_inverse(np.asarray(quat_wxyz, dtype=np.float64), GRAVITY_DOWN)
    cos = float(np.clip(-gravity_b[2], -1.0, 1.0))
    return bool(np.degrees(np.arccos(cos)) > float(limit_deg))


def assets_dir_of(manifest: Manifest) -> Path:
    """The meshes the manifest's scene names: the cited robot bundle's
    `assets/`, found project-first (`bundles.locate`)."""
    from rq_pipeline.bundles.locate import find_bundle  # noqa: PLC0415

    robot = manifest.robot_name
    bundle = find_bundle(robot) if robot else None
    if bundle is None:
        raise FileNotFoundError(
            f"no bundle {robot!r} in the project or the library for the meshes"
        )
    return bundle / "assets"


def load_scene(
    manifest: Manifest, *, assets_dir: Path, dressed: bool = False
) -> mujoco.MjModel:
    """The trained scene as a plain MuJoCo model, its timestep checked
    against the manifest's. `dressed`: the Studio's visual dressing (sky,
    a key light, the checker floor; `tasks.scene.dress`) for a model a
    person watches; physics identical either way, the gate never dresses."""
    assets = {p.name: p.read_bytes() for p in Path(assets_dir).iterdir() if p.is_file()}
    xml = manifest.scene_path.read_text(encoding="utf-8")
    if dressed:
        from rq_pipeline.tasks.scene import dress  # noqa: PLC0415

        spec = mujoco.MjSpec.from_string(xml, assets=assets)
        dress(spec)
        model = spec.compile()
    else:
        model = mujoco.MjModel.from_xml_string(xml, assets=assets)
    expected_dt = manifest.control.physics_timestep_s
    if abs(model.opt.timestep - expected_dt) > TIMESTEP_TOLERANCE_S:
        raise ValueError(
            f"scene timestep {model.opt.timestep} differs from the manifest's "
            f"{expected_dt}"
        )
    return model


def open_runtime(manifest: Manifest, *, assets_dir: Path) -> Runtime:
    """Load the scene with the bundle's meshes and the ONNX policy."""
    import onnxruntime as ort  # noqa: PLC0415

    model = load_scene(manifest, assets_dir=assets_dir)
    data = mujoco.MjData(model)
    session = ort.InferenceSession(
        str(manifest.policy_path), providers=list(ONNX_PROVIDERS)
    )
    runtime = Runtime(manifest=manifest, model=model, data=data, session=session)
    runtime.reset()
    return runtime
