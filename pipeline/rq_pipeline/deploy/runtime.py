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
pose, joint velocities, the last action, the commanded twist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from rq_pipeline.deploy.manifest import Manifest

FREE_JOINT_QPOS = 7
FREE_JOINT_QVEL = 6
TIMESTEP_TOLERANCE_S = 1e-9


@dataclass
class Runtime:
    """One robot, one policy, driven by the manifest."""

    manifest: Manifest
    model: mujoco.MjModel
    data: mujoco.MjData
    session: Any
    command: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    last_action: np.ndarray = field(init=False)
    default_pos: np.ndarray = field(init=False)
    scale: np.ndarray = field(init=False)
    action_to_ctrl: list[int] = field(init=False)
    joint_qpos: np.ndarray = field(init=False)
    joint_qvel: np.ndarray = field(init=False)
    sensors: dict[str, tuple[int, int]] = field(init=False)

    def __post_init__(self) -> None:
        joints = self.manifest.joints
        n = len(joints["policy_order"])
        self.last_action = np.zeros(n, dtype=np.float32)
        self.ticks = 0  # control steps since reset, the gait clock's time
        self.default_pos = np.asarray(joints["default_pos"], dtype=np.float64)
        self.scale = np.asarray(self.manifest.raw["action"]["scale"], dtype=np.float64)
        self.action_to_ctrl = list(joints["action_to_ctrl"])
        # Where each policy-order joint lives in qpos/qvel (after the free joint).
        self.joint_qpos = np.array(
            [
                self.model.jnt_qposadr[self.model.joint(j).id]
                for j in joints["policy_order"]
            ]
        )
        self.joint_qvel = np.array(
            [
                self.model.jnt_dofadr[self.model.joint(j).id]
                for j in joints["policy_order"]
            ]
        )
        self.sensors = {}
        for i in range(self.model.nsensor):
            adr, dim = int(self.model.sensor_adr[i]), int(self.model.sensor_dim[i])
            self.sensors[self.model.sensor(i).name] = (adr, dim)

    @property
    def step_dt(self) -> float:
        """The control period: the physics step times the decimation."""
        control = self.manifest.control
        return float(control["physics_timestep_s"]) * int(control["decimation"])

    @property
    def decimation(self) -> int:
        return int(self.manifest.control["decimation"])

    def reset(self, *, keyframe: int = 0) -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.data, keyframe)
        mujoco.mj_forward(self.model, self.data)
        self.last_action[:] = 0.0
        self.ticks = 0

    def observe(self) -> np.ndarray:
        parts = [self._term(t) for t in self.manifest.observations]
        return np.concatenate(parts).astype(np.float32)

    def _term(self, term: dict[str, Any]) -> np.ndarray:
        source = term["source"]
        scale = term.get("scale", 1.0)
        if source.startswith("sensor "):
            adr, dim = self.sensors[source.split(" ", 1)[1].split("/", 1)[-1]]
            value = self.data.sensordata[adr : adr + dim].copy()
        elif source == "projected_gravity":
            quat = self.data.qpos[3:7]  # w x y z of the free joint
            gravity = np.array([0.0, 0.0, -1.0])
            value = _rotate_inverse(quat, gravity)
        elif source == "joint_pos_rel":
            value = self.data.qpos[self.joint_qpos] - self.default_pos
        elif source == "joint_vel_rel":
            value = self.data.qvel[self.joint_qvel]
        elif source == "last_action":
            value = self.last_action.astype(np.float64)
        elif source == "command twist":
            value = self.command.astype(np.float64)
        elif source == "gait_phase":
            params = term.get("params") or {}
            value = gait_phase(
                self.ticks, self.step_dt, float(params["period"]), self.command
            )
        else:  # pragma: no cover - the manifest loader refuses unknown sources
            raise ValueError(f"cannot compute observation source {source!r}")
        value = np.asarray(value, dtype=np.float64) * np.asarray(
            scale, dtype=np.float64
        )
        clip = term.get("clip")
        if clip is not None:
            value = np.clip(value, clip[0], clip[1])
        if int(value.size) != int(term["width"]):
            raise ValueError(
                f"{term['name']}: width {value.size}, manifest says {term['width']}"
            )
        return value

    def act(self, obs: np.ndarray) -> np.ndarray:
        name = self.session.get_inputs()[0].name
        out = self.session.run(None, {name: obs[None, :]})[0][0]
        return np.asarray(out, dtype=np.float32)

    def apply(self, action: np.ndarray) -> None:
        """One control tick: targets from the action, `decimation` physics steps."""
        target = self.default_pos + self.scale * action.astype(np.float64)
        for i, ctrl in enumerate(self.action_to_ctrl):
            self.data.ctrl[ctrl] = target[i]
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)
        self.last_action = action.astype(np.float32)
        self.ticks += 1

    def base_velocity_b(self) -> np.ndarray:
        """The base's linear velocity in its own frame (what the verdict
        compares to the command)."""
        quat = self.data.qpos[3:7]
        return _rotate_inverse(quat, self.data.qvel[0:3].copy())

    def fell_over(self) -> bool:
        limit = self.manifest.raw.get("termination", {}).get("fell_over_deg")
        if limit is None:
            return False
        gravity_b = _rotate_inverse(self.data.qpos[3:7], np.array([0.0, 0.0, -1.0]))
        # The angle between the body's down and gravity: acos(-g_b·z).
        cos = float(np.clip(-gravity_b[2], -1.0, 1.0))
        return np.degrees(np.arccos(cos)) > float(limit)


STANDING_COMMAND = 0.1  # the reference's threshold: no gait clock below it


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


def _rotate_inverse(quat_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a world vector into the frame of the quaternion (w, x, y, z)."""
    out = np.zeros(3)
    conj = np.array([quat_wxyz[0], -quat_wxyz[1], -quat_wxyz[2], -quat_wxyz[3]])
    mujoco.mju_rotVecQuat(out, np.asarray(v, dtype=np.float64), conj)
    return out


def assets_dir_of(manifest: Manifest) -> Path:
    """The meshes the manifest's scene names: the cited robot bundle's
    `assets/`, found project-first (`bundles.locate`)."""
    from rq_pipeline.bundles.locate import find_bundle  # noqa: PLC0415

    robot = str(manifest.raw.get("robot", "")).split("@", 1)[0]
    bundle = find_bundle(robot) if robot else None
    if bundle is None:
        raise FileNotFoundError(
            f"no bundle {robot!r} in the project or the library for the meshes"
        )
    return bundle / "assets"


def load_scene(manifest: Manifest, *, assets_dir: Path) -> mujoco.MjModel:
    """The trained scene as a plain MuJoCo model, its timestep checked
    against the manifest's."""
    assets = {p.name: p.read_bytes() for p in Path(assets_dir).iterdir() if p.is_file()}
    model = mujoco.MjModel.from_xml_string(
        manifest.scene_path.read_text(), assets=assets
    )
    expected_dt = float(manifest.control["physics_timestep_s"])
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
        str(manifest.policy_path), providers=["CPUExecutionProvider"]
    )
    runtime = Runtime(manifest=manifest, model=model, data=data, session=session)
    runtime.reset()
    return runtime
