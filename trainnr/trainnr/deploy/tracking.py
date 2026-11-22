"""A tracking policy in plain MuJoCo (docs/87 §8): its reference, its
observations, its bounds, and the gate's trials through the clip.

mjlab's tracking task, re-stated from its source (mjlab 1.6,
`tasks/tracking/mdp`) for a runtime with no torch and no mjlab:

- the clock: the robot starts at reference frame 0 and mjlab's clock
  advances once at reset, so at control tick k the policy observes frame
  k + 1; the physics steps; the bounds are checked against frame k + 1;
  then the clock moves on (`ManagerBasedRlEnv.step`'s order; checked
  against mjlab's own rollout, term by term);
- the observations: `command motion` is the reference's joint positions
  and velocities at the frame; `motion_anchor_pos_b` the reference
  torso's position in the robot torso's frame; `motion_anchor_ori_b` the
  first two columns of the rotation from the robot torso to the
  reference torso, row by row;
- the bounds: the torso's height against the reference's, the z of the
  torso's projected gravity against the reference's, and each end
  effector's height against the reference's (mjlab compares the reference
  carried to the robot's position and heading; a turn about the vertical
  leaves heights as they are, so the reference's own heights are it).

A trial starts at the clip's first frame with the start drawn as training
drew it, is pushed and lagged as the cited evaluation was, and succeeds
when the clip plays to its end with no bound broken.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from trainnr.deploy.manifest import (
    SOURCE_MOTION_ANCHOR_ORI,
    SOURCE_MOTION_ANCHOR_POS,
    SOURCE_MOTION_COMMAND,
    Manifest,
    Tracking,
)
from trainnr.deploy.runtime import Runtime, register_source

GRAVITY_DOWN = np.array([0.0, 0.0, -1.0])
POSE_AXES = ("x", "y", "z", "roll", "pitch", "yaw")
REFERENCE_KEYS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)
ANCHOR_Z = "anchor height"
ANCHOR_TILT = "anchor tilt"
END_EFFECTOR_Z = "end effector height"


@dataclass(frozen=True)
class Reference:
    """The clip as the runtime reads it: `motion.npz` beside the policy."""

    joint_pos: np.ndarray  # (T, J), the policy's joint order
    joint_vel: np.ndarray
    body_pos_w: np.ndarray  # (T, B, 3), `body_names` order
    body_quat_w: np.ndarray  # (T, B, 4), w x y z
    body_lin_vel_w: np.ndarray
    body_ang_vel_w: np.ndarray

    @property
    def frames(self) -> int:
        return int(self.joint_pos.shape[0])

    @classmethod
    def load(cls, path: Path) -> Reference:
        with np.load(path) as data:
            missing = [k for k in REFERENCE_KEYS if k not in data]
            if missing:
                raise ValueError(f"{path}: the reference lacks {missing}")
            return cls(
                **{k: np.asarray(data[k], dtype=np.float64) for k in REFERENCE_KEYS}
            )


class Clip:
    """A runtime's reference and the indices it reads through: the frame
    at the runtime's tick, the reference torso, the robot's bodies."""

    def __init__(self, runtime: Runtime, tracking: Tracking) -> None:
        self.tracking = tracking
        self.reference = Reference.load(runtime.manifest.root / tracking.motion)
        if self.reference.frames != tracking.frames:
            raise ValueError(
                f"the reference holds {self.reference.frames} frames, the manifest "
                f"says {tracking.frames}"
            )
        model = runtime.model
        self.anchor = tracking.body_names.index(tracking.anchor_body)
        self.robot_anchor = int(model.body(tracking.anchor_body).id)
        self.effectors = [tracking.body_names.index(b) for b in tracking.end_effectors]
        self.robot_effectors = [int(model.body(b).id) for b in tracking.end_effectors]

    def frame(self, ticks: int) -> int:
        """The frame observed at a tick: one ahead (mjlab's reset advance)."""
        return min(int(ticks) + 1, self.reference.frames - 1)


def clip_of(runtime: Runtime) -> Clip:
    """The runtime's clip, built on first use (a walk's runtime has none)."""
    clip = getattr(runtime, "_clip", None)
    if clip is None:
        tracking = runtime.manifest.tracking
        if tracking is None:
            raise ValueError("this deployment is not a tracking policy: no reference")
        clip = Clip(runtime, tracking)
        object.__setattr__(runtime, "_clip", clip)
    return clip


def _quat_inv(q: np.ndarray) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_negQuat(out, np.asarray(q, dtype=np.float64))
    return out


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mulQuat(
        out, np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    )
    return out


def _rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    out = np.zeros(3)
    mujoco.mju_rotVecQuat(
        out, np.asarray(v, dtype=np.float64), np.asarray(q, dtype=np.float64)
    )
    return out


def _robot_anchor(runtime: Runtime, clip: Clip) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array(runtime.data.xpos[clip.robot_anchor]),
        np.array(runtime.data.xquat[clip.robot_anchor]),
    )


def motion_command(runtime: Runtime, _term: Any) -> np.ndarray:
    clip = clip_of(runtime)
    f = clip.frame(runtime.ticks)
    return np.concatenate([clip.reference.joint_pos[f], clip.reference.joint_vel[f]])


def motion_anchor_pos_b(runtime: Runtime, _term: Any) -> np.ndarray:
    clip = clip_of(runtime)
    f = clip.frame(runtime.ticks)
    pos, quat = _robot_anchor(runtime, clip)
    return _rotate(_quat_inv(quat), clip.reference.body_pos_w[f, clip.anchor] - pos)


def motion_anchor_ori_b(runtime: Runtime, _term: Any) -> np.ndarray:
    clip = clip_of(runtime)
    f = clip.frame(runtime.ticks)
    _pos, quat = _robot_anchor(runtime, clip)
    relative = _quat_mul(_quat_inv(quat), clip.reference.body_quat_w[f, clip.anchor])
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, relative)
    return mat.reshape(3, 3)[:, :2].reshape(-1)


register_source(SOURCE_MOTION_COMMAND, motion_command)
register_source(SOURCE_MOTION_ANCHOR_POS, motion_anchor_pos_b)
register_source(SOURCE_MOTION_ANCHOR_ORI, motion_anchor_ori_b)


def broken_bound(runtime: Runtime, frame: int) -> str | None:
    """The tracking bound the robot breaks against `frame`, or None."""
    clip = clip_of(runtime)
    t, ref = clip.tracking, clip.reference
    pos, quat = _robot_anchor(runtime, clip)
    if abs(ref.body_pos_w[frame, clip.anchor, 2] - pos[2]) > t.anchor_z_m:
        return ANCHOR_Z
    ref_g = _rotate(_quat_inv(ref.body_quat_w[frame, clip.anchor]), GRAVITY_DOWN)
    robot_g = _rotate(_quat_inv(quat), GRAVITY_DOWN)
    if abs(ref_g[2] - robot_g[2]) > t.anchor_tilt:
        return ANCHOR_TILT
    for k, body in zip(clip.effectors, clip.robot_effectors, strict=True):
        if (
            abs(ref.body_pos_w[frame, k, 2] - runtime.data.xpos[body][2])
            > t.end_effector_z_m
        ):
            return END_EFFECTOR_Z
    return None


def _uniform(
    rng: np.random.Generator, ranges: dict[str, Any], axes: tuple[str, ...]
) -> np.ndarray:
    lo = np.array([float((ranges.get(a) or (0.0, 0.0))[0]) for a in axes])
    hi = np.array([float((ranges.get(a) or (0.0, 0.0))[1]) for a in axes])
    return rng.uniform(lo, hi)


def _euler_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """mjlab's `quat_from_euler_xyz`: yaw, then pitch, then roll."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array(
        [
            cy * cr * cp + sy * sr * sp,
            cy * sr * cp - sy * cr * sp,
            cy * cr * sp + sy * sr * cp,
            sy * cr * cp - cy * sr * sp,
        ]
    )


def _set_root_velocity(runtime: Runtime, lin_w: np.ndarray, ang_w: np.ndarray) -> None:
    """The free joint's velocity from world-frame linear and angular parts
    (MuJoCo keeps the angular part in the body's own frame)."""
    v = runtime.base_qvel
    runtime.data.qvel[v : v + 3] = lin_w
    runtime.data.qvel[v + 3 : v + 6] = _rotate(_quat_inv(runtime.quat), ang_w)


def start_at_clip(runtime: Runtime, rng: np.random.Generator, *, drawn: bool) -> None:
    """The robot at the clip's first frame (mjlab's `_resample_command` at
    frame 0): the root pose and velocity, the joints clipped to their soft
    limits; with `drawn`, the offsets training drew around them."""
    runtime.reset()
    clip = clip_of(runtime)
    ref, start = clip.reference, clip.tracking.start
    pose = (
        _uniform(rng, start.get("pose_range") or {}, POSE_AXES)
        if drawn
        else np.zeros(6)
    )
    vel = (
        _uniform(rng, start.get("velocity_range") or {}, POSE_AXES)
        if drawn
        else np.zeros(6)
    )
    joints = np.array(ref.joint_pos[0])
    if drawn and start.get("joint_position_range"):
        off_lo, off_hi = (float(v) for v in start["joint_position_range"])
        joints = joints + rng.uniform(off_lo, off_hi, size=joints.shape)
    low, high = _soft_limits(runtime, float(start.get("soft_limit_factor", 1.0)))
    joints = np.clip(joints, low, high)
    q = runtime.base_qpos
    runtime.data.qpos[q : q + 3] = ref.body_pos_w[0, 0] + pose[:3]
    runtime.data.qpos[q + 3 : q + 7] = _quat_mul(
        _euler_xyz(*pose[3:]), ref.body_quat_w[0, 0]
    )
    runtime.data.qpos[runtime.joint_qpos] = joints
    _set_root_velocity(
        runtime, ref.body_lin_vel_w[0, 0] + vel[:3], ref.body_ang_vel_w[0, 0] + vel[3:]
    )
    runtime.data.qvel[runtime.joint_qvel] = ref.joint_vel[0]
    mujoco.mj_forward(runtime.model, runtime.data)


def _soft_limits(runtime: Runtime, factor: float) -> tuple[np.ndarray, np.ndarray]:
    """The joints' ranges narrowed about their middle by `factor` (mjlab's
    soft joint limits), in the policy order."""
    model = runtime.model
    ids = [model.joint(j).id for j in runtime.manifest.joints.policy_order]
    lo, hi = model.jnt_range[ids, 0], model.jnt_range[ids, 1]
    mid, half = (lo + hi) / 2, (hi - lo) / 2 * factor
    limited = model.jnt_limited[ids].astype(bool)
    return np.where(limited, mid - half, -np.inf), np.where(limited, mid + half, np.inf)


@dataclass(frozen=True)
class ClipTrial:
    """One trial through the clip: how far it got and why it stopped."""

    steps: int  # control ticks played
    frames: int
    success: bool
    bound: str | None  # the bound it broke, when it failed
    lag_ticks: int
    pushes: int

    @property
    def fell(self) -> bool:
        return not self.success

    def row(self) -> dict[str, Any]:
        return asdict(self) | {"fell": self.fell}


@dataclass(frozen=True)
class Conditions:
    """What a trial is judged under: the cited evaluation's, else the
    manifest's training conditions with no lag."""

    pushes: bool
    lag_ms: float
    noise: bool
    drawn_start: bool

    @classmethod
    def of(cls, tracking: Tracking, certificate: dict[str, Any] | None) -> Conditions:
        protocol = (certificate or {}).get("protocol") or {}
        return cls(
            pushes=bool(protocol.get("pushes", tracking.pushes is not None)),
            lag_ms=float(protocol.get("lag_ms", tracking.lag_ms)),
            noise=bool(protocol.get("observation_noise", bool(tracking.noise))),
            drawn_start=True,
        )


def lag_of(lag_ms: float, step_dt: float, rng: np.random.Generator) -> int:
    """A trial's command lag in whole control ticks, uniform in [0, lag_ms]."""
    most = round(lag_ms / 1000 / step_dt)
    return int(rng.integers(0, most + 1))


def run_clip_trial(
    manifest: Manifest,
    runtime: Runtime,
    conditions: Conditions,
    rng: np.random.Generator,
    *,
    on_tick: Any = None,
) -> ClipTrial:
    """One trial from the clip's first frame to its end, or to the first
    bound broken: observed with noise, acted with lag, pushed, as stated."""
    tracking = manifest.tracking
    assert tracking is not None
    dt = manifest.control.step_dt
    lag = lag_of(conditions.lag_ms, dt, rng)
    randomize(runtime, tracking.randomization, rng)  # before: setConst scratches data
    start_at_clip(runtime, rng, drawn=conditions.drawn_start)
    history: list[np.ndarray] = []
    pushes = 0
    push = tracking.pushes if conditions.pushes else None
    next_push = _next_push(push, rng)
    elapsed = 0.0
    frames = tracking.frames
    for k in range(frames):
        obs = runtime.observe()
        if conditions.noise:
            obs = obs + _noise(manifest, tracking, rng)
        history = [runtime.act(obs.astype(np.float32)), *history][: lag + 1]
        action = history[min(lag, len(history) - 1)]
        runtime.apply(action)
        if on_tick is not None:
            on_tick(runtime)
        if k < frames - 1:
            bound = broken_bound(runtime, k + 1)
            if bound is not None:
                return ClipTrial(k + 1, frames, False, bound, lag, pushes)
        elapsed += dt
        if push is not None and next_push is not None and elapsed >= next_push:
            kick = _uniform(rng, push.get("velocity_range") or {}, POSE_AXES)
            v = runtime.base_qvel
            lin_w = runtime.data.qvel[v : v + 3] + kick[:3]
            ang_w = _rotate(runtime.quat, runtime.data.qvel[v + 3 : v + 6]) + kick[3:]
            _set_root_velocity(runtime, lin_w, ang_w)
            pushes += 1
            elapsed, next_push = 0.0, _next_push(push, rng)
    return ClipTrial(frames, frames, True, None, lag, pushes)


def randomize(
    runtime: Runtime, draws: dict[str, Any], rng: np.random.Generator
) -> None:
    """Training's start-up draws, once per trial, on a model restored to
    its own values first: an encoder bias per joint, an offset of a body's
    centre of mass per axis, one friction shared by the feet."""
    model = runtime.model
    saved = getattr(runtime, "_saved_model", None)
    if saved is None:
        saved = (model.body_ipos.copy(), model.geom_friction.copy())
        object.__setattr__(runtime, "_saved_model", saved)
    model.body_ipos[:], model.geom_friction[:] = saved
    bias = draws.get("encoder_bias")
    n = len(runtime.joint_qpos)
    runtime.encoder_bias = (
        rng.uniform(float(bias[0]), float(bias[1]), size=n) if bias else np.zeros(n)
    )
    com = draws.get("body_com")
    if com:
        body = model.body(com["body"]).id
        for axis, (lo, hi) in com["ranges"].items():
            model.body_ipos[body, int(axis)] += rng.uniform(float(lo), float(hi))
    friction = draws.get("foot_friction")
    if friction:
        value = rng.uniform(float(friction["range"][0]), float(friction["range"][1]))
        for geom in friction["geoms"]:
            model.geom_friction[model.geom(geom).id, 0] = value
    mujoco.mj_setConst(model, runtime.data)


def _next_push(push: dict[str, Any] | None, rng: np.random.Generator) -> float | None:
    if push is None:
        return None
    lo, hi = (float(v) for v in push["interval_s"])
    return float(rng.uniform(lo, hi))


def _noise(
    manifest: Manifest, tracking: Tracking, rng: np.random.Generator
) -> np.ndarray:
    """Uniform noise per observation term, as training's corruption."""
    parts = []
    for term in manifest.observations:
        lo, hi = tracking.noise.get(term.name, (0.0, 0.0))
        parts.append(rng.uniform(lo, hi, size=term.width))
    return np.concatenate(parts)
