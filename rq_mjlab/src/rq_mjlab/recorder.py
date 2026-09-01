"""The first real RecorderTerm: mjlab rollouts streaming to a Rerun viewer.

mjlab's recorder API ships with zero implementations and its FAQ wishes
for training-time visualization (docs/e2e-research/56 §5-§6); this term
fills both with no new dependency for the framework. Point it at any
Rerun gRPC endpoint — the Studio's own :9876 first (the window the
operator already watches), a bare `rerun` viewer, or a `.rrd` sink the
caller arranged — and every training run narrates itself: the watched
world's joint positions, the reward, the episode length, on an
`env_steps` timeline beside wall time, with episode boundaries marked
at reset.

Scalars AND the scene: beside the series, the watched world mirrors
itself in 3D (`rq_pipeline.viz.RigMirror` — mesh geoms logged once,
one transform per sample; primitives as oriented solids) under
`world/robot`, so the Studio's 3D view follows whatever robot is
ACTUALLY training — the operator's rule that everything flows from a
3D representation of the physics, not a canned scene (2026-09-01).
`mirror=False` returns the recorder to scalars-only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import mujoco
import torch
from mjlab.managers.recorder_manager import RecorderTerm, RecorderTermCfg

DEFAULT_ADDRESS = "rerun+http://127.0.0.1:9876/proxy"  # the Studio's server


def _require_rerun() -> Any:
    try:
        import rerun as rr  # noqa: PLC0415 - the viz extra
    except ImportError as error:  # pragma: no cover - the helpful error
        raise ImportError(
            "the Rerun recorder needs the 'viz' extra: uv sync --extra viz"
        ) from error
    return rr


class RerunRecorder(RecorderTerm):
    """Stream one watched world's training story to a Rerun endpoint."""

    def __init__(self, cfg: RerunRecorderCfg, env: Any) -> None:
        super().__init__(cfg, env)
        rr = _require_rerun()
        self._rr = rr
        self._cfg = cfg
        rr.init(cfg.app_id, spawn=False)
        rr.connect_grpc(cfg.address)
        # (name, qpos address) per SCALAR joint — hinge/slide only. The
        # first cut indexed qpos by JOINT index, which plots freejoint
        # quaternion components as "joints" on any floating-base robot
        # (review, 2026-09-01); a free/ball joint is named in the config
        # card instead of silently mis-plotted.
        mj_model = env.sim.mj_model
        scalar = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
        self._joints = [
            (mj_model.joint(j).name or f"joint{j}", int(mj_model.jnt_qposadr[j]))
            for j in range(mj_model.njnt)
            if int(mj_model.jnt_type[j]) in scalar
        ]
        self._skipped_joints = [
            mj_model.joint(j).name or f"joint{j}"
            for j in range(mj_model.njnt)
            if int(mj_model.jnt_type[j]) not in scalar
        ]
        self._qpos = None  # torch view, bound lazily (data exists post-init)
        self._geom_views = None  # (xpos, xmat) torch views, bound lazily
        self._render = None  # (mujoco.Renderer, MjData) for the camera leg
        self._mirror = None
        if cfg.mirror:
            from rq_pipeline.viz import RigMirror  # noqa: PLC0415

            self._mirror = RigMirror(mj_model, model_colors=True)
        self._said_no_reward = False
        self._began = time.time()
        rr.log(
            "recorder/config",
            rr.TextDocument(
                f"watched world {cfg.watched_env}, every {cfg.every} steps\n"
                f"joints: {', '.join(name for name, _ in self._joints)}"
                + (
                    f"\nskipped (multi-dof): {', '.join(self._skipped_joints)}"
                    if self._skipped_joints
                    else ""
                )
            ),
            static=True,
        )

    def _clock(self) -> None:
        self._rr.set_time("env_steps", sequence=int(self._env.common_step_counter))
        self._rr.set_time("wall", duration=time.time() - self._began)

    def record_post_step(self) -> None:
        env = self._env
        if int(env.common_step_counter) % self._cfg.every:
            return
        rr = self._rr
        self._clock()
        watched = self._cfg.watched_env
        if self._qpos is None:
            from rq_mjlab.actuator import as_torch  # noqa: PLC0415

            self._qpos = as_torch(env.sim.data.qpos)
        qpos = self._qpos[watched]
        for name, qpos_adr in self._joints:
            rr.log(f"train/qpos/{name}", rr.Scalars(float(qpos[qpos_adr])))
        reward = getattr(env, "reward_buf", None)
        if isinstance(reward, torch.Tensor) and reward.numel() > watched:
            rr.log("train/reward", rr.Scalars(float(reward[watched])))
        elif not self._said_no_reward:
            # Once, by name — and the two absences are different: a
            # missing buffer is an env without rewards, a short one is a
            # watched world past its end (second review, 2026-09-01).
            self._said_no_reward = True
            why = (
                "no reward_buf on this env"
                if not isinstance(reward, torch.Tensor)
                else f"reward_buf has {reward.numel()} entries, watched world is "
                f"{watched}"
            )
            rr.log("recorder/notes", rr.TextLog(f"{why}: reward trace omitted"))
        rr.log(
            "train/episode_length",
            rr.Scalars(float(env.episode_length_buf[watched])),
        )
        if self._mirror is not None:
            self._log_mirror(watched)
        if (
            self._cfg.frames
            and int(env.common_step_counter) % self._cfg.frame_every == 0
        ):
            self._log_frame(watched)

    def _log_mirror(self, watched: int) -> None:
        """One world's geoms out of the batched engine, into 3D."""
        if self._geom_views is None:
            from rq_mjlab.actuator import as_torch  # noqa: PLC0415

            data = self._env.sim.data
            self._geom_views = (as_torch(data.geom_xpos), as_torch(data.geom_xmat))

        class _World:
            geom_xpos = self._geom_views[0][watched].cpu().numpy()
            geom_xmat = self._geom_views[1][watched].cpu().numpy()

        self._mirror.log(_World(), path="world/robot")

    def _log_frame(self, watched: int) -> None:
        """The watched world through MuJoCo's own renderer — the lit,
        shadowed camera image, rendered INSIDE the training process from
        the live state (the batched engine's memory is invisible to any
        outside renderer, which is why the Studio's panel cannot draw
        it). Needs a GL context (MUJOCO_GL=egl on a headless GPU box);
        refusal is once, by name, and frames are simply absent."""
        if self._render is None:
            try:
                model = self._env.sim.mj_model
                self._render = (
                    mujoco.Renderer(model, height=360, width=640),
                    mujoco.MjData(model),
                )
            except Exception as error:
                self._cfg.frames = False
                self._rr.log(
                    "recorder/notes",
                    self._rr.TextLog(f"camera disabled: {error} (set MUJOCO_GL=egl?)"),
                )
                return
        renderer, mj_data = self._render
        qpos = self._qpos[watched].cpu().numpy()
        mj_data.qpos[:] = qpos
        mujoco.mj_forward(self._env.sim.mj_model, mj_data)
        renderer.update_scene(mj_data)
        self._rr.log("camera/watched", self._rr.Image(renderer.render()))

    def record_pre_reset(self, env_ids: torch.Tensor) -> None:
        if (env_ids == self._cfg.watched_env).any():
            self._clock()
            self._rr.log(
                "recorder/episodes",
                self._rr.TextLog(
                    f"world {self._cfg.watched_env} resets at step "
                    f"{int(self._env.common_step_counter)} after "
                    f"{int(self._env.episode_length_buf[self._cfg.watched_env])} ticks"
                ),
            )

    def record_post_reset(self, env_ids: torch.Tensor) -> None:
        del env_ids

    def close(self) -> None:
        self._rr.disconnect()


@dataclass(kw_only=True)
class RerunRecorderCfg(RecorderTermCfg):
    """Where to stream and what to watch. `func` is the term class,
    per mjlab's own contract (recorder terms are stateful)."""

    func: type[RerunRecorder] = RerunRecorder
    address: str = DEFAULT_ADDRESS
    app_id: str = "rq-mjlab-train"
    watched_env: int = 0
    mirror: bool = True  # the 3D scene beside the series
    frames: bool = True  # MuJoCo-rendered camera images of the watched world
    frame_every: int = 25  # control steps between camera frames (renders cost ~30 ms)
    every: int = (
        10  # control steps between samples: the viewer's rate, not the trainer's
    )
