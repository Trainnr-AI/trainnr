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

Deliberately scalar-first: a mesh-true 3D mirror needs per-geom
transforms shipped per tick and belongs to a later slice; a training
run's story — is the reward moving, is the episode surviving, what is
world 0 doing — is legible in the series alone, and this is the shape
mjlab's own wandb charts already speak.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

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
        self._joint_names = [
            env.sim.mj_model.joint(j).name or f"joint{j}"
            for j in range(env.sim.mj_model.njnt)
        ]
        self._qpos = None  # torch view, bound lazily (data exists post-init)
        self._began = time.time()
        rr.log(
            "recorder/config",
            rr.TextDocument(
                f"watched world {cfg.watched_env}, every {cfg.every} steps\n"
                f"joints: {', '.join(self._joint_names)}"
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
        for index, name in enumerate(self._joint_names):
            rr.log(f"train/qpos/{name}", rr.Scalars(float(qpos[index])))
        reward = getattr(env, "reward_buf", None)
        if isinstance(reward, torch.Tensor) and reward.numel() > watched:
            rr.log("train/reward", rr.Scalars(float(reward[watched])))
        rr.log(
            "train/episode_length",
            rr.Scalars(float(env.episode_length_buf[watched])),
        )

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
    every: int = (
        10  # control steps between samples: the viewer's rate, not the trainer's
    )
