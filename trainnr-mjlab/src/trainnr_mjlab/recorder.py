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
itself in 3D (`trainnr.viz.RigMirror` — mesh geoms logged once,
one transform per sample; primitives as oriented solids) under
`world/robot`, so the Studio's 3D view follows whatever robot is
ACTUALLY training — the operator's rule that everything flows from a
3D representation of the physics, not a canned scene (2026-09-01).
`mirror=False` returns the recorder to scalars-only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mujoco
import torch

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from trainnr.viz import RigMirror
from mjlab.managers.recorder_manager import RecorderTerm, RecorderTermCfg

# One home for the Studio's ingest address (trainnr is a hard dep).
from trainnr.viz import STUDIO_ADDRESS as DEFAULT_ADDRESS
from trainnr.viz import VISUAL_ONLY_SKIP_GROUPS, open_stream

from trainnr_mjlab.walks import TERRAIN_SCAN_GROUP


def _require_rerun() -> Any:
    try:
        import rerun as rr  # noqa: PLC0415 - the viz extra
    except ImportError as error:  # pragma: no cover - the helpful error
        raise ImportError(
            "the Rerun recorder needs the 'viz' extra: uv sync --extra viz"
        ) from error
    return rr


FRAME_JPEG_QUALITY = 85  # camera frames go over the wire encoded


class RerunRecorder(RecorderTerm):
    """Stream one watched world's training story to a Rerun endpoint."""

    def __init__(self, cfg: RerunRecorderCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        rr = _require_rerun()
        self._rr = rr
        self._cfg = cfg
        # The one seam (trainnr.viz.open_stream): the Studio's server
        # and, when the run has a folder, its saved stream too (docs/76 §10.5).
        open_stream(cfg.app_id, address=cfg.address, file=cfg.file, rr=rr)
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
        # Bound lazily: the batched data exists only post-init.
        self._qpos: torch.Tensor | None = None
        self._geom_views: tuple[torch.Tensor, torch.Tensor] | None = None
        self._render: tuple[mujoco.Renderer, mujoco.MjData, mujoco.MjvCamera] | None = (
            None
        )
        self._mirror: RigMirror | None = None
        if cfg.mirror:
            from trainnr.viz import RigMirror  # noqa: PLC0415

            # Mirror what MuJoCo's own visualizer shows (groups 0-2):
            # collision geoms share surfaces with the visual meshes and
            # z-fight them into shimmering shades (the duck: 5 opaque
            # group-3 collision meshes over 70 visual ones, 2026-09-01).
            # ...and a captured scene's ground, which sits in a group of its
            # own for the height scans: skipped, the Studio showed the robots
            # on nothing (2026-09-25). Only a model with a heightfield in that
            # group; any other walk mirrors exactly as before.
            ground = bool(
                (
                    (mj_model.geom_type == mujoco.mjtGeom.mjGEOM_HFIELD)
                    & (mj_model.geom_group == TERRAIN_SCAN_GROUP)
                ).any()
            )
            self._mirror = RigMirror(
                mj_model,
                model_colors=True,
                skip_groups=tuple(
                    g
                    for g in VISUAL_ONLY_SKIP_GROUPS
                    if not (ground and g == TERRAIN_SCAN_GROUP)
                ),
            )
        self._said_no_reward = False
        self._began = time.time()
        if cfg.layout:
            self._send_layout()
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

    def _send_layout(self) -> None:
        """A purposeful window: the 3D mirror, the camera, the series —
        each in its own view, so the viewer's auto-layout never wedges
        the 2D camera image into a 3D view (the pinhole complaint,
        2026-09-01)."""
        try:
            import rerun.blueprint as rrb  # noqa: PLC0415 - viz extra

            self._rr.send_blueprint(
                rrb.Blueprint(
                    rrb.Grid(
                        rrb.Spatial3DView(origin="world", name="physics"),
                        rrb.Spatial2DView(origin="camera", name="camera"),
                        rrb.TimeSeriesView(
                            origin="train",
                            name="training",
                            contents=["+ $origin/**", "- $origin/reward_terms/**"],
                        ),
                        rrb.TimeSeriesView(
                            origin="train/reward_terms", name="reward terms"
                        ),
                        rrb.TextLogView(origin="recorder", name="events"),
                    ),
                    collapse_panels=False,
                )
            )
        except Exception as error:
            self._rr.log(
                "recorder/notes", self._rr.TextLog(f"layout not sent: {error}")
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
            from trainnr_mjlab.actuator import as_torch  # noqa: PLC0415

            # mjlab types sim.data as an alias of mujoco_warp's Data, which
            # has no type information: the fields are read as Any.
            data: Any = env.sim.data
            self._qpos = as_torch(data.qpos)
        qpos = self._qpos[watched]
        for name, qpos_adr in self._joints:
            rr.log(f"train/qpos/{name}", rr.Scalars(float(qpos[qpos_adr])))
        reward = getattr(env, "reward_buf", None)
        if isinstance(reward, torch.Tensor) and reward.numel() > watched:
            rr.log("train/reward", rr.Scalars(float(reward[watched])))
            if self._cfg.terms:
                self._log_terms(watched)
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
            self._log_mirror(self._mirror, watched)
        if int(env.common_step_counter) % self._cfg.frame_every == 0:
            if self._cfg.frames:
                self._log_frame(watched)
            self._log_cameras(watched)

    def _log_cameras(self, watched: int) -> None:
        """What the watched world's own camera sensors see (a scene's
        head camera, docs/78 E2): the policy's picture, not the
        renderer's; nothing when the env has none."""
        from mjlab.sensor.camera_sensor import CameraSensor  # noqa: PLC0415

        for name, sensor in self._env.scene.sensors.items():
            if not isinstance(sensor, CameraSensor) or sensor.data.rgb is None:
                continue
            frame = sensor.data.rgb[watched].cpu().numpy()
            self._rr.log(
                f"camera/{name}",
                self._rr.Image(frame).compress(jpeg_quality=FRAME_JPEG_QUALITY),
            )

    def _log_terms(self, watched: int) -> None:
        """Every reward term's value this step for the watched world -
        mjlab's reward manager keeps them per step (the hook Isaac Lab's
        live visualizer reads too); the total alone said nothing about
        WHY a curve moved (the Go2's step at iteration 5000, 2026-09-11)."""
        manager = getattr(self._env, "reward_manager", None)
        terms = getattr(manager, "get_active_iterable_terms", None)
        if terms is None:
            return
        for name, values in terms(watched):
            self._rr.log(
                f"train/reward_terms/{name}", self._rr.Scalars(float(values[0]))
            )

    def _log_mirror(self, mirror: RigMirror, watched: int) -> None:
        """One world's geoms out of the batched engine, into 3D."""
        if self._geom_views is None:
            from trainnr_mjlab.actuator import as_torch  # noqa: PLC0415

            data: Any = self._env.sim.data  # untyped mujoco_warp Data (above)
            self._geom_views = (as_torch(data.geom_xpos), as_torch(data.geom_xmat))
        xpos_view, xmat_view = self._geom_views

        class _World:
            geom_xpos = xpos_view[watched].cpu().numpy()
            geom_xmat = xmat_view[watched].cpu().numpy()

        mirror.log(_World(), path="world/robot")
        # A ground patch that FOLLOWS the robot: context underfoot
        # without an origin-pinned plane skewing the view's bounds.
        center = _World.geom_xpos.mean(axis=0)
        self._rr.log(
            "world/ground",
            self._rr.Boxes3D(
                centers=[[float(center[0]), float(center[1]), -0.005]],
                half_sizes=[[1.0, 1.0, 0.005]],
                colors=[[70, 80, 90]],
                fill_mode="solid",
            ),
        )

    def _log_frame(self, watched: int) -> None:
        """The watched world through MuJoCo's own renderer — the lit,
        shadowed camera image, rendered INSIDE the training process from
        the live state (the batched engine's memory is invisible to any
        outside renderer, which is why the Studio's panel cannot draw
        it). Needs a GL context (MUJOCO_GL=egl on a headless GPU box);
        refusal is once, by name, and frames are simply absent."""
        model = self._env.sim.mj_model
        if self._render is None:
            try:
                camera = mujoco.MjvCamera()
                mujoco.mjv_defaultCamera(camera)
                # A chase camera on the robot, not MuJoCo's default free
                # look-at-nothing. Distance comes from the ROBOT's real
                # size at the first frame — model.stat.extent is inflated
                # by the infinite terrain plane (2 m for a 15 cm duck put
                # the camera 5 m out; the operator saw only floor,
                # 2026-09-01).
                camera.distance = 0.0  # set from the geom cloud below
                camera.elevation = -20.0
                camera.azimuth = 120.0
                self._render = (
                    mujoco.Renderer(model, height=360, width=640),
                    mujoco.MjData(model),
                    camera,
                )
            except Exception as error:
                self._cfg.frames = False
                self._rr.log(
                    "recorder/notes",
                    self._rr.TextLog(f"camera disabled: {error} (set MUJOCO_GL=egl?)"),
                )
                return
        renderer, mj_data, camera = self._render
        qpos_view = self._qpos
        assert qpos_view is not None  # bound by record_post_step before any frame
        mj_data.qpos[:] = qpos_view[watched].cpu().numpy()
        mujoco.mj_forward(model, mj_data)
        if model.nbody > 1:
            camera.lookat[:] = mj_data.xpos[1]
        if camera.distance <= 0.0:
            import numpy as np  # noqa: PLC0415

            robot = model.geom_bodyid > 0  # world geoms (terrain) excluded
            spread = np.linalg.norm(mj_data.geom_xpos[robot] - mj_data.xpos[1], axis=1)
            camera.distance = 4.0 * float(max(spread.max(), 0.05))
        renderer.update_scene(mj_data, camera=camera)
        # JPEG on the way in: a raw 640x360 frame is 690 KB and a play
        # session grew to 1.2 GiB in the viewer in half an hour (2026-09-11);
        # Rerun encodes it here, ~25x smaller, the viewer decodes.
        self._rr.log(
            "camera/watched",
            self._rr.Image(renderer.render()).compress(jpeg_quality=FRAME_JPEG_QUALITY),
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
        # The GL renderer goes first, while EGL is still up: left to the
        # interpreter's exit it prints an EGL_NOT_INITIALIZED traceback
        # after every run (seen on every scene smoke, 2026-09-23).
        if self._render is not None:
            renderer, _data, _camera = self._render
            self._render = None
            renderer.close()
        self._rr.disconnect()


@dataclass(kw_only=True)
class RerunRecorderCfg(RecorderTermCfg):
    """Where to stream and what to watch. `func` is the term class,
    per mjlab's own contract (recorder terms are stateful)."""

    func: type[RerunRecorder] = RerunRecorder
    address: str = DEFAULT_ADDRESS
    app_id: str = "trainnr-mjlab-train"
    # The saved stream, inside the run's folder (docs/76 §10.5); None: live only.
    file: str | None = None
    watched_env: int = 0
    mirror: bool = True  # the 3D scene beside the series
    frames: bool = True  # MuJoCo-rendered camera images of the watched world
    layout: bool = True  # send a purposeful view layout on connect
    terms: bool = True  # every reward term per step beside the total
    frame_every: int = 25  # control steps between camera frames (renders cost ~30 ms)
    every: int = (
        10  # control steps between samples: the viewer's rate, not the trainer's
    )
