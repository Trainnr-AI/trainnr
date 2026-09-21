"""The gate's robot in the Studio: whatever drives the policy - our
MuJoCo runtime, or Unitree's simulator and controller over DDS - the
Studio shows the same picture, the base and every joint mirrored into
the deployment's own scene, with the command and the measured velocity
beside it. Law 0 (docs/76): everything streams to the Studio; until
2026-09-12 the DDS gate showed only in their simulator's window.

The runtime says where the robot is (`pose()`: base position, base
quaternion w x y z, joints in the manifest's policy order); the mirror
puts that into a MuJoCo model of the deployment's scene and logs the
geoms through `rq_pipeline.viz.RigMirror`, the path every rig tool
uses. Missing Rerun (no `viz` extra) or an unloadable scene makes the
mirror a no-op that says so once: the gate's number never depends on
its picture.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from rq_pipeline.deploy.manifest import TWIST_SHORT, Key, Manifest
from rq_pipeline.viz import (
    MIRROR_HZ,
    SIM_TIMELINE,
    STUDIO_ADDRESS,
    VISUAL_ONLY_SKIP_GROUPS,
    RigMirror,
    gaussians,
    open_stream,
)

APP_ID = "rq-gate"
TIMELINE = SIM_TIMELINE


def ticks_per_frame(step_dt: float) -> int:
    """How many control ticks pass between mirrored frames so the mirror
    stays near its budget (`viz.MIRROR_HZ`) whatever the manifest's rate:
    every frame is one log call per geom, and rr.log blocks when it floods."""
    return max(1, round(1.0 / (max(step_dt, 1e-6) * MIRROR_HZ)))


def _rerun() -> Any | None:
    try:
        import rerun as rr  # noqa: PLC0415 - the viz extra
    except ImportError:
        return None
    return rr


SCENE_PATH = "world/scene/splat"
CAMERAS_PATH = "cameras"  # the stage cameras' pictures, one entity each
CAMERA_HZ = 2  # pictures a second: the ray tracer on a CPU takes seconds a frame


class GateMirror:
    """One gate run's picture: the deployment's scene driven by the
    runtime's poses, one Rerun recording named for the runtime."""

    def __init__(  # noqa: PLR0913 - the picture's own knobs, each named
        self,
        manifest: Manifest,
        model: Any,
        runtime_name: str,
        *,
        rr: Any,
        file: Path | None = None,
        scene_dir: Path | None = None,
    ) -> None:
        import mujoco  # noqa: PLC0415 - the sim extra

        self._rr = rr
        self.runtime_name = runtime_name
        self.model = model
        self.data = mujoco.MjData(model)
        self.joint_qpos = np.array(
            [model.jnt_qposadr[model.joint(j).id] for j in manifest.joints.policy_order]
        )
        self._mirror = RigMirror(
            model, model_colors=True, skip_groups=VISUAL_ONLY_SKIP_GROUPS
        )
        self.every = ticks_per_frame(manifest.control.step_dt)
        self.ticks = 0
        self.frames = 0
        self.seconds = 0.0
        self.manifest = manifest
        self._cameras: Any = None
        open_stream(
            f"{APP_ID}-{runtime_name}", address=STUDIO_ADDRESS, file=file, rr=rr
        )
        self._layout()
        if scene_dir is not None:
            self._scene(scene_dir)

    @classmethod
    def open(
        cls,
        manifest: Manifest,
        runtime_name: str,
        *,
        log: TextIO = sys.stderr,
        file: Path | None = None,
        scene_dir: Path | None = None,
    ) -> GateMirror | None:
        """The mirror, or None with a note when it cannot be one; with a
        `scene_dir` (a staged deployment) the scene's splat is the
        picture's ground, the collision parts stay hidden (group 3)."""
        rr = _rerun()
        if rr is None:
            print("[gate] no Rerun SDK: the picture stays in the runtime", file=log)
            return None
        from rq_pipeline.deploy.runtime import (  # noqa: PLC0415
            assets_dir_of,
            load_scene,
        )

        try:
            model = load_scene(manifest, assets_dir=assets_dir_of(manifest))
            return cls(
                manifest, model, runtime_name, rr=rr, file=file, scene_dir=scene_dir
            )
        except (FileNotFoundError, ValueError, KeyError) as why:
            # an unloadable scene, or a joint the manifest names and the
            # scene lacks: the gate runs, the picture does not
            print(f"[gate] no mirror: {why!r}", file=log)
            return None

    def _scene(self, scene_dir: Path) -> None:
        """The captured scene's splat under the robot, once, static; and
        the stage's cameras seeing it, when the instrument can render
        them (`scenes.cameras`), else a note saying why not."""
        from rq_pipeline.scenes.cameras import open_cameras  # noqa: PLC0415
        from rq_pipeline.scenes.record import SPLAT_FILE  # noqa: PLC0415
        from rq_pipeline.scenes.splat import read_ply  # noqa: PLC0415

        rr = self._rr
        splats = read_ply(Path(scene_dir) / SPLAT_FILE)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rr.log(SCENE_PATH, gaussians(rr, splats), static=True)
        names = self._camera_names()
        self._cameras, why = (
            open_cameras(self.model, splats, names=names) if names else (None, "")
        )
        if self._cameras is None:
            rr.log(
                "gate/notes", rr.TextLog(f"no camera pictures: {why or 'no cameras'}")
            )
        else:
            rr.log(
                "gate/notes",
                rr.TextLog(
                    f"cameras {', '.join(names)} at {CAMERA_HZ} Hz from "
                    f"{self._cameras.splats} splats on {self._cameras.instrument}"
                ),
            )

    def _camera_names(self) -> tuple[str, ...]:
        """The stage's cameras, named by its manifest; none for a plane."""
        return tuple((self.manifest.raw.get(Key.SCENE) or {}).get("cameras") or ())

    def _pictures(self) -> None:
        """One frame per stage camera, at the camera cadence."""
        if self._cameras is None or (self.frames % max(MIRROR_HZ // CAMERA_HZ, 1)):
            return
        rr = self._rr
        for name, frame in self._cameras.render(self.model, self.data).items():
            rr.log(f"{CAMERAS_PATH}/{name}", rr.Image(frame))

    def _layout(self) -> None:
        rr = self._rr
        try:
            import rerun.blueprint as rrb  # noqa: PLC0415

            rr.send_blueprint(
                rrb.Blueprint(
                    rrb.Horizontal(
                        rrb.Spatial3DView(
                            origin="world", name=f"gate · {self.runtime_name}"
                        ),
                        rrb.Vertical(
                            rrb.Horizontal(
                                *(
                                    rrb.Spatial2DView(
                                        origin=f"{CAMERAS_PATH}/{name}", name=name
                                    )
                                    for name in self._camera_names()
                                )
                            ),
                            rrb.TimeSeriesView(origin="gate/command", name="command"),
                            rrb.TimeSeriesView(origin="gate/velocity", name="measured"),
                            rrb.TextLogView(origin="gate/notes", name="trials"),
                        ),
                        column_shares=[3, 2],
                    ),
                    collapse_panels=True,
                )
            )
        except Exception as error:  # the layout is decoration
            rr.log("gate/notes", rr.TextLog(f"layout not sent: {error}"))

    def trial(self, index: int, command: np.ndarray) -> None:
        cmd = ", ".join(
            f"{a} {float(c):+.2f}" for a, c in zip(TWIST_SHORT, command, strict=True)
        )
        self._rr.set_time(TIMELINE, duration=self.seconds)
        self._rr.log("gate/notes", self._rr.TextLog(f"trial {index}: {cmd}"))

    def tick(
        self,
        dt: float,
        pose: tuple[np.ndarray, np.ndarray, np.ndarray],
        command: np.ndarray,
        velocity_b: np.ndarray,
    ) -> None:
        """One control tick: the runtime's pose into the scene, a frame
        every self.every, the command and the measured planar velocity."""
        self.ticks += 1
        self.seconds += float(dt)
        if self.ticks % self.every:
            return
        import mujoco  # noqa: PLC0415

        from rq_pipeline.deploy.runtime import FREE_JOINT_QPOS  # noqa: PLC0415

        rr = self._rr
        position, quat, joints = pose
        self.data.qpos[0:3] = position
        self.data.qpos[3:FREE_JOINT_QPOS] = quat
        self.data.qpos[self.joint_qpos] = joints
        mujoco.mj_forward(self.model, self.data)
        rr.set_time(TIMELINE, duration=self.seconds)
        self._mirror.log(self.data)
        self.frames += 1
        self._pictures()
        for axis, value in zip(TWIST_SHORT, command, strict=True):
            rr.log(f"gate/command/{axis}", rr.Scalars(float(value)))
        for axis, value in zip(TWIST_SHORT[:2], velocity_b[:2], strict=True):
            rr.log(f"gate/velocity/{axis}", rr.Scalars(float(value)))
