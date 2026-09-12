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
from typing import Any

import numpy as np

from rq_pipeline.deploy.manifest import TWIST_SHORT, Manifest

APP_ID = "rq-gate"
TIMELINE = "sim"
# A frame every second control tick: 25 a second at the Go2's 50 Hz,
# near the render stream's 20 Hz mirror budget (its MIRROR_HZ); every
# frame is one log call per geom, and rr.log blocks when it floods.
EVERY_TICKS = 2


def _rerun() -> Any | None:
    try:
        import rerun as rr  # noqa: PLC0415 - the viz extra
    except ImportError:
        return None
    return rr


class GateMirror:
    """One gate run's picture: the deployment's scene driven by the
    runtime's poses, one Rerun recording named for the runtime."""

    def __init__(
        self,
        manifest: Manifest,
        model: Any,
        runtime_name: str,
        *,
        rr: Any,
    ) -> None:
        import mujoco  # noqa: PLC0415 - the sim extra

        from rq_pipeline.viz import (  # noqa: PLC0415
            STUDIO_ADDRESS,
            RigMirror,
            leave_cleanly_on_term,
        )

        self._rr = rr
        self.runtime_name = runtime_name
        self.model = model
        self.data = mujoco.MjData(model)
        self.joint_qpos = np.array(
            [model.jnt_qposadr[model.joint(j).id] for j in manifest.joints.policy_order]
        )
        self._mirror = RigMirror(model, model_colors=True, skip_groups=(3, 4, 5))
        self.ticks = 0
        self.seconds = 0.0
        rr.init(f"{APP_ID}-{runtime_name}", spawn=False)
        rr.connect_grpc(STUDIO_ADDRESS)
        leave_cleanly_on_term(rr)
        self._layout()

    @classmethod
    def open(
        cls, manifest: Manifest, runtime_name: str, *, log: Any = sys.stderr
    ) -> GateMirror | None:
        """The mirror, or None with a note when it cannot be one."""
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
            return cls(manifest, model, runtime_name, rr=rr)
        except (FileNotFoundError, ValueError, KeyError) as why:
            # an unloadable scene, or a joint the manifest names and the
            # scene lacks: the gate runs, the picture does not
            print(f"[gate] no mirror: {why!r}", file=log)
            return None

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
        every EVERY_TICKS, the command and the measured planar velocity."""
        self.ticks += 1
        self.seconds += float(dt)
        if self.ticks % EVERY_TICKS:
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
        for axis, value in zip(TWIST_SHORT, command, strict=True):
            rr.log(f"gate/command/{axis}", rr.Scalars(float(value)))
        for axis, value in zip(TWIST_SHORT[:2], velocity_b[:2], strict=True):
            rr.log(f"gate/velocity/{axis}", rr.Scalars(float(value)))
