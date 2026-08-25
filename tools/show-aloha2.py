"""The ALOHA 2 bundle in both viewers: rig, servos, contacts, cameras.

    cd pipeline && WGPU_BACKEND=vulkan uv run --extra sim --extra viz \
        python ../tools/show-aloha2.py

Three laps, each narrated as a stage note:

    NEUTRAL   hold the bundle's neutral_pose keyframe under gravity
    TRAVEL    both arms to the contract test's offset pose and back
    JAM       the finding: reset to qpos=0 (arms straight up), command
              neutral, watch the grippers meet at the top centre

MuJoCo shows the live rig. Rerun collects, on the sim_time clock:

    world/rig                 the mesh-true mirror (visual geoms only)
    cameras/<name>            the four D405-matched cameras, 10 Hz
    servo/<arm>/<joint>       measured position (rad) and its command
    contact/max_force_N       largest contact normal force this step
    stage                     lap notes

Close the MuJoCo window (or Ctrl-C) to stop. Linux/WSL runs the viewer
under plain python; macOS needs mjpython.
"""

import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import rerun as rr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _rig3d import RigMirror

BUNDLE = Path(__file__).resolve().parents[1] / "robots" / "aloha2-nominal"
NEUTRAL = [0, -0.96, 1.16, 0, -0.3, 0, 0.0084] * 2
OFFSET = [0.2, -0.7, 0.9, 0.0, -0.5, 0.3, 0.02] * 2
SERVOS = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
    "gripper",
]
ARMS = ["left", "right"]
CAMERAS = ["overhead_cam", "worms_eye_cam", "wrist_cam_left", "wrist_cam_right"]
CAM_W, CAM_H = 480, 270
SIM_HZ = 500
LOG_EVERY = 10  # 50 Hz plots and mirror
CAM_EVERY = 50  # 10 Hz images
COLLISION_GROUP = 3


def max_contact_force(model, data):
    best = 0.0
    force = np.zeros(6)
    for i in range(data.ncon):
        mujoco.mj_contactForce(model, data, i, force)
        best = max(best, float(force[0]))
    return best


class Session:
    """One model, both viewers: steps physics, syncs MuJoCo, feeds Rerun."""

    def __init__(self, model, data, viewer):
        self.model, self.data, self.viewer = model, data, viewer
        self.renderer = mujoco.Renderer(model, CAM_H, CAM_W)
        self.mirror = RigMirror(
            model, model_colors=True, skip_groups=(COLLISION_GROUP,)
        )
        self.clock = 0.0
        self.steps = 0

    def run(self, seconds, ctrl, stage):
        self.note(stage)
        self.data.ctrl[:] = ctrl
        for step in range(int(seconds * SIM_HZ)):
            mujoco.mj_step(self.model, self.data)
            self.tick()
            if not self.viewer.is_running():
                raise SystemExit("viewer closed")
            if step % 10 == 0:
                self.viewer.sync()
                time.sleep(0.004)

    def note(self, text):
        rr.set_time("sim_time", duration=self.clock)
        rr.log("stage", rr.TextLog(text))
        print(text)

    def tick(self):
        self.clock += 1.0 / SIM_HZ
        rr.set_time("sim_time", duration=self.clock)
        if self.steps % LOG_EVERY == 0:
            self._plots()
        if self.steps % CAM_EVERY == 0:
            self._cameras()
        self.steps += 1

    def _plots(self):
        data = self.data
        self.mirror.log(data, path="world/rig")
        for a, arm in enumerate(ARMS):
            for j, servo in enumerate(SERVOS):
                k = a * len(SERVOS) + j
                rr.log(f"servo/{arm}/{servo}/measured", rr.Scalars(data.sensordata[k]))
                rr.log(f"servo/{arm}/{servo}/command", rr.Scalars(data.ctrl[k]))
        rr.log("contact/max_force_N", rr.Scalars(max_contact_force(self.model, data)))

    def _cameras(self):
        for cam in CAMERAS:
            self.renderer.update_scene(self.data, camera=cam)
            rr.log(f"cameras/{cam}", rr.Image(self.renderer.render()))


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(BUNDLE / "aloha2.xml"))
    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 720
    data = mujoco.MjData(model)

    rr.init("robotiq-aloha2", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    mujoco.mj_resetDataKeyframe(model, data, 0)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 2.2
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20
        viewer.cam.lookat[:] = [0.0, 0.0, 0.2]
        session = Session(model, data, viewer)
        session.run(1.5, NEUTRAL, "NEUTRAL: hold neutral_pose under gravity")
        session.run(2.0, OFFSET, "TRAVEL: both arms to the offset pose")
        session.run(2.0, NEUTRAL, "TRAVEL: back to neutral")

        mujoco.mj_resetData(model, data)  # qpos = 0: both arms straight up
        viewer.sync()
        session.run(3.0, NEUTRAL, "JAM: from qpos=0 to neutral - grippers meet")
        peak = max_contact_force(model, data)
        session.note(
            f"JAM: contact force now {peak:.0f} N - close the MuJoCo window to finish"
        )
        while viewer.is_running():
            time.sleep(0.2)


if __name__ == "__main__":
    main()
