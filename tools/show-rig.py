"""The fetch loop: a cube on the floor, and a robot that goes and gets it.

Run from trainnr/ (needs the sim extra and macOS's mjpython):

    cd trainnr && uv run mjpython ../tools/show-rig.py        # fetch demo
    cd trainnr && uv run mjpython ../tools/show-rig.py car    # car alone
    cd trainnr && uv run mjpython ../tools/show-rig.py arm    # arm alone

Cycle: SEEK the cube (bearing control) -> CREEP onto it (pulsed inchworm
with lateral servo, +-4 mm; retreats and retries on a lateral miss) ->
PICK it off the floor -> HAUL it somewhere else -> PLACE it -> BACK OFF
-> seek again. The cube's position each lap is wherever the last lap
left it, so the patrol wanders honestly. Close the window to stop.
"""

import math
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
from trainnr.tasks.components import (
    CROUCH_POSE,
    GROUND_GRASP_POINT,
    GROUND_HOLDUP,
    GROUND_PICK_SEQUENCE,
    GROUND_PLACE_SEQUENCE,
    add_floor_cube,
    compose,
)
from trainnr.tasks.scene import NominalOptions

mode = sys.argv[1] if len(sys.argv) > 1 else "both"
scene = compose(car=mode in ("car", "both"), arm=mode in ("arm", "both"))
if mode == "both":
    add_floor_cube(scene, (0.9, 0.35))
model = scene.compile()
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

HOME = [0.0, -1.57, 1.57, 1.57, -1.57, 0.0]
WAVE = [0.6, -1.2, 1.2, 0.5, 1.2, 1.0]
TIMESTEP = NominalOptions.TIMESTEP
STEPS_PER_SYNC = 8
GX, GY = GROUND_GRASP_POINT
HAS_CAR = mode in ("car", "both")


def chassis_pose():
    w, _qx, _qy, qz = data.qpos[3:7]
    return data.qpos[0], data.qpos[1], 2 * math.atan2(qz, w)


def cube_in_chassis():
    x, y, yaw = chassis_pose()
    dx, dy = data.qpos[15] - x, data.qpos[16] - y
    c, s = math.cos(-yaw), math.sin(-yaw)
    return c * dx - s * dy, s * dx + c * dy


class Fetch:
    def __init__(self):
        self.state = "seek"
        self.pulse_t = 0
        self.arm_from = list(CROUCH_POSE)
        self.arm_to = list(CROUCH_POSE)
        self.arm_started = 0.0
        self.seq = None
        self.seq_index = 0
        self.seq_started = 0.0
        self.haul_until = 0.0
        self.backoff_until = 0.0
        # Drop points cycle around the origin so the fetch loop stays
        # centred on the grey floor instead of compounding outward.
        self.drops = [(1.0, 0.0), (0.4, 0.9), (-0.7, 0.5), (-0.3, -0.9), (0.8, -0.6)]
        self.drop_index = 0

    def _set_arm(self, t, target):
        if list(target) != self.arm_to:
            self.arm_from = self.arm_now(t)
            self.arm_to = list(target)
            self.arm_started = t

    def arm_now(self, t):
        alpha = min(1.0, (t - self.arm_started) / 1.2)
        return list(
            (1 - alpha) * np.array(self.arm_from) + alpha * np.array(self.arm_to)
        )

    def _run_sequence(self, t):
        pose, duration = self.seq[self.seq_index]
        self._set_arm(t, pose)
        if t - self.seq_started >= duration:
            self.seq_index += 1
            self.seq_started = t
            return self.seq_index >= len(self.seq)
        return False

    def wheels(self, t, step):
        cx, cy = cube_in_chassis()
        x, y, yaw = chassis_pose()
        speed = math.hypot(data.qvel[0], data.qvel[1])
        gap, lat = cx - GX, cy - GY

        if self.state == "seek":
            self._set_arm(t, CROUCH_POSE)
            bearing = math.atan2(data.qpos[16] - y, data.qpos[15] - x)
            err = (bearing - yaw + math.pi) % (2 * math.pi) - math.pi
            dist = math.hypot(data.qpos[15] - x, data.qpos[16] - y)
            if dist < 0.5 and abs(err) < 0.12:
                self.state = "creep"
                return (0.0, 0.0)
            if abs(err) > 0.12:
                return (-0.8, 0.8) if err > 0 else (0.8, -0.8)
            trim = max(-0.25, min(0.25, 2.0 * err))
            return (0.7 - trim, 0.7 + trim)

        if self.state == "creep":
            trim = max(-0.2, min(0.2, 5.0 * lat))
            if abs(gap) <= 0.004:
                if speed < 0.01:
                    if abs(lat) <= 0.004:
                        self.state = "pick"
                        self.seq = list(GROUND_PICK_SEQUENCE)
                        self.seq_index = 0
                        self.seq_started = t
                    else:
                        self.state = "retreat"
                        self.pulse_t = step
                return (0.0, 0.0)
            if gap > 0.12:
                return (0.5 - trim, 0.5 + trim) if speed < 0.25 else (0.0, 0.0)
            phase = (step - self.pulse_t) % 280
            direction = 1.0 if gap > 0 else -1.0
            if phase < 80 and speed < 0.06:
                return (direction * 0.55 - trim, direction * 0.55 + trim)
            return (0.0, 0.0)

        if self.state == "retreat":
            if gap >= 0.07:
                self.state = "creep"
                self.pulse_t = step
                return (0.0, 0.0)
            phase = (step - self.pulse_t) % 280
            return (-0.55, -0.55) if (phase < 80 and speed < 0.08) else (0.0, 0.0)

        if self.state == "pick":
            if self._run_sequence(t):
                self.state = "haul"
                self._set_arm(t, GROUND_HOLDUP)
            return (0.0, 0.0)

        if self.state == "haul":
            self._set_arm(t, GROUND_HOLDUP)
            tx, ty = self.drops[self.drop_index]
            dist = math.hypot(tx - x, ty - y)
            if dist < 0.3:
                if speed < 0.02:
                    self.drop_index = (self.drop_index + 1) % len(self.drops)
                    self.state = "place"
                    self.seq = list(GROUND_PLACE_SEQUENCE)
                    self.seq_index = 0
                    self.seq_started = t
                return (0.0, 0.0)
            bearing = math.atan2(ty - y, tx - x)
            err = (bearing - yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(err) > 0.25:
                return (-0.6, 0.6) if err > 0 else (0.6, -0.6)
            trim = max(-0.2, min(0.2, 1.5 * err))
            return (0.55 - trim, 0.55 + trim)

        if self.state == "place":
            if self._run_sequence(t):
                self.state = "backoff"
                self.backoff_until = t + 2.0
                self._set_arm(t, CROUCH_POSE)
            return (0.0, 0.0)

        # backoff: reverse clear of the cube, then seek again
        self._set_arm(t, CROUCH_POSE)
        if t >= self.backoff_until:
            self.state = "seek"
            return (0.0, 0.0)
        return (-0.5, -0.5)


fetch = Fetch()
ARM_SCRIPT = [(3.0, HOME), (2.5, WAVE), (2.5, CROUCH_POSE)]
ARM_TOTAL = sum(d for d, _ in ARM_SCRIPT)

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.lookat[:] = [0.4, 0.1, 0.1]
    viewer.cam.distance = 2.0
    viewer.cam.elevation = -25
    viewer.cam.azimuth = 140
    step = 0
    while viewer.is_running():
        frame_start = time.time()
        for _ in range(STEPS_PER_SYNC):
            if step % 10 == 0:
                t = step * TIMESTEP
                if mode == "both":
                    wheels = fetch.wheels(t, step)
                    data.ctrl[:] = list(wheels) + fetch.arm_now(t)
                elif mode == "car":
                    cycle = t % 8.0
                    data.ctrl[:] = (
                        [0.7, 0.7]
                        if cycle < 3.0
                        else ([0.5, -0.5] if cycle < 5.0 else [0.7, 0.7])
                    )
                else:
                    acc, target = 0.0, CROUCH_POSE
                    cycle = t % ARM_TOTAL
                    for duration, pose in ARM_SCRIPT:
                        if cycle < acc + duration:
                            target = pose
                            break
                        acc += duration
                    data.ctrl[:] = target
            mujoco.mj_step(model, data)
            step += 1
        if HAS_CAR:
            viewer.cam.lookat[:2] = 0.92 * viewer.cam.lookat[:2] + 0.08 * data.qpos[:2]
        viewer.sync()
        rest = STEPS_PER_SYNC * TIMESTEP - (time.time() - frame_start)
        if rest > 0:
            time.sleep(rest)
