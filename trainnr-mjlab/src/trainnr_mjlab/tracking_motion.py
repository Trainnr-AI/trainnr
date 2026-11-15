"""A retargeted robot motion as mjlab's tracking task reads it, and a
picture of it beside the video it came from.

The SOMA Retargeter writes a CSV with a header: the root in centimetres,
its rotation as extrinsic X-Y-Z Euler degrees, the joints in degrees.
mjlab reads one without a header: metres, a quaternion x y z w, radians.
Video motion jitters from frame to frame (GEM-X on the court clip: joint
speeds to 31 rad/s), so a zero-phase low-pass is applied first; the
dance's own speed stays.

mjlab's `scripts/csv_to_npz.py` replays the motion on its G1 scene to get
every body's pose and velocity, then uploads the file to Weights &
Biases and keeps it at `/tmp/motion.npz`; `write_motion` is the same
replay through mjlab's own `MotionLoader`, written where it is asked.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

CUTOFF_HZ = 6.0
MOTION_FPS = 50.0  # mjlab's tracking control rate
REPLAY_FPS = 30.0
REPLAY_SIZE = 480


def write_mjlab_csv(
    path: Path, pos: np.ndarray, quat: np.ndarray, q: np.ndarray
) -> None:
    np.savetxt(path, np.hstack([pos, quat, q]), delimiter=",", fmt="%.7f")


def robot_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """The retargeter's CSV: root position (m), root quaternion (x, y, z, w),
    joints (rad), and the joint names in its column order."""
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    with Path(path).open(encoding="utf-8") as f:
        rows = list(csv.reader(f))
    head, body = rows[0], np.asarray(rows[1:], dtype=np.float64)
    col = {name: i for i, name in enumerate(head)}
    pos = body[:, [col[f"root_translate{a}"] for a in "XYZ"]] * 0.01
    euler = body[:, [col[f"root_rotate{a}"] for a in "XYZ"]]
    quat = Rotation.from_euler("xyz", euler, degrees=True).as_quat()
    joints = head[7:]
    return pos, quat, np.deg2rad(body[:, [col[j] for j in joints]]), joints


def smooth(
    pos: np.ndarray, quat: np.ndarray, q: np.ndarray, fps: float, cutoff_hz: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A zero-phase Butterworth low-pass along time on all three; the
    quaternions kept in one hemisphere first, renormalized after."""
    from scipy.signal import butter, filtfilt  # noqa: PLC0415

    b, a = butter(4, cutoff_hz / (fps / 2))
    quat = quat.copy()
    for i in range(1, len(quat)):
        if np.dot(quat[i], quat[i - 1]) < 0:
            quat[i] = -quat[i]
    quat = filtfilt(b, a, quat, axis=0)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    return filtfilt(b, a, pos, axis=0), quat, filtfilt(b, a, q, axis=0)


def write_motion(
    mjlab_csv: Path,
    out: Path,
    input_fps: float,
    joints: list[str],
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Replay the motion (its joint columns named by `joints`) on mjlab's G1
    tracking scene and save every body's pose and velocity at `MOTION_FPS`
    to `out` (the npz the task reads)."""
    from mjlab.scene import Scene  # noqa: PLC0415
    from mjlab.scripts.csv_to_npz import MotionLoader  # noqa: PLC0415
    from mjlab.sim.sim import Simulation, SimulationCfg  # noqa: PLC0415
    from mjlab.tasks.tracking.config.g1.env_cfgs import (  # noqa: PLC0415
        unitree_g1_flat_tracking_env_cfg,
    )

    sim_cfg = SimulationCfg()
    sim_cfg.mujoco.timestep = 1.0 / MOTION_FPS
    scene = Scene(unitree_g1_flat_tracking_env_cfg().scene, device=device)
    sim = Simulation(num_envs=1, cfg=sim_cfg, model=scene.compile(), device=device)
    scene.initialize(sim.mj_model, sim.model, sim.data)
    robot = scene["robot"]
    index = robot.find_joints(joints, preserve_order=True)[0]
    motion = MotionLoader(
        motion_file=str(mjlab_csv),
        input_fps=round(input_fps),
        output_fps=int(MOTION_FPS),
        device=sim.device,
    )
    keys = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w")
    keys += ("body_lin_vel_w", "body_ang_vel_w")
    log: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    for _ in range(motion.output_frames):
        (pos, rot, lin, ang, dof, dof_vel), _ = motion.get_next_state()
        root = robot.data.default_root_state.clone()
        root[:, :3] = pos
        root[:, :2] += scene.env_origins[:, :2]
        root[:, 3:7], root[:, 7:10], root[:, 10:] = rot, lin, ang
        robot.write_root_state_to_sim(root)
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, index], joint_vel[:, index] = dof, dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        sim.forward()
        scene.update(sim.mj_model.opt.timestep)
        data = robot.data
        for key, value in (
            ("joint_pos", data.joint_pos),
            ("joint_vel", data.joint_vel),
            ("body_pos_w", data.body_link_pos_w),
            ("body_quat_w", data.body_link_quat_w),
            ("body_lin_vel_w", data.body_link_lin_vel_w),
            ("body_ang_vel_w", data.body_link_ang_vel_w),
        ):
            log[key].append(value[0].cpu().numpy().copy())
    arrays = {k: np.stack(v) for k, v in log.items()}
    np.savez(out, fps=[MOTION_FPS], **arrays)  # type: ignore[arg-type]
    return {
        "frames": int(motion.output_frames),
        "fps": MOTION_FPS,
        "joints": list(robot.joint_names),
        "bodies": list(robot.body_names),
    }


def render_replay(motion_npz: Path, frames_dir: Path) -> int:
    """The motion posed on the G1 frame by frame (no physics), from a
    camera that follows the pelvis, as PNGs at `REPLAY_FPS`."""
    import mujoco  # noqa: PLC0415
    from mjlab.asset_zoo.robots.unitree_g1 import g1_constants  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    motion = np.load(motion_npz)
    fps = float(motion["fps"][0])
    spec = mujoco.MjSpec.from_file(str(g1_constants.G1_XML))
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], rgba=[0.35, 0.4, 0.38, 1]
    )
    spec.worldbody.add_light(pos=[0, 0, 4], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
    model = spec.compile()
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, REPLAY_SIZE, REPLAY_SIZE)
    camera = mujoco.MjvCamera()
    camera.distance, camera.elevation, camera.azimuth = 3.2, -10.0, 135.0
    frames_dir.mkdir(parents=True, exist_ok=True)
    total = len(motion["joint_pos"])
    picks = np.round(np.arange(0, total / fps, 1 / REPLAY_FPS) * fps).astype(int)
    for k, i in enumerate(picks.clip(0, total - 1)):
        data.qpos[:3] = motion["body_pos_w"][i, 0]
        data.qpos[3:7] = motion["body_quat_w"][i, 0]
        data.qpos[7:] = motion["joint_pos"][i]
        mujoco.mj_forward(model, data)
        camera.lookat[:] = data.qpos[:3]
        camera.lookat[2] = 0.8
        renderer.update_scene(data, camera)
        Image.fromarray(renderer.render()).save(frames_dir / f"{k:05d}.png")
    return len(picks)
