#!/usr/bin/env python3
"""gripper-pick in both viewers: the USD-imported 2F-85 on its carriage,
the acceptance ladder's rungs (pick, no-close, limp) replayed from the
exact states the referee judged — the MuJoCo viewer live, the Rerun
stream into the Studio (and a file), stills of the grasp and the hold.

    cd trainnr && uv run --no-sync --extra sim --extra viz \\
        python ../tools/show-gripper-pick.py --stills ../docs/figures/gripper-pick
    # --trial 2          another paired start (the band's corners, 0-3)
    # --rungs pick       one rung only;  --no-viewer  headless (stills + stream)

The episodes are the ladder's own (`tasks/gripper_pick.LADDER`), run
through the task's expert once per rung; the viewers REPLAY their
FULLPHYSICS rows, so what is on screen is what was judged, not a second
simulation. Rerun, one recording on one clock (sim_time), narrated at
10 Hz (docs/e2e-research/55):

    world/rig                     the pick rung's mesh-true mirror
    cameras/front                 the task's front camera, pick rung
    ladder/<rung>/cube_lift_m     cube height above its spawn, per rung
    ladder/<rung>/finger_rad      the finger joint, per rung
    ladder/<rung>/verdict         the referee's verdict and the funnel
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from _lab import bootstrap, frame_viewer, hold_until_closed

bootstrap()

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from trainnr.envs.gymnasium_env import bundle_source  # noqa: E402
from trainnr.evaluate.harness import home_state  # noqa: E402
from trainnr.physics.mujoco_backend import MuJoCoBackend  # noqa: E402
from trainnr.protocol import events_for  # noqa: E402
from trainnr.tasks.gripper_pick import (  # noqa: E402
    BASE_BODY,
    EXPERT_RUNG,
    FINGER_SENSOR,
    GRIPPER_PICK_SPEC,
    LADDER,
    Layout,
    PickChoreography,
    build_gripper_pick,
)
from trainnr.tasks.scene import GeomGroup  # noqa: E402
from trainnr.viz import RigMirror, open_stream, viewer_file  # noqa: E402

APP_ID = "trainnr-show-gripper-pick"


@dataclass(frozen=True)
class Show:
    """The replay's pacing and the stills' framing."""

    narrate_hz: float = 10.0  # the viz catalog's live cap
    camera_hz: float = 5.0
    viewer_hz: float = 50.0  # replay at real time, synced this often
    frame_w: int = 640
    frame_h: int = 480
    # stills: (file stem, rung, seconds into the episode; negative counts
    # back from its end, so a longer or shorter episode still has them)
    moments: tuple[tuple[str, str, float], ...] = (
        ("pick-grasp", "pick", PickChoreography.CLOSE_S - 0.05),
        ("pick-hold", "pick", -0.5),
        ("no-close-end", "no-close", -0.5),
        ("limp-end", "limp", -0.5),
    )
    close_distance: float = 0.42
    close_azimuth: float = 160.0
    close_elevation: float = -10.0


SHOW = Show()


def seat(model, data, row) -> None:
    mujoco.mj_setState(model, data, row, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    mujoco.mj_forward(model, data)


def close_camera(data, model, show: Show) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    base = data.xpos[model.body(BASE_BODY).id]
    cam.lookat[:] = (base[0], base[1], base[2] - PickChoreography.PAD_REACH_M * 0.7)
    cam.distance = show.close_distance
    cam.azimuth = show.close_azimuth
    cam.elevation = show.close_elevation
    return cam


def visual_option() -> mujoco.MjvOption:
    opt = mujoco.MjvOption()
    opt.geomgroup[GeomGroup.COLLISION] = 0  # the hulls, not the visuals
    return opt


@dataclass(frozen=True)
class Episode:
    """One rung's judged episode: its rows, the verdict, the milestones."""

    states: np.ndarray
    sensors: np.ndarray
    success: bool
    fired: list[str]


def run_ladder(task, model, start, rungs) -> dict[str, Episode]:
    """Each rung through the task's own expert, judged by its referee."""
    protocol = task.protocol
    layout = Layout.of(model)
    z = layout.cube_pos.start + 2
    episodes = {}
    for rung in rungs:
        states, sensors = LADDER[rung](model, start, spec=task.task_spec)[:2]
        episode = Episode(
            states,
            sensors,
            bool(protocol.success(states, sensors)),
            [e["name"] for e in events_for(protocol, states, sensors)],
        )
        episodes[rung] = episode
        print(
            f"  {rung:9s} {'SUCCESS' if episode.success else 'failure'}  milestones "
            f"{episode.fired or 'none'}  cube rise at the end "
            f"{(states[-1, z] - states[0, z]) * 1000:.1f} mm",
            flush=True,
        )
    return episodes


def stream(rr, model, episodes: dict[str, Episode], show: Show) -> None:
    """The Rerun narration: one clock, 10 Hz, the expert rung mirrored."""
    layout = Layout.of(model)
    z = layout.cube_pos.start + 2
    mirror = RigMirror(model, model_colors=True, skip_groups=(GeomGroup.COLLISION,))
    renderer = mujoco.Renderer(model, show.frame_h, show.frame_w)
    data = mujoco.MjData(model)
    dt = float(model.opt.timestep)
    narrate = max(1, round(1.0 / (show.narrate_hz * dt)))
    frames = max(1, round(1.0 / (show.camera_hz * dt)))
    for rung, ep in episodes.items():
        for k in range(0, len(ep.states), narrate):
            rr.set_time("sim_time", duration=k * dt)
            rise = ep.states[k, z] - ep.states[0, z]
            rr.log(f"ladder/{rung}/cube_lift_m", rr.Scalars([rise]))
            finger = ep.sensors[k, FINGER_SENSOR]
            rr.log(f"ladder/{rung}/finger_rad", rr.Scalars([finger]))
            if rung != EXPERT_RUNG:
                continue
            seat(model, data, ep.states[k])
            mirror.log(data)
            if k % frames == 0:
                renderer.update_scene(data, camera="front")
                rr.log("cameras/front", rr.Image(renderer.render()).compress(85))
        rr.set_time("sim_time", duration=(len(ep.states) - 1) * dt)
        verdict = "SUCCESS" if ep.success else "failure"
        milestones = ", ".join(ep.fired) or "none"
        rr.log(
            f"ladder/{rung}/verdict",
            rr.TextLog(f"{rung}: {verdict}; milestones {milestones}"),
        )
    renderer.close()


def stills(model, episodes: dict[str, Episode], folder: Path, show: Show) -> None:
    """A close-up and the task's front camera at each named instant."""
    folder.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, show.frame_h, show.frame_w)
    data = mujoco.MjData(model)
    dt = float(model.opt.timestep)
    for stem, rung, at_s in show.moments:
        if rung not in episodes:
            continue
        # a negative instant indexes from the episode's end
        seat(model, data, episodes[rung].states[round(at_s / dt)])
        renderer.update_scene(data, close_camera(data, model, show), visual_option())
        path = folder / f"{stem}.png"
        Image.fromarray(renderer.render()).save(path)
        renderer.update_scene(data, camera="front")
        Image.fromarray(renderer.render()).save(folder / f"{stem}-front.png")
        print(f"  still {path}", flush=True)
    renderer.close()


def replay(model, episodes: dict[str, Episode], show: Show, *, hold: bool) -> None:
    """The MuJoCo viewer, replaying each rung's judged rows at real time."""
    data = mujoco.MjData(model)
    dt = float(model.opt.timestep)
    every = max(1, round(1.0 / (show.viewer_hz * dt)))
    with mujoco.viewer.launch_passive(model, data) as viewer:
        frame_viewer(viewer, 0.9, azimuth=120, elevation=-20, lookat=(0, 0, 0.1))
        for rung, ep in episodes.items():
            print(f"  viewer: {rung}", flush=True)
            for k in range(0, len(ep.states), every):
                if not viewer.is_running():
                    return
                seat(model, data, ep.states[k])
                viewer.sync()
                time.sleep(every * dt)
        if hold:
            hold_until_closed(viewer)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--trial", type=int, default=0, choices=range(GRIPPER_PICK_SPEC.trials)
    )
    parser.add_argument("--rungs", nargs="+", default=list(LADDER), choices=LADDER)
    parser.add_argument("--stills", type=Path, default=None)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--hold", action="store_true", help="keep the window open")
    parser.add_argument(
        "--stream-file",
        type=Path,
        default=viewer_file("runs/show-gripper-pick", "ladder"),
    )
    args = parser.parse_args(argv)
    task = build_gripper_pick()
    backend = MuJoCoBackend()
    backend.load_spec(task.spec)
    model = backend.model
    start = task.protocol.perturb(args.trial, home_state(backend, task.protocol))
    source = bundle_source(task.bundle_dir)
    print(f"{task.stamp} on {source}, trial {args.trial}", flush=True)
    episodes = run_ladder(task, model, start, args.rungs)
    rr = open_stream(APP_ID, file=args.stream_file)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    note = f"{task.stamp} on {source}, trial {args.trial}: {task.instruction}"
    rr.log("task", rr.TextDocument(note), static=True)
    stream(rr, model, episodes, SHOW)
    if args.stills is not None:
        stills(model, episodes, args.stills, SHOW)
    if not args.no_viewer:
        replay(model, episodes, SHOW, hold=args.hold)
    rr.get_global_data_recording().flush(timeout_sec=10.0)
    # the ladder as the critic reads it: the expert passes, no other rung
    graded = all(ep.success == (rung == EXPERT_RUNG) for rung, ep in episodes.items())
    return 0 if graded else 1


if __name__ == "__main__":
    sys.exit(main())
