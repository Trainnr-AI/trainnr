"""Footage to a G1 reference motion, as one trainnr job (docs/86 §5.1):

    footage --GEM-X--> SOMA motion --gem_to_bvh--> BVH --SOMA Retargeter-->
    G1 CSV --smooth--> mjlab's motion (motion.npz) --> a recording

    uv run python -m trainnr_mjlab.video_reference <footage folder> <recording folder>

GEM-X and the SOMA Retargeter run as external programs in their own
environments (`trainnr.video.nvidia`, installed by tools/video-setup.sh).
Each stage shows in the Studio's Running now; the result is a recording
(collection `video`): the G1's joints and root over time as channels,
and in `raw/` everything the stages wrote: GEM-X's prediction, the BVH,
the retargeted CSV, the motion the tracking task reads, and the motion
replayed on the G1 beside the video (`replay.mp4`).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from trainnr.mcp_jobs import default_jobs_root, jobs_dir_of, track
from trainnr.robots.recording import (
    BASE_POSE,
    JOINT_POSITION,
    POSE_COMPONENTS,
    Channel,
    Recording,
)
from trainnr.video.footage import FOOTAGE_FILE, Footage
from trainnr.video.nvidia import gemx_dir, missing, soma_dir
from trainnr.viz import open_stream

from trainnr_mjlab.tracking_motion import (
    CUTOFF_HZ,
    MOTION_FPS,
    REPLAY_FPS,
    render_replay,
    robot_csv,
    smooth,
    write_mjlab_csv,
    write_motion,
)

KIND = "retarget-video"
ROBOT = "unitree_g1"  # the retargeter's name for its target
ADAPTER = "gem-x+soma-retargeter"
COLLECTION_VIDEO = "video"
GEMX_CONFIG = "weights/body_video_native77.yaml"
SOMA_SKELETON = "soma_retargeter/assets/soma/soma_zero_frame0.bvh"
RAW = "raw"
# The job's live stream into the Studio, kept in the recording's raw/.
STREAM_PREFIX = "trainnr-video-"
STREAM_FILE = "stream.rrd"
TIMELINE = "video"
# The video at the replay's rate and height, the G1 to its right.
SIDE_BY_SIDE = "[1:v]fps=30,scale=-2:480[v];[v][0:v]hstack=inputs=2"


def _run(argv: list[str], cwd: Path, rr: Any) -> str:
    """One external program, its output line by line into the job's log
    (Running now's log) and its stage lines into the stream; the last
    line returned. Refused by name with the last lines on failure."""
    import sys  # noqa: PLC0415

    print(f"$ {' '.join(argv)}", flush=True)
    lines: list[str] = []
    with subprocess.Popen(
        argv,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    ) as child:
        assert child.stdout is not None
        for line in child.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            text = line.strip().split("\r")[-1]
            if text:
                lines.append(text)
                if not text.startswith(("Module ", "  ", "Warp ")):
                    rr.log("stages/programs", rr.TextLog(text[:200]))
    if child.returncode != 0:
        raise SystemExit(
            f"{Path(argv[0]).name} failed ({child.returncode}): {lines[-5:]}"
        )
    return lines[-1] if lines else ""


def _layout() -> Any:
    """The stream's picture: the video and the stages, the G1's reference
    in 3D, every joint over time (raw from the retargeter, smoothed)."""
    import rerun.blueprint as rrb  # noqa: PLC0415

    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial2DView(origin="video", name="video"),
                rrb.TextLogView(origin="stages", name="stages"),
            ),
            rrb.Spatial3DView(origin="g1", name="the G1's reference"),
            rrb.TimeSeriesView(origin="joints", name="joints (rad)"),
        ),
        collapse_panels=True,
    )


def _log_reference(  # noqa: PLR0913 - the reference's parts, each named
    rr: Any,
    video: Path,
    *,
    motion_npz: Path,
    motion: dict[str, Any],
    raw_q: np.ndarray,
    raw_fps: float,
) -> None:
    """The reference on the stream's timeline: the video's frames, the G1's
    bodies and bones in 3D frame by frame, each joint raw and smoothed."""
    import mujoco  # noqa: PLC0415
    from mjlab.asset_zoo.robots.unitree_g1 import g1_constants  # noqa: PLC0415
    from trainnr.project.present import log_frames  # noqa: PLC0415

    log_frames(rr, "video/frames", video)
    npz = np.load(motion_npz)
    model = mujoco.MjModel.from_xml_path(str(g1_constants.G1_XML))
    bodies = motion["bodies"]
    index = {name: i for i, name in enumerate(bodies)}
    bones = [
        (index[model.body(int(model.body(name).parentid[0])).name], i)
        for i, name in enumerate(bodies)
        if model.body(int(model.body(name).parentid[0])).name in index
    ]
    for i, pos in enumerate(npz["body_pos_w"]):
        rr.set_time(TIMELINE, duration=i / MOTION_FPS)
        rr.log("g1/bodies", rr.Points3D(pos, radii=0.025))
        rr.log("g1/bones", rr.LineStrips3D([[pos[a], pos[b]] for a, b in bones]))
    smooth_t = rr.TimeColumn(
        TIMELINE, duration=np.arange(len(npz["joint_pos"])) / MOTION_FPS
    )
    raw_t = rr.TimeColumn(TIMELINE, duration=np.arange(len(raw_q)) / raw_fps)
    for j, name in enumerate(motion["joints"]):
        short = name.removesuffix("_joint")
        rr.send_columns(
            f"joints/{short}/smoothed",
            indexes=[smooth_t],
            columns=rr.Scalars.columns(scalars=npz["joint_pos"][:, j]),
        )
        rr.send_columns(
            f"joints/{short}/raw",
            indexes=[raw_t],
            columns=rr.Scalars.columns(scalars=raw_q[:, j]),
        )


def _commit(root: Path) -> str:
    out = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return out.stdout.strip() or "unrecorded"


def main() -> None:  # noqa: PLR0915 - one pipeline, each stage named
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("footage", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--project", default=None)
    parser.add_argument("--cutoff-hz", type=float, default=CUTOFF_HZ)
    args = parser.parse_args()
    why = missing()
    if why:
        raise SystemExit(why)
    footage = Footage.read(args.footage / FOOTAGE_FILE)
    video = args.footage / footage.video
    out = args.out.resolve()
    if out.exists():
        raise SystemExit(f"{out} already exists")
    gemx, soma = gemx_dir(), soma_dir()
    timings: dict[str, float] = {}
    stream = f"{STREAM_PREFIX}{out.name}"

    with (
        tempfile.TemporaryDirectory(prefix="trainnr-video-") as tmp,
        track(
            KIND,
            jobs_dir=jobs_dir_of(args.project or default_jobs_root()),
            name=out.name,
            viewer=str(out / RAW / STREAM_FILE),
            stream=stream,
        ) as run,
    ):
        work = Path(tmp)
        rr = open_stream(stream, file=work / STREAM_FILE)
        rr.send_blueprint(_layout())

        def stage(n: int, text: str) -> float:
            run.progress(n, 6, "stages", text)
            run.stage(text)
            rr.log("stages", rr.TextLog(text))
            return time.perf_counter()

        t = stage(0, f"GEM-X: the person's motion in {footage.name}")
        _run(
            [
                str(gemx / ".venv/bin/gem-infer"),
                "motion-video",
                "--task",
                "body",
                "--input",
                str(video.resolve()),
                "--config",
                GEMX_CONFIG,
                "--device",
                "cuda",
                "--postprocess",
                "--soma-assets",
                "assets/soma",
                "--output",
                str(work / "gem-x.pt"),
            ],
            gemx,
            rr,
        )
        timings["gem_x_s"] = time.perf_counter() - t

        t = stage(1, "SOMA motion to the retargeter's skeleton, on a level floor")
        bvh_dir = work / "bvh"
        bvh_dir.mkdir()
        level = json.loads(
            _run(
                [
                    str(gemx / ".venv/bin/python"),
                    str(Path(__file__).with_name("gem_to_bvh.py")),
                    str(work / "gem-x.pt"),
                    str(soma / SOMA_SKELETON),
                    str(bvh_dir / "soma.bvh"),
                    "assets/soma",
                ],
                gemx,
                rr,
            )
        )
        rr.log("stages", rr.TextLog(f"floor: {level['floor']}"))
        timings["to_bvh_s"] = time.perf_counter() - t

        t = stage(2, "SOMA Retargeter: onto the G1's joints")
        config = work / "retarget.json"
        config.write_text(
            json.dumps(
                {
                    "import_folder": str(bvh_dir),
                    "export_folder": str(work / "csv"),
                    "batch_size": 1,
                    "retargeter": "Newton",
                    "retarget_source": "soma",
                    "retarget_target": ROBOT,
                    "retarget_source_facing_direction": "Mujoco",
                    "extra_robot_paths": [],
                }
            )
        )
        _run(
            [
                str(soma / ".venv/bin/python"),
                "app/bvh_to_csv_converter.py",
                "--config",
                str(config),
            ],
            soma,
            rr,
        )
        timings["retarget_s"] = time.perf_counter() - t

        t = stage(
            3,
            f"the tracking task's motion, smoothed at {args.cutoff_hz:g} Hz",
        )
        pos, quat, raw_q, joints = robot_csv(work / "csv" / "soma.csv")
        pos, quat, q = smooth(pos, quat, raw_q, level["fps"], args.cutoff_hz)
        write_mjlab_csv(work / "robot.csv", pos, quat, q)
        motion = write_motion(
            work / "robot.csv", work / "motion.npz", level["fps"], joints
        )
        timings["motion_s"] = time.perf_counter() - t
        _log_reference(
            rr,
            video,
            motion_npz=work / "motion.npz",
            motion=motion,
            raw_q=raw_q,
            raw_fps=level["fps"],
        )

        t = stage(4, "the G1 replayed beside the video")
        render_replay(work / "motion.npz", work / "frames")
        subprocess.run(
            [
                "ffmpeg", "-v", "error", "-y", "-framerate", str(REPLAY_FPS),
                "-i", str(work / "frames" / "%05d.png"), "-i", str(video),
                "-filter_complex", SIDE_BY_SIDE,
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-shortest",
                str(work / "replay.mp4"),
            ],
            check=True,
            capture_output=True,
        )  # fmt: skip
        timings["replay_s"] = time.perf_counter() - t

        stage(5, f"the reference: {out.name}")
        npz = np.load(work / "motion.npz")
        times = np.arange(len(npz["joint_pos"])) / MOTION_FPS
        base = np.hstack([npz["body_pos_w"][:, 0], npz["body_quat_w"][:, 0]])
        recording = Recording(
            source=footage.name,
            adapter=ADAPTER,
            collection=COLLECTION_VIDEO,
            channels={
                JOINT_POSITION: Channel(
                    JOINT_POSITION,
                    times,
                    npz["joint_pos"],
                    unit="rad",
                    components=tuple(motion["joints"]),
                ),
                BASE_POSE: Channel(
                    BASE_POSE,
                    times,
                    base,
                    unit="m, unit quaternion",
                    components=POSE_COMPONENTS,
                ),
            },
            census=_census(
                footage,
                motion=motion,
                level=level,
                cutoff_hz=args.cutoff_hz,
                gemx=gemx,
                soma=soma,
                timings=timings,
            ),
            notes=[
                "a person's motion estimated from video (GEM-X) and retargeted onto "
                "the G1 (SOMA Retargeter): kinematic, never measured on a robot",
                f"floor levelled: tilted {level['floor']['tilt_deg']} degrees, lifted "
                f"{level['floor']['lift_m_min']} to {level['floor']['lift_m_max']} m",
            ],
            provenance={"origin": footage.origin, "licence": footage.licence},
        )
        recording.write(out)
        raw = out / RAW
        raw.mkdir()
        for name, src in (
            ("gem-x.pt", work / "gem-x.pt"),
            ("soma.bvh", bvh_dir / "soma.bvh"),
            ("retargeted.csv", work / "csv" / "soma.csv"),
            ("motion.npz", work / "motion.npz"),
            ("replay.mp4", work / "replay.mp4"),
        ):
            shutil.move(str(src), raw / name)
        rr.log("stages", rr.TextLog(f"done: {out.name}"))
        rr.disconnect()
        shutil.move(str(work / STREAM_FILE), raw / STREAM_FILE)
        run.stage(f"done: {out.name}, {motion['frames']} frames at {MOTION_FPS:g} fps")
        print(f"[video] {out}")


def _census(  # noqa: PLR0913 - what made the reference, each part named
    footage: Footage,
    *,
    motion: dict[str, Any],
    level: dict[str, Any],
    cutoff_hz: float,
    gemx: Path,
    soma: Path,
    timings: dict[str, float],
) -> dict[str, Any]:
    return {
        "robot": ROBOT,
        "footage": footage.name,
        "frames": motion["frames"],
        "fps": MOTION_FPS,
        "estimator": {"name": "GEM-X", "commit": _commit(gemx), "config": GEMX_CONFIG},
        "retargeter": {
            "name": "SOMA Retargeter",
            "commit": _commit(soma),
            "target": ROBOT,
        },
        "floor": level["floor"],
        "smoothing": f"zero-phase Butterworth, 4th order, {cutoff_hz:g} Hz",
        "seconds": {k: round(v, 1) for k, v in timings.items()},
    }


if __name__ == "__main__":
    main()
