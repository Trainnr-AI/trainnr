"""The shared lab bench — the rituals every tool used to hand-roll.

    from _lab import bootstrap, rr_session

Import-order contract: `bootstrap()` must run before any `rq_pipeline`
import (it is what puts <repo>/pipeline on sys.path), so callers keep
`# noqa: E402` on the imports that follow the call.
"""

import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
TOOLS = Path(__file__).resolve().parent
# Frame decimation the tools share: a 50 Hz control loop previewed at 10 Hz.
PREVIEW_EVERY_TICKS = 5
# Camera frames go to the viewer JPEG-encoded: a raw 640x480 RGB frame is
# 0.9 MB, ten a second is 9 MB/s down a pod's SSH tunnel and 200 MB per
# played episode in a saved .rrd (measured 2026-08-28); 85 keeps the
# pads and the parts legible at ~1/15 the bytes.
PREVIEW_JPEG_QUALITY = 85


class LeRobotScripts:
    """LeRobot's entry points, run as modules in whatever interpreter
    launched the tool — no venv name, no bin/ layout baked in."""

    TRAIN = "lerobot.scripts.lerobot_train"
    EVAL = "lerobot.scripts.lerobot_eval"


class LeRobotDefaults:
    """The knobs every LeRobot command line from this bench shares."""

    LOG_FREQ = 50  # trainer log lines, in steps
    # Synchronous vector envs: the rollouts are render-bound and `spawn`
    # workers buy nothing on one renderer.
    SYNC_ENVS = "--eval.use_async_envs=false"


VIEWER_HOLD_PERIOD_S = 0.2  # how often a held viewer window is polled


def load_dotenv(path: Path = REPO / ".env") -> list[str]:
    """Read KEY=VALUE lines from `path` into the environment (existing
    variables win) and return the NAMES set — never the values. The
    repo's `.env` holds API credentials; tools load it, nothing prints
    it, and `cloud-gpu push` never ships it."""

    names: list[str] = []
    if not path.is_file():
        return names
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value
            names.append(key)
    return names


def bootstrap() -> None:
    """Put <repo>/pipeline and <repo>/tools at the front of sys.path."""
    for root in (str(REPO / "tools"), str(REPO / "pipeline")):
        if root not in sys.path:
            sys.path.insert(0, root)


def frame_viewer(
    viewer: Any,
    distance: float,
    *,
    azimuth: float = 90.0,
    elevation: float = -40.0,
    lookat: Sequence[float] = (0.0, 0.0, 0.0),
) -> None:
    """Point a passive MuJoCo viewer's camera — the four lines every rig
    tool wrote for itself."""
    viewer.cam.distance = distance
    viewer.cam.azimuth = azimuth
    viewer.cam.elevation = elevation
    viewer.cam.lookat[:] = list(lookat)


def viewer_executable() -> str | None:
    """Where the Rerun viewer binary is when it is not on PATH: the sim
    venv installs it beside the SDK (`rerun-sdk`), the train venv only the
    SDK — so tools launched from the train venv used to need
    `PATH="$PWD/.venv/bin:$PATH"` (2026-08-26). None means "on PATH"."""

    from rq_pipeline.mcp_actions import venv_bin  # noqa: PLC0415 - after bootstrap

    if shutil.which("rerun"):
        return None
    candidate = venv_bin(REPO / "pipeline" / ".venv", "rerun")
    return str(candidate) if candidate.exists() else None


# The Rerun viewer decodes H.264 (LeRobot's eval videos) through an
# external `ffmpeg` and wants at least this version; Ubuntu 22.04 ships
# 4.4.2, and the viewer showed "Failed to decode" (2026-08-28).
VIEWER_FFMPEG_MIN = (5, 1)
VIEWER_BIN = REPO / "pipeline" / "runs" / ".viewer-bin"  # runs/ is git-ignored


def ffmpeg_on_path() -> str | None:
    """A modern `ffmpeg` on PATH for a viewer this process spawns: the
    system one when it is new enough, else imageio-ffmpeg's bundled
    static binary (in both venvs already, under its versioned name),
    exposed under the plain name in a directory prepended to PATH.
    Returns what the viewer will find, or None."""

    system = shutil.which("ffmpeg")
    if system:
        banner = subprocess.run(
            [system, "-version"], capture_output=True, text=True, check=False
        ).stdout
        found = re.search(r"ffmpeg version (\d+)\.(\d+)", banner)
        if found and (int(found.group(1)), int(found.group(2))) >= VIEWER_FFMPEG_MIN:
            return system
    try:
        import imageio_ffmpeg  # noqa: PLC0415
    except ImportError:
        return system
    bundled = Path(imageio_ffmpeg.get_ffmpeg_exe())
    VIEWER_BIN.mkdir(parents=True, exist_ok=True)
    exposed = VIEWER_BIN / ("ffmpeg.exe" if bundled.suffix == ".exe" else "ffmpeg")
    if not exposed.exists():
        try:
            exposed.symlink_to(bundled)
        except OSError:  # a filesystem without symlinks: copy the binary
            shutil.copy2(bundled, exposed)
    os.environ["PATH"] = str(VIEWER_BIN) + os.pathsep + os.environ.get("PATH", "")
    return str(exposed)


def rr_session(
    app_id: str,
    *,
    mode: str = "attach",
    recording_id: str | None = None,
    world_up: bool = True,
) -> None:
    """Open a Rerun stream: "attach" joins a running viewer and spawns one
    if none answers, "spawn" always opens a fresh viewer, "connect" joins
    a viewer known to exist (and fails loudly when it does not). Every
    rig session logs a Z-up world; `world_up=False` for streams that
    log no 3D."""
    import rerun as rr  # noqa: PLC0415 — keep the bench importable without viz extras

    kwargs = {"recording_id": recording_id} if recording_id is not None else {}
    rr.init(app_id, spawn=False, **kwargs)
    ffmpeg_on_path()  # before the viewer is spawned: it inherits PATH
    spawn = {"executable_path": viewer_executable()}
    if mode == "spawn":
        rr.spawn(**spawn)
    elif mode == "attach":
        try:
            rr.connect_grpc()
        except Exception:
            rr.spawn(**spawn)
    elif mode == "connect":
        rr.connect_grpc()
    else:
        raise ValueError(f"unknown rr_session mode {mode!r}")
    if world_up:
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)


def load_demo_actions(dataset_id: str, episode: int):
    """One episode's action rows from a LeRobot dataset (train extra) —
    the nine lines two tools used to carry separately."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

    table = LeRobotDataset(dataset_id).hf_dataset
    episodes = np.asarray([int(e) for e in table["episode_index"]])
    rows = np.flatnonzero(episodes == episode)
    return np.stack(
        [np.asarray(torch.as_tensor(table[int(i)]["action"])) for i in rows]
    )


def verdict_word(success: bool) -> str:
    return "SUCCESS" if success else "fail"


def lerobot_train_command(  # noqa: PLR0913 - every knob of one command line, named
    *,
    policy: str,
    device: str,
    dataset: str,
    output_dir: Path | str,
    job_name: str,
    steps: int,
    batch_size: int,
    save_freq: int,
    dataset_root: Path | str | None = None,
    log_freq: int = LeRobotDefaults.LOG_FREQ,
    num_workers: int | None = None,
    lr: float | None = None,
    eval_freq: int | None = None,
    eval_episodes: int | None = None,
    eval_batch: int | None = None,
    extra: Sequence[str] = (),
) -> list[str]:
    """`lerobot-train` as an argv, the one place its flags are spelled
    for tools (train-watch's run, e2e-smoke's chain). `eval_freq` turns
    on the trainer's own in-loop evaluation every that many steps with
    `eval_episodes` episodes over `eval_batch` synchronous envs; `extra`
    carries the `--env.*` flags a plugin supplies for it."""
    command = [
        sys.executable,
        "-m",
        LeRobotScripts.TRAIN,
        f"--policy.type={policy}",
        f"--policy.device={device}",
        "--policy.push_to_hub=false",
        f"--dataset.repo_id={dataset}",
        f"--output_dir={output_dir}",
        f"--job_name={job_name}",
        f"--steps={steps}",
        f"--batch_size={batch_size}",
        f"--log_freq={log_freq}",
        *([f"--num_workers={num_workers}"] if num_workers is not None else []),
        *([f"--policy.optimizer_lr={lr}"] if lr is not None else []),
        f"--save_freq={save_freq}",
        "--wandb.enable=false",
    ]
    if dataset_root is not None:
        command.append(f"--dataset.root={dataset_root}")
    if eval_freq is not None:
        episodes = 1 if eval_episodes is None else eval_episodes
        command += [
            f"--env_eval_freq={eval_freq}",
            f"--eval.n_episodes={episodes}",
            f"--eval.batch_size={episodes if eval_batch is None else eval_batch}",
            LeRobotDefaults.SYNC_ENVS,
        ]
    return [*command, *extra]


def lerobot_eval_command(  # noqa: PLR0913 - every knob of one command line, named
    *,
    policy_path: Path | str,
    device: str,
    output_dir: Path | str,
    seed: int,
    episodes: int,
    batch_size: int,
    extra: Sequence[str] = (),
) -> list[str]:
    """`lerobot-eval` as an argv; `extra` carries the plugin's `--env.*`
    flags. Synchronous vector envs: the rollouts are render-bound and
    `spawn` workers buy nothing on one renderer."""
    return [
        sys.executable,
        "-m",
        LeRobotScripts.EVAL,
        f"--policy.path={policy_path}",
        f"--policy.device={device}",
        f"--output_dir={output_dir}",
        f"--seed={seed}",
        f"--eval.n_episodes={episodes}",
        f"--eval.batch_size={batch_size}",
        LeRobotDefaults.SYNC_ENVS,
        *extra,
    ]


def hold_until_closed(viewer: Any, period_s: float = VIEWER_HOLD_PERIOD_S) -> None:
    """Keep a passive MuJoCo viewer's window open until the operator
    closes it — the three-line hold every viewer tool used to carry."""

    while viewer.is_running():
        time.sleep(period_s)


def deployment_args(parser: Any) -> None:
    """The arguments every deployment tool shares: the project, the
    deployment's folder name, the runtime (the registry's names)."""
    from rq_pipeline.deploy.runtimes import (  # noqa: PLC0415
        DEFAULT_RUNTIME,
        runtime_names,
    )

    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True, help="the deployment's folder name")
    parser.add_argument("--runtime", default=DEFAULT_RUNTIME, choices=runtime_names())


def resolve_deployment(args: Any) -> tuple[Any, Path, Any, Path]:
    """(project, deployment folder, manifest, the bundle's assets) from
    `deployment_args`; the project made current. Raises what the loaders
    raise (FileNotFoundError, ValueError) for the tool to refuse by name."""
    from rq_pipeline.deploy.manifest import load_manifest  # noqa: PLC0415
    from rq_pipeline.deploy.runtime import assets_dir_of  # noqa: PLC0415
    from rq_pipeline.project.locate import DEPLOY_FOLDER, Project  # noqa: PLC0415

    project = Project(Path(args.project).resolve()).use()
    deployment = project.folder(DEPLOY_FOLDER) / args.name
    manifest = load_manifest(deployment)
    return project, deployment, manifest, assets_dir_of(manifest)


def running(
    project_root: Path | None,
    *,
    name: str = "",
    viewport: str = "",
    viewer: Path | str = "",
) -> Any:
    """This tool's run in the project's job table while it runs, so the
    Studio's Running now panel shows it (2026-09-25: a gate from a
    terminal showed "idle"). The kind is the script's own name - the same
    word the door that starts it uses. With no project the current one's
    table, else the pipeline's own (`mcp_jobs.default_jobs_root`, where the
    doors put theirs). Use as `with running(...) as run:` and tell it
    `run.stage(...)`, `run.progress(done, total, unit, line)`."""
    from rq_pipeline.mcp_jobs import (  # noqa: PLC0415
        default_jobs_root,
        jobs_dir_of,
        track,
    )

    kind = Path(sys.argv[0]).stem
    root = Path(project_root) if project_root is not None else default_jobs_root()
    return track(
        kind,
        jobs_dir=jobs_dir_of(root),
        name=name,
        argv=sys.argv,
        viewport=viewport,
        viewer=str(viewer),
    )


def trial_reporter(run: Any, unit: str = "trials") -> Any:
    """A gate-shaped loop's per-trial callback (`deploy.gate.TrialProgress`)
    speaking to a `running(...)` run: "trial 7 of 20: tracked"."""

    def told(done: int, total: int, outcome: Any) -> None:
        word = "tracked" if getattr(outcome, "success", False) else "lost"
        run.progress(done, total, unit, f"trial {done} of {total}: {word}")

    return told
