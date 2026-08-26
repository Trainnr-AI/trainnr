"""The shared lab bench — the rituals every tool used to hand-roll.

    from _lab import bootstrap, rr_session

Import-order contract: `bootstrap()` must run before any `rq_pipeline`
import (it is what puts <repo>/pipeline on sys.path), so callers keep
`# noqa: E402` on the imports that follow the call.
"""

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
# Frame decimation the tools share: a 50 Hz control loop previewed at 10 Hz.
PREVIEW_EVERY_TICKS = 5


class LeRobotScripts:
    """LeRobot's entry points, run as modules in whatever interpreter
    launched the tool — no venv name, no bin/ layout baked in."""

    TRAIN = "lerobot.scripts.lerobot_train"
    EVAL = "lerobot.scripts.lerobot_eval"


def bootstrap() -> None:
    """Put <repo>/pipeline and <repo>/tools at the front of sys.path."""
    for root in (str(REPO / "tools"), str(REPO / "pipeline")):
        if root not in sys.path:
            sys.path.insert(0, root)


def viewer_executable() -> str | None:
    """Where the Rerun viewer binary is when it is not on PATH: the sim
    venv installs it beside the SDK (`rerun-sdk`), the train venv only the
    SDK — so tools launched from the train venv used to need
    `PATH="$PWD/.venv/bin:$PATH"` (2026-08-26). None means "on PATH"."""
    import shutil  # noqa: PLC0415

    if shutil.which("rerun"):
        return None
    for candidate in (
        REPO / "pipeline" / ".venv" / "bin" / "rerun",
        REPO / "pipeline" / ".venv" / "Scripts" / "rerun.exe",
    ):
        if candidate.exists():
            return str(candidate)
    return None


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
    log_freq: int = 50,
    extra: Sequence[str] = (),
) -> list[str]:
    """`lerobot-train` as an argv, the one place its flags are spelled
    for tools (train-watch's run, e2e-smoke's chain). `extra` carries
    the `--env.*` flags a plugin supplies for in-loop evaluation."""
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
        f"--save_freq={save_freq}",
        "--wandb.enable=false",
    ]
    if dataset_root is not None:
        command.append(f"--dataset.root={dataset_root}")
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
        "--eval.use_async_envs=false",
        *extra,
    ]


def hold_until_closed(viewer: Any, period_s: float = 0.2) -> None:
    """Keep a passive MuJoCo viewer's window open until the operator
    closes it — the three-line hold every viewer tool used to carry."""
    import time  # noqa: PLC0415

    while viewer.is_running():
        time.sleep(period_s)
