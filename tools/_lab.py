"""The shared lab bench — the rituals every tool used to hand-roll.

    from _lab import bootstrap, rr_session

Import-order contract: `bootstrap()` must run before any `rq_pipeline`
import (it is what puts <repo>/pipeline on sys.path), so callers keep
`# noqa: E402` on the imports that follow the call.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# Frame decimation the tools share: a 50 Hz control loop previewed at 10 Hz.
PREVIEW_EVERY_TICKS = 5


def bootstrap() -> None:
    """Put <repo>/pipeline and <repo>/tools at the front of sys.path."""
    for root in (str(REPO / "tools"), str(REPO / "pipeline")):
        if root not in sys.path:
            sys.path.insert(0, root)


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
    if mode == "spawn":
        rr.spawn()
    elif mode == "attach":
        try:
            rr.connect_grpc()
        except Exception:
            rr.spawn()
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
