"""The shared lab bench — the two rituals every tool used to hand-roll.

    from _lab import bootstrap, rr_session

Import-order contract: `bootstrap()` must run before any `rq_pipeline`
import (it is what puts <repo>/pipeline on sys.path), so callers keep
`# noqa: E402` on the imports that follow the call.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def bootstrap() -> None:
    """Put <repo>/pipeline and <repo>/tools at the front of sys.path."""
    for root in (str(REPO / "tools"), str(REPO / "pipeline")):
        if root not in sys.path:
            sys.path.insert(0, root)


def rr_session(
    app_id: str, *, mode: str = "attach", recording_id: str | None = None
) -> None:
    """Open a Rerun stream: "attach" joins a running viewer and spawns one
    if none answers, "spawn" always opens a fresh viewer, "connect" joins
    a viewer known to exist (and fails loudly when it does not)."""
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
