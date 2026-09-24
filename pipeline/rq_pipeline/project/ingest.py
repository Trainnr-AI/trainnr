"""Ingest into a project: the robot seam writes the recording, the
project layer names the folder and stamps the artifact.

`robots/ingest` knows adapters and files and nothing above them; this
module is what the MCP door and the capture state machine call, so the
tier order holds (robots below project) and a recording's version is
minted by the one stamping door.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rq_pipeline.project.kinds import Kind, stamp_kind
from rq_pipeline.project.locate import INDEX_DIR, Project
from rq_pipeline.robots.capture import (
    DEFAULT_SOURCE,
    LISTENING,
    CaptureState,
    Listener,
    open_capture,
)
from rq_pipeline.robots.capture import status as capture_state
from rq_pipeline.robots.ingest import ingest as ingest_into


def ingest(  # noqa: PLR0913 - a recording's identity: source, name, adapter, whose robot, why
    project: Project,
    source: Path,
    *,
    name: str | None = None,
    adapter: str | None = None,
    provenance: Mapping[str, Any] | None = None,
    basis: str | None = None,
    keep_as: str | None = None,
) -> dict[str, Any]:
    """A source into the project's recordings, stamped AFTER everything
    the recording keeps has landed (`keep_as`: a capture's store moved
    into `raw/`); the record carries `stamp` and the recording's `path`
    relative to the project."""
    record = ingest_into(
        project.recordings,
        source,
        name=name,
        adapter=adapter,
        provenance=provenance,
        basis=basis,
        keep_as=keep_as,
    )
    out = Path(record["path"])
    return {
        "stamp": stamp_kind(Kind.RECORDING, out),
        **record,
        "path": str(out.relative_to(project.root)),
    }


# While a capture listens, the index is rewritten this often so the
# Studio shows it filling; every state change past listening re-indexes.
REINDEX_EVERY_S = 2.0


def capture(
    project: Project, name: str, *, source: str = DEFAULT_SOURCE, **options: Any
) -> Listener:
    """A live capture into the project: the registry's listener for
    `source` pointed at the project's recordings and state directory,
    landing through the stamping ingest above, re-indexing as it runs."""
    last = {"at": 0.0}

    def land(_recordings: Path, source: Path, **kw: Any) -> dict[str, Any]:
        return ingest(project, source, **kw)

    def reindex(state: CaptureState) -> None:
        now = time.monotonic()
        if state.state == LISTENING and now - last["at"] < REINDEX_EVERY_S:
            return
        from rq_pipeline.project.index import (  # noqa: PLC0415 - above this module
            index_project,
            write_index,
        )

        write_index(project, index_project(project))
        last["at"] = now

    return open_capture(
        source,
        project.recordings,
        project.root / INDEX_DIR,
        name,
        ingest=land,
        on_state=reindex,
        **options,
    )


def capture_status(project: Project) -> dict[str, Any]:
    return capture_state(project.root / INDEX_DIR)
