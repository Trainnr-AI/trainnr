"""Ingest into a project: the robot seam writes the recording, the
project layer names the folder and stamps the artifact.

`robots/ingest` knows adapters and files and nothing above them; this
module is what the MCP door and the capture state machine call, so the
tier order holds (robots below project) and a recording's version is
minted by the one stamping door.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rq_pipeline.project.kinds import Kind, stamp_kind
from rq_pipeline.project.locate import INDEX_DIR, Project
from rq_pipeline.robots.capture import WIRE_UDP_PORT, WireUdpCapture
from rq_pipeline.robots.capture import status as capture_state
from rq_pipeline.robots.ingest import ingest as ingest_into


def ingest(
    project: Project,
    source: Path,
    *,
    name: str | None = None,
    adapter: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """A source into the project's recordings, stamped; the record
    carries `stamp` and the recording's `path` relative to the project."""
    record = ingest_into(
        project.recordings, source, name=name, adapter=adapter, provenance=provenance
    )
    out = Path(record["path"])
    return {
        "stamp": stamp_kind(Kind.RECORDING, out),
        **record,
        "path": str(out.relative_to(project.root)),
    }


def capture(
    project: Project, name: str, *, port: int = WIRE_UDP_PORT
) -> WireUdpCapture:
    """A live capture into the project: the seam's listener pointed at the
    project's recordings and state directory, landing through the
    stamping ingest above."""

    def land(_recordings: Path, source: Path, **kw: Any) -> dict[str, Any]:
        return ingest(project, source, **kw)

    return WireUdpCapture(
        project.recordings, project.root / INDEX_DIR, name, port=port, ingest=land
    )


def capture_status(project: Project) -> dict[str, Any]:
    return capture_state(project.root / INDEX_DIR)
