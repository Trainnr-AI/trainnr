"""Ingest: a source, through its adapter, into a project as a stamped
`recording` artifact — the first move of the loop (docs/76 §3).

    from rq_pipeline.robots.ingest import ingest
    ingest(project, Path("go2-walk.mcap"), name="go2-walk")

The recording lands at `<project>/recordings/<name>/` with its manifest
(`recording.json`, the kind's marker) and signals; the returned record
carries the stamp the project index will compute, the adapter that read
it, the census, and the channel list — what the agent needs to choose the
next move without re-reading the files.
"""

from __future__ import annotations

import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rq_pipeline.project.kinds import Kind, stamp_kind
from rq_pipeline.project.locate import Project
from rq_pipeline.robots.adapter import detect, resolve
from rq_pipeline.robots.recording import Recording

RAW_DIR = "raw"  # the source file as received, inside the recording


def ingest(
    project: Project,
    source: Path,
    *,
    name: str | None = None,
    adapter: str | None = None,
) -> dict[str, Any]:
    """Read `source` (with `adapter`, or the one that accepts it) and
    write it into the project. Refuses to overwrite an existing recording
    of the same name — re-ingest under another name or remove it first."""
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"no source at {source}")
    entry = resolve(adapter) if adapter else detect(source)
    recording: Recording = entry.build().read(source)
    label = name or source.stem
    if "@" in label or "/" in label:
        raise ValueError(f"recording names are plain words, got {label!r}")
    out = project.folder("recordings") / label
    if out.exists():
        raise FileExistsError(f"a recording named {label!r} already exists at {out}")
    recording.write(out)
    # The raw source travels with the artifact (a file; a dataset
    # directory stays where it is): an identification method reads the
    # original bytes, and a stranger can re-run the adapter on them.
    raw = None
    if source.is_file():
        raw_dir = out / RAW_DIR
        raw_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, raw_dir / source.name)
        raw = f"{RAW_DIR}/{source.name}"
    identity = stamp_kind(Kind.RECORDING, out)
    manifest = recording.manifest()
    return {
        "stamp": identity,
        "raw": raw,
        "path": str(out.relative_to(project.root)),
        "adapter": entry.name,
        "source": recording.source,
        "duration_s": round(manifest.duration_s, 3),
        "channels": [asdict(c) for c in manifest.channels],
        "census": manifest.census,
        "notes": manifest.notes,
    }
