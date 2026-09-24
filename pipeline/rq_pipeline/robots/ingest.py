"""Ingest: a source, through its adapter, into a directory as a
`recording` artifact — the first move of the loop (docs/76 §3).

    from rq_pipeline.robots.ingest import ingest
    ingest(recordings_dir, Path("go2-walk.mcap"), name="go2-walk")

The recording lands at `<recordings>/<name>/` with its manifest
(`recording.json`, the kind's marker) and signals; the returned record
carries the adapter that read it, the census, and the channel list —
what the agent needs to choose the next move without re-reading the
files. Which project the directory belongs to, and the version the
artifact gets, are the project layer's business (`project/ingest`):
this seam knows robots and files, nothing above them.
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from rq_pipeline.robots.adapter import detect, resolve
from rq_pipeline.robots.recording import BASES, Recording

RAW_DIR = "raw"  # the source file as received, inside the recording
NAME_FORBIDDEN = "@/"


def ingest(  # noqa: PLR0913 - a recording's identity: source, name, adapter, whose robot, why
    recordings: Path,
    source: Path,
    *,
    name: str | None = None,
    adapter: str | None = None,
    provenance: Mapping[str, Any] | None = None,
    basis: str | None = None,
) -> dict[str, Any]:
    """Read `source` (with `adapter`, or the one that accepts it) and
    write it under `recordings`. `basis` says whose robot it was (one of
    `BASES`: the operator's own, a public log, a simulation) and
    overrides the adapter's default; `provenance` carries the facts
    behind it (a public log's source, url, licence, robot). Refused by
    name when the basis is not a known word. Refuses to overwrite an
    existing recording of the same name — re-ingest under another name
    or remove it first. Returns the record with the recording's
    directory under `path`."""
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"no source at {source}")
    if basis is not None and basis not in BASES:
        raise ValueError(f"recording basis {basis!r}; known: {BASES}")
    entry = resolve(adapter) if adapter else detect(source)
    recording: Recording = entry.build().read(source)
    if basis is not None:
        recording = replace(recording, basis=basis)
    if provenance:
        recording = replace(recording, provenance=dict(provenance))
    label = name or source.stem
    if any(c in label for c in NAME_FORBIDDEN):
        raise ValueError(f"recording names are plain words, got {label!r}")
    out = Path(recordings) / label
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
    manifest = recording.manifest()
    return {
        "raw": raw,
        "path": out,
        "adapter": entry.name,
        "source": recording.source,
        "duration_s": round(manifest.duration_s, 3),
        "channels": [asdict(c) for c in manifest.channels],
        "census": manifest.census,
        "notes": manifest.notes,
        "basis": manifest.basis,
        "provenance": manifest.provenance,
    }
