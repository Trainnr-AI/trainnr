"""The evaluation a deployment cites, found by its version among the
project's evaluations - one home for what the gate and the assay tools
both need (the record's k, n and interval travel into the gate's record
beside its own)."""

from __future__ import annotations

import json
from typing import Any

from trainnr.project.kinds import CERTIFICATE_FILE, Kind, stamp_kind
from trainnr.project.locate import CERTIFICATES_FOLDER, Project


def cited_certificate(project: Project, stamp: str | None) -> dict[str, Any] | None:
    """The cited evaluation's record, or None when the manifest cites
    none; refuses by name a citation the project does not hold."""
    if not stamp or "@" not in stamp:
        return None
    wanted = stamp.split("@", 1)[1]
    folder = project.folder(CERTIFICATES_FOLDER)
    for candidate in sorted(folder.iterdir()) if folder.is_dir() else []:
        path = candidate / CERTIFICATE_FILE
        if not path.is_file():
            continue
        if stamp_kind(Kind.CERTIFICATE, candidate).split("@", 1)[1] == wanted:
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        f"the manifest cites evaluation {stamp!r}, which is not in {folder}"
    )
