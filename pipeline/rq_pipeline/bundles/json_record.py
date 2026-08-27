"""One way to write a dataclass to JSON and read it back.

Three sidecars carried the same eight lines (a demo episode's manifest,
a training run's manifest, a dataset's provenance) before this mixin.
Reading is tolerant of keys this version does not know — a manifest
mirrored from a machine running a newer or older tool must still load —
and strict about the rest: a missing required field is the dataclass's
own TypeError, by name.
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T", bound="JsonRecord")


class JsonRecord:
    """Mixin for a dataclass: `write(path)` / `read(path)` as JSON."""

    def as_json(self) -> str:
        return json.dumps(asdict(self), indent=1)

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.as_json(), encoding="utf-8")
        return path

    @classmethod
    def read(cls: type[T], path: Path) -> T:
        raw: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in raw.items() if key in known})
