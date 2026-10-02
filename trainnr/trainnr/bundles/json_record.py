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
from typing import TYPE_CHECKING, Any, TypeVar, cast

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

T = TypeVar("T", bound="JsonRecord")


class JsonRecord:
    """Mixin for a dataclass: `write(path)` / `read(path)` as JSON."""

    def as_json(self) -> str:
        # The mixin contract (dataclass hosts only) is invisible to the
        # checker; the casts state it once instead of per call site.
        return json.dumps(asdict(cast("DataclassInstance", self)), indent=1)

    def write(self, path: Path) -> Path:
        """Atomic: a reader (another shard's datasheet pass, the feed)
        sees the old file or the whole new one, never half of it."""
        import os  # noqa: PLC0415

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        staging.write_text(self.as_json(), encoding="utf-8")
        os.replace(staging, path)
        return path

    @classmethod
    def read(cls: type[T], path: Path) -> T:
        raw: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {field.name for field in fields(cast("type[DataclassInstance]", cls))}
        return cls(**{key: value for key, value in raw.items() if key in known})
