"""A robot's model file into a hash-stamped bundle, by its format.

One door (`onboard`), a registry of model sources keyed by file suffix
(`@model_source`), the way engines, tasks and telemetry adapters
register: MJCF is copied whole and compiled once as the honesty check;
USD goes through Newton's importer and the bundle writer
(`rq_pipeline.robot.usd_import`); a third format enters through the
`rq_pipeline.model_sources` entry-point group. Every source refuses an
option it does not take, by name, so a caller cannot pass a variant to
an MJCF and have it ignored.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rq_pipeline.bundles.bundle import write_bundle_record
from rq_pipeline.bundles.hashing import stamp
from rq_pipeline.plugins import load_group

ENTRY_POINT_GROUP = "rq_pipeline.model_sources"
BUILTIN_MODULES = ("rq_pipeline.robot.usd_import",)
MJCF_SUFFIXES = (".xml",)
MJCF_SOURCE = "mjcf"
USD_SOURCE = "usd"

Onboarder = Callable[[Path, str, Path, Mapping[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class ModelSource:
    """A format the door reads: its suffixes, the function that writes
    the bundle, and the options it takes."""

    name: str
    suffixes: tuple[str, ...]
    onboard: Onboarder
    options: tuple[str, ...]
    doc: str

    def check_options(self, options: Mapping[str, Any]) -> None:
        unknown = sorted(set(options) - set(self.options))
        if unknown:
            takes = ", ".join(self.options) or "no options"
            raise ValueError(f"a {self.name} source takes {takes}; not {unknown}")


_REGISTRY: dict[str, ModelSource] = {}
_LOADED = False


def model_source(
    name: str,
    suffixes: tuple[str, ...],
    *,
    options: tuple[str, ...] = (),
    doc: str = "",
) -> Callable[[Onboarder], Onboarder]:
    """Register the function that onboards one format; refuses a name or
    a suffix already taken."""

    def register(function: Onboarder) -> Onboarder:
        if name in _REGISTRY:
            raise ValueError(f"model source {name!r} is already registered")
        taken = {s: n for n, src in _REGISTRY.items() for s in src.suffixes}
        for suffix in suffixes:
            if suffix in taken:
                raise ValueError(f"suffix {suffix!r} is already {taken[suffix]}'s")
        _REGISTRY[name] = ModelSource(
            name, suffixes, function, options, doc or function.__doc__ or ""
        )
        return function

    return register


def sources() -> dict[str, ModelSource]:
    """Every registered source, the built-ins and the entry points loaded once."""
    global _LOADED  # noqa: PLW0603 - the one load flag
    if not _LOADED:
        _LOADED = True
        load_group(ENTRY_POINT_GROUP, BUILTIN_MODULES)
    return dict(_REGISTRY)


def source_for(path: Path) -> ModelSource:
    """The source that reads `path`, by suffix; refuses naming what is known."""
    suffix = Path(path).suffix.lower()
    for source in sources().values():
        if suffix in source.suffixes:
            return source
    known = ", ".join(f"{n} ({' '.join(s.suffixes)})" for n, s in sources().items())
    raise ValueError(f"no model source reads {suffix!r}; known: {known}")


def onboard(
    source_path: Path,
    name: str,
    destination: Path,
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The door: the file must exist, the bundle must not, the format
    must be known and the options must be its; then the source writes
    the bundle and the reply carries the stamp. The name's spelling is
    the project layer's rule (`project.locate.plain_name`), checked by
    the caller above this tier."""
    source_path = Path(source_path).expanduser()
    if not source_path.is_file():
        raise FileNotFoundError(f"no model file at {source_path}")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(
            f"robots/{name} already exists (stamp: {stamp(name, destination)}) "
            "— onboarding never overwrites a bundle; pick another name or "
            "remove it deliberately"
        )
    source = source_for(source_path)
    options = dict(options or {})
    source.check_options(options)
    return source.onboard(source_path, name, destination, options)


@model_source(
    MJCF_SOURCE,
    MJCF_SUFFIXES,
    doc="an MJCF: its whole directory copied (meshes and includes ride along), "
    "compiled once before a byte lands",
)
def onboard_mjcf(
    source_path: Path, name: str, destination: Path, options: Mapping[str, Any]
) -> dict[str, Any]:
    """MJCF: compile FIRST — a model that does not compile is refused
    before a single byte lands — then the directory whole, then the record."""
    import mujoco  # noqa: PLC0415 - sim extra

    del options  # an MJCF source takes none; `onboard` refused any
    model = mujoco.MjModel.from_xml_path(str(source_path))
    shutil.copytree(source_path.parent, destination)
    write_bundle_record(destination, name, source_path.name, model, source=source_path)
    return {
        "stamp": stamp(name, destination),
        "path": str(destination),
        "model_file": source_path.name,
        "bodies": int(model.nbody),
        "joints": int(model.njnt),
        "actuators": int(model.nu),
    }
