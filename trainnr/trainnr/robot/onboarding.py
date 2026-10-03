"""A robot's model file into a hash-stamped bundle, by its format.

One door (`onboard`), a registry of model sources keyed by file suffix
(`@model_source`), the way engines, tasks and telemetry adapters
register: MJCF is copied whole and compiled once as the honesty check;
URDF goes through MuJoCo's own loader (`trainnr.robot.urdf_import`);
USD goes through Newton's importer and the bundle writer
(`trainnr.robot.usd_import`); a third format enters through the
`trainnr.model_sources` entry-point group. Every source refuses an
option it does not take, by name, so a caller cannot pass a variant to
an MJCF and have it ignored.

After the source has written, the door AUDITS what the importer
changed (`trainnr.robot.import_audit`): the bundle's compiled model
against the description as authored. A change the format's reader
explains goes on the record; one it does not is refused by name and
the half-written bundle removed — unless the caller passes
`accept_changes`, the one option every format takes, and then the
record says the changes were accepted.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trainnr.bundles.bundle import (
    AUDIT_KEY,
    model_file_of,
    write_audit,
    write_bundle_record,
)
from trainnr.bundles.hashing import stamp
from trainnr.plugins import load_group
from trainnr.robot.import_audit import Audit, audit_bundle, require_explained

ENTRY_POINT_GROUP = "trainnr.model_sources"
BUILTIN_MODULES = ("trainnr.robot.urdf_import", "trainnr.robot.usd_import")
MJCF_SUFFIXES = (".xml",)
MJCF_SOURCE = "mjcf"
USD_SOURCE = "usd"
ACCEPT_CHANGES = "accept_changes"  # the door's own option, every format
DOOR_OPTIONS = (ACCEPT_CHANGES,)
ACCEPTED_WORD = "accepted"

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
    accept = bool(options.pop(ACCEPT_CHANGES, False))
    source.check_options(options)
    out = source.onboard(source_path, name, destination, options)
    audit = audit_written(source_path, destination, options, accept=accept)
    out["stamp"] = stamp(name, destination)
    out[AUDIT_KEY] = audit.summary()
    if audit.unexplained:
        out[f"{AUDIT_KEY}_unexplained"] = [c.line() for c in audit.unexplained]
    return out


def audit_written(
    source_path: Path,
    destination: Path,
    options: Mapping[str, Any],
    *,
    accept: bool = False,
) -> Audit:
    """The audit of a bundle the door just wrote, put on its record; an
    unexplained change removes the bundle and refuses, unless accepted,
    and then the record says so."""
    model_file = model_file_of(destination)
    if model_file is None:
        raise FileNotFoundError(f"{destination} holds no MJCF to audit")
    try:
        audit = audit_bundle(source_path, model_file, options)
        if not accept:
            require_explained(audit)
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    record = audit.to_record()
    if accept and audit.unexplained:
        record[ACCEPTED_WORD] = True
    write_audit(destination, record)
    return audit


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
        SHAPE_KEY: shape_census(model),
    }


# The reply's census of what a trainer looks up by name, so an agent
# learns a model's shape here and not from a traceback after `train_walk`
# (docs/77 §3 item 5, done 2026-10-03 after the stranger test).
SHAPE_KEY = "shape"


def shape_census(model: Any) -> dict[str, Any]:
    """The named parts a task reads by name: the collision geoms that carry a
    name (contype or conaffinity set), the sites, and the trunk (the first
    body under the world)."""
    import mujoco  # noqa: PLC0415 - sim extra

    def name_of(obj: Any, i: int) -> str | None:
        return mujoco.mj_id2name(model, obj, i) or None

    collision = [
        name
        for i in range(model.ngeom)
        if (model.geom_contype[i] or model.geom_conaffinity[i])
        and (name := name_of(mujoco.mjtObj.mjOBJ_GEOM, i))
    ]
    sites = [
        name
        for i in range(model.nsite)
        if (name := name_of(mujoco.mjtObj.mjOBJ_SITE, i))
    ]
    trunk = next(
        (
            name_of(mujoco.mjtObj.mjOBJ_BODY, i)
            for i in range(1, model.nbody)
            if model.body_parentid[i] == 0
        ),
        None,
    )
    return {"collision_geoms": collision, "sites": sites, "trunk": trunk}
