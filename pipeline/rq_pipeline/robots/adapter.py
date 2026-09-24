"""The `RobotAdapter` Protocol and its registry — the provider seam's shape.

An adapter answers three questions about one source format: can it read
this path; what does the robot in it report (the census: joints,
sensors, rates); and the recording itself. Adapters are registered by
name with `@adapter(name)`, the built-ins are imported by `adapters()`,
and a third party's adapter enters through the `rq_pipeline.robot_adapters`
entry-point group — one module and one entry point, the way a second GPU
vendor does.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from rq_pipeline.plugins import load_group
from rq_pipeline.robots.recording import Recording

ENTRY_POINT_GROUP = "rq_pipeline.robot_adapters"
BUILTIN_MODULES = (
    "rq_pipeline.robots.adapters.wire",
    "rq_pipeline.robots.adapters.lerobot",
    "rq_pipeline.robots.adapters.mcap",
    "rq_pipeline.robots.adapters.rosbag2",
    "rq_pipeline.robots.adapters.mocap",
    # The rosbag2 SQLite reader lives beside the identification that needed
    # it first (robot/rosbag_sqlite); see its note on where it collapses.
    "rq_pipeline.robot.rosbag_sqlite",
    "rq_pipeline.robot.pt_dict",
)


@runtime_checkable
class RobotAdapter(Protocol):
    """What a source format must do to sit behind the seam."""

    name: str

    def accepts(self, source: Path) -> bool:
        """Whether this adapter reads what is at `source` — by extension
        or marker file, never by trying and failing."""

    def read(self, source: Path) -> Recording:
        """The recording. Refuses, by name, anything it cannot read."""


@dataclass(frozen=True)
class AdapterEntry:
    name: str
    build: Callable[..., Any]  # build() -> a RobotAdapter
    doc: str


_REGISTRY: dict[str, AdapterEntry] = {}


def adapter(
    name: str, *, doc: str = ""
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorate an adapter class (or factory); refuses a duplicate name."""
    if not name or "/" in name or " " in name:
        raise ValueError(f"adapter names are single words, got {name!r}")

    def decorate(build: Callable[..., Any]) -> Callable[..., Any]:
        existing = _REGISTRY.get(name)
        if existing is not None and existing.build is not build:
            raise ValueError(f"adapter {name!r} registered twice")
        line = doc or (build.__doc__ or "").strip().split("\n")[0]
        _REGISTRY[name] = AdapterEntry(name=name, build=build, doc=line)
        return build

    return decorate


def adapters() -> Mapping[str, AdapterEntry]:
    """Every registered adapter: the built-ins plus installed plugins."""
    load_group(ENTRY_POINT_GROUP, BUILTIN_MODULES)
    return dict(_REGISTRY)


def resolve(name: str) -> AdapterEntry:
    known = adapters()
    if name not in known:
        raise KeyError(f"unknown robot adapter {name!r}; known: {sorted(known)}")
    return known[name]


def detect(source: Path) -> AdapterEntry:
    """The one adapter that accepts `source`; refused by name when none
    or more than one does — a source is one format."""
    source = Path(source)
    matches = [entry for entry in adapters().values() if entry.build().accepts(source)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(
            f"no robot adapter reads {source}; known: {sorted(adapters())} — "
            "name one explicitly, or install a plugin for this format"
        )
    names = sorted(m.name for m in matches)
    raise ValueError(f"{source} is accepted by {names}; name one explicitly")
