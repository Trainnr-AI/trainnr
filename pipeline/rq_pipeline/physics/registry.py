"""The engine registry: engines declare themselves; callers resolve by name.

The tasks registry's pattern (rq_pipeline/tasks/registry.py), applied to
the other side of the seam. An engine class registers itself:

    @engine("acme-sim", doc="Acme's rigid-body engine over the same MJCF")
    class AcmeBackend: ...

and an installed package advertises its module through the
`rq_pipeline.engines` entry-point group, which `engines()` imports on
demand. The built-ins are "mujoco" — CPU MuJoCo, the metrology
instrument — and "mjx-warp" — batched MJX-Warp, the throughput
instrument. Registering costs nothing: an engine whose extra is not
installed raises its own ImportError, naming the extra, when it is
CONSTRUCTED, not when it is listed. Every engine enters through the
same gauntlet (tests/test_mjx_backend.py is the pattern) and the same
consumer-side contract (`evaluate.harness.Engine`). Standard library only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from rq_pipeline.plugins import load_group

ENTRY_POINT_GROUP = "rq_pipeline.engines"
# The modules that register the built-in engines; imported by `engines()`
# so a checkout works before its entry points are installed.
BUILTIN_MODULES = (
    "rq_pipeline.physics.mujoco_backend",
    "rq_pipeline.physics.mjx_backend",
)
CPU_ENGINE = "mujoco"
GPU_ENGINE = "mjx-warp"


@dataclass(frozen=True)
class EngineEntry:
    """A registered engine: its name, its constructor, one line about it."""

    name: str
    build: Callable[..., Any]  # build(**kwargs) -> an Engine
    doc: str


_REGISTRY: dict[str, EngineEntry] = {}


def engine(name: str, *, doc: str = ""):
    """Decorate an engine class (or factory); refuses a duplicate name."""
    if not name or "/" in name or " " in name:
        raise ValueError(f"engine names are single words, got {name!r}")

    def decorate(build: Callable[..., Any]) -> Callable[..., Any]:
        existing = _REGISTRY.get(name)
        if existing is not None and existing.build is not build:
            raise ValueError(f"engine {name!r} registered twice")
        line = doc or (build.__doc__ or "").strip().split("\n")[0]
        _REGISTRY[name] = EngineEntry(name=name, build=build, doc=line)
        return build

    return decorate


def engines() -> Mapping[str, EngineEntry]:
    """Every registered engine: the built-ins plus installed plugins."""
    load_group(ENTRY_POINT_GROUP, BUILTIN_MODULES)
    return dict(_REGISTRY)


def resolve(name: str) -> EngineEntry:
    """The entry for `name`; an unknown name is refused with the known ones."""
    known = engines()
    if name not in known:
        raise KeyError(f"unknown engine {name!r}; known: {sorted(known)}")
    return known[name]
