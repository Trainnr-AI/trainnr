"""The actuator library: every servo we can load a friction model for.

`robots/actuators/<slug>/` is data, the same category as a robot
bundle: one JSON per model tier (`m1.json`..`m6.json`, not every
actuator has all six) plus a `PROVENANCE.json` naming where the
numbers came from. This module is the one door that data enters
through — `load_actuator` refuses a directory with no provenance,
an unknown model, or a params file missing the servo-model fields
BAM's schema always carries, the same fail-loudly discipline
`trainnr.robot.model_checks` applies to whole robots.

Growing the library never touches this file: `tools/sync-bam-
actuators.py` vendors new BAM releases, a real bench fit or a
datasheet-only guess lands by hand with the matching `source` tag —
`list_actuators`/`load_actuator` read whatever is on disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trainnr.paths import checkout
from trainnr.robot.friction_budget import FrictionParams

ACTUATORS_ROOT = checkout() / "robots" / "actuators"
PROVENANCE_FILE = "PROVENANCE.json"
REQUIRED_PROVENANCE_FIELDS = ("source", "citation", "license")
# "bam-refit": BAM's own model and optimiser re-run by us on Rhoban's
# public bench logs, with a bootstrap interval (docs/e2e-research/72).
KNOWN_SOURCES = ("bam", "bam-refit", "own-bench", "datasheet")
# BAM's schema always carries these on top of the friction fields
# (bam.model.Model / bam.actuator.VoltageControlledActuator): the
# DC-motor + firmware-control-law side, not the friction budget.
SERVO_FIELDS = (
    "kt",
    "R",
    "armature",
    "q_offset",
    "command_delay",
    "max_velocity",
    "error_gain_ratio",
)


@dataclass(frozen=True)
class ServoParams:
    """The DC-motor + firmware-control-law side of a BAM fit — kept
    alongside the friction budget because a params file identifies
    both together, even though today's friction_budget.py only
    consumes the friction fields. Kept optional per-field: not every
    actuator's JSON carries every servo constant (`error_gain_ratio`
    is absent from some BAM releases' older fits)."""

    kt: float | None = None
    R: float | None = None
    armature: float | None = None
    q_offset: float | None = None
    command_delay: float | None = None
    max_velocity: float | None = None
    error_gain_ratio: float | None = None

    @classmethod
    def from_json(cls, fields: dict[str, Any]) -> ServoParams:
        return cls(**{k: fields[k] for k in SERVO_FIELDS if k in fields})


@dataclass(frozen=True)
class Provenance:
    """Where an actuator's numbers came from — required, not metadata."""

    source: str
    citation: str
    license: str
    raw: dict[str, Any]

    def __post_init__(self) -> None:
        if self.source not in KNOWN_SOURCES:
            raise ValueError(
                f"unknown provenance source {self.source!r}; known: {KNOWN_SOURCES}"
            )


@dataclass(frozen=True)
class ActuatorModel:
    """One (actuator, model-tier) load: the friction budget, the servo
    constants, and the provenance that says whether to trust them."""

    slug: str
    tier: str
    friction: FrictionParams
    servo: ServoParams
    provenance: Provenance


def _actuator_dir(slug: str) -> Path:
    directory = ACTUATORS_ROOT / slug
    if not directory.is_dir():
        available = sorted(list_actuators())
        raise KeyError(f"no actuator {slug!r} under {ACTUATORS_ROOT}; have {available}")
    return directory


def _load_provenance(directory: Path) -> Provenance:
    path = directory / PROVENANCE_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{directory.name} has no {PROVENANCE_FILE} — an actuator model with "
            "unknown provenance is refused, not loaded as if it were trustworthy"
        )
    raw = json.loads(path.read_text())
    missing = [f for f in REQUIRED_PROVENANCE_FIELDS if f not in raw]
    if missing:
        raise ValueError(f"{path} is missing required field(s): {missing}")
    return Provenance(
        source=raw["source"], citation=raw["citation"], license=raw["license"], raw=raw
    )


def list_actuators() -> tuple[str, ...]:
    """Every actuator slug with a provenance-declared directory."""
    if not ACTUATORS_ROOT.is_dir():
        return ()
    return tuple(
        sorted(
            p.name
            for p in ACTUATORS_ROOT.iterdir()
            if p.is_dir() and (p / PROVENANCE_FILE).exists()
        )
    )


def list_models(slug: str) -> tuple[str, ...]:
    """Which model tiers (`m1`..`m6`) exist for one actuator."""
    directory = _actuator_dir(slug)
    # `m6.uncertainty.json` rides beside `m6.json` (the fitted interval,
    # 2026-09-06); a tier is a stem without a dot.
    return tuple(sorted(p.stem for p in directory.glob("m*.json") if "." not in p.stem))


def load_actuator(slug: str, tier: str = "m6") -> ActuatorModel:
    """Load one actuator at one model tier. `tier` defaults to the
    richest fit available for it — BAM's own guidance (docs/e2e-
    research/53) is to compare validation error across M1-M6 and pick
    accordingly, so a caller comparing tiers passes `tier` explicitly."""
    directory = _actuator_dir(slug)
    path = directory / f"{tier}.json"
    if not path.exists():
        raise KeyError(
            f"actuator {slug!r} has no tier {tier!r}; have {list_models(slug)}"
        )
    fields = json.loads(path.read_text())
    return ActuatorModel(
        slug=slug,
        tier=tier,
        friction=FrictionParams.from_json(fields),
        servo=ServoParams.from_json(fields),
        provenance=_load_provenance(directory),
    )
