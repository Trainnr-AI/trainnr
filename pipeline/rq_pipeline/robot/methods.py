"""Identification methods: the third registry seam, for stage ② as a door.

`identify()` fits parameters with intervals; `fit_drivetrain` is the one
concrete method — the rig's ratio-form fit from a wire recording. An
agent cannot call either from a project. This module makes a method a
named thing a project can run on (robot, recording): a Protocol, a
registry with `@method(name)`, an entry-point group
(`rq_pipeline.identification_methods`), and the drivetrain ratio fit as
the built-in. A robot family with no method is refused by name — the
seam is the generality, one method is the truth today (docs/76 §6).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from rq_pipeline.plugins import load_group
from rq_pipeline.robot.identify import IdentificationResult

ENTRY_POINT_GROUP = "rq_pipeline.identification_methods"
BUILTIN_MODULES = ("rq_pipeline.robot.methods",)


@runtime_checkable
class IdentificationMethod(Protocol):
    """What a method must do to sit behind the seam."""

    name: str

    def accepts(self, bundle_dir: Path, recording_dir: Path) -> str | None:
        """None when this method can fit this robot from this recording;
        otherwise the reason it cannot, in a sentence."""

    def fit(
        self, bundle_dir: Path, recording_dir: Path, *, write: bool = True
    ) -> tuple[IdentificationResult, Path | None]:
        """Excite, identify, and (unless `write` is off) record into the
        bundle's `fits/`; returns the result and the record's path."""


@dataclass(frozen=True)
class MethodEntry:
    name: str
    build: Callable[..., Any]  # build() -> an IdentificationMethod
    doc: str


_REGISTRY: dict[str, MethodEntry] = {}


def method(
    name: str, *, doc: str = ""
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorate a method class (or factory); refuses a duplicate name."""
    if not name or "/" in name or " " in name:
        raise ValueError(f"method names are single words, got {name!r}")

    def register(build: Callable[..., Any]) -> Callable[..., Any]:
        if name in _REGISTRY:
            raise ValueError(
                f"an identification method named {name!r} is already registered"
            )
        _REGISTRY[name] = MethodEntry(
            name=name, build=build, doc=doc or (build.__doc__ or "")
        )
        return build

    return register


def methods() -> Mapping[str, MethodEntry]:
    """Every method: the built-ins, then the entry-point group."""
    load_group(ENTRY_POINT_GROUP, BUILTIN_MODULES)
    return dict(_REGISTRY)


def resolve(name: str) -> MethodEntry:
    known = methods()
    if name not in known:
        raise KeyError(f"no identification method {name!r}; one of {sorted(known)}")
    return known[name]


def detect(bundle_dir: Path, recording_dir: Path) -> MethodEntry:
    """The one method that accepts this robot and recording; refused by
    name, with every method's reason, when none does."""
    reasons = {}
    for name, entry in methods().items():
        why = entry.build().accepts(bundle_dir, recording_dir)
        if why is None:
            return entry
        reasons[name] = why
    raise ValueError(
        "no identification method fits this robot from this recording: "
        + "; ".join(f"{name}: {why}" for name, why in reasons.items())
    )


def raw_files(recording_dir: Path, suffix: str) -> list[Path]:
    """The raw source files an ingest kept beside the recording."""
    return sorted((Path(recording_dir) / "raw").glob(f"*{suffix}"))


@method(
    "drivetrain-ratio",
    doc="The rig's two-wheel drivetrain from a .wire sweep: gear and friction "
    "loss per unit damping, damping fixed as the scale reference (ratio form).",
)
class DrivetrainRatio:
    name = "drivetrain-ratio"

    def accepts(self, bundle_dir: Path, recording_dir: Path) -> str | None:
        from rq_pipeline.bundles.profile import load_profile  # noqa: PLC0415

        bundle_dir, recording_dir = Path(bundle_dir), Path(recording_dir)
        try:
            profile = load_profile(bundle_dir)
        except (
            Exception
        ) as why:  # a bundle this method cannot read is a reason, not a crash
            return f"the bundle has no drivetrain profile ({why})"
        if not (bundle_dir / profile.model_file).is_file():
            return f"the bundle's model {profile.model_file!r} is missing"
        if not raw_files(recording_dir, ".wire"):
            return "the recording keeps no raw .wire file (ingest a .wire sweep)"
        return None

    def fit(
        self, bundle_dir: Path, recording_dir: Path, *, write: bool = True
    ) -> tuple[IdentificationResult, Path | None]:
        from rq_pipeline.robot.drivetrain_fit import fit_drivetrain  # noqa: PLC0415

        wire = raw_files(Path(recording_dir), ".wire")[0]
        return fit_drivetrain(Path(bundle_dir), wire, write=write)
