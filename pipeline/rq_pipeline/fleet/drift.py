"""Drift: is the robot we have still the robot the evaluation was judged
on? (docs/76 §9.1, 2026-09-13)

A fresh recording is identified by the bundle's registered method,
without writing a fit record — a check is a question, identification is
a decision — and every parameter's fresh interval is judged against the
REFERENCE interval: the union of the bundle's PINNED fit intervals,
lowest lower bound to highest upper bound. That is the spread rule the
repo already trusts (`robot/fit_record.cross_run_spread`): when runs
disagree beyond their own intervals, trust the spread, never one run. A
record that never pinned a parameter contributes nothing to its
reference: an unbounded interval would make every later check `within`.

Three verdicts, no fourth. `within`: the fresh interval overlaps the
reference. `left`: the fresh interval is pinned (the identifier's own
criterion) and does not overlap. `unresolved`: no verdict can be given,
and the row says why — the fresh interval is not pinned, the fresh fit
did not resolve (a non-finite estimate), no record ever pinned the
parameter, or the fresh fit did not return it. A parameter the method
anchors from outside the data is `anchored`: reported, never judged.
The record is drifted when any parameter left.

On disk the record is strict JSON: an unknown or unbounded interval end
is `null`, never `NaN` or `Infinity` (the fit records learned the same,
2026-09-04).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rq_pipeline.bundles.json_record import JsonRecord
from rq_pipeline.robot.fit_record import FitRecord, code_version, load_fit_records

if TYPE_CHECKING:
    from rq_pipeline.robot.identify import IdentificationResult
    from rq_pipeline.robot.methods import IdentificationMethod

DRIFT_FILE = "drift.json"
DRIFT_SCHEMA = "trainnr-drift/1"

WITHIN = "within"
LEFT = "left"
UNRESOLVED = "unresolved"
ANCHORED = "anchored"
VERDICTS = (WITHIN, LEFT, UNRESOLVED, ANCHORED)
# Why a row is unresolved, in the record's own words.
NO_PINNED_REFERENCE = "no fit record pinned this parameter: nothing to judge against"
FRESH_NOT_PINNED = "this recording did not pin it"
FRESH_NOT_FINITE = "the fresh fit did not resolve (a non-finite estimate)"
MISSING_IN_FRESH = "the fresh fit did not return this parameter"

# The record's word, spoken the same way by the card, the drawer, the
# tile and the viewer.
DRIFTED_WORD = "drifted"
WITHIN_WORD = "within interval"

RULE = (
    "reference = the union of the bundle's PINNED fit-record intervals per "
    "parameter; within = the fresh interval overlaps it; left = the fresh interval "
    "is pinned and does not overlap; unresolved = no verdict (the row says why); "
    "anchored = fixed by the method from outside the data, never judged"
)
RECOMMEND_CLEAN = "no parameter left its identified interval; the evaluation stands"
RECOMMEND_DRIFTED = (
    "re-identify the robot from this recording (identify_system), then "
    "re-evaluate the policy against the new interval"
)
RECOMMEND_UNRESOLVED = (
    "this check could not judge {names}; record a longer or wider excitation, "
    "or identify the robot again, before judging them"
)


def verdict_word(drifted: bool) -> str:
    return DRIFTED_WORD if drifted else WITHIN_WORD


def finite_or_none(value: float | None) -> float | None:
    """A number a strict JSON can carry: a non-finite end is `null`."""
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def _strict(value: Any) -> Any:
    """Every float in a nested value made strict-JSON safe."""
    if isinstance(value, float):
        return finite_or_none(value)
    if isinstance(value, dict):
        return {k: _strict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(v) for v in value]
    return value


@dataclass(frozen=True)
class ParameterDrift:
    """One parameter's fresh interval against its reference, judged.
    `None` in an interval end means unknown (no pinned reference) or
    unresolved (a fit that did not return or did not converge)."""

    name: str
    verdict: str
    reference_lower: float | None
    reference_upper: float | None
    fresh_estimate: float | None
    fresh_half_width: float | None
    fresh_pinned: bool
    # (fresh estimate - reference centre) over the reference half-width:
    # 0 at the centre, ±1 at the edges, None when there is no reference
    # width to measure against.
    shift: float | None
    unit: str = ""
    note: str = ""  # why a row is unresolved; empty otherwise

    @property
    def fresh_lower(self) -> float | None:
        if self.fresh_estimate is None or self.fresh_half_width is None:
            return None
        return finite_or_none(self.fresh_estimate - self.fresh_half_width)

    @property
    def fresh_upper(self) -> float | None:
        if self.fresh_estimate is None or self.fresh_half_width is None:
            return None
        return finite_or_none(self.fresh_estimate + self.fresh_half_width)


@dataclass(frozen=True)
class DriftRecord(JsonRecord):
    """One check, as an artifact: what was compared to what, and the word."""

    robot: str
    recording: str
    method: str
    fit: tuple[str, ...]  # the reference records' recording identities
    parameters: tuple[ParameterDrift, ...]
    drifted: bool
    left: tuple[str, ...]
    unresolved: tuple[str, ...]
    recommendation: str
    anchor: str
    created_utc: str
    code: str
    instrument: str
    schema: str = DRIFT_SCHEMA
    rule: str = RULE
    # 2026-09-25, like with like: whose robot the fresh recording is, the
    # reference fits of ANOTHER recorded basis left out (a public log of
    # someone else's Go2 is not our robot's reference), and the
    # deployment or certificate whose span this check is read against.
    basis: str = ""
    left_out: tuple[str, ...] = ()
    against: str = ""

    @property
    def references(self) -> int:
        return len(self.fit)

    def as_json(self) -> str:
        return json.dumps(_strict(asdict(self)), indent=1, allow_nan=False)

    @classmethod
    def read(cls, path: Path) -> DriftRecord:  # type: ignore[override]
        return load_drift_record(path)


def reference_intervals(
    records: tuple[FitRecord, ...],
) -> dict[str, tuple[float, float]]:
    """Per parameter, the union of every PINNED interval; a parameter no
    record pinned is absent."""
    out: dict[str, tuple[float, float]] = {}
    for record in records:
        for p in record.parameters:
            if not p.pinned or not math.isfinite(p.lower) or not math.isfinite(p.upper):
                continue
            lo, hi = out.get(p.name, (math.inf, -math.inf))
            out[p.name] = (min(lo, p.lower), max(hi, p.upper))
    return out


def _shift(estimate: float, lower: float, upper: float) -> float | None:
    half = (upper - lower) / 2.0
    if half <= 0.0:
        return None
    return (estimate - (lower + upper) / 2.0) / half


def method_anchors(fitter: Any) -> tuple[str, ...]:
    """The parameters a method fixes from outside the data, when it says
    so (`anchored` on the method; optional, a method without it anchors
    nothing as far as this check knows)."""
    return tuple(getattr(fitter, "anchored", ()) or ())


def _names_in_order(
    records: tuple[FitRecord, ...], fresh: IdentificationResult
) -> list[str]:
    """The reference's parameters first, then anything only the fresh fit returned."""
    known: list[str] = []
    for record in records:
        for p in record.parameters:
            if p.name not in known:
                known.append(p.name)
    for p in fresh.parameters:
        if p.name not in known:
            known.append(p.name)
    return known


def judge_parameters(
    records: tuple[FitRecord, ...],
    fresh: IdentificationResult,
    *,
    anchored: tuple[str, ...] = (),
) -> tuple[ParameterDrift, ...]:
    """Every parameter the reference or the fresh fit names, judged."""
    reference = reference_intervals(records)
    units: dict[str, str] = {}
    for record in records:
        units.update(record.units or {})
    fresh_by_name = {p.name: p for p in fresh.parameters}
    judged: list[ParameterDrift] = []
    for name in _names_in_order(records, fresh):
        p = fresh_by_name.get(name)
        ref = reference.get(name)
        lo, hi = ref if ref is not None else (None, None)
        if p is None:
            judged.append(
                ParameterDrift(
                    name=name,
                    verdict=UNRESOLVED,
                    reference_lower=lo,
                    reference_upper=hi,
                    fresh_estimate=None,
                    fresh_half_width=None,
                    fresh_pinned=False,
                    shift=None,
                    unit=units.get(name, ""),
                    note=MISSING_IN_FRESH,
                )
            )
            continue
        finite = math.isfinite(p.estimate) and math.isfinite(p.half_width)
        note = ""
        if name in anchored:
            verdict = ANCHORED
        elif not finite:
            verdict, note = UNRESOLVED, FRESH_NOT_FINITE
        elif ref is None:
            verdict, note = UNRESOLVED, NO_PINNED_REFERENCE
        elif not p.pinned:
            verdict, note = UNRESOLVED, FRESH_NOT_PINNED
        elif p.lower <= ref[1] and p.upper >= ref[0]:
            verdict = WITHIN
        else:
            verdict = LEFT
        judged.append(
            ParameterDrift(
                name=name,
                verdict=verdict,
                reference_lower=lo,
                reference_upper=hi,
                fresh_estimate=finite_or_none(p.estimate),
                fresh_half_width=finite_or_none(p.half_width),
                fresh_pinned=bool(p.pinned),
                shift=(
                    _shift(p.estimate, ref[0], ref[1])
                    if ref is not None and finite
                    else None
                ),
                unit=units.get(name, ""),
                note=note,
            )
        )
    return tuple(judged)


def recommendation_for(left: tuple[str, ...], unresolved: tuple[str, ...]) -> str:
    if left:
        return RECOMMEND_DRIFTED
    if unresolved:
        return RECOMMEND_UNRESOLVED.format(names=", ".join(unresolved))
    return RECOMMEND_CLEAN


def same_basis(
    records: tuple[FitRecord, ...], basis: str
) -> tuple[tuple[FitRecord, ...], tuple[str, ...]]:
    """The reference fits a fresh recording of `basis` is judged against:
    those that measured the same kind of robot, and those that predate the
    basis field (kept, as before 2026-09-25); the rest are left out and
    named - drift of OUR robot against a union that includes other labs'
    Go2s would say nothing about ours."""
    kept, left_out = [], []
    for record in records:
        if record.basis is None or record.basis == basis:
            kept.append(record)
        else:
            left_out.append(f"{record.recording} ({record.basis})")
    return tuple(kept), tuple(left_out)


def judge(  # noqa: PLR0913 - what is judged, against what, each named
    bundle_dir: Path,
    recording_dir: Path,
    fitter: IdentificationMethod,
    *,
    robot: str,
    recording: str,
    basis: str,
    against: str = "",
) -> DriftRecord:
    """Identify the fresh recording without writing, judge it against the
    bundle's records of the same `basis` - the recording's own, which the
    caller reads (`same_basis`). Refuses by name a bundle with no fit
    record, or none of the recording's basis. `against` names the
    deployment or certificate whose span the check is read against,
    recorded as given."""
    bundle_dir, recording_dir = Path(bundle_dir), Path(recording_dir)
    every = load_fit_records(bundle_dir)
    if not every:
        raise ValueError(
            f"{robot}: no fit record to compare against; identify the robot "
            "from a recording first (identify_system)"
        )
    records, left_out = same_basis(every, basis)
    if not records:
        raise ValueError(
            f"{robot}: no fit of the recording's basis ({basis}) to compare "
            f"against; the bundle holds {list(left_out)} - identify this robot "
            "from a recording of the same basis first"
        )
    fresh, _ = fitter.fit(bundle_dir, recording_dir, write=False)
    parameters = judge_parameters(records, fresh, anchored=method_anchors(fitter))
    left = tuple(p.name for p in parameters if p.verdict == LEFT)
    unresolved = tuple(p.name for p in parameters if p.verdict == UNRESOLVED)
    return DriftRecord(
        robot=robot,
        recording=recording,
        method=fitter.name,
        fit=tuple(r.recording for r in records),
        parameters=parameters,
        drifted=bool(left),
        left=left,
        unresolved=unresolved,
        recommendation=recommendation_for(left, unresolved),
        anchor=records[-1].anchor,
        created_utc=datetime.now(timezone.utc).isoformat(),
        code=code_version(),
        instrument=_instrument(),
        basis=basis,
        left_out=left_out,
        against=against,
    )


def load_drift_record(path: Path) -> DriftRecord:
    """A record back from disk, its parameters typed; refuses another
    schema by name."""
    raw: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = raw.get("schema")
    if schema != DRIFT_SCHEMA:
        raise ValueError(
            f"{path}: schema {schema!r}, this reader speaks {DRIFT_SCHEMA!r}"
        )
    raw["parameters"] = tuple(ParameterDrift(**p) for p in raw.get("parameters", []))
    for key in ("fit", "left", "unresolved", "left_out"):
        raw[key] = tuple(raw.get(key, ()))
    known = set(DriftRecord.__dataclass_fields__)
    return DriftRecord(**{k: v for k, v in raw.items() if k in known})


def _instrument() -> str:
    try:
        import mujoco  # noqa: PLC0415
    except ImportError:
        return "unrecorded"
    from rq_pipeline.physics.backend import instrument_stamp  # noqa: PLC0415

    return instrument_stamp("mujoco", mujoco.__version__)
