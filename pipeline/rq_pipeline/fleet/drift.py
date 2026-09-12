"""Drift: is the robot we have still the robot the evaluation was judged
on? (docs/76 §9.1, 2026-09-13)

A fresh recording is identified by the bundle's registered method,
without writing a fit record — a check is a question, identification is
a decision — and every parameter's fresh interval is judged against the
REFERENCE interval: the union of the bundle's fit records, lowest lower
bound to highest upper bound. That is the spread rule the repo already
trusts (`robot/fit_record.cross_run_spread`): when runs disagree beyond
their own intervals, trust the spread, never one run.

Three verdicts, no fourth. `within`: the fresh interval overlaps the
reference. `left`: the fresh interval is pinned (the identifier's own
criterion) and does not overlap. `unresolved`: the fresh interval is not
pinned, so this recording cannot say either way. A parameter the method
anchors from outside the data is `anchored`: reported, never judged. The
record is drifted when any parameter left; its recommendation names them.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rq_pipeline.bundles.json_record import JsonRecord
from rq_pipeline.robot.fit_record import FitRecord, load_fit_records

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

RULE = (
    "reference = the union of the bundle's fit-record intervals per parameter; "
    "within = the fresh interval overlaps it; left = the fresh interval is pinned "
    "and does not overlap; unresolved = the fresh interval is not pinned; "
    "anchored = fixed by the method from outside the data, never judged"
)
RECOMMEND_CLEAN = "no parameter left its identified interval; the evaluation stands"
RECOMMEND_DRIFTED = (
    "re-identify the robot from this recording (identify_system), then "
    "re-evaluate the policy against the new interval"
)
RECOMMEND_UNRESOLVED = (
    "this recording could not pin {names}; record a longer or wider excitation "
    "before judging them"
)


@dataclass(frozen=True)
class ParameterDrift:
    """One parameter's fresh interval against its reference, judged."""

    name: str
    verdict: str
    reference_lower: float
    reference_upper: float
    fresh_estimate: float
    fresh_half_width: float
    fresh_pinned: bool
    # (fresh estimate - reference centre) over the reference half-width:
    # 0 at the centre, ±1 at the edges, None when the reference has no
    # width to measure against (a single record's point, or unbounded).
    shift: float | None
    unit: str = ""

    @property
    def fresh_lower(self) -> float:
        return self.fresh_estimate - self.fresh_half_width

    @property
    def fresh_upper(self) -> float:
        return self.fresh_estimate + self.fresh_half_width


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
    references: int = field(default=0)


def reference_intervals(
    records: tuple[FitRecord, ...],
) -> dict[str, tuple[float, float]]:
    """Per parameter, the union of every record's interval."""
    out: dict[str, tuple[float, float]] = {}
    for record in records:
        for p in record.parameters:
            lo, hi = out.get(p.name, (math.inf, -math.inf))
            out[p.name] = (min(lo, p.lower), max(hi, p.upper))
    return out


def _shift(estimate: float, lower: float, upper: float) -> float | None:
    half = (upper - lower) / 2.0
    if not math.isfinite(half) or half <= 0.0:
        return None
    return (estimate - (lower + upper) / 2.0) / half


def judge_parameters(
    records: tuple[FitRecord, ...],
    fresh: IdentificationResult,
    *,
    anchored: tuple[str, ...] = (),
) -> tuple[ParameterDrift, ...]:
    """Every fresh parameter against its reference; a parameter the
    reference never identified is unresolved by name."""
    reference = reference_intervals(records)
    units: dict[str, str] = {}
    for record in records:
        units.update(record.units or {})
    judged: list[ParameterDrift] = []
    for p in fresh.parameters:
        lo, hi = reference.get(p.name, (math.nan, math.nan))
        if p.name in anchored:
            verdict = ANCHORED
        elif math.isnan(lo) or not p.pinned:
            verdict = UNRESOLVED
        elif p.lower <= hi and p.upper >= lo:
            verdict = WITHIN
        else:
            verdict = LEFT
        judged.append(
            ParameterDrift(
                name=p.name,
                verdict=verdict,
                reference_lower=lo,
                reference_upper=hi,
                fresh_estimate=p.estimate,
                fresh_half_width=p.half_width,
                fresh_pinned=p.pinned,
                shift=None if math.isnan(lo) else _shift(p.estimate, lo, hi),
                unit=units.get(p.name, ""),
            )
        )
    return tuple(judged)


def recommendation_for(left: tuple[str, ...], unresolved: tuple[str, ...]) -> str:
    if left:
        return RECOMMEND_DRIFTED
    if unresolved:
        return RECOMMEND_UNRESOLVED.format(names=", ".join(unresolved))
    return RECOMMEND_CLEAN


def judge(
    bundle_dir: Path,
    recording_dir: Path,
    fitter: IdentificationMethod,
    *,
    robot: str,
    recording: str,
) -> DriftRecord:
    """Identify the fresh recording without writing, judge it against the
    bundle's records. Refuses by name a bundle with no fit record."""
    bundle_dir, recording_dir = Path(bundle_dir), Path(recording_dir)
    records = load_fit_records(bundle_dir)
    if not records:
        raise ValueError(
            f"{robot}: no fit record to compare against; identify the robot "
            "from a recording first (identify_system)"
        )
    fresh, _ = fitter.fit(bundle_dir, recording_dir, write=False)
    parameters = judge_parameters(records, fresh, anchored=fitter.anchored)
    left = tuple(p.name for p in parameters if p.verdict == LEFT)
    unresolved = tuple(p.name for p in parameters if p.verdict == UNRESOLVED)
    return DriftRecord(
        robot=robot,
        recording=recording,
        method=fitter.name,
        fit=tuple(r.recording for r in records),
        references=len(records),
        parameters=parameters,
        drifted=bool(left),
        left=left,
        unresolved=unresolved,
        recommendation=recommendation_for(left, unresolved),
        anchor=records[-1].anchor,
        created_utc=datetime.now(timezone.utc).isoformat(),
        code=_code_version(),
        instrument=_instrument(),
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
    for key in ("fit", "left", "unresolved"):
        raw[key] = tuple(raw.get(key, ()))
    known = {f for f in DriftRecord.__dataclass_fields__}
    return DriftRecord(**{k: v for k, v in raw.items() if k in known})


def _code_version() -> str:
    from rq_pipeline.robot.fit_record import _code_version as version  # noqa: PLC0415

    return version()


def _instrument() -> str:
    try:
        import mujoco  # noqa: PLC0415
    except ImportError:
        return "unrecorded"
    return f"mujoco-{mujoco.__version__}"
