"""The fit record: a measurement written into the robot's bundle.

`identify()` produces a result in memory; this module makes it an
artifact. Each excitation run's fit lands as one JSON file under the
bundle's `fits/` directory, named after the recording it was fitted
from, carrying: every parameter's estimate, interval and pinned verdict;
which recording (`name@hash` — unstamped is refused, same rule as the
certificate); and the anchor statement.

The anchor is REQUIRED, not optional metadata. The torque scale is
structurally unobservable from duty→angle data alone (the rehearsal's
first finding — see robots/rig-drivetrain/README.md), so a fit that does
not say which parameter was anchored from outside the data, and from
where, is not auditable and is refused.

Repeat runs are the protocol, not an option: single-run tick-quantization
bias is systematic (~2% gear, ~5% damping) and does not average away, so
`cross_run_spread` reports the estimate spread ACROSS records beside the
per-run intervals — when the spread dwarfs the intervals, the intervals
are lying and the spread is the truth.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from rq_pipeline.robot.identify import IdentificationResult, IdentifiedParameter

FITS_DIRECTORY = "fits"


@dataclass(frozen=True)
class FitRecord:
    """One excitation run's fit, bound to the exact bytes it came from."""

    robot: str
    recording: str
    anchor: str
    confidence: float
    parameters: tuple[IdentifiedParameter, ...]
    created_utc: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def write_fit_record(
    bundle_dir: Path,
    result: IdentificationResult,
    *,
    robot: str,
    recording: str,
    anchor: str,
) -> Path:
    """Record a fit into the bundle. One file per recording; refitting the
    same recording overwrites (git history keeps the old fit)."""
    if "@" not in recording:
        raise ValueError(
            f"recording identity must be name@hash, got {recording!r} — "
            "stamp it with rq_pipeline.bundles.stamp first"
        )
    if "/" in recording or "\\" in recording:
        raise ValueError(
            f"recording identity must not contain path separators, got "
            f"{recording!r} — the record's filename is derived from it"
        )
    if not anchor.strip():
        raise ValueError(
            "a fit without an anchor statement is not auditable: the torque "
            "scale is unobservable from the data alone, so say which "
            "parameter was anchored and from what source"
        )
    record = FitRecord(
        robot=robot,
        recording=recording,
        anchor=anchor,
        confidence=result.confidence,
        parameters=result.parameters,
        created_utc=datetime.now(timezone.utc).isoformat(),
    )
    fits = Path(bundle_dir) / FITS_DIRECTORY
    fits.mkdir(parents=True, exist_ok=True)
    path = fits / f"{recording}.json"
    path.write_text(record.to_json())
    return path


def load_fit_records(bundle_dir: Path) -> tuple[FitRecord, ...]:
    """Every fit the bundle carries, oldest first by creation time."""
    fits = Path(bundle_dir) / FITS_DIRECTORY
    if not fits.is_dir():
        return ()
    records = []
    for path in sorted(fits.glob("*.json")):
        raw = json.loads(path.read_text())
        raw["parameters"] = tuple(
            IdentifiedParameter(**parameter) for parameter in raw["parameters"]
        )
        records.append(FitRecord(**raw))
    return tuple(sorted(records, key=lambda record: record.created_utc))


def cross_run_spread(records: tuple[FitRecord, ...]) -> dict[str, tuple[float, float]]:
    """Per parameter, (lowest, highest) estimate across all records.

    The Paper 0 protocol number: quantization bias is systematic within a
    run, so agreement BETWEEN runs — not the width of any single run's
    interval — is what supports trusting an estimate.
    """
    estimates: dict[str, list[float]] = {}
    for record in records:
        for parameter in record.parameters:
            estimates.setdefault(parameter.name, []).append(parameter.estimate)
    return {name: (min(values), max(values)) for name, values in estimates.items()}


def spread_summary(records: tuple[FitRecord, ...]) -> str:
    """Human-readable cross-run report: spread beside mean interval width."""
    if not records:
        return "no fit records"
    lines = [f"{len(records)} fit run(s)"]
    widths: dict[str, list[float]] = {}
    for record in records:
        for parameter in record.parameters:
            widths.setdefault(parameter.name, []).append(parameter.half_width)
    for name, (low, high) in cross_run_spread(records).items():
        mean_half_width = sum(widths[name]) / len(widths[name])
        verdict = (
            "spread EXCEEDS per-run intervals — trust the spread"
            if high - low > 2.0 * mean_half_width
            else "runs agree within their intervals"
        )
        lines.append(
            f"  {name}: estimates span [{low:.6g}, {high:.6g}], "
            f"mean half-width {mean_half_width:.6g} — {verdict}"
        )
    return "\n".join(lines)
