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
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rq_pipeline.bundles.basis import BASES
from rq_pipeline.bundles.hashing import require_stamp, stamp
from rq_pipeline.robot.identify import (
    DEFAULT_PINNED_FRACTION,
    IdentificationResult,
    IdentifiedParameter,
)

FITS_DIRECTORY = "fits"
SPREAD_FILENAME = "SPREAD.json"


def code_version() -> str:
    """The git sha the fit ran under; 'unknown' outside a checkout."""
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                cwd=Path(__file__).parent,
            ).stdout.strip()
            or "unknown"
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@dataclass(frozen=True)
class FitRecord:
    """One excitation run's fit, bound to the exact bytes it came from.

    The 2026-08-25 artifact review walked the flagship hash chain and
    found one present link of five — the record named its recording and
    nothing else it was produced under. The four absent links are now
    fields: `profile` and `model` (name@hash of the bundle files whose
    values the fit consumed), `code` (git sha), and `units` (a bare
    float is not a measurement). `pinned_criterion` spells out what
    "pinned" means, because range-relative pinning can mark an interval
    nine times its estimate as pinned and a recipient deserves to know.
    Records written before that date lack the fields (None on load).
    """

    robot: str
    recording: str
    anchor: str
    confidence: float
    parameters: tuple[IdentifiedParameter, ...]
    created_utc: str
    profile: str | None = None
    model: str | None = None
    code: str | None = None
    units: dict[str, str] | None = None
    pinned_criterion: str | None = None
    # 2026-09-24, the legged fit on public logs: whose robot the data was
    # (`bundles.basis.BASES`), where it came from, and how well each
    # part fitted. None on records written before.
    basis: str | None = None
    provenance: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None

    def to_json(self) -> str:
        # Strict JSON: `Infinity` is not RFC 8259 and broke JSON.parse
        # on exactly the honesty feature (an unbounded half-width). An
        # unbounded interval serializes as null; the loader restores it.
        payload = asdict(self)
        for parameter in payload["parameters"]:
            if parameter["half_width"] == float("inf"):
                parameter["half_width"] = None
        return json.dumps(payload, indent=2, allow_nan=False)


def write_fit_record(  # noqa: PLR0913 - each argument is a refusal rule
    bundle_dir: Path,
    result: IdentificationResult,
    *,
    robot: str,
    recording: str,
    anchor: str,
    units: dict[str, str],
    basis: str | None = None,
    provenance: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> Path:
    """Record a fit into the bundle. One file per recording; refitting the
    same recording overwrites (git history keeps the old fit)."""
    require_stamp(recording, "recording identity")
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
    missing_units = [
        parameter.name for parameter in result.parameters if parameter.name not in units
    ]
    if missing_units:
        raise ValueError(
            f"every parameter needs a units entry; missing {missing_units} — "
            "a bare float is not a measurement"
        )
    if basis is not None and basis not in BASES:
        raise ValueError(f"basis must be one of {BASES}, got {basis!r}")
    bundle = Path(bundle_dir)
    profile_path = bundle / "profile.json"
    model_files = sorted(bundle.glob("*.xml"))
    record = FitRecord(
        robot=robot,
        recording=recording,
        anchor=anchor,
        confidence=result.confidence,
        parameters=result.parameters,
        created_utc=datetime.now(timezone.utc).isoformat(),
        profile=(
            stamp(profile_path.name, profile_path) if profile_path.is_file() else None
        ),
        model=(stamp(model_files[0].name, model_files[0]) if model_files else None),
        code=code_version(),
        units=dict(units),
        pinned_criterion=(
            f"half_width <= {DEFAULT_PINNED_FRACTION} * allowed_range "
            "(range-relative; a pinned interval can still span zero)"
        ),
        basis=basis,
        provenance=dict(provenance) if provenance is not None else None,
        metrics=dict(metrics) if metrics is not None else None,
    )
    fits = Path(bundle_dir) / FITS_DIRECTORY
    fits.mkdir(parents=True, exist_ok=True)
    path = fits / f"{recording}.json"
    path.write_text(record.to_json(), encoding="utf-8")
    return path


def load_fit_records(bundle_dir: Path) -> tuple[FitRecord, ...]:
    """Every fit the bundle carries, oldest first by creation time."""
    fits = Path(bundle_dir) / FITS_DIRECTORY
    if not fits.is_dir():
        return ()
    records = []
    for path in sorted(fits.glob("*.json")):
        if path.name == SPREAD_FILENAME:
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        for parameter in raw["parameters"]:
            if parameter["half_width"] is None:
                parameter["half_width"] = float("inf")
        raw["parameters"] = tuple(
            IdentifiedParameter(**parameter) for parameter in raw["parameters"]
        )
        try:
            records.append(FitRecord(**raw))
        except TypeError as error:
            raise ValueError(
                f"fit record {path} has unknown field(s) — written by newer "
                f"code? ({error})"
            ) from error
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


@dataclass(frozen=True)
class SpreadVerdict:
    """One parameter's cross-run judgement — the single spread truth.

    The review (2026-08-26) caught the summary and the artifact
    encoding this rule TWICE with a divergence: the artifact filtered
    unbounded half-widths before averaging, the summary did not — so a
    parameter with one unbounded interval made the summary's mean
    infinite and it printed "runs agree" for exactly the parameter
    that was never pinned. One implementation now; both faces render it.
    """

    lowest: float
    highest: float
    mean_half_width: float | None  # None when every interval is unbounded

    @property
    def exceeds(self) -> bool:
        if self.mean_half_width is None:
            return False
        return self.highest - self.lowest > 2.0 * self.mean_half_width

    @property
    def verdict(self) -> str:
        return (
            "spread EXCEEDS per-run intervals — trust the spread"
            if self.exceeds
            else "runs agree within their intervals"
        )


def spread_verdicts(records: tuple[FitRecord, ...]) -> dict[str, SpreadVerdict]:
    """Per parameter: estimate span vs mean FINITE interval width."""
    widths: dict[str, list[float]] = {}
    for record in records:
        for parameter in record.parameters:
            widths.setdefault(parameter.name, []).append(parameter.half_width)
    verdicts = {}
    for name, (low, high) in cross_run_spread(records).items():
        finite = [width for width in widths[name] if width != float("inf")]
        verdicts[name] = SpreadVerdict(
            lowest=low,
            highest=high,
            mean_half_width=sum(finite) / len(finite) if finite else None,
        )
    return verdicts


def spread_summary(records: tuple[FitRecord, ...]) -> str:
    """Human-readable cross-run report: spread beside mean interval width."""
    if not records:
        return "no fit records"
    lines = [f"{len(records)} fit run(s)"]
    for name, judged in spread_verdicts(records).items():
        width = (
            f"{judged.mean_half_width:.6g}"
            if judged.mean_half_width is not None
            else "unbounded"
        )
        lines.append(
            f"  {name}: estimates span [{judged.lowest:.6g}, {judged.highest:.6g}], "
            f"mean half-width {width} — {judged.verdict}"
        )
    return "\n".join(lines)


MIN_FITS_FOR_SPREAD = 2  # a spread of one run is not a spread


def write_spread_record(bundle_dir: Path) -> Path:
    """Persist the cross-run verdict — the number the house calls the truth.

    `fits/SPREAD.json`: per-parameter spread beside mean interval width
    with the exceeds/agrees verdict, bound to the records it summarizes.
    """
    records = load_fit_records(bundle_dir)
    if len(records) < MIN_FITS_FOR_SPREAD:
        raise ValueError(
            f"cross-run spread needs at least {MIN_FITS_FOR_SPREAD} fit records, "
            f"got {len(records)}"
        )
    payload = {
        "summarizes": [record.recording for record in records],
        "spread": {
            name: {
                "lowest_estimate": judged.lowest,
                "highest_estimate": judged.highest,
                "mean_half_width": judged.mean_half_width,
                "verdict": judged.verdict,
            }
            for name, judged in spread_verdicts(records).items()
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code": code_version(),
    }
    path = Path(bundle_dir) / FITS_DIRECTORY / SPREAD_FILENAME
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return path
