"""The measurement report, one command — the artifact the product IS.

    cd pipeline && uv run --extra sim python ../tools/fit-report.py \
        ../robots/rig-drivetrain [--fit ../recordings/<sweep>.wire ...]

Renders a bundle's fit records the way a customer should meet them:
every parameter with its estimate, interval, units and PINNED /
NOT PINNED verdict; the anchor statement verbatim (a fit that cannot
say what anchored its scale is not auditable); the cross-run spread
verdict beside the per-run intervals — and the field's counterpart,
the one copied constant everyone else ships, as the foil.

`--fit` first runs the drivetrain ratio-form fit on the given wire
recording(s), writing new records + SPREAD.json into the bundle — the
whole record → excite → identify → report chain in one invocation.
"""

import argparse
import sys
from pathlib import Path

from _lab import bootstrap

bootstrap()
from rq_pipeline.robot.fit_record import (  # noqa: E402
    load_fit_records,
    spread_summary,
    write_spread_record,
)

# The foil, cited (docs/e2e-research/30 §3.3-3.4): the SO-101 constants
# shipped by the field, byte-identical across two companies' repos.
MIN_FITS_FOR_SPREAD = 2  # a spread needs two fits to disagree


FOIL = (
    "The field's counterpart:  kp=17.8, damping=0.60 — one guess for six\n"
    "different joints, byte-identical in Lightwheel's and Positronic's\n"
    "trees; Lightwheel's own two repos disagree on the same arm by 56x in\n"
    "stiffness. No intervals, no verdicts, no anchor, anywhere.\n"
    "(docs/e2e-research/30-the-pipeline.md, read in their sources.)"
)


def render(bundle: Path) -> str:
    records = load_fit_records(bundle)
    if not records:
        return f"{bundle}: no fit records (fits/ empty or absent)"
    lines = [f"MEASUREMENT REPORT — {bundle.name}", "=" * 64]
    for record in records:
        lines.append("")
        lines.append(f"run {record.recording}")
        lines.append(f"  fitted against {record.model} + {record.profile}")
        lines.append(f"  code {record.code} · confidence {record.confidence:.0%}")
        for parameter in record.parameters:
            verdict = "pinned" if parameter.pinned else "NOT PINNED"
            width = (
                "± unbounded"
                if parameter.half_width == float("inf")
                else f"± {parameter.half_width:.3g}"
            )
            unit = (record.units or {}).get(parameter.name, "units unstated")
            lines.append(
                f"    {parameter.name:22s} {parameter.estimate:12.6g} "
                f"{width:>14s}  [{verdict}]  {unit}"
            )
        lines.append(f"  anchor: {record.anchor}")
    lines.append("")
    lines.append("CROSS-RUN VERDICT (the number to trust)")
    lines.append(spread_summary(records))
    lines.append("")
    lines.append("-" * 64)
    lines.append(FOIL)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bundle", type=Path, help="robot bundle directory")
    parser.add_argument(
        "--fit",
        type=Path,
        action="append",
        default=[],
        metavar="WIRE",
        help="fit this sweep recording first (drivetrain ratio form)",
    )
    arguments = parser.parse_args()
    if arguments.fit:
        from rq_pipeline.robot.drivetrain_fit import fit_drivetrain  # noqa: PLC0415

        for wire in arguments.fit:
            _, path = fit_drivetrain(arguments.bundle, wire)
            print(f"fitted {wire.name} -> {path.name}", file=sys.stderr)
        if len(load_fit_records(arguments.bundle)) >= MIN_FITS_FOR_SPREAD:
            write_spread_record(arguments.bundle)
    print(render(arguments.bundle))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
