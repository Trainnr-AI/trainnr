"""Run the synthetic STS3215 identifiability matrix and write the artifact.

    cd pipeline && uv run --extra sim python ../tools/sts-study.py

Regenerates data/sts3215-synthetic-identifiability.json — the committed
generator the artifact review found missing (an artifact whose producer
is not in the repo cannot be reproduced). The JSON carries its own
schema notes, units, and generator reference; docs/26 narrates the
findings; tools/sts-figure.py renders the heatmap.
"""

import json
import sys
from pathlib import Path

from _lab import bootstrap

bootstrap()
from rq_pipeline.robot.sts_synth import (  # noqa: E402
    STUDY_MATRIX,
    TRUE_ARMATURE,
    TRUE_DAMPING,
    TRUE_FRICTIONLOSS,
    TRUE_KP,
    run_condition,
)

TRUTH = {
    "servo_kp": TRUE_KP,
    "damping": TRUE_DAMPING,
    "frictionloss": TRUE_FRICTIONLOSS,
    "armature": TRUE_ARMATURE,
}
UNITS = {
    "servo_kp": "N*m/rad (position-loop P gain)",
    "damping": "N*m*s/rad (joint damping)",
    "frictionloss": "N*m (Coulomb friction torque)",
    "armature": "kg*m^2 (reflected rotor inertia)",
}
OUTPUT = Path(__file__).resolve().parent.parent / "data"
OUTPUT_FILE = OUTPUT / "sts3215-synthetic-identifiability.json"


def main() -> int:
    rows = []
    for condition in STUDY_MATRIX:
        result = run_condition(condition, seconds=6.0)
        cells = {}
        for parameter in result.parameters:
            truth = TRUTH[parameter.name]
            cells[parameter.name] = {
                "estimate": round(parameter.estimate, 6),
                "half_width": round(parameter.half_width, 6),
                "pinned": parameter.pinned,
                "error_pct": round(100.0 * (parameter.estimate - truth) / truth, 2),
                "truth_covered": bool(parameter.lower <= truth <= parameter.upper),
            }
        rows.append({"label": condition.label, "parameters": cells})
        print(condition.label, file=sys.stderr)
    payload = {
        "generator": "tools/sts-study.py (rq_pipeline.robot.sts_synth)",
        "design_date": "2026-08-25",
        "truth": TRUTH,
        "units": UNITS,
        "corruption_source": (
            "third-party video bench test of one STS3215-12V, "
            "operator-supplied summary (docs/e2e-research/27 s5) — "
            "vendor/reported-grade evidence, never our measurement"
        ),
        "rng_seed": 20260824,
        "legend": (
            "pinned = half_width <= 0.1*allowed_range. error_pct is signed "
            "vs truth. Clean cells recover truth to machine precision; "
            "their truth_covered can read false because a near-zero "
            "interval fails float-equality coverage — that is a numerics "
            "footnote, not a miss (see docs/26)."
        ),
        "conditions": rows,
    }
    OUTPUT_FILE.write_text(
        json.dumps(payload, indent=1, allow_nan=False), encoding="utf-8"
    )
    print(f"wrote {OUTPUT_FILE}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
