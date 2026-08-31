"""The certified actuator bundle: BAM's fit, wrapped, never rewritten.

The interchange artifact of the cross-framework design (docs/
e2e-research/58 §1, dialect contract §9): one self-contained JSON per
(actuator, model tier) that any BAM loader can still consume — the
inner ``params`` dict is the published fit VERBATIM, no key renamed,
BAM's own ``model``/``actuator`` keys untouched — while everything BAM
discards rides in sibling sections: provenance, metrics, uncertainty,
electrical/firmware context, and sanity checks. The envelope is
content-addressed with the same ``name@hash`` stamp every other
artifact in this repo carries.

Honesty over completeness: a section the wrap cannot source is ABSENT,
never fabricated — ``verify`` names what is missing as advisories
(our vendored fits, imported from BAM's published point estimates,
legitimately lack metrics and uncertainty until re-fit from logs) and
raises only on violations: a tampered stamp, a params dict without
BAM's identity keys, an unknown top-level section. Unknown keys INSIDE
``params`` are allowed — BAM's key set varies per motor by design
(docs/e2e-research/57 §3) and verbatim means verbatim; unknown keys in
OUR envelope are refused — silent tolerance of typos is how BAM's own
loader loses fields.

Stdlib only, deliberately: a bundle must verify on an auditor's laptop.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rq_pipeline.bundles.hashing import STAMP_SEPARATOR, fields_hash
from rq_pipeline.robot.actuator_library import (
    ACTUATORS_ROOT,
    PROVENANCE_FILE,
    list_actuators,
    list_models,
)

SCHEMA = "robotiq-actuator-bundle/1"

# Every section the envelope may carry. A key outside this set is a
# refusal, not a warning — the one lesson BAM's silently-tolerant
# loader teaches by counterexample.
SECTIONS = frozenset(
    {
        "schema",
        "params",
        "provenance",
        "context",
        "metrics",
        "uncertainty",
        "checks",
        "wrap",
        "stamp",
    }
)

# Sections whose absence is legal but must be SAID: verify reports each
# as an advisory so a consumer knows what this bundle cannot promise.
OPTIONAL_SECTIONS = ("context", "metrics", "uncertainty")

# BAM's params files always carry these two identity keys on top of the
# numeric fit fields; a dict without them is not a BAM fit.
PARAM_IDENTITY_KEYS = ("model", "actuator")

# Observed search bounds from BAM's published fits — NOT their declared
# Parameter(min,max) (which the params files do not record), but rails
# their shipped values demonstrably sit on: xl330/m4 ships
# alpha = 9.999999997 (docs/e2e-research/57 §3). A fitted value within
# RAIL_TOLERANCE of a bound earns a `near_search_bound` check flag — a
# certification signal, not an error.
OBSERVED_SEARCH_BOUNDS: Mapping[str, float] = {"alpha": 10.0}
RAIL_TOLERANCE = 0.01  # fraction of the bound

# A parameter this small is the optimizer's floor, not a measurement —
# BAM ships load terms at ~1e-13 (docs/e2e-research/57 §3).
FLOOR = 1e-9

# Fields excluded from the content hash: the stamp itself, and the wrap
# metadata (WHEN it was wrapped must not change WHAT it is — the same
# fit wrapped twice is the same artifact).
UNHASHED = ("stamp", "wrap")


def wrap(
    slug: str,
    tier: str,
    *,
    context: Mapping[str, Any] | None = None,
    wrapped_on: str | None = None,
) -> dict[str, Any]:
    """Build the bundle for one vendored (actuator, tier) fit.

    `context` is caller-supplied electrical/firmware facts (vin, kp_fw,
    actuator class) — BAM keeps them in class-constructor defaults, not
    in the params file (57 §4), so the wrap records only what the
    caller states rather than guessing.
    """
    directory = ACTUATORS_ROOT / slug
    params_path = directory / f"{tier}.json"
    if not params_path.exists():
        raise FileNotFoundError(
            f"no {tier}.json for actuator {slug!r} under {directory} — "
            f"tiers available: {list(list_models(slug))}"
        )
    params = json.loads(params_path.read_text())
    missing = [key for key in PARAM_IDENTITY_KEYS if key not in params]
    if missing:
        raise ValueError(
            f"{params_path} is missing BAM identity key(s) {missing} — "
            "not a BAM fit, refusing to wrap it as one"
        )
    provenance = json.loads((directory / PROVENANCE_FILE).read_text())

    bundle: dict[str, Any] = {
        "schema": SCHEMA,
        "params": params,
        "provenance": provenance,
        "checks": run_checks(params),
    }
    if context:
        bundle["context"] = dict(context)
    if wrapped_on is not None:
        bundle["wrap"] = {"date": wrapped_on}
    bundle["stamp"] = _stamp(bundle)
    return bundle


def run_checks(params: Mapping[str, Any]) -> dict[str, list[str]]:
    """The sanity signals BAM computes nothing like: fields at the
    optimizer's floor, and fields sitting on an observed search rail."""
    numeric = {
        key: value
        for key, value in params.items()
        if key not in PARAM_IDENTITY_KEYS and isinstance(value, (int, float))
    }
    at_floor = sorted(key for key, value in numeric.items() if 0 <= abs(value) <= FLOOR)
    near_bound = sorted(
        key
        for key, bound in OBSERVED_SEARCH_BOUNDS.items()
        if key in numeric and abs(numeric[key] - bound) <= RAIL_TOLERANCE * bound
    )
    return {"at_floor": at_floor, "near_search_bound": near_bound}


def verify(bundle: Mapping[str, Any]) -> list[str]:
    """Raise on any violation; return the advisories (absent optional
    sections) a consumer should read before trusting the bundle."""
    unknown = sorted(set(bundle) - SECTIONS)
    if unknown:
        raise ValueError(
            f"bundle carries unknown section(s) {unknown} — "
            f"the schema {SCHEMA} defines {sorted(SECTIONS)}"
        )
    for required in ("schema", "params", "provenance", "stamp"):
        if required not in bundle:
            raise ValueError(f"bundle is missing required section {required!r}")
    if bundle["schema"] != SCHEMA:
        raise ValueError(
            f"bundle schema is {bundle['schema']!r}; this verifier speaks {SCHEMA!r}"
        )

    params = bundle["params"]
    missing = [key for key in PARAM_IDENTITY_KEYS if key not in params]
    if missing:
        raise ValueError(f"params is missing BAM identity key(s) {missing}")
    bad_types = sorted(
        key
        for key, value in params.items()
        if key not in PARAM_IDENTITY_KEYS and not isinstance(value, (int, float))
    )
    if bad_types:
        raise ValueError(f"non-numeric fit value(s) for {bad_types}")

    expected = _stamp(bundle)
    if bundle["stamp"] != expected:
        raise ValueError(
            f"stamp mismatch: bundle says {bundle['stamp']!r} but its content "
            f"hashes to {expected!r} — the artifact was edited after stamping"
        )

    return [
        f"no {section!r} section: this bundle cannot promise "
        + {
            "context": "the electrical/firmware operating point (vin, kp)",
            "metrics": "fit quality (train or held-out MAE)",
            "uncertainty": "parameter intervals (point estimates only)",
        }[section]
        for section in OPTIONAL_SECTIONS
        if section not in bundle
    ]


def write_bundle(bundle: Mapping[str, Any], out_dir: Path) -> Path:
    """`<name>.<tier>.bundle.json` under `out_dir`, verified first —
    this module never writes an artifact it would refuse to read.
    The filename prefers the provenance slug over BAM's `actuator` key:
    the key drops the operating voltage (`sts3215` for the 7.4 V fit,
    57 §3), and two voltage variants must not collide on disk."""
    verify(bundle)
    params = bundle["params"]
    name = bundle["provenance"].get("slug", params["actuator"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.{params['model']}.bundle.json"
    path.write_text(json.dumps(dict(bundle), indent=2, sort_keys=True) + "\n")
    return path


def read_bundle(path: Path) -> dict[str, Any]:
    bundle = json.loads(Path(path).read_text())
    verify(bundle)
    return bundle


def wrap_all(out_dir: Path, *, wrapped_on: str | None = None) -> list[Path]:
    """Every vendored (actuator, tier) fit → a verified bundle file."""
    return [
        write_bundle(wrap(slug, tier, wrapped_on=wrapped_on), out_dir)
        for slug in list_actuators()
        for tier in list_models(slug)
    ]


def _stamp(bundle: Mapping[str, Any]) -> str:
    """Stamp name and hash both derive from bundle CONTENT alone, so a
    bundle verifies self-contained — deliberately NOT the directory
    slug, which BAM lets disagree with the `actuator` key (57 §3)."""
    params = bundle["params"]
    hashed = {key: value for key, value in bundle.items() if key not in UNHASHED}
    return (
        f"{params['actuator']}-{params['model']}{STAMP_SEPARATOR}{fields_hash(hashed)}"
    )
