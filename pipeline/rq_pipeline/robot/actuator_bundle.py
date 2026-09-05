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
from typing import TYPE_CHECKING, Any

from rq_pipeline.bundles.hashing import STAMP_SEPARATOR, fields_hash

if TYPE_CHECKING:  # annotation-only: this module stays stdlib-pure
    from numpy.random import Generator
from rq_pipeline.robot.actuator_library import (
    ACTUATORS_ROOT,
    PROVENANCE_FILE,
    list_actuators,
    list_models,
)

SCHEMA = "robotiq-actuator-bundle/1"

# The committed store of wrapped bundles, beside the library they wrap —
# the ONE spelling (the MCP server, the CLI and the e2e preflight all
# import it; it was spelled three ways once, review 2026-09-01).
BUNDLE_STORE = ACTUATORS_ROOT.parent / "actuator-bundles"

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
# as an advisory naming what the bundle cannot promise. OPTIONAL_SECTIONS
# derives from this map so the two cannot drift (review 2026-09-01).
ABSENCE_ADVISORIES = {
    "context": "the electrical/firmware operating point (vin, kp)",
    "metrics": "fit quality (train or held-out MAE)",
    "uncertainty": "parameter intervals (point estimates only)",
}
OPTIONAL_SECTIONS = tuple(ABSENCE_ADVISORIES)

# BAM's params files always carry these two identity keys on top of the
# numeric fit fields; a dict without them is not a BAM fit.
PARAM_IDENTITY_KEYS = ("model", "actuator")

# Observed search bounds from BAM's published fits — NOT their declared
# Parameter(min,max) (which the params files do not record), but rails
# their shipped values demonstrably sit on: xl330/m4 ships
# alpha = 9.999999997 (docs/e2e-research/57 §3). A fitted value within
# RAIL_TOLERANCE of a bound earns a `near_search_bound` check flag — a
# certification signal, not an error.
# POLICY (second review, 2026-09-01): checks are derived, so they sit
# outside the stamp — but verify recomputes them, which means changing
# FLOOR/RAIL_TOLERANCE makes every committed bundle FAIL verify (a
# measured 10/48 on a FLOOR bump) until `tools/actuator-bundle.py wrap
# --all` refreshes the store. Deliberate: a loud fleet-wide refusal
# with a named recovery beats silently re-stamping unchanged fits.
OBSERVED_SEARCH_BOUNDS: Mapping[str, float] = {"alpha": 10.0}
RAIL_TOLERANCE = 0.01  # fraction of the bound

# A parameter this small is the optimizer's floor, not a measurement —
# BAM ships load terms at ~1e-13 (docs/e2e-research/57 §3).
FLOOR = 1e-9

# Parameters that are NOT the motor's to randomise: q_offset and
# command_delay are the identification RIG's (mount bias, bus latency —
# BAM's own docs say so, 57 §7), and max_velocity is the firmware's
# internal rate-limit register. A span over "the params" must never
# jitter these — jittering a bench constant is physically meaningless.
# error_gain_ratio is deliberately NOT here: it is a FITTED correction
# (present in every params JSON with a fitted value) — the first cut
# excluded it and contradicted rq_mjlab's SCALABLE list, which draws
# it (second review, 2026-09-01). The two lists are pinned disjoint by
# rq_mjlab/tests/test_kernel_parity.py.
RIG_AND_FIRMWARE_PARAMS = frozenset({"q_offset", "command_delay", "max_velocity"})

# Fields excluded from the content hash: the stamp itself; the wrap
# metadata (WHEN it was wrapped must not change WHAT it is — the same
# fit wrapped twice is the same artifact); and the checks, which are
# DERIVED from params under this module's constants — hashing them
# coupled every bundle's identity to FLOOR/RAIL_TOLERANCE, so
# tightening a checker re-stamped unchanged fits (review 2026-09-01).
# `verify` recomputes checks instead, so tampering is still caught.
UNHASHED = ("stamp", "wrap", "checks")


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
    # A fit that carries its interval ships `<tier>.uncertainty.json`
    # beside `<tier>.json` (tools/bam-bootstrap.py --emit-uncertainty,
    # 2026-09-05): {"uncertainty": {param: {low, high}}, "metrics": {...}}.
    # Vendored point fits have none, and verify says so as an advisory.
    interval_path = directory / f"{tier}.uncertainty.json"
    if interval_path.exists():
        interval = json.loads(interval_path.read_text())
        bundle["uncertainty"] = interval["uncertainty"]
        if "metrics" in interval:
            bundle["metrics"] = interval["metrics"]
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
    for required in ("schema", "params", "provenance", "checks", "stamp"):
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

    for section, shape in (("checks", dict), ("wrap", dict), ("context", dict)):
        if section in bundle and not isinstance(bundle[section], shape):
            raise ValueError(
                f"section {section!r} must be a {shape.__name__}, got "
                f"{type(bundle[section]).__name__}"
            )
    if "checks" in bundle and bundle["checks"] != run_checks(params):
        raise ValueError(
            "the bundle's stored checks disagree with the checks its params "
            "earn under this verifier — the artifact was edited, or the "
            "checker's constants changed; refresh the store with "
            "tools/actuator-bundle.py wrap --all"
        )

    expected = _stamp(bundle)
    if bundle["stamp"] != expected:
        raise ValueError(
            f"stamp mismatch: bundle says {bundle['stamp']!r} but its content "
            f"hashes to {expected!r} — the artifact was edited after stamping"
        )

    return [
        f"no {section!r} section: this bundle cannot promise "
        + ABSENCE_ADVISORIES[section]
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


def dr_ranges(bundle: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    """The per-parameter sampling region this bundle DECLARES —
    `uncertainty: {param: {"low": x, "high": y}}` — and nothing else.
    A point-estimate bundle has no region; that is a refusal here, not
    a fabricated ±10% (the microduck lesson, 57 §5: fitted-treated-as-
    exact beside hand-guessed spans, with nothing marking which)."""
    uncertainty = bundle.get("uncertainty")
    if not uncertainty:
        raise ValueError(
            f"bundle {bundle.get('stamp', '<unstamped>')} carries no "
            "'uncertainty' section (point estimates only) — there is no "
            "identified region to sample; pass sample_dynamics a "
            "fallback_span to DECLARE a span instead, and the basis will "
            "say so"
        )
    ranges = {
        param: (float(region["low"]), float(region["high"]))
        for param, region in uncertainty.items()
        if param not in RIG_AND_FIRMWARE_PARAMS
    }
    if not ranges:
        raise ValueError(
            f"bundle {bundle.get('stamp', '<unstamped>')}'s uncertainty "
            "section covers only rig/firmware parameters — nothing the "
            "motor's dynamics may draw"
        )
    return ranges


def sample_dynamics(
    bundle: Mapping[str, Any],
    rng: Generator,
    *,
    fallback_span: float | None = None,
) -> tuple[dict[str, float], str]:
    """One episode's dynamics draw from the bundle, with its BASIS —
    the string a press manifest records so a dataset says whether its
    randomisation was identified or declared (docs/e2e-research/60 §3).

    With an `uncertainty` section: uniform over each parameter's
    identified interval, basis "identified-interval". Without one, and
    only with an explicit `fallback_span`: uniform over ±span around
    the point estimates, basis naming the span as caller-declared.
    `rng` is any numpy Generator — the press already owns one."""
    ranges, basis = declared_ranges(bundle, fallback_span=fallback_span)
    return {
        param: float(rng.uniform(low, high)) for param, (low, high) in ranges.items()
    }, basis


def declared_ranges(
    bundle: Mapping[str, Any], *, fallback_span: float | None = None
) -> tuple[dict[str, tuple[float, float]], str]:
    """The sampling region and its BASIS, spelled once for every
    consumer (the press draws episodes from it; `rq_mjlab` draws
    per-world scales from it): the bundle's identified intervals when it
    declares them, else — only with an explicit `fallback_span` — a
    caller-declared span around the point estimates, with the basis
    string saying exactly which."""
    try:
        return dr_ranges(bundle), "identified-interval"
    except ValueError:
        if fallback_span is None:
            raise
    numeric = {
        key: float(value)
        for key, value in bundle["params"].items()
        if key not in PARAM_IDENTITY_KEYS
        and key not in RIG_AND_FIRMWARE_PARAMS
        and isinstance(value, (int, float))
    }
    # min/max, not (1-s, 1+s) order: a NEGATIVE parameter (q_offset
    # is -0.068 in the shipped sts3215 fit) inverts the endpoints —
    # caught by the first test that sampled a real bundle.
    ranges = {
        param: (
            min(value * (1 - fallback_span), value * (1 + fallback_span)),
            max(value * (1 - fallback_span), value * (1 + fallback_span)),
        )
        for param, value in numeric.items()
    }
    return ranges, (
        f"caller-declared span ±{fallback_span:g} (bundle is point estimates)"
    )


def as_scales(
    dynamics: Mapping[str, float], bundle: Mapping[str, Any]
) -> dict[str, float]:
    """A draw re-expressed as multipliers of the bundle's point
    estimates — the shape scale-based randomisers (kitting's
    `scale_dynamics`, mjlab's `operation="scale"` DR) consume."""
    params = bundle["params"]
    zeroes = sorted(p for p in dynamics if p in params and float(params[p]) == 0.0)
    if zeroes:
        raise ValueError(
            f"cannot express {zeroes} as scales: the bundle's point "
            "estimate is zero — a multiplier of zero is undefined, and "
            "silently dropping the parameter would un-randomise it"
        )
    return {
        param: value / float(params[param])
        for param, value in dynamics.items()
        if param in params
    }


def _stamp(bundle: Mapping[str, Any]) -> str:
    """Stamp name and hash both derive from bundle CONTENT alone, so a
    bundle verifies self-contained — deliberately NOT the directory
    slug, which BAM lets disagree with the `actuator` key (57 §3)."""
    params = bundle["params"]
    hashed = {key: value for key, value in bundle.items() if key not in UNHASHED}
    return (
        f"{params['actuator']}-{params['model']}{STAMP_SEPARATOR}{fields_hash(hashed)}"
    )
