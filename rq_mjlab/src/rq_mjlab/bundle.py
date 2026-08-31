"""Verified-bundle loading: the refusal at the door.

`rq_mjlab` builds actuators from the certified store in
`robots/actuator-bundles/` (schema robotiq-actuator-bundle/1, A1) and
from nowhere else. The reader and verifier are the pipeline's own
(`rq_pipeline.robot.actuator_bundle`) — one implementation of the hash,
the rails and the floors, never a second copy here. An unverifiable
bundle is refused with every problem named; "which fit was this policy
trained against" is answerable because the stamp travels with the
config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rq_pipeline.robot.actuator_bundle import read_bundle, verify


class BundleRefused(ValueError):
    """A bundle that cannot be trusted, with the verifier's findings."""


def verified_bundle(path: str | Path) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Read and verify one bundle file; refuse on any violation.

    `verify`'s contract (A1): it RAISES on a violation — a tampered
    parameter, a stamp mismatch, a missing required section — and
    RETURNS the advisories (absent optional sections: context, metrics,
    uncertainty). A violation becomes a `BundleRefused` naming the file;
    the advisories come back with the bundle, because what a bundle
    cannot promise is part of consuming it honestly, never a reason to
    invent the missing sections.
    """
    path = Path(path)
    try:
        bundle = read_bundle(path)  # read_bundle verifies the stamp itself
        advisories = verify(bundle)
    except ValueError as violation:
        raise BundleRefused(
            f"{path.name} failed verification; refusing to build an "
            f"actuator from it: {violation}"
        ) from violation
    return bundle, tuple(advisories)
