"""What a robot bundle says about itself: `bundle.json`, written at
onboarding — the model file that was compiled (a directory may hold
several: a base model, an MJX twin, scenes), where it came from, and
the census the compile gave. The rig's `profile.json` (drivetrain
constants) stays separate: a profile is a measurement, this is a record
of provenance.

No timestamps inside: a bundle's version is a hash over its files, and
two onboardings of the same directory must be the same version.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BUNDLE_FILE = "bundle.json"
BUNDLE_SCHEMA = "trainnr-robot/1"


def write_bundle_record(
    bundle_dir: Path, name: str, model_file: str, model: Any, *, source: Path
) -> Path:
    """Record the bundle's model file, source and census."""
    record = {
        "schema": BUNDLE_SCHEMA,
        "name": name,
        "model_file": model_file,
        "source": str(Path(source).expanduser().resolve()),
        "census": {
            "bodies": int(model.nbody),
            "joints": int(model.njnt),
            "dofs": int(model.nv),
            "actuators": int(model.nu),
            "sensors": int(model.nsensor),
            "geoms": int(model.ngeom),
            "meshes": int(model.nmesh),
            "keyframes": int(model.nkey),
        },
    }
    out = Path(bundle_dir) / BUNDLE_FILE
    out.write_text(
        json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out


def read_bundle_record(bundle_dir: Path) -> dict[str, Any]:
    """The record, or an empty mapping for a bundle onboarded before it
    existed (the library's rigs)."""
    path = Path(bundle_dir) / BUNDLE_FILE
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}


def model_file_of(bundle_dir: Path) -> Path | None:
    """The MJCF a bundle names: `bundle.json` first, then the rig
    profile's `model_file`, else the largest XML at the root; None for a
    bundle with no XML at all."""
    bundle_dir = Path(bundle_dir)
    named = read_bundle_record(bundle_dir).get("model_file")
    if not named:
        profile = bundle_dir / "profile.json"
        if profile.is_file():
            try:
                named = json.loads(profile.read_text(encoding="utf-8")).get(
                    "model_file"
                )
            except ValueError:
                named = None
    if named and (bundle_dir / named).is_file():
        return bundle_dir / named
    # A library rig onboarded before records existed: the largest XML at
    # the root (a scene includes its robot, so it is the biggest file).
    candidates = sorted(
        bundle_dir.glob("*.xml"), key=lambda p: p.stat().st_size, reverse=True
    )
    return candidates[0] if candidates else None
