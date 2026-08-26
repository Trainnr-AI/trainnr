"""The robot's numbers live in the bundle, typed, with provenance.

One source of truth per robot: `profile.json` inside its bundle
directory. Code never hardcodes a robot constant — it loads the profile
and reads a named field, so changing a value in one place fixes it
everywhere, and every value carries a provenance string saying where it
came from (including the honest "UNVERIFIED" when that is the truth).

Deliberately a stdlib dataclass with explicit validation, not pydantic:
the bundles layer must stay dependency-free so a signed report is
recomputable on any Python. The guarantees the type provides — named
fields, validation on construction, immutability — are the same.

Deliberately NOT here: `STATUS_HZ`. The 50 Hz status clock is a property
of the wire protocol (the firmware contract), not of a robot, and its
single definition lives in `rq_pipeline.collect.frames`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path

PROFILE_FILE = "profile.json"


@dataclass(frozen=True)
class RobotProfile:
    """One robot's physical constants, named and validated.

    `provenance` maps field names to where-this-number-came-from strings;
    every key must name a real field, so a typo cannot silently document
    nothing.
    """

    name: str
    ticks_per_revolution: float
    camera_fps: int
    servo_pulse_floor_us: float
    servo_pulse_ceiling_us: float
    duty_ceiling_percent: float
    model_file: str = "model.xml"
    provenance: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Types before ranges: JSON will happily deliver "4" where 4 was
        # meant, and a string survives every `<=` check only to blow up
        # deep inside a consumer (R12).
        for field_name in ("name", "model_file"):
            if not isinstance(getattr(self, field_name), str):
                raise ValueError(f"{field_name} must be a string")
        if isinstance(self.camera_fps, bool) or not isinstance(self.camera_fps, int):
            raise ValueError(f"camera_fps must be an integer, got {self.camera_fps!r}")
        for field_name in (
            "ticks_per_revolution",
            "servo_pulse_floor_us",
            "servo_pulse_ceiling_us",
            "duty_ceiling_percent",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field_name} must be a number, got {value!r}")
        positives = {
            "ticks_per_revolution": self.ticks_per_revolution,
            "camera_fps": self.camera_fps,
            "servo_pulse_floor_us": self.servo_pulse_floor_us,
            "duty_ceiling_percent": self.duty_ceiling_percent,
        }
        for field_name, value in positives.items():
            if value <= 0:
                raise ValueError(f"{field_name} must be positive, got {value}")
        if self.servo_pulse_ceiling_us <= self.servo_pulse_floor_us:
            raise ValueError(
                "servo pulse band is empty: floor "
                f"{self.servo_pulse_floor_us} >= ceiling "
                f"{self.servo_pulse_ceiling_us}"
            )
        field_names = {entry.name for entry in fields(self)}
        unknown = set(self.provenance) - field_names
        if unknown:
            raise ValueError(f"provenance names unknown fields: {sorted(unknown)}")


def load_profile(bundle_dir: Path) -> RobotProfile:
    """Read and validate a bundle's `profile.json`.

    Unknown keys are an error, not a warning — a misspelled field would
    otherwise fall back to a default and lie quietly.
    """
    path = Path(bundle_dir) / PROFILE_FILE
    raw = json.loads(path.read_text(encoding="utf-8"))
    known = {entry.name for entry in fields(RobotProfile)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"{path} has unknown fields: {sorted(unknown)} — "
            f"known fields are {sorted(known)}"
        )
    return RobotProfile(**raw)
