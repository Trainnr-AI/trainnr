"""The deploy manifest (`deploy.json`, schema `trainnr-deploy/1`) as the
runtime reads it. Written by `rq_mjlab.walk_export` from the BUILT
training environment; validated here so a manifest a hand edited, or
one from an older exporter, is refused by name before a robot moves."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA = "trainnr-deploy/1"
MANIFEST_FILE = "deploy.json"
GATE_FILE = "gate.json"
GATE_SCHEMA = "trainnr-gate/1"
# The observation sources the plain-MuJoCo runtime can compute. A manifest
# naming another is refused: the runtime never guesses a term.
KNOWN_SOURCES = (
    "sensor robot/imu_lin_vel",
    "sensor robot/imu_ang_vel",
    "projected_gravity",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
    "command twist",
    "gait_phase",
)


@dataclass(frozen=True)
class Manifest:
    raw: dict[str, Any]
    root: Path

    @property
    def observations(self) -> list[dict[str, Any]]:
        return list(self.raw["observations"])

    @property
    def joints(self) -> dict[str, Any]:
        return dict(self.raw["joints"])

    @property
    def control(self) -> dict[str, Any]:
        return dict(self.raw["control"])

    @property
    def policy_path(self) -> Path:
        return self.root / self.raw["onnx"]["file"]

    @property
    def scene_path(self) -> Path:
        return self.root / self.raw["scene"]["file"]


def load_manifest(deployment_dir: Path) -> Manifest:
    """Read and validate a deployment's manifest; refuses by name."""
    root = Path(deployment_dir)
    path = root / MANIFEST_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{root}: no {MANIFEST_FILE}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(
            f"{path}: schema {raw.get('schema')!r}, this runtime speaks "
            f"{MANIFEST_SCHEMA!r}"
        )
    for key in (
        "robot",
        "run",
        "control",
        "joints",
        "action",
        "observations",
        "onnx",
        "scene",
    ):
        if key not in raw:
            raise ValueError(f"{path}: missing {key!r}")
    unknown = [
        t["name"] for t in raw["observations"] if t.get("source") not in KNOWN_SOURCES
    ]
    if unknown:
        raise ValueError(
            f"{path}: observation terms this runtime cannot compute: {unknown}; "
            f"known sources are {', '.join(KNOWN_SOURCES)}"
        )
    for f in (raw["onnx"]["file"], raw["scene"]["file"]):
        if not (root / f).is_file():
            raise FileNotFoundError(f"{root}: missing {f}")
    widths = sum(int(t["width"]) for t in raw["observations"])
    if widths != int(raw["onnx"]["input_width"]):
        raise ValueError(
            f"{path}: observation widths sum to {widths}, the policy takes "
            f"{raw['onnx']['input_width']}"
        )
    return Manifest(raw=raw, root=root)
