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
GATE_FILE = "gate.json"  # the plain-MuJoCo gate's record
DDS_GATE_FILE = "gate-dds.json"  # the reference's simulator and controller over DDS
GATE_SCHEMA = "trainnr-gate/1"
# Every gate record a deployment can hold, by the runtime that wrote it,
# and the instrument each stands for, in the field's words.
GATE_RECORDS = {"mujoco": GATE_FILE, "dds": DDS_GATE_FILE}
GATE_INSTRUMENTS = {
    "mujoco": "plain MuJoCo",
    "dds": "Unitree's simulator and controller over DDS",
}


def read_gates(folder: Path) -> dict[str, dict[str, Any]]:
    """The gate records present beside a manifest, by runtime, in the
    order of `GATE_RECORDS`."""
    out: dict[str, dict[str, Any]] = {}
    for runtime, name in GATE_RECORDS.items():
        path = Path(folder) / name
        if path.is_file():
            out[runtime] = json.loads(path.read_text())
    return out


def gate_word(record: dict[str, Any]) -> str:
    """A gate record's verdict in one word: passed, failed, or reported
    (no certificate was cited, so nothing was judged)."""
    passed = (record.get("verdict") or {}).get("passed")
    if passed is True:
        return "passed"
    return "failed" if passed is False else "reported, not judged"


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
