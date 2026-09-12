"""The deploy manifest (`deploy.json`, schema `trainnr-deploy/1`) as the
runtime reads it. Written by `rq_mjlab.walk_export` from the BUILT
training environment; validated here so a manifest a hand edited, or
one from an older exporter, is refused by name before a robot moves.

The manifest's sections are typed (`Control`, `Joints`, `Action`,
`Observation`, `Commands`, `Termination`, `UnitreeFacts`); the raw
mapping stays on `Manifest.raw` for the provenance strings and the
index. The key names are declared once here and the exporter writes
them from here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rq_pipeline.deploy.runtimes import RUNTIMES

MANIFEST_SCHEMA = "trainnr-deploy/1"
MANIFEST_FILE = "deploy.json"
GATE_SCHEMA = "trainnr-gate/1"
# The gate records a deployment can hold, by the runtime that wrote them,
# and the instrument each stands for — one table, `deploy.runtimes`.
GATE_RECORDS = {name: spec.record_file for name, spec in RUNTIMES.items()}
GATE_INSTRUMENTS = {name: spec.instrument for name, spec in RUNTIMES.items()}
GATE_FILE = GATE_RECORDS["mujoco"]
DDS_GATE_FILE = GATE_RECORDS["dds"]


class Key:
    """The manifest's top-level keys, spelled once."""

    SCHEMA = "schema"
    STAMP_OF = "stamp_of"
    POLICY = "policy"
    CHECKPOINT = "checkpoint"
    RUN = "run"
    ROBOT = "robot"
    ACTUATOR = "actuator"
    TASK = "task"
    DR_BASIS = "dr_basis"
    SEED = "seed"
    CERTIFICATE = "certificate"
    CONTROL = "control"
    JOINTS = "joints"
    ACTION = "action"
    OBSERVATIONS = "observations"
    ONNX = "onnx"
    SCENE = "scene"
    COMMANDS = "commands"
    COMMAND_BASIS = "command_basis"
    TERMINATION = "termination"
    UNITREE = "unitree"


# Every key a manifest may carry; one it must.
KNOWN_KEYS = frozenset(
    v for k, v in vars(Key).items() if k.isupper() and isinstance(v, str)
)
REQUIRED_KEYS = (
    Key.ROBOT,
    Key.RUN,
    Key.CONTROL,
    Key.JOINTS,
    Key.ACTION,
    Key.OBSERVATIONS,
    Key.ONNX,
    Key.SCENE,
    Key.COMMANDS,
)
# An observation term's keys; the runtime never guesses at an extra one.
TERM_KEYS = frozenset(
    {"name", "width", "source", "params", "scale", "clip", "history_length"}
)
TWIST_AXES = ("lin_vel_x", "lin_vel_y", "ang_vel_z")
# The same axes as a door and the Studio name them (forward, left, turn).
TWIST_SHORT = ("vx", "vy", "wz")

# The observation sources the plain-MuJoCo runtime can compute. A manifest
# naming another is refused: the runtime never guesses a term.
SOURCE_IMU_LIN_VEL = "sensor robot/imu_lin_vel"
SOURCE_IMU_ANG_VEL = "sensor robot/imu_ang_vel"
SOURCE_PROJECTED_GRAVITY = "projected_gravity"
SOURCE_JOINT_POS_REL = "joint_pos_rel"
SOURCE_JOINT_VEL_REL = "joint_vel_rel"
SOURCE_LAST_ACTION = "last_action"
SOURCE_COMMAND_TWIST = "command twist"
SOURCE_GAIT_PHASE = "gait_phase"
KNOWN_SOURCES = (
    SOURCE_IMU_LIN_VEL,
    SOURCE_IMU_ANG_VEL,
    SOURCE_PROJECTED_GRAVITY,
    SOURCE_JOINT_POS_REL,
    SOURCE_JOINT_VEL_REL,
    SOURCE_LAST_ACTION,
    SOURCE_COMMAND_TWIST,
    SOURCE_GAIT_PHASE,
)


@dataclass(frozen=True)
class Control:
    """The rates: physics step, decimation, the control rate they make,
    the episode length. `step_dt` and `episode_ticks` are THE period and
    THE length every reader uses; `control_hz` is the rounded rate a
    record prints."""

    physics_timestep_s: float
    decimation: int
    control_hz: float
    episode_length_s: float

    @property
    def step_dt(self) -> float:
        return self.physics_timestep_s * self.decimation

    @property
    def episode_ticks(self) -> int:
        return round(self.episode_length_s / self.step_dt)

    @classmethod
    def of(cls, raw: dict[str, Any]) -> Control:
        return cls(
            physics_timestep_s=float(raw["physics_timestep_s"]),
            decimation=int(raw["decimation"]),
            control_hz=float(raw["control_hz"]),
            episode_length_s=float(raw["episode_length_s"]),
        )


@dataclass(frozen=True)
class Joints:
    """The policy's joint order, the simulator's actuator order, the map
    between them, the gains and the home pose in policy order, and the
    vendor SDK's order when the robot has one."""

    policy_order: tuple[str, ...]
    action_to_ctrl: tuple[int, ...]
    default_pos: tuple[float, ...]
    ctrl_order: tuple[str, ...] = ()
    stiffness: tuple[float, ...] = ()
    damping: tuple[float, ...] = ()
    effort_limit: tuple[float, ...] = ()
    sdk_order_map: tuple[int, ...] | None = None
    sdk_order_source: str | None = None

    @classmethod
    def of(cls, raw: dict[str, Any]) -> Joints:
        sdk = raw.get("sdk_order_map")
        return cls(
            policy_order=tuple(raw["policy_order"]),
            action_to_ctrl=tuple(int(i) for i in raw["action_to_ctrl"]),
            default_pos=tuple(float(v) for v in raw["default_pos"]),
            ctrl_order=tuple(raw.get("ctrl_order", ())),
            stiffness=tuple(float(v) for v in raw.get("stiffness", ())),
            damping=tuple(float(v) for v in raw.get("damping", ())),
            effort_limit=tuple(float(v) for v in raw.get("effort_limit", ())),
            sdk_order_map=tuple(int(i) for i in sdk) if sdk is not None else None,
            sdk_order_source=raw.get("sdk_order_source"),
        )


@dataclass(frozen=True)
class Action:
    """joint position target = offset + scale * action, clipped first
    when `clip` is set (a symmetric bound, rsl_rl's `clip_actions`)."""

    scale: tuple[float, ...]
    offset: tuple[float, ...] = ()
    clip: tuple[float, float] | None = None
    kind: str = ""

    @classmethod
    def of(cls, raw: dict[str, Any]) -> Action:
        clip = raw.get("clip")
        return cls(
            scale=tuple(float(v) for v in raw["scale"]),
            offset=tuple(float(v) for v in raw.get("offset", ())),
            clip=(float(clip[0]), float(clip[1])) if clip is not None else None,
            kind=str(raw.get("kind", "")),
        )


@dataclass(frozen=True)
class Observation:
    """One term of the observation vector, in concat order."""

    name: str
    width: int
    source: str
    params: dict[str, Any]
    scale: float | list[float]
    clip: tuple[float, float] | None
    history_length: int

    @classmethod
    def of(cls, raw: dict[str, Any]) -> Observation:
        clip = raw.get("clip")
        return cls(
            name=str(raw["name"]),
            width=int(raw["width"]),
            source=str(raw.get("source", "")),
            params=dict(raw.get("params") or {}),
            scale=raw.get("scale", 1.0),
            clip=(float(clip[0]), float(clip[1])) if clip is not None else None,
            history_length=int(raw.get("history_length", 1)),
        )


@dataclass(frozen=True)
class Commands:
    """The twist ranges the policy trained under: what a gate draws from."""

    lin_vel_x: tuple[float, float]
    lin_vel_y: tuple[float, float]
    ang_vel_z: tuple[float, float]
    heading: tuple[float, float] | None = None

    @property
    def twist(self) -> dict[str, tuple[float, float] | None]:
        return {
            "lin_vel_x": self.lin_vel_x,
            "lin_vel_y": self.lin_vel_y,
            "ang_vel_z": self.ang_vel_z,
            "heading": self.heading,
        }

    @classmethod
    def of(cls, raw: dict[str, Any], *, where: str = "commands") -> Commands:
        twist = raw.get("twist") or {}
        missing = [k for k in TWIST_AXES if twist.get(k) is None]
        if missing:
            raise ValueError(f"{where}: twist ranges missing {missing}")
        heading = twist.get("heading")
        return cls(
            lin_vel_x=(float(twist["lin_vel_x"][0]), float(twist["lin_vel_x"][1])),
            lin_vel_y=(float(twist["lin_vel_y"][0]), float(twist["lin_vel_y"][1])),
            ang_vel_z=(float(twist["ang_vel_z"][0]), float(twist["ang_vel_z"][1])),
            heading=(float(heading[0]), float(heading[1])) if heading else None,
        )


@dataclass(frozen=True)
class Termination:
    fell_over_deg: float | None = None

    @classmethod
    def of(cls, raw: dict[str, Any] | None) -> Termination:
        value = (raw or {}).get("fell_over_deg")
        return cls(fell_over_deg=float(value) if value is not None else None)


@dataclass(frozen=True)
class UnitreeFacts:
    """What Unitree's own simulator and controller need to run this robot
    (their `unitree_rl_mjlab` checkout): the robot's name in their
    tree, their controller binary, the scene their simulator loads
    (relative to their checkout), and where those facts were read."""

    robot: str
    controller: str
    scene: str
    source: str

    @classmethod
    def of(cls, raw: dict[str, Any]) -> UnitreeFacts:
        return cls(
            robot=str(raw["robot"]),
            controller=str(raw["controller"]),
            scene=str(raw["scene"]),
            source=str(raw.get("source", "unrecorded")),
        )


@dataclass(frozen=True)
class Manifest:
    raw: dict[str, Any]
    root: Path

    @property
    def observations(self) -> list[Observation]:
        return [Observation.of(t) for t in self.raw[Key.OBSERVATIONS]]

    @property
    def joints(self) -> Joints:
        return Joints.of(self.raw[Key.JOINTS])

    @property
    def control(self) -> Control:
        return Control.of(self.raw[Key.CONTROL])

    @property
    def action(self) -> Action:
        return Action.of(self.raw[Key.ACTION])

    @property
    def commands(self) -> Commands:
        return Commands.of(self.raw[Key.COMMANDS], where=f"{self.root}: commands")

    @property
    def termination(self) -> Termination:
        return Termination.of(self.raw.get(Key.TERMINATION))

    @property
    def unitree(self) -> UnitreeFacts | None:
        raw = self.raw.get(Key.UNITREE)
        return UnitreeFacts.of(raw) if raw else None

    @property
    def robot_name(self) -> str:
        """The cited robot bundle's name, without its version."""
        return str(self.raw.get(Key.ROBOT, "")).split("@", 1)[0]

    @property
    def policy_path(self) -> Path:
        return self.root / self.raw[Key.ONNX]["file"]

    @property
    def scene_path(self) -> Path:
        return self.root / self.raw[Key.SCENE]["file"]


def load_manifest(deployment_dir: Path) -> Manifest:
    """Read and validate a deployment's manifest; refuses by name."""
    root = Path(deployment_dir)
    path = root / MANIFEST_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{root}: no {MANIFEST_FILE}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get(Key.SCHEMA) != MANIFEST_SCHEMA:
        raise ValueError(
            f"{path}: schema {raw.get(Key.SCHEMA)!r}, this runtime speaks "
            f"{MANIFEST_SCHEMA!r}"
        )
    stray = sorted(set(raw) - KNOWN_KEYS)
    if stray:
        raise ValueError(f"{path}: keys this runtime does not know: {stray}")
    for key in REQUIRED_KEYS:
        if key not in raw:
            raise ValueError(f"{path}: missing {key!r}")
    for term in raw[Key.OBSERVATIONS]:
        extra = sorted(set(term) - TERM_KEYS)
        if extra:
            raise ValueError(
                f"{path}: observation {term.get('name')!r} carries keys this "
                f"runtime does not know: {extra}"
            )
    unknown = [
        t["name"] for t in raw[Key.OBSERVATIONS] if t.get("source") not in KNOWN_SOURCES
    ]
    if unknown:
        raise ValueError(
            f"{path}: observation terms this runtime cannot compute: {unknown}; "
            f"known sources are {', '.join(KNOWN_SOURCES)}"
        )
    for f in (raw[Key.ONNX]["file"], raw[Key.SCENE]["file"]):
        if not (root / f).is_file():
            raise FileNotFoundError(f"{root}: missing {f}")
    widths = sum(int(t["width"]) for t in raw[Key.OBSERVATIONS])
    if widths != int(raw[Key.ONNX]["input_width"]):
        raise ValueError(
            f"{path}: observation widths sum to {widths}, the policy takes "
            f"{raw[Key.ONNX]['input_width']}"
        )
    manifest = Manifest(raw=raw, root=root)
    _ = manifest.commands  # the twist ranges are required, by name
    return manifest


def read_gates(folder: Path) -> dict[str, dict[str, Any]]:
    """The gate records present beside a manifest, by runtime, in the
    order of `GATE_RECORDS`; a record of another schema is refused."""
    out: dict[str, dict[str, Any]] = {}
    for runtime, name in GATE_RECORDS.items():
        path = Path(folder) / name
        if not path.is_file():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("schema") != GATE_SCHEMA:
            raise ValueError(
                f"{path}: schema {record.get('schema')!r}, this reader speaks "
                f"{GATE_SCHEMA!r}"
            )
        out[runtime] = record
    return out


def gate_word(record: dict[str, Any]) -> str:
    """A gate record's verdict in one word: passed, failed, or reported
    (no evaluation was cited, so nothing was judged)."""
    passed = (record.get("verdict") or {}).get("passed")
    if passed is True:
        return "passed"
    return "failed" if passed is False else "reported, not judged"
