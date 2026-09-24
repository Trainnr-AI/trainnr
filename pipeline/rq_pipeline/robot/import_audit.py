"""What the importer changed: a compiled bundle against the description
it came from, said per body, per joint and per element class.

Every onboarding door converts something — MuJoCo's URDF loader makes
the root link the world body and, before 3.13.0, dropped every mimic
without a word; Newton's bridge writes hulls for meshes and prim paths
for names; an MJCF include can resolve differently in a copy. Nobody in
the field checks a converted asset against its source (docs/e2e-research/78
§1 item 2: IsaacSim #841, an NVIDIA-confirmed silent loss across three
backends; mujoco #3559, an inertial frame lost for years). This module
does: a SOURCE READER per format (a registry by suffix, the way the
onboarding door dispatches) reads the description as authored into a
`Snapshot`; the bundle's compiled model reads into the same shape; the
two are compared with declared tolerances; every difference is a
`Change` that the reader either EXPLAINS (a documented conversion, or a
loader loss at this mujoco build) or does not. The door refuses an
unexplained change by name; an explained one goes on the record and
the robot card.

The audit is data: `Audit.to_record()` is what `bundle.json` carries
under `AUDIT_KEY`, and `tools/audit-bundle.py` prints it for any bundle
whose source is at hand.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rq_pipeline.plugins import load_group

AUDIT_SCHEMA = "trainnr-import-audit/1"
ENTRY_POINT_GROUP = "rq_pipeline.source_readers"
BUILTIN_MODULES = ("rq_pipeline.robot.urdf_import", "rq_pipeline.robot.usd_import")
WORLD = "world"

# The kinds of change, spelled once; the record and the drawer use these words.
MASS = "mass"
COM = "centre of mass"
INERTIA = "inertia"
JOINT_TYPE = "joint type"
JOINT_AXIS = "joint axis"
JOINT_RANGE = "joint range"
JOINT_LIMITED = "joint limited"
ARMATURE = "armature"
DAMPING = "damping"
FRICTIONLOSS = "friction loss"
STIFFNESS = "stiffness"
FORCE_RANGE = "force range"
BODY_MISSING = "body missing"
BODY_ADDED = "body added"
JOINT_MISSING = "joint missing"
JOINT_ADDED = "joint added"
JOINT_ORDER = "joint order"
COUNT = "count"
UNIT = "unit"
ANY = "*"  # an explanation that covers every element of its kind

HINGE, SLIDE, BALL, FREE = "hinge", "slide", "ball", "free"
# The unit words a snapshot declares, spelled once: what an MJCF bundle
# states, and what a source states unless it says otherwise.
RADIAN, DEGREE, METRE, KILOGRAM = "radian", "degree", "meter", "kilogram"
BUNDLE_UNITS = {"angle": RADIAN, "length": METRE, "mass": KILOGRAM}
# MuJoCo's joint types by the XML's words, and the equality a mimic
# becomes; read off MuJoCo's own enums (`_joint_kinds`, `_eq_joint`).
JOINT_KIND_NAMES = {
    "mjJNT_FREE": FREE,
    "mjJNT_BALL": BALL,
    "mjJNT_SLIDE": SLIDE,
    "mjJNT_HINGE": HINGE,
}


def _joint_kinds() -> dict[int, str]:
    import mujoco  # noqa: PLC0415 - sim extra

    return {int(getattr(mujoco.mjtJoint, k)): v for k, v in JOINT_KIND_NAMES.items()}


def _eq_joint() -> int:
    import mujoco  # noqa: PLC0415 - sim extra

    return int(mujoco.mjtEq.mjEQ_JOINT)


# The element classes a source may count; a bundle count that differs
# is a COUNT change on that class.
MIMICS = "mimics"
EQUALITIES = "equalities"
TENDONS = "tendons"
SITES = "sites"
SENSORS = "sensors"
KEYFRAMES = "keyframes"
ACTUATORS = "actuators"
MESHES = "meshes"
CLASSES = (MIMICS, EQUALITIES, TENDONS, SITES, SENSORS, KEYFRAMES, ACTUATORS, MESHES)


@dataclass(frozen=True)
class Tolerances:
    """How different two numbers may be before the audit calls it a
    change: masses and inertias relative, positions and angles absolute."""

    mass_rel: float = 1e-4
    length_abs: float = 1e-5
    inertia_rel: float = 1e-3
    inertia_abs: float = 1e-9
    angle_abs: float = 1e-5
    param_rel: float = 1e-6
    param_abs: float = 1e-9


TOLERANCES = Tolerances()


@dataclass(frozen=True)
class BodyFacts:
    """A body as a description states it: mass, centre of mass in the
    body frame, and the full inertia tensor in the body frame (row-major,
    nine numbers) — the frame is compared, not only the eigenvalues, so
    a lost inertial rotation is a change."""

    mass: float
    com: tuple[float, ...] | None = None  # three numbers
    inertia: tuple[float, ...] | None = None  # nine numbers


@dataclass(frozen=True)
class JointFacts:
    kind: str
    axis: tuple[float, ...] | None = None  # three numbers, in the body frame
    limited: bool = False
    range: tuple[float, ...] = (0.0, 0.0)  # lower, upper
    armature: float = 0.0
    damping: float = 0.0
    frictionloss: float = 0.0
    stiffness: float = 0.0
    force_range: tuple[float, ...] | None = None  # lower, upper


@dataclass(frozen=True)
class Snapshot:
    """A description in one shape, whatever wrote it. `joints` keeps the
    document's order (dicts do), `counts` only the classes the source
    can count, `units` what the source declared."""

    bodies: dict[str, BodyFacts]
    joints: dict[str, JointFacts]
    counts: dict[str, int] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Explanation:
    """A change the reader expected: the kind, the element it covers
    (`ANY` for the whole kind) and why the conversion is right."""

    kind: str
    element: str
    reason: str

    def covers(self, kind: str, element: str) -> bool:
        return self.kind == kind and self.element in (ANY, element)


@dataclass(frozen=True)
class SourceRead:
    """What a source reader hands back: the snapshot, the name map from
    the source's names to the bundle's (identity when absent), the
    changes it expects, and what the source declares that no MJCF slot
    holds (an advisory, not a change)."""

    snapshot: Snapshot
    names: Mapping[str, str] = field(default_factory=dict)
    explanations: tuple[Explanation, ...] = ()
    advisories: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Change:
    kind: str
    element: str
    source: Any
    bundle: Any
    explanation: str | None = None

    @property
    def explained(self) -> bool:
        return self.explanation is not None

    def line(self) -> str:
        why = f" — {self.explanation}" if self.explanation else " — UNEXPLAINED"
        return f"{self.kind} of {self.element}: {self.source} → {self.bundle}{why}"


@dataclass(frozen=True)
class Audit:
    """The audit as the record carries it."""

    format: str
    source: str
    mujoco: str
    changes: tuple[Change, ...]
    advisories: tuple[str, ...] = ()
    reader: Mapping[str, Any] = field(default_factory=dict)
    schema: str = AUDIT_SCHEMA

    @property
    def unexplained(self) -> tuple[Change, ...]:
        return tuple(c for c in self.changes if not c.explained)

    @property
    def explained(self) -> tuple[Change, ...]:
        return tuple(c for c in self.changes if c.explained)

    def summary(self) -> str:
        """One line for a card: nothing changed, n explained, n UNEXPLAINED."""
        if not self.changes:
            return "nothing"
        parts = []
        if self.explained:
            parts.append(f"{len(self.explained)} explained")
        if self.unexplained:
            parts.append(f"{len(self.unexplained)} UNEXPLAINED")
        return ", ".join(parts)

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "format": self.format,
            "source": self.source,
            "mujoco": self.mujoco,
            "reader": dict(self.reader),
            "changes": [asdict(c) for c in self.changes],
            "advisories": list(self.advisories),
            "summary": self.summary(),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> Audit:
        return cls(
            format=str(record.get("format", "")),
            source=str(record.get("source", "")),
            mujoco=str(record.get("mujoco", "")),
            changes=tuple(Change(**c) for c in record.get("changes", ())),
            advisories=tuple(record.get("advisories", ())),
            reader=dict(record.get("reader", {})),
            schema=str(record.get("schema", AUDIT_SCHEMA)),
        )


class ImportAuditError(ValueError):
    """The importer changed something no reader explains."""


# -- the readers -------------------------------------------------------

Reader = Callable[[Path, Mapping[str, Any]], SourceRead]


@dataclass(frozen=True)
class SourceReader:
    name: str
    suffixes: tuple[str, ...]
    read: Reader


_READERS: dict[str, SourceReader] = {}
_LOADED = False


def source_reader(name: str, suffixes: tuple[str, ...]) -> Callable[[Reader], Reader]:
    """Register the reader of one format; refuses a name or suffix taken."""

    def register(function: Reader) -> Reader:
        if name in _READERS:
            raise ValueError(f"source reader {name!r} is already registered")
        taken = {s: n for n, r in _READERS.items() for s in r.suffixes}
        for suffix in suffixes:
            if suffix in taken:
                raise ValueError(f"suffix {suffix!r} is already {taken[suffix]}'s")
        _READERS[name] = SourceReader(name, suffixes, function)
        return function

    return register


def readers() -> dict[str, SourceReader]:
    global _LOADED  # noqa: PLW0603 - the one load flag
    if not _LOADED:
        _LOADED = True
        load_group(ENTRY_POINT_GROUP, BUILTIN_MODULES)
    return dict(_READERS)


def reader_for(path: Path) -> SourceReader:
    suffix = Path(path).suffix.lower()
    for reader in readers().values():
        if suffix in reader.suffixes:
            return reader
    known = ", ".join(f"{n} ({' '.join(r.suffixes)})" for n, r in readers().items())
    raise ValueError(f"no source reader reads {suffix!r}; known: {known}")


# -- snapshots ---------------------------------------------------------


def quat_to_matrix(quat: Sequence[float]) -> np.ndarray:
    """A unit quaternion (w, x, y, z) as a rotation matrix: MuJoCo's own
    `mju_quat2Mat`, the one rotation every module here uses."""
    import mujoco  # noqa: PLC0415 - sim extra

    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, np.asarray(quat, dtype=np.float64))
    return out.reshape(3, 3)


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF's fixed-axis roll-pitch-yaw as a rotation matrix (R = Rz·Ry·Rx)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def tensor_in_frame(
    principal: Sequence[float], rotation: np.ndarray
) -> tuple[float, ...]:
    """A diagonal inertia in its own frame, expressed in the parent frame."""
    full = rotation @ np.diag([float(v) for v in principal]) @ rotation.T
    return tuple(float(v) for v in full.reshape(-1))


def full_tensor(moments: Sequence[float]) -> np.ndarray:
    """(ixx, iyy, izz, ixy, ixz, iyz), URDF's order, as the symmetric matrix."""
    ixx, iyy, izz, ixy, ixz, iyz = (float(v) for v in moments)
    return np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]], dtype=float)


def snapshot_model(model: Any) -> Snapshot:
    """A compiled MjModel in the audit's shape: the world body left out,
    every joint in model order, the classes MuJoCo counts."""
    import mujoco  # noqa: PLC0415 - sim extra

    def name(kind: Any, i: int) -> str:
        return mujoco.mj_id2name(model, kind, i) or f"#{i}"

    bodies: dict[str, BodyFacts] = {}
    for b in range(1, model.nbody):
        bodies[name(mujoco.mjtObj.mjOBJ_BODY, b)] = BodyFacts(
            mass=float(model.body_mass[b]),
            com=tuple(float(v) for v in model.body_ipos[b]),
            inertia=tensor_in_frame(
                model.body_inertia[b], quat_to_matrix(model.body_iquat[b])
            ),
        )
    joints: dict[str, JointFacts] = {}
    for j in range(model.njnt):
        dof = int(model.jnt_dofadr[j])
        limited = bool(model.jnt_limited[j])
        joints[name(mujoco.mjtObj.mjOBJ_JOINT, j)] = JointFacts(
            kind=_joint_kinds()[int(model.jnt_type[j])],
            axis=tuple(float(v) for v in model.jnt_axis[j]),
            limited=limited,
            range=tuple(float(v) for v in model.jnt_range[j])
            if limited
            else (0.0, 0.0),
            armature=float(model.dof_armature[dof]),
            damping=float(model.dof_damping[dof]),
            frictionloss=float(model.dof_frictionloss[dof]),
            stiffness=float(model.jnt_stiffness[j]),
            force_range=(
                tuple(float(v) for v in model.jnt_actfrcrange[j])
                if model.jnt_actfrclimited[j]
                else None
            ),
        )
    counts = {
        MIMICS: int(sum(1 for t in model.eq_type if int(t) == _eq_joint())),
        EQUALITIES: int(model.neq),
        TENDONS: int(model.ntendon),
        SITES: int(model.nsite),
        SENSORS: int(model.nsensor),
        KEYFRAMES: int(model.nkey),
        ACTUATORS: int(model.nu),
        MESHES: int(model.nmesh),
    }
    return Snapshot(bodies, joints, counts, dict(BUNDLE_UNITS))


def snapshot_mjcf(path: Path) -> Snapshot:
    """An MJCF compiled where it stands — includes and assets resolved
    from its own directory."""
    import mujoco  # noqa: PLC0415 - sim extra

    return snapshot_model(mujoco.MjModel.from_xml_path(str(path)))


# -- the comparison ----------------------------------------------------


def _close_rel(a: float, b: float, rel: float, abs_: float) -> bool:
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b)))


def _close_seq(a: Sequence[float], b: Sequence[float], tol: float) -> bool:
    return len(a) == len(b) and all(
        abs(x - y) <= tol for x, y in zip(a, b, strict=True)
    )


def _close_tensor(a: Sequence[float], b: Sequence[float], tol: Tolerances) -> bool:
    scale = max(abs(v) for v in (*a, *b, 0.0))
    return len(a) == len(b) and all(
        abs(x - y) <= max(tol.inertia_abs, tol.inertia_rel * scale)
        for x, y in zip(a, b, strict=True)
    )


def _body_changes(
    name: str, src: BodyFacts, dst: BodyFacts, tol: Tolerances
) -> list[Change]:
    out = []
    if not _close_rel(src.mass, dst.mass, tol.mass_rel, tol.param_abs):
        out.append(Change(MASS, name, src.mass, dst.mass))
    if (
        src.com is not None
        and dst.com is not None
        and not _close_seq(src.com, dst.com, tol.length_abs)
    ):
        out.append(Change(COM, name, src.com, dst.com))
    if (
        src.inertia is not None
        and dst.inertia is not None
        and not _close_tensor(src.inertia, dst.inertia, tol)
    ):
        out.append(Change(INERTIA, name, _round(src.inertia), _round(dst.inertia)))
    return out


def _joint_changes(
    name: str, src: JointFacts, dst: JointFacts, tol: Tolerances
) -> list[Change]:
    out = []
    if src.kind != dst.kind:
        out.append(Change(JOINT_TYPE, name, src.kind, dst.kind))
    if (
        src.axis is not None
        and dst.axis is not None
        and not _close_seq(src.axis, dst.axis, tol.angle_abs)
    ):
        out.append(Change(JOINT_AXIS, name, src.axis, dst.axis))
    if src.limited != dst.limited:
        out.append(Change(JOINT_LIMITED, name, src.limited, dst.limited))
    elif src.limited and not _close_seq(src.range, dst.range, tol.angle_abs):
        out.append(Change(JOINT_RANGE, name, src.range, dst.range))
    for kind, a, b in (
        (ARMATURE, src.armature, dst.armature),
        (DAMPING, src.damping, dst.damping),
        (FRICTIONLOSS, src.frictionloss, dst.frictionloss),
        (STIFFNESS, src.stiffness, dst.stiffness),
    ):
        if not _close_rel(a, b, tol.param_rel, tol.param_abs):
            out.append(Change(kind, name, a, b))
    if src.force_range is not None and (
        dst.force_range is None
        or not _close_seq(src.force_range, dst.force_range, tol.param_abs)
    ):
        out.append(Change(FORCE_RANGE, name, src.force_range, dst.force_range))
    return out


def _round(values: Iterable[float], digits: int = 6) -> tuple[float, ...]:
    return tuple(round(float(v), digits) for v in values)


def compare(
    source: Snapshot,
    bundle: Snapshot,
    names: Mapping[str, str] | None = None,
    tolerances: Tolerances = TOLERANCES,
) -> list[Change]:
    """Every difference between what the source states and what the
    bundle compiled to. `names` maps a source name to the bundle's (the
    USD writer's leaf renaming); unmapped names compare as themselves."""
    names = dict(names or {})

    def to_bundle(n: str) -> str:
        return names.get(n, n)

    changes: list[Change] = []
    mapped_bodies = {to_bundle(n): n for n in source.bodies}
    for src_name, facts in source.bodies.items():
        dst = bundle.bodies.get(to_bundle(src_name))
        if dst is None:
            changes.append(Change(BODY_MISSING, src_name, facts.mass, None))
            continue
        changes += _body_changes(src_name, facts, dst, tolerances)
    changes += [
        Change(BODY_ADDED, n, None, facts.mass)
        for n, facts in bundle.bodies.items()
        if n not in mapped_bodies
    ]
    mapped_joints = {to_bundle(n): n for n in source.joints}
    for src_name, jfacts in source.joints.items():
        djoint = bundle.joints.get(to_bundle(src_name))
        if djoint is None:
            changes.append(Change(JOINT_MISSING, src_name, jfacts.kind, None))
            continue
        changes += _joint_changes(src_name, jfacts, djoint, tolerances)
    changes += [
        Change(JOINT_ADDED, n, None, jf.kind)
        for n, jf in bundle.joints.items()
        if n not in mapped_joints
    ]
    src_order = [to_bundle(n) for n in source.joints if to_bundle(n) in bundle.joints]
    dst_order = [n for n in bundle.joints if n in mapped_joints]
    if src_order != dst_order:
        changes.append(Change(JOINT_ORDER, ANY, src_order, dst_order))
    for cls, n in source.counts.items():
        m = bundle.counts.get(cls)
        if m is not None and m != n:
            changes.append(Change(COUNT, cls, n, m))
    for unit, word in source.units.items():
        theirs = bundle.units.get(unit)
        if theirs is not None and theirs != word:
            changes.append(Change(UNIT, unit, word, theirs))
    return changes


def explain(
    changes: Iterable[Change], explanations: Iterable[Explanation]
) -> tuple[Change, ...]:
    """Every change with the reader's reason attached where one covers it."""
    explanations = tuple(explanations)
    out = []
    for change in changes:
        reason = next(
            (e.reason for e in explanations if e.covers(change.kind, change.element)),
            None,
        )
        out.append(
            Change(change.kind, change.element, change.source, change.bundle, reason)
        )
    return tuple(out)


# -- the audit ---------------------------------------------------------


def audit_bundle(
    source_path: Path,
    model_file: Path,
    options: Mapping[str, Any] | None = None,
    tolerances: Tolerances = TOLERANCES,
) -> Audit:
    """The audit of one bundle: its source read as authored, its MJCF
    compiled from the file, the two compared, the reader's explanations
    applied. `options` are the door's (a USD's variants), so the reader
    sees the same description the importer did."""
    import mujoco  # noqa: PLC0415 - sim extra

    from rq_pipeline.bundles.bundle import source_locator  # noqa: PLC0415

    source_path = Path(source_path)
    reader = reader_for(source_path)
    read = reader.read(source_path, dict(options or {}))
    bundle = snapshot_mjcf(Path(model_file))
    changes = explain(
        compare(read.snapshot, bundle, read.names, tolerances), read.explanations
    )
    return Audit(
        format=reader.name,
        source=source_locator(source_path),
        mujoco=mujoco.__version__,
        changes=changes,
        advisories=tuple(read.advisories),
        reader=dict(read.provenance),
    )


def require_explained(audit: Audit) -> None:
    """Refuse by name when the importer changed something no reader explains."""
    if audit.unexplained:
        lines = "\n  ".join(c.line() for c in audit.unexplained)
        raise ImportAuditError(
            f"the {audit.format} importer changed {len(audit.unexplained)} thing(s) "
            f"no reader explains (mujoco {audit.mujoco}):\n  {lines}\n"
            "pass accept_changes=True to onboard anyway; the record will say so"
        )


def version_tuple(version: str) -> tuple[int, ...]:
    """'3.13.0' → (3, 13, 0); a suffix after a dash is dropped."""
    head = version.split("-", 1)[0].split("+", 1)[0]
    return tuple(int(p) for p in head.split(".") if p.isdigit())


# -- the MJCF reader (the door that copies) -------------------------------


@source_reader("mjcf", (".xml",))
def read_mjcf(path: Path, options: Mapping[str, Any]) -> SourceRead:
    """An MJCF says what it compiles to where it stands; the bundle is a
    copy, so the audit expects nothing to change — an include or an
    asset that resolved differently in the copy is what it catches."""
    del options  # an MJCF takes none; the door refused any
    return SourceRead(snapshot_mjcf(Path(path)))
