"""A robot authored in USD (Isaac Sim, Omniverse) into a bundle.

Newton reads the stage (every UsdPhysics joint, mass and collider, the
`mjc:`, `newton:` and PhysX vendor schemas, mimics and loop closures)
and its MuJoCo solver bridges the model to an MjSpec; measured on the
Robotiq 2F-85 the bridge keeps every joint range and mass of the USD
layer (docs/e2e-research/77 §4). This module is what the bridge does
not do: it refuses a stage a missing or empty layer would refuse from
deep inside the library, selects the variants the caller names, renames
the prim paths Newton writes into leaf names, moves the inline convex
hulls into files and puts the USD's own visual meshes beside them, adds
the sensors the harness observes by contract and a `home` keyframe,
gives every position servo the range of its joint, and records where
all of it came from. The written MJCF is compiled from its file — the
file is the bundle's truth, never the spec in memory.

Optional extra: `usd` (Newton's importer) beside `gpu` (mujoco_warp,
which Newton's solver module imports even when it steps on the CPU);
a machine without them is refused with the install line.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import math
import platform
import tempfile
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rq_pipeline.bundles.bundle import source_locator, write_bundle_record
from rq_pipeline.bundles.hashing import stamp
from rq_pipeline.robot.asset_fetch import read_marker
from rq_pipeline.robot.import_audit import (
    ANY,
    BALL,
    COUNT,
    DEGREE,
    EQUALITIES,
    HINGE,
    JOINT_MISSING,
    JOINT_ORDER,
    KEYFRAMES,
    KILOGRAM,
    MESHES,
    METRE,
    MIMICS,
    SENSORS,
    SLIDE,
    UNIT,
    BodyFacts,
    Explanation,
    JointFacts,
    Snapshot,
    SourceRead,
    quat_to_matrix,
    source_reader,
    tensor_in_frame,
)
from rq_pipeline.robot.onboarding import USD_SOURCE, model_source
from rq_pipeline.scenes.record import UNRECORDED
from rq_pipeline.scenes.tooling import INSTALL_HINTS, install_hint

USD_SUFFIXES = (".usd", ".usda", ".usdc", ".usdz")
USD_EXTRA = "usd"
# The modules the reader needs, in the order they are checked; each one's
# per-OS install line is in the shared table (`scenes.tooling.INSTALL_HINTS`).
REQUIRED_MODULES = ("pxr", "newton", "newton_usd_schemas", "mujoco_warp")
DARWIN_LINE = INSTALL_HINTS["mujoco_warp"]["Darwin"]

NEWTON_VERSION_WARNING = "MuJoCo dependency version mismatch"

ROOT_FIXED = (
    "fixed"  # welded to the world: a gripper on a bench, or attached under an arm
)
ROOT_FREE = "free"  # a free joint at the root: a body that falls
ROOT_KINDS = (ROOT_FIXED, ROOT_FREE)

ASSETS_DIR = "assets"
HOME_KEY = "home"
HULL_WORD = "hull"
VISUAL_WORD = "visual"
DRIVE_WORD = "drive"
LICENSE_FILE = "LICENSE"
# Menagerie's group convention: 2 visual, 3 collision.
VISUAL_GROUP = 2
COLLISION_GROUP = 3
VISUAL_RGBA = (0.55, 0.55, 0.6, 1.0)
# The licence a text declares, by the phrase its first lines carry.
LICENSE_PHRASES: dict[str, str] = {
    "Attribution 4.0 International": "CC-BY-4.0",
    "Apache License": "Apache-2.0",
    "BSD 3-Clause": "BSD-3-Clause",
    "BSD 2-Clause": "BSD-2-Clause",
    "MIT License": "MIT",
}
LICENSE_CANDIDATES = (
    "PACKAGE-LICENSES/LICENSE",
    "LICENSE",
    "LICENSE.txt",
    "LICENSE.md",
)
# The option fields Newton's bridge sets from ITS defaults, not from the
# stage (a stage without a physics scene declares none): reset to
# MuJoCo's, so the bundle states only what the asset and the caller said.
NEWTON_OPTION_FIELDS = (
    "integrator",
    "enableflags",
    "disableflags",
    "ls_tolerance",
    "ccd_tolerance",
    "sleep_tolerance",
)
# The USD attribute that carries a joint's authored position (degrees for
# a revolute joint), when the asset states one.
JOINT_STATE_ATTRIBUTE = "state:angular:physics:position"


@dataclass(frozen=True)
class GripOptions:
    """What Menagerie's 2F-85 and Robotiq's own MuJoCo scripts declare at
    runtime and the USD does not carry: an elliptic cone with a high
    impedance ratio, for grip without slip. Declared into the bundle
    only when the caller asks."""

    impratio: float = 10.0
    cone: str = "elliptic"


GRIP_OPTIONS = GripOptions()


@dataclass(frozen=True)
class ImportSettings:
    """What the caller decides about an import; everything else comes
    from the stage."""

    variants: Mapping[str, str] = field(default_factory=dict)
    root: str = ROOT_FIXED
    mesh_maxhullvert: int = 64  # Newton's default, named so the record carries it
    grip_options: bool = False

    def __post_init__(self) -> None:
        if self.root not in ROOT_KINDS:
            raise ValueError(f"root is one of {ROOT_KINDS}, got {self.root!r}")


@dataclass(frozen=True)
class UsdRead:
    """What Newton read and wrote: the bridge's MJCF text and the prim
    paths behind each element."""

    xml: str
    bodies: Mapping[str, int]
    joints: Mapping[str, int]
    shapes: Mapping[str, int]
    shape_scale: Mapping[str, tuple[float, float, float]]
    census: Mapping[str, int]
    versions: Mapping[str, str]
    variants: Mapping[str, str]


class UsdLayerError(FileNotFoundError):
    """A layer the stage composes is missing or empty; named."""


def missing_line(system: str | None = None) -> str | None:
    """The refusal when this interpreter cannot read USD: the first
    missing module and its install line, else None."""
    system = system or platform.system()
    for module in REQUIRED_MODULES:
        if importlib.util.find_spec(module) is None:
            return f"USD import needs {module}: {install_hint(module, system)}"
    return None


def versions() -> dict[str, str]:
    """The reader's own versions, for the record."""
    out = {}
    for package in ("newton", "usd-core", "newton-usd-schemas", "warp-lang", "mujoco"):
        try:
            out[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            out[package] = UNRECORDED
    return out


# -- the stage ----------------------------------------------------------

SCHEMA_PACKAGE = "newton_usd_schemas"


SCHEMA_WITNESS = "NewtonMimicAPI"  # a Newton API type pxr knows only through the plugin
POISONED_LINE = (
    "pxr's schema registry was built in this process before Newton's USD "
    f"schemas were registered ({SCHEMA_WITNESS} is unknown to it): a stage "
    "opened through pxr directly, earlier, did that. pxr builds the registry "
    "once, so import in a fresh process (tools/import-usd.py, or the Studio's "
    "door) — an import here would read every mimic as nothing"
)


def register_schemas() -> None:
    """Newton's USD schema plugin (`mjc:`, `newton:` API schemas) into
    pxr's plugin registry BEFORE any stage is opened in this process,
    and a refusal when it is too late. pxr builds its schema registry
    once, on first use: a stage opened earlier — a plain look at the
    file, a refused variant — left the registry without
    `NewtonMimicAPI`, and every later import in that process read the
    asset's mimic as nothing, silently (2026-09-24, found by the test
    order). Every pxr entry point here calls this first; a process
    already poisoned is refused by name rather than read wrong."""
    if importlib.util.find_spec(SCHEMA_PACKAGE) is None:
        # Opening a stage without the plugin is the poisoning this guards
        # against: refused by name, never a silent return (review 2026-09-24:
        # the audit's reader opened a stage after that return).
        raise RuntimeError(
            f"USD reading needs {SCHEMA_PACKAGE}: {install_hint(SCHEMA_PACKAGE)}"
        )
    importlib.import_module(SCHEMA_PACKAGE)  # its import registers the plugin
    from pxr import Tf, Usd  # noqa: PLC0415

    known = Usd.SchemaRegistry().GetTypeFromSchemaTypeName(SCHEMA_WITNESS)
    if not known or known == Tf.Type.Unknown:
        raise RuntimeError(POISONED_LINE)


def check_layers(root: Path) -> list[Path]:
    """Every layer the root composes (sublayers, references, payloads,
    recursively), each present and non-empty; refuses the first that is
    not, by name, before the stage opens — an empty sublayer is a
    composition error that refuses everything (2026-09-24)."""
    register_schemas()
    from pxr import Sdf  # noqa: PLC0415

    seen: list[Path] = []
    pending = [Path(root)]
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        if not path.is_file():
            raise UsdLayerError(f"layer {path} is missing (composed by {root.name})")
        if path.stat().st_size == 0:
            raise UsdLayerError(
                f"layer {path} is empty (0 bytes; composed by {root.name})"
            )
        seen.append(path)
        if path.suffix == ".usdz":
            continue  # a package: checked as one file
        layer = Sdf.Layer.FindOrOpen(str(path))
        if layer is None:
            raise UsdLayerError(f"layer {path} does not open as USD")
        for dependency in layer.GetCompositionAssetDependencies():
            if dependency:  # an empty asset path is a reference within the layer
                pending.append(Path(layer.ComputeAbsolutePath(dependency)))
    return seen


def select_variants(stage: Any, variants: Mapping[str, str]) -> dict[str, str]:
    """Apply the caller's variant selections on the default prim, in the
    stage's SESSION layer: the asset's own layer stays as read, so two
    stages over one file in one process never see each other's choice
    (authored into the shared root layer, a PhysX selection outlived
    its stage and the next Newton import lost its mimic, 2026-09-24).
    Refuses an unknown set or choice naming what exists. Returns every
    set's selection after the choice (the record carries them all)."""
    from pxr import Usd  # noqa: PLC0415

    prim = stage.GetDefaultPrim()
    if not prim:
        raise ValueError("the stage has no default prim to select variants on")
    sets = prim.GetVariantSets()
    names = list(sets.GetNames())
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for name, choice in variants.items():
            if name not in names:
                raise ValueError(f"no variant set {name!r}; the stage has {names}")
            one = sets.GetVariantSet(name)
            choices = list(one.GetVariantNames())
            if choice not in choices:
                raise ValueError(f"variant set {name!r} has {choices}, not {choice!r}")
            one.SetVariantSelection(choice)
    return {name: sets.GetVariantSet(name).GetVariantSelection() for name in names}


def open_stage(path: Path, variants: Mapping[str, str]) -> tuple[Any, dict[str, str]]:
    """The composed stage with the variants applied, and the selections."""
    register_schemas()
    from pxr import Usd  # noqa: PLC0415

    check_layers(path)
    stage = Usd.Stage.Open(str(path))
    return stage, select_variants(stage, variants)


# -- Newton -------------------------------------------------------------


def read_usd(path: Path, settings: ImportSettings) -> UsdRead:
    """The stage through Newton's importer and its MuJoCo bridge: the
    MJCF text Newton writes, with the prim path behind every element."""
    line = missing_line()
    if line:
        raise ImportError(line)
    import newton  # noqa: PLC0415
    from newton.solvers import SolverMuJoCo  # noqa: PLC0415
    from newton.usd import (  # noqa: PLC0415
        SchemaResolverMjc,
        SchemaResolverNewton,
        SchemaResolverPhysx,
    )

    path = Path(path)
    stage, selected = open_stage(path, settings.variants)
    builder = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(builder)
    result = builder.add_usd(
        stage,
        schema_resolvers=[
            SchemaResolverMjc(),
            SchemaResolverPhysx(),
            SchemaResolverNewton(),
        ],
        collapse_fixed_joints=False,
        mesh_maxhullvert=settings.mesh_maxhullvert,
    )
    if builder.joint_count == 0:
        raise ValueError(
            f"{path.name} with variants {selected} has no joints "
            "(Newton's bridge needs one); pick a physics variant"
        )
    census = {
        "bodies": int(builder.body_count),
        "joints": int(builder.joint_count),
        "shapes": int(builder.shape_count),
        "dofs": int(builder.joint_dof_count),
    }
    model = builder.finalize(device="cpu")
    with tempfile.TemporaryDirectory() as tmp, warnings.catch_warnings():
        # Newton 1.6 declares mujoco 3.12 and warns at every bridge on our
        # locked 3.11.0; the bridge measured equal to the USD layer on
        # 3.11.0 (docs/77 §4) and the versions are in the record, so the
        # warning says nothing the bundle does not.
        warnings.filterwarnings("ignore", message=NEWTON_VERSION_WARNING)
        out = Path(tmp) / "newton.xml"
        SolverMuJoCo(model, save_to_mjcf=str(out), use_mujoco_cpu=True)
        xml = out.read_text(encoding="utf-8")
    return UsdRead(
        xml=xml,
        bodies=dict(result["path_body_map"]),
        joints=dict(result["path_joint_map"]),
        shapes=dict(result["path_shape_map"]),
        shape_scale={
            k: tuple(v) for k, v in result.get("path_shape_scale", {}).items()
        },
        census=census,
        versions=versions(),
        variants=selected,
    )


# -- the writer ---------------------------------------------------------


def leaf(prim_path: str) -> str:
    return prim_path.rstrip("/").rsplit("/", 1)[-1]


def unique(name: str, taken: set[str]) -> str:
    """`name`, or `name_2`, `name_3`… when a leaf repeats."""
    candidate, n = name, 1
    while candidate in taken:
        n += 1
        candidate = f"{name}_{n}"
    taken.add(candidate)
    return candidate


def _newton_name(prim_path: str) -> str:
    return prim_path.replace("/", "_")


def rename_elements(spec: Any, read: UsdRead) -> dict[str, str]:
    """Bodies and joints from Newton's `_path_with_underscores` to their
    leaf names, and every reference to them (excludes, equalities,
    actuator targets) re-pointed. Returns old → new."""
    renamed: dict[str, str] = {}
    taken: set[str] = set()
    by_newton = {_newton_name(p): p for p in read.bodies}
    for body in spec.bodies:
        if body.name in by_newton:
            renamed[body.name] = body.name = unique(leaf(by_newton[body.name]), taken)
    taken = set()
    joint_paths = {_newton_name(p): p for p in read.joints}
    for joint in spec.joints:
        if joint.name in joint_paths:
            renamed[joint.name] = joint.name = unique(
                leaf(joint_paths[joint.name]), taken
            )
    for exclude in spec.excludes:
        exclude.bodyname1 = renamed.get(exclude.bodyname1, exclude.bodyname1)
        exclude.bodyname2 = renamed.get(exclude.bodyname2, exclude.bodyname2)
    for equality in spec.equalities:
        equality.name1 = renamed.get(equality.name1, equality.name1)
        equality.name2 = renamed.get(equality.name2, equality.name2)
    for actuator in spec.actuators:
        actuator.target = renamed.get(actuator.target, actuator.target)
    return renamed


def geom_prim_paths(read: UsdRead) -> dict[str, str]:
    """Newton names a mesh geom `<shape prim path>_<shape index>`; the
    prim path behind each such name."""
    return {f"{path}_{index}": path for path, index in read.shapes.items()}


def _triangles(counts: Any, indices: Any) -> list[tuple[int, int, int]]:
    """Polygons fanned into triangles."""
    out: list[tuple[int, int, int]] = []
    at = 0
    for count in counts:
        polygon = indices[at : at + count]
        at += count
        out.extend(
            (polygon[0], polygon[i], polygon[i + 1]) for i in range(1, count - 1)
        )
    return out


def write_obj(path: Path, vertices: Any, faces: Any) -> None:
    """A Wavefront OBJ, triangles only, one-based indices."""
    lines = [f"v {x:.6g} {y:.6g} {z:.6g}" for x, y, z in vertices]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _visual_mesh(
    stage: Any, prim_path: str, scale: tuple[float, ...]
) -> tuple[Any, Any] | None:
    """The USD mesh at a shape's path (points scaled as Newton scaled
    the hull), or None when the shape is not a mesh prim."""
    from pxr import UsdGeom  # noqa: PLC0415

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsA(UsdGeom.Mesh):
        return None
    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get() or []
    counts = mesh.GetFaceVertexCountsAttr().Get() or []
    indices = mesh.GetFaceVertexIndicesAttr().Get() or []
    if not points or not counts:
        return None
    sx, sy, sz = (*scale, 1.0, 1.0, 1.0)[:3]
    vertices = [(p[0] * sx, p[1] * sy, p[2] * sz) for p in points]
    return vertices, _triangles(list(counts), list(indices))


def meshes_to_files(
    spec: Any, stage: Any, read: UsdRead, assets: Path
) -> dict[str, str]:
    """Newton's inline hulls become `<body>_hull.obj` (collision group),
    and the USD's own mesh at the same prim `<body>_visual.obj` (visual
    group, no contact) beside it. Returns the visual's word per body:
    the file, or `hull` when the prim carries no mesh."""
    import mujoco  # noqa: PLC0415

    assets.mkdir(parents=True, exist_ok=True)
    spec.meshdir = ASSETS_DIR
    shape_paths = geom_prim_paths(read)
    visuals: dict[str, str] = {}
    counts: dict[str, int] = {}
    stale = []
    for geom in list(spec.geoms):
        if geom.type != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        body = geom.parent.name
        counts[body] = counts.get(body, 0) + 1
        suffix = "" if counts[body] == 1 else f"_{counts[body]}"
        hull_name = f"{body}_{HULL_WORD}{suffix}"
        old = spec.mesh(geom.meshname)
        flat_vertices, flat_faces = list(old.uservert), list(old.userface)
        vertices = [
            tuple(flat_vertices[i : i + 3]) for i in range(0, len(flat_vertices), 3)
        ]
        faces = [tuple(flat_faces[i : i + 3]) for i in range(0, len(flat_faces), 3)]
        write_obj(assets / f"{hull_name}.obj", vertices, faces)
        spec.add_mesh(
            name=hull_name, file=f"{hull_name}.obj", maxhullvert=old.maxhullvert
        )
        stale.append(old)
        prim_path = shape_paths.get(geom.name, geom.name)
        geom.meshname = hull_name
        geom.name = hull_name
        geom.group = COLLISION_GROUP
        visual = _visual_mesh(
            stage, prim_path, read.shape_scale.get(prim_path, (1.0, 1.0, 1.0))
        )
        if visual is None:
            visuals[body] = HULL_WORD
            continue
        visual_name = f"{body}_{VISUAL_WORD}{suffix}"
        write_obj(assets / f"{visual_name}.obj", *visual)
        spec.add_mesh(name=visual_name, file=f"{visual_name}.obj")
        geom.parent.add_geom(
            name=visual_name,
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=visual_name,
            pos=geom.pos,
            quat=geom.quat,
            group=VISUAL_GROUP,
            contype=0,
            conaffinity=0,
            rgba=VISUAL_RGBA,
        )
        visuals[body] = f"{visual_name}.obj"
    for old in stale:
        spec.delete(old)
    return visuals


def name_actuators(spec: Any) -> None:
    """Position servos get their joint's range as `ctrlrange` and a name
    after the joint; Newton leaves both blank."""
    import mujoco  # noqa: PLC0415

    ranges = {j.name: tuple(j.range) for j in spec.joints}
    for actuator in spec.actuators:
        if actuator.trntype != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        if not actuator.name:
            actuator.name = f"{actuator.target}_{DRIVE_WORD}"
        if (
            actuator.biastype == mujoco.mjtBias.mjBIAS_AFFINE
            and actuator.target in ranges
        ):
            actuator.ctrlrange = ranges[actuator.target]
            actuator.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE


def add_sensors(spec: Any) -> int:
    """jointpos and jointvel per hinge and slide joint — what the
    harness's policies observe by contract."""
    import mujoco  # noqa: PLC0415

    kinds = (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
    added = 0
    for joint in spec.joints:
        if joint.type not in kinds:
            continue
        for word, kind in (
            ("pos", mujoco.mjtSensor.mjSENS_JOINTPOS),
            ("vel", mujoco.mjtSensor.mjSENS_JOINTVEL),
        ):
            spec.add_sensor(
                name=f"{joint.name}_{word}",
                type=kind,
                objtype=mujoco.mjtObj.mjOBJ_JOINT,
                objname=joint.name,
            )
            added += 1
    return added


def name_equalities(spec: Any) -> None:
    import mujoco  # noqa: PLC0415

    words = {
        mujoco.mjtEq.mjEQ_CONNECT: "connect",
        mujoco.mjtEq.mjEQ_WELD: "weld",
        mujoco.mjtEq.mjEQ_JOINT: "mimic",
    }
    taken: set[str] = set()
    for equality in spec.equalities:
        if not equality.name:
            word = words.get(equality.type, "equality")
            equality.name = unique(f"{word}_{equality.name1}_{equality.name2}", taken)


# MuJoCo's own words for an equality's softness, as the mjcPhysics schema
# authors them on a mimic joint (its `mjc:target` names the leader).
MIMIC_SOLREF = "mjc:solref"
MIMIC_SOLIMP = "mjc:solimp"


def mimic_softness(spec: Any, stage: Any, renamed: Mapping[str, str]) -> list[str]:
    """Each mimic's authored `mjc:solref` / `mjc:solimp` onto the joint
    equality Newton wrote for it. Newton's bridge carried them onto the
    loop closures' connects but left the mimic at MuJoCo's defaults
    (0.02 s against the 2F-85's authored 0.005 s, 2026-09-25). A value
    the USD does not author stays MuJoCo's default. Returns the names of
    the equalities set."""
    import mujoco  # noqa: PLC0415

    authored: dict[str, dict[str, list[float]]] = {}
    for prim in stage.Traverse():
        if not _is_mimic(prim):
            continue
        values = {
            field: [float(v) for v in value]
            for field, name in (("solref", MIMIC_SOLREF), ("solimp", MIMIC_SOLIMP))
            if (value := _attr(prim, name)) is not None
        }
        if values:
            follower = _newton_name(str(prim.GetPath()))
            authored[renamed.get(follower, follower)] = values
    done = []
    for equality in spec.equalities:
        if equality.type == mujoco.mjtEq.mjEQ_JOINT and equality.name1 in authored:
            for field, value in authored[equality.name1].items():
                setattr(equality, field, value)
            done.append(equality.name1)
    return done


def set_root(spec: Any, kind: str) -> str:
    """The root body Newton wrote as mocap becomes a plain welded body
    (`fixed`) or gets a free joint (`free`). Returns the root's name."""
    roots = list(spec.worldbody.bodies)  # the world's own parent is not readable
    if not roots:
        raise ValueError("the bridge wrote no body under the world")
    root = roots[0]
    root.mocap = False
    if kind == ROOT_FREE:
        root.add_freejoint()
    return root.name


def reset_newton_options(spec: Any) -> None:
    import mujoco  # noqa: PLC0415

    defaults = mujoco.MjSpec().option
    for name in NEWTON_OPTION_FIELDS:
        setattr(spec.option, name, getattr(defaults, name))


CONES = {"pyramidal": "mjCONE_PYRAMIDAL", "elliptic": "mjCONE_ELLIPTIC"}


def declare_grip_options(spec: Any, options: GripOptions = GRIP_OPTIONS) -> None:
    import mujoco  # noqa: PLC0415 - sim extra

    spec.option.impratio = options.impratio
    spec.option.cone = getattr(mujoco.mjtCone, CONES[options.cone])


def home_keyframe(spec: Any, stage: Any, read: UsdRead) -> None:
    """`home`: every joint at the position the USD states (radians;
    zero when it states none), every position servo holding it."""
    import math  # noqa: PLC0415

    import mujoco  # noqa: PLC0415

    by_leaf = {leaf(p): p for p in read.joints}
    model = spec.compile()
    qpos = model.qpos0.copy()
    for i in range(model.njnt):
        if model.jnt_type[i] not in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        prim = stage.GetPrimAtPath(by_leaf[name]) if name in by_leaf else None
        attribute = prim.GetAttribute(JOINT_STATE_ATTRIBUTE) if prim else None
        value = attribute.Get() if attribute and attribute.HasValue() else None
        if value is not None:
            radians = (
                math.radians(value)
                if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
                else value
            )
            qpos[model.jnt_qposadr[i]] = radians
    ctrl = [0.0] * model.nu
    for a in range(model.nu):
        if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
            ctrl[a] = float(qpos[model.jnt_qposadr[model.actuator_trnid[a][0]]])
    spec.add_key(name=HOME_KEY, qpos=qpos.tolist(), ctrl=ctrl)


def licence_folders(source: Path) -> list[Path]:
    """Where an asset's licence may sit: its own folder and, inside a
    fetched tree, every folder up to the tree's root (the fetch marker's
    folder) and no further. A local asset is looked for beside itself
    only: walking three folders up adopted the enclosing checkout's own
    LICENSE as the asset's (review 2026-09-24)."""
    from rq_pipeline.robot.asset_fetch import MARKER_FILE  # noqa: PLC0415

    folders = []
    for folder in source.parents:
        folders.append(folder)
        if (folder / MARKER_FILE).is_file():
            return folders  # the fetched tree's root: no further
    return [source.parent]  # not inside a fetched tree: beside the file only


def find_license(source: Path) -> tuple[Path | None, str]:
    """The licence file for the asset (`licence_folders`) and the licence
    it declares; the path found is recorded beside it."""
    for folder in licence_folders(source):
        for candidate in LICENSE_CANDIDATES:
            path = folder / candidate
            if path.is_file():
                return path, license_of(path)
    return None, UNRECORDED


def license_of(path: Path) -> str:
    head = path.read_text(encoding="utf-8", errors="replace")[:2000]
    for phrase, spdx in LICENSE_PHRASES.items():
        if phrase in head:
            return spdx
    return UNRECORDED


def provenance(
    source: Path, read: UsdRead, settings: ImportSettings, spdx: str
) -> dict[str, Any]:
    marker = read_marker(source.parent) or {}
    return {
        "format": "usd",
        "repository": marker.get("repository", UNRECORDED),
        "commit": marker.get("commit", UNRECORDED),
        "file": source.name,
        "variants": dict(read.variants),
        "root": settings.root,
        "mesh_maxhullvert": settings.mesh_maxhullvert,
        "grip_options": settings.grip_options,
        "license": spdx,
        "newton_census": dict(read.census),
        "versions": dict(read.versions),
    }


def readme_text(
    name: str, source: Path, prov: Mapping[str, Any], visuals: Mapping[str, str]
) -> str:
    variants = ", ".join(f"{k}={v}" for k, v in prov["variants"].items()) or "none"
    hull_only = [b for b, v in visuals.items() if v == HULL_WORD]
    versions = prov["versions"]
    commit = str(prov["commit"])[:7]
    lines = [
        f"# {name} — imported from USD",
        "",
        f"Source: `{source.name}` ({prov['repository']} @ {commit}), "
        f"variants {variants}, licence {prov['license']} "
        f"(the upstream text is `{LICENSE_FILE}`).",
        "",
        "Read by Newton's USD importer and bridged to MuJoCo by its solver "
        f"(newton {versions.get('newton')}, usd-core {versions.get('usd-core')}); "
        "the bundle writer renamed prim paths to leaf names, moved the inline "
        f"convex hulls ({prov['mesh_maxhullvert']} vertices at most) into "
        f"`{ASSETS_DIR}/*_{HULL_WORD}.obj` beside the USD's own visual meshes "
        f"(`{ASSETS_DIR}/*_{VISUAL_WORD}.obj`), added jointpos and jointvel "
        f"sensors, a `{HOME_KEY}` keyframe, and each position servo's joint "
        "range as its control range. docs/e2e-research/77 §6 names every rule. "
        "Contact type and affinity masks on the hulls are Newton's encoding of "
        "the USD's collision filters, kept as written.",
        "",
        f"Root: {prov['root']}. Runtime grip options declared: {prov['grip_options']}.",
        "",
        "A mimic's softness is carried when the USD authors it in MuJoCo's words "
        f"(`{MIMIC_SOLREF}`, `{MIMIC_SOLIMP}`); Newton's bridge wrote "
        "MuJoCo's defaults. "
        "What the USD carries that the bundle does NOT: a PhysX mimic's natural "
        "frequency and damping ratio (no exact MuJoCo equivalent is claimed); "
        "PhysX-only attributes. Newton's own solver defaults were reset to MuJoCo's.",
    ]
    if hull_only:
        lines += [
            "",
            f"Bodies whose visual is the hull (no mesh prim): {', '.join(hull_only)}.",
        ]
    return "\n".join(lines) + "\n"


def write_usd_bundle(
    source: Path, name: str, destination: Path, settings: ImportSettings | None = None
) -> dict[str, Any]:
    """The whole path: read, rewrite, write the files, compile the
    written MJCF, record, stamp. Never overwrites."""
    import mujoco  # noqa: PLC0415

    settings = settings or ImportSettings()
    source = Path(source).expanduser()
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"{destination} already exists")
    read = read_usd(source, settings)
    stage, _ = open_stage(source, settings.variants)
    spec = mujoco.MjSpec.from_string(read.xml)
    spec.modelname = name
    renamed = rename_elements(spec, read)
    mimic_softness(spec, stage, renamed)
    destination.mkdir(parents=True)
    visuals = meshes_to_files(spec, stage, read, destination / ASSETS_DIR)
    spec.modelfiledir = str(destination)  # the in-memory compile finds the files
    name_actuators(spec)
    name_equalities(spec)
    root = set_root(spec, settings.root)
    reset_newton_options(spec)
    if settings.grip_options:
        declare_grip_options(spec)
    sensors = add_sensors(spec)
    home_keyframe(spec, stage, read)
    model_file = f"{name}.xml"
    (destination / model_file).write_text(spec.to_xml(), encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(
        str(destination / model_file)
    )  # the file is the truth
    license_path, spdx = find_license(source)
    if license_path is not None:
        (destination / LICENSE_FILE).write_bytes(license_path.read_bytes())
    prov = provenance(source, read, settings, spdx)
    (destination / "README.md").write_text(
        readme_text(name, source, prov, visuals), encoding="utf-8"
    )
    write_bundle_record(
        destination, name, model_file, model, source=source, provenance=prov
    )
    return {
        "stamp": stamp(name, destination),
        "path": str(destination),
        "model_file": model_file,
        "bodies": int(model.nbody),
        "joints": int(model.njnt),
        "actuators": int(model.nu),
        "equalities": int(model.neq),
        "sensors": sensors,
        "root": root,
        "source": source_locator(source),
        "provenance": prov,
    }


@model_source(
    USD_SOURCE,
    USD_SUFFIXES,
    options=("variants", "root", "mesh_maxhullvert", "grip_options"),
    doc="a USD asset read by Newton's importer and written as a bundle: "
    "variants select its variant sets, root is fixed or free",
)
def onboard_usd(
    source_path: Path, name: str, destination: Path, options: Mapping[str, Any]
) -> dict[str, Any]:
    """The onboarding door's USD source: the options become settings
    (a wrong value is refused by name), then the bundle is written."""
    return write_usd_bundle(source_path, name, destination, ImportSettings(**options))


# -- the audit reader (docs/e2e-research/78 §1 item 2) ---------------------

USD_JOINT_KINDS = {  # the prim type names UsdPhysics gives its joints
    "PhysicsRevoluteJoint": HINGE,
    "PhysicsPrismaticJoint": SLIDE,
    "PhysicsSphericalJoint": BALL,
}
MJC_JOINT_ATTRIBUTES = {
    "armature": "mjc:armature",
    "damping": "mjc:damping",
    "frictionloss": "mjc:frictionloss",
    "stiffness": "mjc:stiffness",
}
# A joint that follows another: Newton's, PhysX's or MuJoCo's mimic
# schema applied (a multiple-apply schema carries an instance suffix).
MIMIC_SCHEMAS = ("NewtonMimicAPI", "PhysxMimicJointAPI", "MjcPhysicsEqualityJointAPI")
LOOP_ATTRIBUTE = "physics:excludeFromArticulation"
# UsdPhysics drives by the joint's motion: angular for a hinge, linear
# for a slide (the audit read angular only; review 2026-09-24).
DRIVE_FAMILY = {HINGE: "angular", SLIDE: "linear"}
DRIVE_STIFFNESS = "drive:{family}:physics:stiffness"
DRIVE_MAX_FORCE = "drive:{family}:physics:maxForce"
COM_ATTRIBUTE = "physics:centerOfMass"
DIAGONAL_INERTIA = "physics:diagonalInertia"
PRINCIPAL_AXES = "physics:principalAxes"
USD_EXPLANATIONS = (
    Explanation(UNIT, "angle", "USD states joint limits in degrees; MJCF in radians"),
    Explanation(
        COUNT, SENSORS, "jointpos and jointvel sensors added by the bundle writer"
    ),
    Explanation(COUNT, KEYFRAMES, "the home keyframe added by the bundle writer"),
    Explanation(
        COUNT, MESHES, "a convex hull for collision written beside each visual mesh"
    ),
    Explanation(
        JOINT_ORDER,
        ANY,
        "Newton orders joints by the kinematic tree, depth first; the USD "
        "authors them under a scope in its own order",
    ),
)


def _attr(prim: Any, name: str, default: Any = None) -> Any:
    attribute = prim.GetAttribute(name)
    if attribute and attribute.HasAuthoredValue():
        value = attribute.Get()
        return default if value is None else value
    return default


def _usd_body(prim: Any) -> BodyFacts:
    from pxr import UsdPhysics  # noqa: PLC0415

    mass = (
        float(_attr(prim, "physics:mass", 0.0))
        if prim.HasAPI(UsdPhysics.MassAPI)
        else 0.0
    )
    com = _attr(prim, COM_ATTRIBUTE)
    diagonal = _attr(prim, DIAGONAL_INERTIA)
    axes = _attr(prim, PRINCIPAL_AXES)
    inertia = None
    if diagonal is not None:
        quat = (1.0, 0.0, 0.0, 0.0)
        if axes is not None:
            imaginary = axes.GetImaginary()
            quat = (axes.GetReal(), imaginary[0], imaginary[1], imaginary[2])
        inertia = tensor_in_frame(tuple(diagonal), quat_to_matrix(quat))
    return BodyFacts(
        mass=mass,
        com=tuple(float(v) for v in com) if com is not None else None,
        inertia=inertia,
    )


def _usd_joint(prim: Any, kind: str) -> JointFacts:
    lower = float(_attr(prim, "physics:lowerLimit", -math.inf))
    upper = float(_attr(prim, "physics:upperLimit", math.inf))
    limited = math.isfinite(lower) and math.isfinite(upper) and lower <= upper
    if kind == HINGE:
        lower, upper = math.radians(lower), math.radians(upper)
    family = DRIVE_FAMILY.get(kind)
    max_force = (
        float(_attr(prim, DRIVE_MAX_FORCE.format(family=family), 0.0))
        if family
        else 0.0
    )
    params = {
        field_: float(_attr(prim, attribute, 0.0))
        for field_, attribute in MJC_JOINT_ATTRIBUTES.items()
    }
    return JointFacts(
        kind=kind,
        axis=None,  # the USD axis lives in the joint frame; the bundle's in the body's
        limited=limited,
        range=(lower, upper) if limited else (0.0, 0.0),
        force_range=(-max_force, max_force) if max_force > 0 else None,
        **params,
    )


def _is_mimic(prim: Any) -> bool:
    return any(name.startswith(MIMIC_SCHEMAS) for name in prim.GetAppliedSchemas())


def snapshot_stage(stage: Any) -> tuple[Snapshot, dict[str, str], tuple[str, ...]]:
    """The composed stage as the audit reads it: every rigid body, every
    joint prim in traversal order (loop closures included, so their
    absence from the bundle is a change the reader explains), the mimic
    and drive counts. Returns the snapshot, the prim-leaf name map, and
    the loop closures' names."""
    from pxr import Usd, UsdPhysics  # noqa: PLC0415

    bodies: dict[str, BodyFacts] = {}
    joints: dict[str, JointFacts] = {}
    names: dict[str, str] = {}
    taken_bodies: set[str] = set()
    taken_joints: set[str] = set()
    loops: list[str] = []
    mimics = drives = meshes = 0
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        path = prim.GetPath().pathString
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            name = unique(leaf(path), taken_bodies)
            names[path] = name
            bodies[path] = _usd_body(prim)
        kind = USD_JOINT_KINDS.get(prim.GetTypeName())
        if kind is not None:
            name = unique(leaf(path), taken_joints)
            names[path] = name
            joints[path] = _usd_joint(prim, kind)
            if _attr(prim, LOOP_ATTRIBUTE, False):
                loops.append(path)
            if _is_mimic(prim):
                mimics += 1
            family = DRIVE_FAMILY.get(kind)
            stiffness = DRIVE_STIFFNESS.format(family=family) if family else ""
            if family and float(_attr(prim, stiffness, 0.0)) > 0:
                drives += 1
        if prim.GetTypeName() == "Mesh":
            meshes += 1
    counts = {MIMICS: mimics, EQUALITIES: mimics + len(loops), MESHES: meshes}
    counts.update({SENSORS: 0, KEYFRAMES: 0})
    if drives:
        counts["actuators"] = drives
    snapshot = Snapshot(bodies, joints, counts, stage_units(stage))
    return snapshot, names, tuple(loops)


def stage_units(stage: Any) -> dict[str, str]:
    """The units the stage declares: joint angles in degrees (UsdPhysics),
    lengths and masses by its `metersPerUnit` and `kilogramsPerUnit`.
    A stage in centimetres or grams reads as such, so the audit sees a
    unit change the bundle writer does not explain and refuses it by
    name (it used to say "meter" whatever the stage said; review
    2026-09-24)."""
    from pxr import UsdGeom, UsdPhysics  # noqa: PLC0415

    metres = float(UsdGeom.GetStageMetersPerUnit(stage))
    kilograms = float(UsdPhysics.GetStageKilogramsPerUnit(stage))
    return {
        "angle": DEGREE,
        "length": METRE if metres == 1.0 else f"{metres:g} {METRE} per unit",
        "mass": KILOGRAM if kilograms == 1.0 else f"{kilograms:g} {KILOGRAM} per unit",
    }


@source_reader(USD_SOURCE, USD_SUFFIXES)
def audit_usd(path: Path, options: Mapping[str, Any]) -> SourceRead:
    """The stage as authored, with the bundle writer's conversions and
    Newton's loop-closure rewrite as the explanations."""
    settings = ImportSettings(**options)
    stage, selected = open_stage(Path(path), settings.variants)
    snapshot, names, loops = snapshot_stage(stage)
    explanations = list(USD_EXPLANATIONS) + [
        Explanation(
            JOINT_MISSING,
            loop,
            "a loop closure (excludeFromArticulation) written as a connect equality",
        )
        for loop in loops
    ]
    return SourceRead(
        snapshot,
        names=names,
        explanations=tuple(explanations),
        provenance={"variants": selected, "versions": versions()},
    )
