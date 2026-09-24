"""A robot authored in URDF into a bundle, through MuJoCo's own loader,
with the loader's known losses named by the mujoco build that runs.

Two things live here, both registered: the ONBOARDING SOURCE for
`.urdf` (MuJoCo's loader parses the file into an MjSpec, the bundle
holds the MJCF the spec writes beside the URDF and its meshes) and the
AUDIT READER that parses the URDF as authored — with the standard
library, not through MuJoCo — so the audit compares what the file says
against what the loader made of it.

What MuJoCo's loader does to a URDF, measured on this box (docs/e2e-research/78 §1):
the root link becomes the world body and its inertial is dropped
(3.11.0 and 3.13.0); a `<mimic>` was dropped silently until 3.13.0
(google-deepmind/mujoco PR #3530, merged 2026-09-08); an inertial
origin's rotation was lost until 3.14.0 (#3559, fixed 2026-09-17); a
`<limit velocity>` has no MJCF slot; `<transmission>` and `<gazebo>`
are ignored. Each is an explanation the audit applies only while the
running build has the loss, so a fixed build makes the same change
UNEXPLAINED — which is the point.
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from rq_pipeline.bundles.bundle import source_locator, write_bundle_record
from rq_pipeline.bundles.hashing import stamp
from rq_pipeline.robot.import_audit import (
    ANY,
    BODY_MISSING,
    COUNT,
    HINGE,
    INERTIA,
    MIMICS,
    SLIDE,
    BodyFacts,
    Explanation,
    JointFacts,
    Snapshot,
    SourceRead,
    full_tensor,
    rpy_to_matrix,
    source_reader,
    version_tuple,
)
from rq_pipeline.robot.onboarding import model_source

URDF_SOURCE = "urdf"
URDF_SUFFIXES = (".urdf",)
DEFAULT_AXIS = (1.0, 0.0, 0.0)  # URDF's default joint axis
# URDF joint types and the MuJoCo joint they become; `fixed` becomes no
# joint (a welded body), `floating` a free joint, `planar` has no MJCF form.
JOINT_KINDS = {
    "revolute": HINGE,
    "continuous": HINGE,
    "prismatic": SLIDE,
    "floating": "free",
}
UNLIMITED_KINDS = ("continuous",)
NO_JOINT_KINDS = ("fixed",)
UNSUPPORTED_KINDS = ("planar",)
IGNORED_TAGS = ("transmission", "gazebo")


@dataclass(frozen=True)
class LoaderLoss:
    """A documented loss of MuJoCo's URDF loader, gone from `fixed_in`."""

    kind: str
    element: str
    fixed_in: str | None
    issue: str
    reason: str

    def active(self, version: str) -> bool:
        return self.fixed_in is None or version_tuple(version) < version_tuple(
            self.fixed_in
        )


ROOT_LINK_LOSS = LoaderLoss(
    BODY_MISSING,
    ANY,
    None,
    "MuJoCo's URDF loader",
    "the URDF root link becomes the world body; its inertial is dropped "
    "(measured on mujoco 3.11.0 and 3.13.0)",
)
LOADER_LOSSES = (
    LoaderLoss(
        COUNT,
        MIMICS,
        "3.13.0",
        "google-deepmind/mujoco#3530",
        "a <mimic> was dropped silently by MuJoCo's URDF loader before 3.13.0",
    ),
    LoaderLoss(
        INERTIA,
        ANY,
        "3.14.0",
        "google-deepmind/mujoco#3559",
        "an inertial origin's rotation was lost by MuJoCo's URDF loader before 3.14.0",
    ),
)


def _floats(text: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if not text:
        return default
    return tuple(float(v) for v in text.split())


def _body(link: ET.Element) -> BodyFacts:
    inertial = link.find("inertial")
    if inertial is None:
        return BodyFacts(mass=0.0)
    mass_el = inertial.find("mass")
    mass = float(mass_el.get("value", "0")) if mass_el is not None else 0.0
    origin = inertial.find("origin")
    xyz = _floats(origin.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0))
    rpy = _floats(origin.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0))
    inertia = inertial.find("inertia")
    tensor = None
    if inertia is not None:
        raw = full_tensor(
            [
                float(inertia.get(k, "0"))
                for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
            ]
        )
        rotation = rpy_to_matrix(*rpy)
        tensor = tuple(float(v) for v in (rotation @ raw @ rotation.T).reshape(-1))
    return BodyFacts(mass=mass, com=(xyz[0], xyz[1], xyz[2]), inertia=tensor)


def _joint(joint: ET.Element) -> JointFacts | None:
    kind = joint.get("type", "")
    if kind in NO_JOINT_KINDS or kind in UNSUPPORTED_KINDS:
        return None
    axis_el = joint.find("axis")
    axis = _floats(axis_el.get("xyz") if axis_el is not None else None, DEFAULT_AXIS)
    limit = joint.find("limit")
    lower = float(limit.get("lower", "0")) if limit is not None else 0.0
    upper = float(limit.get("upper", "0")) if limit is not None else 0.0
    effort = float(limit.get("effort", "0")) if limit is not None else 0.0
    dynamics = joint.find("dynamics")
    limited = (
        kind not in UNLIMITED_KINDS
        and limit is not None
        and (lower, upper) != (0.0, 0.0)
    )
    return JointFacts(
        kind=JOINT_KINDS.get(kind, kind),
        axis=(axis[0], axis[1], axis[2]),
        limited=limited,
        range=(lower, upper) if limited else (0.0, 0.0),
        damping=float(dynamics.get("damping", "0")) if dynamics is not None else 0.0,
        frictionloss=float(dynamics.get("friction", "0"))
        if dynamics is not None
        else 0.0,
        force_range=(-effort, effort) if effort > 0 else None,
    )


def parse_urdf(path: Path) -> tuple[Snapshot, str, tuple[str, ...]]:
    """The URDF as authored: bodies, joints in document order, the
    mimic count; the root link's name; the advisories (what the file
    declares that no MJCF slot holds)."""
    root = ET.parse(path).getroot()
    bodies = {str(link.get("name")): _body(link) for link in root.findall("link")}
    joints: dict[str, JointFacts] = {}
    children: set[str] = set()
    mimics = 0
    advisories: list[str] = []
    for joint in root.findall("joint"):
        name = str(joint.get("name"))
        child = joint.find("child")
        if child is not None:
            children.add(str(child.get("link")))
        if joint.find("mimic") is not None:
            mimics += 1
        limit = joint.find("limit")
        if limit is not None and limit.get("velocity"):
            velocity = limit.get("velocity")
            advisories.append(
                f"joint {name}: <limit velocity={velocity}> has no MJCF slot"
            )
        if joint.get("type") in UNSUPPORTED_KINDS:
            advisories.append(
                f"joint {name}: type {joint.get('type')} has no MJCF form"
            )
        facts = _joint(joint)
        if facts is not None:
            joints[name] = facts
    for tag in IGNORED_TAGS:
        n = len(root.findall(tag))
        if n:
            advisories.append(f"{n} <{tag}> element(s) ignored by MuJoCo's loader")
    roots = [n for n in bodies if n not in children]
    root_link = roots[0] if roots else ""
    snapshot = Snapshot(bodies, joints, {MIMICS: mimics}, {"angle": "radian"})
    return snapshot, root_link, tuple(advisories)


@source_reader(URDF_SOURCE, URDF_SUFFIXES)
def read_urdf(path: Path, options: Mapping[str, Any]) -> SourceRead:
    """The URDF parsed as authored, with the loader's losses at THIS
    mujoco build as the explanations."""
    import mujoco  # noqa: PLC0415 - sim extra

    del options
    snapshot, root_link, advisories = parse_urdf(Path(path))
    explanations = [Explanation(ROOT_LINK_LOSS.kind, root_link, ROOT_LINK_LOSS.reason)]
    active = [loss for loss in LOADER_LOSSES if loss.active(mujoco.__version__)]
    explanations += [
        Explanation(loss.kind, loss.element, f"{loss.reason} ({loss.issue})")
        for loss in active
    ]
    return SourceRead(
        snapshot,
        explanations=tuple(explanations),
        advisories=advisories,
        provenance={
            "root_link": root_link,
            "loader_losses": [loss.issue for loss in active],
        },
    )


@model_source(
    URDF_SOURCE,
    URDF_SUFFIXES,
    doc="a URDF through MuJoCo's own loader: its directory copied whole "
    "(meshes ride along), the loader's MJCF written beside it",
)
def onboard_urdf(
    source_path: Path, name: str, destination: Path, options: Mapping[str, Any]
) -> dict[str, Any]:
    """URDF: the loader parses FIRST (a file it cannot read is refused
    before a byte lands), the directory is copied whole, the spec's MJCF
    is written beside the URDF and compiled from its file."""
    import mujoco  # noqa: PLC0415 - sim extra

    del options  # a URDF source takes none; `onboard` refused any
    spec = mujoco.MjSpec.from_file(str(source_path))
    spec.modelname = name
    shutil.copytree(source_path.parent, destination)
    model_file = f"{name}.xml"
    (destination / model_file).write_text(spec.to_xml(), encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(str(destination / model_file))
    _, root_link, _ = parse_urdf(source_path)
    write_bundle_record(
        destination,
        name,
        model_file,
        model,
        source=source_path,
        provenance={
            "format": URDF_SOURCE,
            "file": source_path.name,
            "root_link": root_link,
            "mujoco": mujoco.__version__,
        },
    )
    return {
        "stamp": stamp(name, destination),
        "path": str(destination),
        "model_file": model_file,
        "bodies": int(model.nbody),
        "joints": int(model.njnt),
        "actuators": int(model.nu),
        "source": source_locator(source_path),
    }
