"""Neverwhere's scenes as ours (docs/78 §4 E1/E4; docs/e2e-research/75
§3). The Neverwhere Visual Parkour Benchmark (arXiv 2609.16443, MIT)
ships each scene as a folder: `3dgs/model.splat` (the web splat) and
`3dgs/model.pt` (the gsplat checkpoint), `geometry/collision_mesh.obj`
already in the world frame, `geometry/collision_tf.json` (the
similarity that takes the capture frame - the splat's, the visual
mesh's - into the world frame), the poses and the dense reconstruction
when the zip is the full one, and a MuJoCo XML that places the
collision mesh as a signed-distance geom with a declared friction.

Importing one: the splat moved into the world frame by that
similarity and written as our PLY, the collision mesh copied as the
proxy and wrapped as MJCF, the gap measured, the friction their XML
declares recorded as DECLARED with the span this project declares,
and every fact their folder does not carry marked unrecorded.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from trainnr.robot.fit_record import code_version
from trainnr.scenes import gap as gap_audit
from trainnr.scenes.obj import FACE, VERTEX
from trainnr.scenes.proxy import PARTS_FILE, ensure_parts
from trainnr.scenes.record import (
    DECLARED,
    DECLARED_FRICTION_SPAN,
    FLOOR_FRICTION,
    PROXY_FILE,
    PROXY_MJCF,
    SCENE_FILE,
    SPLAT_FILE,
    UNRECORDED,
    Alignment,
    Capture,
    Physics,
    SceneRecord,
    Tool,
    proxy_mjcf,
)
from trainnr.scenes.splat import (
    describe,
    euler_xyz_matrix,
    read_any,
    write_ply,
)

SOURCE = "neverwhere"
PAPER = "arXiv 2609.16443 (Neverwhere Visual Parkour Benchmark, IROS 2026), MIT"
WEB_SPLAT = Path("3dgs") / "model.splat"
CHECKPOINT = Path("3dgs") / "model.pt"
COLLISION_MESH = Path("geometry") / "collision_mesh.obj"
COLLISION_TF = Path("geometry") / "collision_tf.json"
# Their XML declares the floor's friction on the collision geom; the
# span around it is THIS project's declaration (`record.DECLARED_FRICTION_SPAN`).
FRICTION_RE = re.compile(r'name="collision_mesh_geom"[^>]*friction="([^"]+)"')
FRICTION_CITES = (
    f"{SOURCE}: the scene XML's collision geom (sliding, torsional, rolling); "
    "the span is this project's declaration"
)
# What every import of theirs must say about itself.
IMPORT_NOTES = (
    "the web splat is opacity- and rotation-quantised and carries no harmonics "
    "beyond degree 0; the gsplat checkpoint beside it (model.pt) holds degree 3 "
    "and is the source for a finer export",
    "their simulator collides against this mesh as a signed-distance geom (sdflib); "
    "MuJoCo's plain mesh geom collides against its convex hull - a hurdle is not "
    "convex, so a task here decomposes the proxy (CoACD) or uses the SDF plugin, "
    "and says which",
)


def _triple(values: Any, what: str) -> tuple[float, float, float]:
    if len(values) != 3:  # noqa: PLR2004 - a 3-vector
        raise ValueError(f"{what}: three numbers expected, got {values!r}")
    return float(values[0]), float(values[1]), float(values[2])


def is_scene_folder(folder: Path) -> str | None:
    """None when `folder` is a Neverwhere scene, else why it is not."""
    folder = Path(folder)
    for needed in (WEB_SPLAT, COLLISION_MESH, COLLISION_TF):
        if not (folder / needed).is_file():
            return f"no {needed} under {folder}"
    return None


def _declared_friction(folder: Path) -> list[float] | None:
    for xml in sorted(folder.glob("*.xml")):
        m = FRICTION_RE.search(xml.read_text(encoding="utf-8", errors="replace"))
        if m:
            return [float(v) for v in m.group(1).split()]
    return None


WAYPOINT_BODY = re.compile(
    r'<body\s+name="waypoint-(\d+)"[^>]*?\bpos="([^"]+)"', re.IGNORECASE
)
WAYPOINT_SOURCE = "the scene's XML: its mocap bodies named waypoint-N, in order"


def _waypoints(source: Path) -> list[list[float]]:
    """The course's waypoint positions from the scene's own XML, in the
    order their names give; empty when the scene lays out none."""
    for xml in sorted(source.glob("*.xml")):
        found = WAYPOINT_BODY.findall(xml.read_text(encoding="utf-8", errors="replace"))
        if found:
            ordered = sorted(found, key=lambda m: int(m[0]))
            return [[float(v) for v in pos.split()] for _, pos in ordered]
    return []


def _mesh_facts(obj: Path) -> dict[str, Any]:
    """Vertex and face counts from the OBJ itself; watertightness from
    Open3D when the scene extra is there, else unrecorded."""
    vertices = faces = 0
    with Path(obj).open("r", encoding="utf-8", errors="replace") as src:
        for line in src:
            if line.startswith(VERTEX):
                vertices += 1
            elif line.startswith(FACE):
                faces += 1
    facts: dict[str, Any] = {"file": PROXY_FILE, "vertices": vertices, "faces": faces}
    try:
        import open3d as o3d  # noqa: PLC0415

        mesh = o3d.io.read_triangle_mesh(obj)
        facts["watertight"] = bool(mesh.is_watertight())
        facts["edge_manifold"] = bool(mesh.is_edge_manifold())
        v = np.asarray(mesh.vertices)
        facts["extent_m"] = [
            [round(float(x), 3) for x in v.min(0)],
            [round(float(x), 3) for x in v.max(0)],
        ]
    except ImportError:
        facts["watertight"] = UNRECORDED
    return facts


def import_scene(source: Path, out_dir: Path, *, name: str) -> Path:
    """One Neverwhere scene folder into `out_dir` as a scene artifact;
    returns the record's path. Refuses by name a folder that is not one
    or an `out_dir` that exists (a scene is never overwritten)."""
    source, out_dir = Path(source), Path(out_dir)
    why = is_scene_folder(source)
    if why:
        raise ValueError(f"{source} is not a Neverwhere scene: {why}")
    if out_dir.exists():
        raise ValueError(
            f"{out_dir} exists; a scene is an artifact and is never overwritten"
        )
    tf = json.loads((source / COLLISION_TF).read_text(encoding="utf-8"))
    scale = float(tf["mesh_scale"])
    euler = _triple(tf["mesh_euler"], "mesh_euler")
    pos = _triple(tf["mesh_pos"], "mesh_pos")
    splats = read_any(source / WEB_SPLAT).transformed(
        scale=scale,
        rotation=euler_xyz_matrix(np.array(euler)),
        translation=np.array(pos),
    )
    out_dir.mkdir(parents=True)
    write_ply(splats, out_dir / SPLAT_FILE)
    shutil.copy2(source / COLLISION_MESH, out_dir / PROXY_FILE)
    friction = _declared_friction(source)
    (out_dir / PROXY_MJCF).write_text(
        proxy_mjcf(PROXY_FILE, friction), encoding="utf-8"
    )
    try:
        measured = gap_audit.measure(splats, out_dir / PROXY_FILE)
    except ImportError as missing:
        measured = gap_audit.unmeasured(str(missing))
    proxy_facts = _mesh_facts(out_dir / PROXY_FILE)
    try:
        parts = ensure_parts(out_dir)
        proxy_facts["parts"] = parts.parts
        proxy_facts["parts_file"] = PARTS_FILE
    except ImportError as missing:
        proxy_facts["parts"] = UNRECORDED
        proxy_facts["parts_note"] = str(missing)
    physics: list[Physics] = []
    if friction is not None:
        physics.append(
            Physics(
                name=FLOOR_FRICTION,
                value=friction,
                basis=DECLARED,
                span=DECLARED_FRICTION_SPAN,
                cites=FRICTION_CITES,
                unit="MuJoCo friction triple",
            )
        )
    notes = list(IMPORT_NOTES)
    if not (source / CHECKPOINT).is_file():
        notes.append(
            "no gsplat checkpoint in this folder: the web splat is the only source"
        )
    record = SceneRecord(
        name=name,
        source=f"{SOURCE}/{source.name}",
        # The small zips carry no capture; the full ones carry Polycam or raw frames.
        capture=Capture(),
        tools=(
            Tool(
                name="Neverwhere",
                version=UNRECORDED,
                license="MIT",
                role="the scene's author: capture, splat, collision mesh, alignment",
            ),
            Tool(
                name="gsplat",
                version=UNRECORDED,
                license="Apache-2.0",
                role="their splat trainer (model.pt)",
            ),
            Tool(
                name="OpenMVS + their mesh tools",
                version=UNRECORDED,
                license="AGPL-3.0 (OpenMVS); their own tools unrecorded",
                role="their collision mesh (docs/e2e-research/75 §2)",
            ),
            Tool(
                name="trainnr.scenes",
                version=code_version(),
                license="Apache-2.0",
                role="the import, the world-frame splat, the gap",
            ),
        ),
        splat={"file": SPLAT_FILE, "from": str(WEB_SPLAT), **describe(splats)},
        proxy=proxy_facts,
        alignment=Alignment(
            scale=scale,
            rotation_euler_xyz=euler,
            translation=pos,
            source=str(COLLISION_TF),
        ),
        gap=measured,
        physics=tuple(physics),
        created_utc=datetime.now(timezone.utc).isoformat(),
        code=code_version(),
        notes=tuple(notes),
        course={"waypoints": waypoints, "source": WAYPOINT_SOURCE}
        if (waypoints := _waypoints(source))
        else {},
    )
    return record.write(out_dir / SCENE_FILE)
