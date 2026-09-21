"""The mesh-true Rerun 3D mirror: any MuJoCo model, drawn from its own
physics state.

Every geom becomes what the physics uses — mesh geoms as real Mesh3D
entities (geometry logged once, one Transform3D per frame), primitives
as oriented solids (cylinders, capsules and spheres by their bounding
box) — posed from geom_xpos/xmat. Live dashboards, replays and sim
predictions all draw the same shapes the solver collides, and none of
them invents a skeleton. Grown from the yellow rig's `tools/_rig3d.py`
(which now re-exports from here); promoted 2026-09-01 so the mjlab
recorder can mirror ANY entity the trainer runs — the Studio's 3D view
follows what is actually happening, not a canned scene.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

RIG_PATH = "world/rig"  # where every rig tool logs the mirror
# The simulation clock every narrator logs on (the render stream, the walk
# view, the gate's mirror): one name, so the viewer lines them up.
SIM_TIMELINE = "sim"
# MuJoCo geom groups a visual mirror leaves out: 3 collision, 4 hidden, 5
# markers/sites (the convention mjlab and Menagerie scenes follow).
VISUAL_ONLY_SKIP_GROUPS: tuple[int, ...] = (3, 4, 5)
# A mirror's frame budget: rr.log per geom per frame blocks when it floods.
MIRROR_HZ = 20
# The Studio's Rerun ingest door — the ONE home for the address every
# feed and recorder connects to (the Rust shell binds the same port;
# crates/studio-shell/src/main.rs stays a documented mirror).
STUDIO_ADDRESS = "rerun+http://127.0.0.1:9876/proxy"


def _host_port(address: str) -> tuple[str, int]:
    """The host and port an address names, for a socket."""
    from urllib.parse import urlsplit  # noqa: PLC0415

    parts = urlsplit(address.replace("rerun+", "", 1))
    if parts.hostname is None or parts.port is None:
        raise ValueError(f"{address!r} names no host:port")
    return parts.hostname, parts.port


STUDIO_HOST, STUDIO_PORT = _host_port(STUDIO_ADDRESS)
TERM_EXIT_STATUS = 128 + 15  # a process ended by SIGTERM, as a shell reports it

# The headless rule (docs/76 §10.5): a feed that narrates an artifact also
# writes its stream into that artifact, under a hidden folder — outside
# the artifact's hash and the index's walk, so a picture never moves a
# version and a reindex never reads one.
VIEWER_DIR = ".viewer"
VIEWER_SUFFIX = ".rrd"
# One knob for disk: "0", "off", "no" or "false" keeps the live stream only.
VIEWER_FILE_ENV = "TRAINNR_VIEWER_FILE"
_OFF = ("0", "off", "no", "false")


def viewer_file(folder: Path | str, name: str) -> Path:
    """Where the named stream of the artifact at `folder` is saved."""
    return Path(folder) / VIEWER_DIR / f"{name}{VIEWER_SUFFIX}"


def viewer_files(folder: Path | str) -> list[Path]:
    """Every saved stream the artifact at `folder` carries, by name."""
    root = Path(folder) / VIEWER_DIR
    return sorted(root.glob(f"*{VIEWER_SUFFIX}")) if root.is_dir() else []


def viewer_file_wanted() -> bool:
    import os  # noqa: PLC0415

    return os.environ.get(VIEWER_FILE_ENV, "1").strip().lower() not in _OFF


# How long a feed waits to learn whether a Studio listens at the address
# before it opens: one TCP connect, on loopback, far below a frame.
LISTEN_PROBE_S = 0.5


def studio_listening(
    address: str = STUDIO_ADDRESS, timeout_s: float = LISTEN_PROBE_S
) -> bool:
    """Whether something accepts connections at the Studio's address now.
    A feed toward a viewer that never answers keeps every chunk in a
    bounded queue; once that queue is full the SDK's shutdown flush waits
    for an acknowledgement that never comes and the process never exits
    (a gate under the job runner, 2026-09-13: verdict printed, process
    alive for ten minutes). So the question is asked before the sink is
    chosen, not after."""
    import socket  # noqa: PLC0415

    try:
        with socket.create_connection(_host_port(address), timeout=timeout_s):
            return True
    except (OSError, ValueError):
        return False


def sinks(
    rr: Any,
    *,
    address: str = STUDIO_ADDRESS,
    file: Path | str | None = None,
    listening: bool | None = None,
    wanted: bool | None = None,
) -> list[Any]:
    """The sinks a feed streams to: the Studio's server when one listens,
    plus the file when one is named and wanted. Both at once on purpose:
    `rr.save()` alone REPLACES the viewer connection and the window goes
    dark while the file fills (measured 2026-08-28). With a file and no
    Studio, the file alone — a headless run must end (see
    `studio_listening`). With no file and no Studio the server sink stays,
    as every feed behaved before the file existed. `wanted` overrides the
    environment knob: a file the operator named on a command line is
    written whatever the knob says."""
    saving = viewer_file_wanted() if wanted is None else wanted
    heard = studio_listening(address) if listening is None else listening
    out: list[Any] = []
    if heard or file is None or not saving:
        out.append(rr.GrpcSink(address))
    if file is not None and saving:
        path = Path(file)
        path.parent.mkdir(parents=True, exist_ok=True)
        out.append(rr.FileSink(str(path)))
    return out


def open_stream(  # noqa: PLR0913 - the stream's own knobs, each named
    app_id: str,
    *,
    address: str = STUDIO_ADDRESS,
    file: Path | str | None = None,
    recording_id: str | None = None,
    on_term: bool = True,
    rr: Any = None,
) -> Any:
    """The one way a feed opens its stream: the global recording named
    `app_id`, sent to the Studio and, when `file` names one, saved there
    too; the TERM handler installed so a stopped job closes cleanly.
    `rr` is the SDK module when the caller already holds one (a feed that
    took it by injection, a test's fake); else the real one. Returns the
    module for the caller's logging."""
    import sys  # noqa: PLC0415

    if rr is None:
        import rerun  # noqa: PLC0415 - the viz extra

        rr = rerun
    kwargs = {"recording_id": recording_id} if recording_id is not None else {}
    rr.init(app_id, spawn=False, **kwargs)
    chosen = sinks(rr, address=address, file=file)
    kinds = [type(sink).__name__ for sink in chosen]
    if kinds == ["GrpcSink"]:
        rr.connect_grpc(address)
    else:
        rr.set_sinks(*chosen)
    if "GrpcSink" not in kinds:
        print(
            f"[{app_id}] no Studio listens at {address}: saving the stream to "
            f"{file} only (headless, docs/76 §10.5)",
            file=sys.stderr,
            flush=True,
        )
    if on_term:
        leave_cleanly_on_term(rr)
    return rr


def leave_cleanly_on_term(rr: Any) -> None:
    """A process streaming into the Studio's own Rerun server dies by
    TERM when the window closes or a job is stopped (the shell reaps its
    children; the job runner signals the group); a plain TERM cut the
    gRPC stream mid-message and the server logged "h2 protocol error:
    error reading a body from connection" at every close (2026-09-12).
    Close the connection first, then go. One home for every process
    that connects: the render stream's narrator, the walk view, the
    recorder, the verdict, the gate's mirror.

    The exit status stays what TERM means (128 + 15, the shell's
    convention): a stopped training job or gate must not read as one
    that finished. The exit is hard, like TERM's default action: the
    render stream's stdin reader holds a lock a normal shutdown trips
    over. A handler can only be installed from the main thread; from
    any other the default action stays."""
    import os  # noqa: PLC0415
    import signal  # noqa: PLC0415
    import threading  # noqa: PLC0415

    if threading.current_thread() is not threading.main_thread():
        return

    def _leave(*_: object) -> None:
        rr.disconnect()
        os._exit(TERM_EXIT_STATUS)

    signal.signal(signal.SIGTERM, _leave)


def mat_to_xyzw(flat: Any) -> Any:
    """Rotation matrix -> xyzw quaternion, via MuJoCo's own routine
    (mju_mat2Quat is wxyz; Rerun wants xyzw)."""
    import mujoco  # noqa: PLC0415 - sim extra
    import numpy as np  # noqa: PLC0415

    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.asarray(flat, dtype=float).reshape(9))
    return [quat[1], quat[2], quat[3], quat[0]]


def _bounding_half(mujoco: Any, kind: int, size: Any) -> list | None:
    """A primitive geom's bounding half-sizes; None for planes."""
    if kind == int(mujoco.mjtGeom.mjGEOM_PLANE):
        return None
    if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
        return size.tolist()
    if kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        return [size[0], size[0], size[1]]
    if kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        return [size[0], size[0], size[1] + size[0]]
    return [size[0]] * 3


class RigMirror:
    """Collects a model's geoms once; logs their poses per frame.

    `log(data, path)` accepts anything exposing per-geom `geom_xpos`
    (ngeom, 3) and `geom_xmat` ((ngeom, 9) or (ngeom, 3, 3)) as numpy
    arrays — a raw `mujoco.MjData`, or one world sliced out of a
    batched engine. MuJoCo compiles mesh vertices into the geom frame,
    so geom_xpos/xmat is the whole transform.

    model_colors=True paints every geom with its geom_rgba (matching
    the MuJoCo render); the default keeps the yellow-rig name-prefix
    palette the standing dashboards were tuned on.
    """

    def __init__(
        self,
        model: Any,
        skip: Sequence[str] = ("floor",),
        model_colors: bool = False,
        skip_groups: Sequence[int] = (),
        skip_prefixes: Sequence[str] = (),
    ) -> None:
        import mujoco  # noqa: PLC0415 - sim extra

        self.geoms: list[int] = []  # geom ids drawn as boxes
        self.half_sizes: list[list[float]] = []
        self.colors: list[list[int]] = []  # rgb per box geom
        # (geom id, entity name, vertices, faces, rgba)
        self.meshes: list[tuple[int, str, Any, Any, list[int]]] = []
        self._mesh_logged: set[str] = set()
        for g in range(model.ngeom):
            name = model.geom(g).name
            if name in skip or int(model.geom_group[g]) in skip_groups:
                continue
            if any(name.startswith(prefix) for prefix in skip_prefixes):
                continue  # e.g. every duck but the narrated one
            size = model.geom_size[g]
            kind = int(model.geom_type[g])
            # A geom with a material takes the material's colour (the
            # geom's own rgba is the grey default then).
            mat = int(model.geom_matid[g])
            rgba_f = model.mat_rgba[mat] if mat >= 0 else model.geom_rgba[g]
            rgba = [int(c * 255) for c in rgba_f]
            if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
                mid = int(model.geom_dataid[g])
                v0, nv = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
                f0, nf = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
                self.meshes.append(
                    (
                        g,
                        name or f"geom{g}",
                        model.mesh_vert[v0 : v0 + nv].copy(),
                        model.mesh_face[f0 : f0 + nf].copy(),
                        rgba,
                    )
                )
                continue
            half = _bounding_half(mujoco, kind, size)
            if half is None or max(half) <= 0.0:
                # Planes (infinite; mirrored as a zero box they would
                # ANCHOR the view's bounds at their own position, so a
                # robot a meter away shrinks to a corner of the frame —
                # measured on the duck, 2026-09-01) and zero-size geoms
                # draw nothing; the caller supplies its own ground.
                continue
            self.geoms.append(g)
            self.half_sizes.append(half)
            if model_colors:
                self.colors.append(rgba[:3])
            elif name.startswith("yarm"):
                self.colors.append([255, 190, 40])
            elif name.startswith("prop"):
                self.colors.append([230, 40, 40])
            else:
                self.colors.append([90, 130, 220])

    def log(self, data: Any, path: str = RIG_PATH, *, static: bool = False) -> None:
        """`static=True` for a snapshot (an artifact shown as itself): the
        poses then hold at every time, instead of landing at the clock's
        current instant and vanishing behind a paused cursor (2026-09-09)."""
        import rerun as rr  # noqa: PLC0415 - viz extra

        if self.geoms:
            rr.log(
                path,
                rr.Boxes3D(
                    centers=data.geom_xpos[self.geoms],
                    half_sizes=self.half_sizes,
                    quaternions=[
                        rr.Quaternion(xyzw=mat_to_xyzw(data.geom_xmat[g]))
                        for g in self.geoms
                    ],
                    colors=self.colors,
                    fill_mode="solid",
                ),
                static=static,
            )
        for g, name, verts, faces, rgba in self.meshes:
            entity = f"{path}/{name}"
            if entity not in self._mesh_logged:
                rr.log(
                    entity,
                    rr.Mesh3D(
                        vertex_positions=verts,
                        triangle_indices=faces,
                        albedo_factor=rgba,
                    ),
                    static=True,
                )
                self._mesh_logged.add(entity)
            rr.log(
                entity,
                rr.Transform3D(
                    translation=data.geom_xpos[g],
                    mat3x3=data.geom_xmat[g].reshape(3, 3),
                ),
                static=static,
            )


def gaussians(rr: Any, splats: Any) -> Any:
    """A scene's splat as Rerun's archetype: centres, scales, rotations
    reordered to Rerun's (x, y, z, w), colour and opacity as bytes."""
    import numpy as np  # noqa: PLC0415

    return rr.GaussianSplats3D(
        centers=splats.means,
        scales=splats.scales,
        quaternions=splats.quats[:, [1, 2, 3, 0]],
        colors=np.concatenate(
            [
                (splats.colors * 255).astype(np.uint8),
                (splats.opacities * 255).astype(np.uint8)[:, None],
            ],
            axis=1,
        ),
    )
