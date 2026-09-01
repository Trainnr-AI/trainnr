"""Stream MuJoCo's own render to stdout, one frame at a time, for a native
app that has no in-process MuJoCo binding of its own. Reads camera-orbit
commands back on stdin, so the viewer on the other end can rotate and zoom.

    cd pipeline && uv run --extra sim python ../tools/studio-render-stream.py [task]

Why this exists: `mujoco-rs` (the Rust FFI binding) needs a patched fork of
`glutin` to render on macOS at all, plus its own separate MuJoCo 3.9.0 install
— a real dependency-trust and version-skew cost this repo chose not to pay
(docs/e2e-research/54 and the studio-shell build log record the attempt).
MuJoCo already renders correctly from Python everywhere this repo runs it
(`camera-match.py`, `kitting_demos.py`, `show-aloha2.py`); this script is
that same `mujoco.Renderer` in a loop, framed onto stdout so any process in
any language can display it without touching MuJoCo's C API directly.

Frame transport (2026-09-02, the Rust piping rebuild): with `--shm
<path>` frames go through a MEMORY-MAPPED ring the controller created —
16-byte header (magic u32, seq u32, width u32, height u32, all LE) then
raw RGB8 pixels — under a seqlock (seq odd while writing, even when
stable; a reader that sees the seq move mid-copy discards and retries).
One 0xF7 byte on stdout per published frame is the wake-up token, so
the controller never polls. Raw frames over the pipe (the previous
wire, kept as the no-`--shm` fallback below) measured ~6 MB/frame — the
lag the ring exists to remove.

Fallback wire format, stdout, per frame, flushed immediately:

    width  : u32 little-endian
    height : u32 little-endian
    pixels : width * height * 3 raw RGB8 bytes, row-major, no padding

Wire format, stdin — TAGGED messages, one u8 tag then a fixed payload:

    0x01 camera : f32 d_azimuth (degrees to ADD, MuJoCo's convention),
                  f32 d_elevation (clamped here), f32 d_distance
                  (metres, clamped here), u32 width, u32 height
                  (absolute pixels the renderer should produce)
    0x02 select : f32 x, f32 y — pointer in the RENDERED image,
                  normalized [0,1], top-left origin; starts a
                  perturbation on the body under the pointer
    0x03 drag   : f32 dx, f32 dy — normalized pointer deltas moving the
                  active perturbation (MuJoCo's own mjv_movePerturb)
    0x04 release: end the perturbation
    0x05 pause  : toggle the physics loop

Camera values arrive as DELTAS and this side integrates them: every
absolute camera fact — the per-rig starting pose, the clamps — lives
here and only here. The first wire carried absolutes, which required the
viewer to duplicate the defaults, and the copies drifted the day per-rig
framing landed (the viewer's first drag snapped a 2.2 m ALOHA frame to
its stale 1.0 m). No pan yet — orbit-and-zoom only. Size is absolute;
the render loop recreates `mujoco.Renderer` only when it actually
changes, since a `Renderer` is fixed-size once constructed. A dedicated
thread reads updates off stdin so a slow/absent controller never blocks
rendering.
"""

import math
import os
import pathlib
import struct
import sys
import threading
import time

# Must run before `import mujoco` — this repo's own convention everywhere
# else offscreen rendering happens on Linux (tools/e2e-smoke.py,
# tools/cloud-gpu.py): EGL is the GPU-accelerated offscreen backend there.
# macOS's default (CGL) already works without it; `setdefault` still lets
# an operator override either platform's choice via their own environment.
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

from _lab import bootstrap

bootstrap()

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from rq_pipeline.tasks.registry import tasks  # noqa: E402

BUILDERS = {entry.name: entry for entry in tasks().values()}
# Tasks whose accepted scripted expert drives the sim for real; anything
# else gets the idle sinusoid. Grows as experts land (the SO-101 ones
# live on the rl-engineering branch until its merge).
TASKS_WITH_EXPERTS = ("kitting",)
DEFAULT_TASK = "kitting"
WIDTH, HEIGHT = 1024, 576
# 60 with the shared-memory ring (a frame is one memcpy); the stdout
# fallback stays honest at 30 (6 MB/frame through a pipe, measured).
TARGET_HZ = 60.0
FALLBACK_HZ = 30.0
SHM_HEADER = 16  # magic u32, seq u32, width u32, height u32 - all LE
SHM_MAGIC = 0x524A4D51  # "QMJR"
FRAME_TOKEN = b"\xf7"  # one byte on stdout per published shm frame

# Free-camera framing per rig, seeded from each rig's own viewer tools
# (show-aloha2's frame_viewer; the SO-101 numbers tuned by eye earlier) —
# a starting pose the operator immediately corrects by dragging.
RIG_CAMERAS = {
    "microduck": {
        "azimuth": 120.0,
        "elevation": -15.0,
        "distance": 0.8,
        "lookat": (0.0, 0.0, 0.12),
    },
    "so101": {
        "azimuth": 90.0,
        "elevation": -20.0,
        "distance": 1.0,
        "lookat": (0.0, 0.0, 0.15),
    },
    "aloha2": {
        "azimuth": 90.0,
        "elevation": -20.0,
        "distance": 2.2,
        "lookat": (0.0, 0.0, 0.2),
    },
}

# The physics narration into the Studio's embedded Rerun viewer: the same
# `mj_step` loop that renders the pixels also logs the twin, every named
# joint, every actuator and the contacts — one clock, so the plots can
# never drift from the picture. Everything the MuJoCo viewer's own panels
# show, but as named, legible Rerun views a blueprint lays out.
# Strictly best-effort: viz must never kill rendering (the same doctrine
# the resize clamp bought), so a missing SDK or an absent viewer just
# means no narration.
# Measured the hard way: at 30 Hz this narration is ~750 log calls/s,
# which outran the embedded viewer's ingest (debug build), filled its
# 128 MiB / 1024-message quota channel one minute after launch, and
# WEDGED it — the viewer kept drawing stale data while every later
# connection's messages (probes, the instrument one-shots) queued behind
# the block forever, delivered at the TCP level and never displayed.
# 10 Hz is ample for glanceable telemetry and stays far under the drain
# rate; the pixels keep their full frame rate regardless.
NARRATE_HZ = 10.0  # scalar series: plots need no more
MIRROR_HZ = 20.0  # the 3D twin: fluid motion; 30 Hz of per-mesh
# messages (~275 ms/s of Python serialization) blew the loop's realtime
# budget and slowed BOTH panes (2026-09-01)
# Past this geom count the shadow pass costs more than it lights (43 vs
# 10.5 ms/frame on the 20-duck flock, 2026-09-01).
SHADOW_GEOM_BUDGET = 400


class PhysicsNarrator:
    """The sim's state into Rerun, per step, on the `sim` timeline."""

    def __init__(self, model: "mujoco.MjModel", task_name: str) -> None:
        import rerun as rr  # noqa: PLC0415 - viz extra
        import rerun.blueprint as rrb  # noqa: PLC0415
        from rq_pipeline.viz import RigMirror  # noqa: PLC0415

        self.rr = rr
        rr.init(f"robotiq-sim-{task_name}", spawn=False)
        rr.connect_grpc()  # default 127.0.0.1:9876 — the Studio itself
        # Narrate ONE robot even when the scene holds a flock: rr.log
        # BLOCKS when the channel floods, and twenty ducks' series plus
        # 700 mesh transforms per tick froze the whole sim loop inside
        # a log call ("Sender has been blocked", 2026-09-01). The
        # pixels show the flock; the twin narrates specimen zero.
        flock = sorted(
            {
                model.joint(j).name.split("/")[0]
                for j in range(model.njnt)
                if "/" in (model.joint(j).name or "")
            }
        )
        self._narrated = flock[0] + "/" if len(flock) > 1 else ""
        others = tuple(f"{prefix}/" for prefix in flock[1:])
        self.mirror = RigMirror(model, model_colors=True, skip_prefixes=others)

        # Hinges and slides get scalar series; a free joint's 7-wide qpos
        # is pose, not a signal, and the mirror already shows it.
        scalar_types = (
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        )
        self.joints = [
            (
                model.joint(j).name or f"joint{j}",
                model.jnt_qposadr[j],
                model.jnt_dofadr[j],
            )
            for j in range(model.njnt)
            if int(model.jnt_type[j]) in scalar_types
            and (model.joint(j).name or "").startswith(self._narrated)
        ]
        self._last_series = -1.0
        self.actuators = [
            (model.actuator(a).name or f"actuator{a}", a)
            for a in range(model.nu)
            if (model.actuator(a).name or "").startswith(self._narrated)
        ]
        rr.send_blueprint(self._blueprint(rrb))

    def _blueprint(self, rrb: "object") -> "object":
        """One curated layout: the twin beside four small-multiple views,
        each a single unit family (positions, velocities, commands,
        forces) so no axis ever mixes rad with N·m."""
        # Origin-only views: the entity paths themselves split by unit
        # family (`joints/position/<name>`), because a `contents` path
        # filter silently matched nothing on first live contact while the
        # filterless contacts view worked — structure beats query.
        return rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(origin="world", name="physics twin"),
                rrb.Vertical(
                    rrb.Horizontal(
                        rrb.TimeSeriesView(
                            origin="joints/position", name="joint positions (rad)"
                        ),
                        rrb.TimeSeriesView(
                            origin="joints/velocity", name="joint velocities (rad/s)"
                        ),
                    ),
                    rrb.Horizontal(
                        rrb.TimeSeriesView(
                            origin="actuators/command", name="actuator commands"
                        ),
                        rrb.TimeSeriesView(
                            origin="actuators/force", name="actuator forces (N·m)"
                        ),
                    ),
                    rrb.TimeSeriesView(origin="contacts", name="contacts in scene"),
                ),
                column_shares=[3, 2],
            ),
            collapse_panels=True,
        )

    def log(self, data: "mujoco.MjData", sim_time: float) -> None:
        """`sim_time` is monotonic across episodes (the caller adds an
        offset) — `data.time` alone rewinds at every episode reset and a
        timeline must not. Called at MIRROR_HZ; the scalar series
        rate-limit themselves to NARRATE_HZ (a duck marching at 10 Hz
        beside 30 fps pixels read as 'very low frames', 2026-09-01)."""
        rr = self.rr
        rr.set_time("sim", duration=sim_time)
        self.mirror.log(data)
        now_series = sim_time - self._last_series >= 1.0 / NARRATE_HZ
        if not now_series:
            if data.ncon:
                rr.log(
                    "world/contacts",
                    rr.Points3D(data.contact.pos[: data.ncon], radii=0.004),
                )
            return
        self._last_series = sim_time
        for name, qpos_adr, dof_adr in self.joints:
            rr.log(f"joints/position/{name}", rr.Scalars(float(data.qpos[qpos_adr])))
            rr.log(f"joints/velocity/{name}", rr.Scalars(float(data.qvel[dof_adr])))
        for name, index in self.actuators:
            rr.log(f"actuators/command/{name}", rr.Scalars(float(data.ctrl[index])))
            rr.log(
                f"actuators/force/{name}", rr.Scalars(float(data.actuator_force[index]))
            )
        rr.log("contacts/count", rr.Scalars(float(data.ncon)))
        if data.ncon:
            rr.log(
                "world/contacts",
                rr.Points3D(data.contact.pos[: data.ncon], radii=0.004),
            )
        else:
            rr.log("world/contacts", rr.Clear(recursive=False))


def narrator_for(model: "mujoco.MjModel", task_name: str) -> "PhysicsNarrator | None":
    if "--no-rerun" in sys.argv:
        return None
    try:
        return PhysicsNarrator(model, task_name)
    except Exception as err:
        print(f"physics narration disabled: {err}", file=sys.stderr)
        return None


# The task's own offscreen budget is sized to its cameras (1280x720, the
# ArmnetBench wrist camera) — a full-screen viewer asks for more, and
# `mujoco.Renderer` REFUSES a size beyond the model's framebuffer rather
# than clamping (measured live: "Image width 1376 > framebuffer width
# 1280", the stream died, the viewport froze on its last frame). The
# budget is raised to the viewer's own cap before compile, and requests
# are clamped to the compiled framebuffer regardless, so no size a viewer
# sends can kill the stream. 1920 matches MAX_RENDER_SIDE in
# crates/studio-shell/src/viewport.rs — duplicated across the language
# boundary like the rest of this wire contract; change both together.
MAX_RENDER_SIDE = 1920

# Seeded from the `front` ArmnetBench camera's own placement
# (pos=[0, -0.85, 0.25], rq_pipeline/tasks/so101.py) so the free camera's
# first frame looks close to that fixed one — approximate by eye, not
# derived, since the viewer immediately lets a human correct it.


# The one home for every absolute camera fact (see the module docstring's
# deltas-not-absolutes rationale).
MAX_ELEVATION_DEG = 89.0
MIN_DISTANCE_M = 0.15
MAX_DISTANCE_M = 6.0


class OrbitCamera:
    """The camera's absolute state, integrated from the viewer's deltas —
    lock-protected: read by the render loop, written by
    `_read_camera_updates`'s thread."""

    def __init__(self, defaults: dict) -> None:
        self._lock = threading.Lock()
        self.azimuth = defaults["azimuth"]
        self.elevation = defaults["elevation"]
        self.distance = defaults["distance"]
        self.lookat = defaults["lookat"]
        self.width = WIDTH
        self.height = HEIGHT

    def apply_to(self, cam: "mujoco.MjvCamera") -> None:
        with self._lock:
            cam.azimuth, cam.elevation, cam.distance = (
                self.azimuth,
                self.elevation,
                self.distance,
            )

    def size(self) -> tuple[int, int]:
        with self._lock:
            return self.width, self.height

    def apply_deltas(
        self,
        d_azimuth: float,
        d_elevation: float,
        d_distance: float,
        width: int,
        height: int,
    ) -> None:
        with self._lock:
            self.azimuth += d_azimuth
            self.elevation = max(
                -MAX_ELEVATION_DEG, min(MAX_ELEVATION_DEG, self.elevation + d_elevation)
            )
            self.distance = max(
                MIN_DISTANCE_M, min(MAX_DISTANCE_M, self.distance + d_distance)
            )
            self.width, self.height = width, height


class Perturber:
    """A viewer-grade perturbation, split across the three threads that
    each own a piece of it: the stdin reader QUEUES select/drag/release,
    the render lane RESOLVES them (selection and mjv_movePerturb need
    the freshly updated scene and camera), and the physics loop APPLIES
    the force each step (mjv_applyPerturbForce writes xfrc_applied).
    The native viewer does all three in one loop; across a process
    boundary the queue is the seam."""

    def __init__(self, model: "mujoco.MjModel") -> None:
        self._lock = threading.Lock()
        self._model = model
        self.pert = mujoco.MjvPerturb()
        self._live_xpos = np.zeros((model.nbody, 3))
        self._pending_select: tuple[float, float] | None = None
        self._pending_drag = [0.0, 0.0]
        self._release = False
        self.paused = False

    # -- stdin reader side ------------------------------------------------
    def queue_select(self, x: float, y: float) -> None:
        with self._lock:
            self._pending_select = (x, y)

    def queue_drag(self, dx: float, dy: float) -> None:
        with self._lock:
            self._pending_drag[0] += dx
            self._pending_drag[1] += dy

    def queue_release(self) -> None:
        with self._lock:
            self._release = True

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    # -- render-lane side -------------------------------------------------
    def resolve(self, data: "mujoco.MjData", scene, vopt, aspect: float) -> None:
        """Selection and drag against the CURRENT scene — MuJoCo's own
        math: mjv_select ray-casts the pointer, mjv_movePerturb moves
        the reference point in the camera plane."""
        with self._lock:
            select, self._pending_select = self._pending_select, None
            dx, dy = self._pending_drag
            self._pending_drag = [0.0, 0.0]
            release, self._release = self._release, False
        if release:
            self.pert.active = 0
            self.pert.select = 0
            return
        if select is not None:
            x, y = select
            point = np.zeros(3)
            geom_id = np.array([-1], dtype=np.int32)
            flex_id = np.array([-1], dtype=np.int32)
            skin_id = np.array([-1], dtype=np.int32)
            # mjv_select's rely runs bottom-up; the wire sends top-down.
            body = mujoco.mjv_select(
                self._model,
                data,
                vopt,
                aspect,
                x,
                1.0 - y,
                scene,
                point,
                geom_id,
                flex_id,
                skin_id,
            )
            if body > 0:
                self.pert.select = body
                self.pert.active = mujoco.mjtPertBit.mjPERT_TRANSLATE
                mujoco.mjv_initPerturb(self._model, data, scene, self.pert)
        if self.pert.active and (dx or dy):
            mujoco.mjv_movePerturb(
                self._model,
                data,
                mujoco.mjtMouse.mjMOUSE_MOVE_H,
                dx,
                dy,
                scene,
                self.pert,
            )

    def draw(self, scene) -> None:
        """The pull, visible: a connector from the body to the reference
        point (the Renderer's update_scene never passes pert down, so
        the built-in mjVIS_PERTFORCE arrow cannot draw itself)."""
        if not self.pert.active or scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3),
            np.zeros(3),
            np.zeros(9),
            np.array([1.0, 0.35, 0.1, 0.8], dtype=np.float32),
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            0.004,
            self.pert.refpos,
            self._body_pos(),
        )
        scene.ngeom += 1

    def _body_pos(self) -> "np.ndarray":
        return self._live_xpos[self.pert.select].copy()

    # -- physics side -----------------------------------------------------
    def apply(self, data: "mujoco.MjData") -> None:
        """Each physics step: the standard simulate.cc ritual — clear,
        then let MuJoCo turn the reference offset into a force."""
        data.xfrc_applied[:] = 0.0
        if self.pert.active:
            mujoco.mjv_applyPerturbForce(self._model, data, self.pert)
        # The render lane draws the connector from the LIVE body pose.
        self._live_xpos = data.xpos


# The stdin protocol's tags, one home (mirrored by viewport.rs).
TAG_CAMERA, TAG_SELECT, TAG_DRAG, TAG_RELEASE, TAG_PAUSE = 1, 2, 3, 4, 5
TAG_PAYLOAD_BYTES = {
    TAG_CAMERA: 20,
    TAG_SELECT: 8,
    TAG_DRAG: 8,
    TAG_RELEASE: 0,
    TAG_PAUSE: 0,
}


def _read_control_messages(
    camera: OrbitCamera, perturber, poke, exit_on_eof: bool = False
) -> None:
    """The tagged stdin protocol (module docstring): camera deltas,
    perturbation gestures, pause. `poke` wakes the render lane so a
    drag re-renders NOW instead of at the next physics tick.

    `exit_on_eof`: under a controller (--shm), a closed stdin means the
    Studio is GONE — exit, hard. A SIGTERMed controller never runs its
    Drop reaper, and the orphan kept a core at 100% for 46 minutes
    before anyone looked (measured 2026-09-02). Standalone (no --shm),
    EOF keeps rendering at the last pose."""
    stdin = sys.stdin.buffer
    while True:
        tag_raw = stdin.read(1)
        if not tag_raw:
            if exit_on_eof:
                os._exit(0)  # daemon thread; no cleanup owed
            return  # standalone: keep rendering at the last pose
        tag = tag_raw[0]
        size = TAG_PAYLOAD_BYTES.get(tag)
        if size is None:
            return  # protocol desync: better a frozen camera than garbage
        payload = stdin.read(size) if size else b""
        if len(payload) < size:
            return
        if tag == TAG_CAMERA:
            camera.apply_deltas(*struct.unpack("<fffII", payload))
        elif tag == TAG_SELECT:
            perturber.queue_select(*struct.unpack("<ff", payload))
        elif tag == TAG_DRAG:
            perturber.queue_drag(*struct.unpack("<ff", payload))
        elif tag == TAG_RELEASE:
            perturber.queue_release()
        elif tag == TAG_PAUSE:
            perturber.toggle_pause()
        poke()


class FrameSink:
    """Where finished pixels go: the shared-memory ring when the
    controller gave us one (seqlock write, then a one-byte stdout
    token), the raw-stdout wire otherwise."""

    def __init__(self, shm_path: str | None) -> None:
        self._mm = None
        if shm_path:
            import mmap  # noqa: PLC0415

            handle = open(shm_path, "r+b")  # noqa: SIM115 - lives forever
            self._mm = mmap.mmap(handle.fileno(), 0)
            self._seq = 0
            struct.pack_into("<I", self._mm, 0, SHM_MAGIC)

    @property
    def shared(self) -> bool:
        return self._mm is not None

    def ship(self, frame: "np.ndarray", width: int, height: int) -> None:
        out = sys.stdout.buffer
        if self._mm is None:
            out.write(struct.pack("<II", width, height))
            out.write(frame.tobytes())
            out.flush()
            return
        self._seq += 1
        struct.pack_into("<I", self._mm, 4, self._seq)  # odd: writing
        struct.pack_into("<II", self._mm, 8, width, height)
        pixels = frame.tobytes()
        self._mm[SHM_HEADER : SHM_HEADER + len(pixels)] = pixels
        self._seq += 1
        struct.pack_into("<I", self._mm, 4, self._seq)  # even: stable
        out.write(FRAME_TOKEN)
        out.flush()


class RenderPump:
    """Everything one observed physics step needs: resize, orbit, render,
    frame out, narration — shared by the expert's `on_control` hook and
    the no-expert idle loop, so both paths behave identically."""

    def __init__(
        self,
        model: "mujoco.MjModel",
        orbit: OrbitCamera,
        narrator,
        perturber: Perturber | None = None,
        sink: FrameSink | None = None,
    ) -> None:
        self.model = model
        self.orbit = orbit
        self.narrator = narrator
        self.perturber = perturber or Perturber(model)
        self.sink = sink or FrameSink(None)
        self.hz = TARGET_HZ if self.sink.shared else FALLBACK_HZ
        self.last_narrated = 0.0
        # Episodes reset `data.time` to zero; the narration timeline must
        # not rewind with them, so it runs on an offset the episode loop
        # advances at each boundary.
        self.time_offset = 0.0
        self.last_sim_time = 0.0
        # The render lane: its own THREAD with its own MjData — physics
        # never waits for the GPU, drags track at true frame rate, and
        # the two big budget lines overlap instead of queueing (the
        # leanest fix, 2026-09-01: mj_step/render/pipe-write all release
        # the GIL, so a second thread is real parallelism). The staging
        # MjData carries the latest state; the render thread copies it
        # under the lock, forwards, and draws. The EGL context is
        # thread-affine, so the Renderer is BUILT in the render thread.
        self._staging = mujoco.MjData(model)
        self._staging_lock = threading.Lock()
        self._fresh = threading.Event()
        # macOS: Cocoa wants GL on the main thread (MuJoCo's offscreen
        # path rides GLFW there) — render inline instead of in a lane.
        self._threaded = sys.platform != "darwin"
        if self._threaded:
            threading.Thread(target=self._render_lane, daemon=True).start()
        else:
            self._inline_state = None  # built lazily by _render_once

    def tick(self, data: "mujoco.MjData", pace_seconds: float) -> None:
        """Observe one step: narrate (rate-limited), hand the render lane
        a snapshot, then sleep toward real time — `pace_seconds` is how
        much simulated time this step advanced."""
        self.last_sim_time = data.time
        # The shove, if one is active: xfrc for the caller's NEXT steps.
        self.perturber.apply(data)
        now = time.monotonic()
        if self.narrator is not None and now - self.last_narrated >= 1.0 / MIRROR_HZ:
            self.last_narrated = now
            self.narrator.log(data, self.time_offset + data.time)

        with self._staging_lock:
            self._staging.qpos[:] = data.qpos
            if self.model.nmocap:
                self._staging.mocap_pos[:] = data.mocap_pos
                self._staging.mocap_quat[:] = data.mocap_quat
        self._fresh.set()
        if not self._threaded:
            self._render_inline(now)

        # Pace toward real time: sleep off whatever of this step's
        # simulated duration wall time hasn't already consumed.
        remaining = pace_seconds - (time.monotonic() - now)
        if remaining > 0:
            time.sleep(remaining)

    def _render_inline(self, now: float) -> None:
        """The Darwin path: one render lane's body, run synchronously at
        the pump's rate inside tick (Cocoa's main-thread GL rule)."""
        if self._inline_state is None:
            self._inline_state = {"last": 0.0}
        if now - self._inline_state["last"] < 1.0 / self.hz:
            return
        self._inline_state["last"] = now
        self._render_step(self._inline_state)

    def _render_lane(self) -> None:
        state: dict = {}
        interval = 1.0 / self.hz
        while True:
            self._fresh.wait(timeout=interval)
            self._fresh.clear()
            began = time.monotonic()
            self._render_step(state)
            # Hold the lane to the target rate.
            leftover = interval - (time.monotonic() - began)
            if leftover > 0:
                time.sleep(leftover)

    def _render_step(self, state: dict) -> None:
        """One frame: snapshot -> forward -> (re)size -> render -> ship.
        `state` persists the renderer/camera between calls; built on
        first use IN THE CALLING THREAD (GL contexts are thread-affine).
        Shadow budget: measured on the 20-duck flock at 1300x400,
        43 ms/frame with shadows vs 10.5 without (2026-09-01)."""
        model = self.model
        if "renderer" not in state:
            state["width"], state["height"] = WIDTH, HEIGHT
            state["renderer"] = mujoco.Renderer(
                model, height=state["height"], width=state["width"]
            )
            state["shadows"] = model.ngeom <= SHADOW_GEOM_BUDGET
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat = list(self.orbit.lookat)
            state["cam"] = cam
            state["local"] = mujoco.MjData(model)
        local = state["local"]
        with self._staging_lock:
            local.qpos[:] = self._staging.qpos
            if model.nmocap:
                local.mocap_pos[:] = self._staging.mocap_pos
                local.mocap_quat[:] = self._staging.mocap_quat
        mujoco.mj_forward(model, local)
        want_width, want_height = self.orbit.size()
        # Clamp to the compiled framebuffer no matter what the viewer
        # asked: `mujoco.Renderer` refuses (raises) beyond it, and an
        # exception here kills the whole stream.
        want_width = min(want_width, int(model.vis.global_.offwidth))
        want_height = min(want_height, int(model.vis.global_.offheight))
        if (want_width, want_height) != (state["width"], state["height"]):
            # `Renderer` is fixed-size once constructed — a size change
            # means a rebuild, not a resize.
            state["renderer"].close()
            state["width"], state["height"] = want_width, want_height
            state["renderer"] = mujoco.Renderer(
                model, height=state["height"], width=state["width"]
            )
        if "vopt" not in state:
            # The physics made visible (rung 1, 2026-09-02): the same
            # scene-option flags the native viewer toggles with F -
            # contact forces as arrows, drawn by mjv_updateScene itself.
            vopt = mujoco.MjvOption()
            vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            state["vopt"] = vopt
        self.orbit.apply_to(state["cam"])
        renderer = state["renderer"]
        renderer.update_scene(local, camera=state["cam"], scene_option=state["vopt"])
        self.perturber.resolve(
            local, renderer.scene, state["vopt"], state["width"] / state["height"]
        )
        self.perturber.draw(renderer.scene)
        if not state["shadows"]:
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        frame = renderer.render()  # HxWx3 uint8, C-contiguous
        self.sink.ship(frame, state["width"], state["height"])


def run_expert_forever(task: "object", pump: RenderPump) -> None:
    """The real thing: the task's accepted scripted expert drives the sim,
    cycling the protocol's own paired trial starts — the same
    `perturb(trial, home)` draws the acceptance verdict ran on. The pump
    rides `on_control` (the hook the expert grew for exactly this), so
    the pixels and the narration show a genuine pick-and-place, contact
    events and grasp forces included."""
    from rq_pipeline.evaluate.harness import home_state  # noqa: PLC0415
    from rq_pipeline.physics.mujoco_backend import MuJoCoBackend  # noqa: PLC0415
    from rq_pipeline.tasks.aloha2 import scripted_kitting_episode  # noqa: PLC0415

    backend = MuJoCoBackend()
    backend.load_model(pump.model)
    protocol = task.protocol
    home = home_state(backend, protocol)
    tick_seconds = pump.model.opt.timestep * protocol.control_interval

    trial = 0
    while True:
        start = protocol.perturb(trial, home)

        def on_control(step: int, data: "mujoco.MjData") -> None:
            pump.tick(data, tick_seconds)

        _states, _sensors, _actions = scripted_kitting_episode(
            pump.model, start, on_control=on_control, spec=task.task_spec
        )
        # Advance the timeline by the episode's OBSERVED duration, not an
        # assumed `protocol.steps * timestep` — the two agree today, but
        # an early-ending expert would silently skew every later episode.
        pump.time_offset += pump.last_sim_time
        trial = (trial + 1) % protocol.trials


def run_idle_forever(model: "mujoco.MjModel", pump: RenderPump) -> None:
    """No expert registered for this task: a slow sinusoid on every
    actuator — visible, harmless placeholder motion so 'streaming' and
    'stalled' can be told apart at a glance."""
    data = mujoco.MjData(model)
    period = 0.0
    dt = model.opt.timestep
    while True:
        period += dt
        data.ctrl[:] = 0.2 * math.sin(period)
        mujoco.mj_step(model, data)
        pump.tick(data, dt)


DUCK = "duck"  # preview-only scene, not a registry task (no referee)


def duck_scene() -> "object":
    """The microduck bundle on a plain floor — a preview scene, built
    here rather than registered: a scene with no referee has no
    business in the task census. The XML's own kp=0.55 position servos
    hold it under the idle sinusoid."""
    import mujoco  # noqa: PLC0415

    repo = pathlib.Path(__file__).resolve().parent.parent
    scene = mujoco.MjSpec()
    scene.modelname = "microduck-preview"
    # A decorative parade needs no 2 ms integration: 4 ms halves the
    # physics share of the realtime budget (375 -> ~190 ms/s).
    scene.option.timestep = 0.004
    scene.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[3.0, 3.0, 0.1],
        rgba=[0.45, 0.5, 0.55, 1.0],
    )
    scene.worldbody.add_light(pos=[0.4, -0.4, 1.2], dir=[-0.3, 0.3, -1.0])
    # A flock on museum stands (the operator's ask, 2026-09-01). The
    # bare XML cannot STAND: only the feet carry floor collision (the
    # other collision meshes are self-collision-only, 2/2) and the
    # kp=0.55 placeholder servos - replaced by the BAM actuator at
    # train time - are too weak to hold posture. So each trunk welds at
    # standing height, limbs free under the idle sinusoid. Stand
    # height measured 2026-09-01: at the zero pose every mesh vertex
    # sits ~0.144 m ABOVE the root origin (onshape's export frame), so
    # the stand goes just below zero and the weld's sag rests the feet
    # onto ground contact.
    count, spacing = 20, 0.4
    columns = 5
    xml = str(repo / "robots" / "microduck" / "robot_walk.xml")
    for index in range(count):
        duck = mujoco.MjSpec.from_file(xml)
        trunk = duck.worldbody.first_body()
        row, col = divmod(index, columns)
        frame = scene.worldbody.add_frame(
            pos=[
                (col - (columns - 1) / 2) * spacing,
                (row - (count / columns - 1) / 2) * spacing,
                -0.008,
            ]
        )
        frame.attach_body(trunk, f"duck{index}/", "")
        # The stand is a MOCAP body, not the world: the parade loop
        # glides it forward, so the flock marches while the weld keeps
        # each duck upright (free walking needs the trained policy -
        # that is G3's job, not a preview's).
        stand = scene.worldbody.add_body(
            name=f"stand{index}",
            mocap=True,
            pos=[
                (col - (columns - 1) / 2) * spacing,
                (row - (count / columns - 1) / 2) * spacing,
                0.112,
            ],
        )
        weld = scene.add_equality()
        weld.type = mujoco.mjtEq.mjEQ_WELD
        weld.objtype = mujoco.mjtObj.mjOBJ_BODY
        weld.name1 = trunk.name  # attach already prefixed it
        weld.name2 = stand.name
    return scene


GAIT_HZ = 1.6  # step frequency of the parade waddle
PARADE_SPEED = 0.12  # m/s along +x, wrapping at the floor's edge
PARADE_WRAP_X = 2.4


def run_flock_parade_forever(model: "mujoco.MjModel", pump: RenderPump) -> None:
    """The duck parade: a scripted waddle on every duck's leg servos
    (phase-offset per duck) while each mocap stand glides forward -
    the legs are real physics under weak real servos; the forward
    motion is the stand's, honestly staged. Free walking is a trained
    policy's job (flagship G3)."""
    data = mujoco.MjData(model)

    def targets(name: str) -> "list[tuple[int, float, float]]":
        """(ctrl index, amplitude, phase) for one duck's gait."""
        gait = {
            "left_hip_pitch": (0.45, 0.0),
            "right_hip_pitch": (0.45, math.pi),
            "left_knee": (0.6, math.pi / 2),
            "right_knee": (0.6, math.pi / 2 + math.pi),
            "left_ankle": (0.3, math.pi),
            "right_ankle": (0.3, 0.0),
            "head_pitch": (0.12, math.pi / 2),
        }
        rows = []
        for a in range(model.nu):
            actuator = model.actuator(a).name or ""
            if not actuator.startswith(name):
                continue
            joint = actuator.removeprefix(name)
            if joint in gait:
                amplitude, phase = gait[joint]
                rows.append((a, amplitude, phase))
        return rows

    import numpy as np  # noqa: PLC0415

    # Vectorized gait: one sin() call per step instead of ~140 Python
    # float writes at 500 Hz - the interpreter loop was a real share of
    # the realtime budget (the drag-lag session, 2026-09-01).
    indices, amplitudes, phases = [], [], []
    mocap_ids = []
    duck_index = 0
    for index in range(model.nbody):
        body = model.body(index)
        if int(body.mocapid[0]) < 0:
            continue
        mocap_ids.append(int(body.mocapid[0]))
        for ctrl_index, amplitude, phase in targets(f"duck{duck_index}/"):
            indices.append(ctrl_index)
            amplitudes.append(amplitude)
            phases.append(phase + duck_index * 0.7)
        duck_index += 1
    indices = np.array(indices)
    amplitudes = np.array(amplitudes)
    phases = np.array(phases)
    mocap_ids = np.array(mocap_ids)
    omega = 2.0 * math.pi * GAIT_HZ
    dt = model.opt.timestep
    substeps = 4  # observe every 4th step: the tick's own overhead x4 less
    t = 0.0
    while True:
        for _ in range(substeps):
            t += dt
            data.ctrl[indices] = amplitudes * np.sin(omega * t + phases)
            x = data.mocap_pos[mocap_ids, 0] + PARADE_SPEED * dt
            x[x > PARADE_WRAP_X] = -PARADE_WRAP_X
            data.mocap_pos[mocap_ids, 0] = x
            mujoco.mj_step(model, data)
        pump.tick(data, dt * substeps)


def stream(task_name: str, shm_path: str | None) -> None:
    if task_name == DUCK:
        task, spec, rig = None, duck_scene(), "microduck"
    else:
        entry = BUILDERS[task_name]
        task = entry.build()
        spec, rig = task.spec, entry.rig
    # Raise (never lower) the offscreen budget to the viewer's cap — see
    # MAX_RENDER_SIDE's comment for the measured failure without this.
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, MAX_RENDER_SIDE)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, MAX_RENDER_SIDE)
    model = spec.compile()

    orbit = OrbitCamera(RIG_CAMERAS.get(rig, RIG_CAMERAS["so101"]))
    perturber = Perturber(model)
    pump = RenderPump(
        model, orbit, narrator_for(model, task_name), perturber, FrameSink(shm_path)
    )
    # `poke` = the render lane's own event: a camera or perturb gesture
    # re-renders NOW, not at the next physics tick (the native viewer's
    # decoupling, reproduced across the process boundary).
    threading.Thread(
        target=_read_control_messages,
        args=(orbit, perturber, pump._fresh.set, pump.sink.shared),
        daemon=True,
    ).start()

    if task is not None and task_name in TASKS_WITH_EXPERTS:
        run_expert_forever(task, pump)
    elif task_name == DUCK:
        run_flock_parade_forever(model, pump)
    else:
        run_idle_forever(model, pump)


if __name__ == "__main__":
    arguments = [a for a in sys.argv[1:] if not a.startswith("--")]
    shm = None
    for flag in sys.argv[1:]:
        if flag.startswith("--shm="):
            shm = flag.removeprefix("--shm=")
    task_name = arguments[0] if arguments else DEFAULT_TASK
    if task_name != DUCK and task_name not in BUILDERS:
        sys.exit(f"unknown task {task_name!r}; one of {sorted([*BUILDERS, DUCK])}")
    stream(task_name, shm)
