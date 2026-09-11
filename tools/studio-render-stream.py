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

Process shape (2026-09-09): TWO processes. This one renders; it spawns
itself again with `--physics=<ring>` for the physics loop, and the two
meet in a memory-mapped state ring (`StateRing`): physics publishes
time/qpos/mocap each step, the renderer publishes the perturbation
wrench each frame. One interpreter could not do both — MuJoCo's render
holds the GIL (finding studio-viewport-pipe-2026-09-09).

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

Flags: `--shm=<ring>` (the controller's frame ring), `--shadows=on|off|auto`
(auto, the default, keeps shadows while the measured render fits one
60 Hz frame), `--no-rerun` (no narration into the viewer).

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
elif sys.platform == "darwin":
    # CGL, not GLFW: a CGL offscreen context is not bound to Cocoa's main
    # thread, so the render lane can be a thread here too (measured
    # 2026-09-09: 12 ms/frame from a background thread on Apple Silicon;
    # finding studio-viewport-pipe-2026-09-09). An operator's own
    # MUJOCO_GL still wins.
    os.environ.setdefault("MUJOCO_GL", "cgl")

from _lab import bootstrap

bootstrap()

# Two threads share the interpreter: physics with the scripted policy
# (long Python stretches) and the render lane (a few short Python
# stretches between C calls that release the GIL). At CPython's default
# 5 ms switch interval each of the lane's GIL acquisitions can wait 5 ms
# behind the policy thread — measured 2026-09-09 on kitting: a 10.6 ms
# render took 25 ms per lane frame (finding studio-viewport-pipe). A
# shorter interval hands the GIL over sooner at a cost the physics thread
# never notices (mj_step releases the GIL).
sys.setswitchinterval(0.0005)

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
# Above the display's 60 on purpose: Event.wait overshoots its timeout by
# 2-3 ms on macOS (measured 2026-09-09: a 2.6 ms render paced to 60 Hz
# delivered 52 fps), and the Studio's vsync caps what is drawn anyway.
TARGET_HZ = 75.0
FALLBACK_HZ = 30.0
SHM_HEADER = 16  # magic u32, seq u32, width u32, height u32 - all LE
SHM_MAGIC = 0x524A4D51  # "QMJR"
FRAME_TOKEN = b"\xf7"  # one byte on stdout per published shm frame

# Free-camera framing per rig, seeded from each rig's own viewer tools
# (show-aloha2's frame_viewer; the SO-101 numbers tuned by eye earlier) —
# a starting pose the operator immediately corrects by dragging.
DEFAULT_CAMERA = "default"  # a rig with no preset of its own
RIG_CAMERAS = {
    DEFAULT_CAMERA: {  # a metre-scale scene seen from the front, slightly above
        "azimuth": 90.0,
        "elevation": -20.0,
        "distance": 1.0,
        "lookat": (0.0, 0.0, 0.15),
    },
    "microduck-rl": {  # the RL view: nine worlds on a grid, seen from above the corner
        "azimuth": 120.0,
        "elevation": -20.0,
        "distance": 3.0,
        "lookat": (0.0, 0.0, 0.1),
    },
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


def rig_camera(rig: str | None) -> dict:
    """The rig's own framing, else the default preset."""
    return RIG_CAMERAS.get(rig or DEFAULT_CAMERA, RIG_CAMERAS[DEFAULT_CAMERA])


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
NARRATE_SHARE = 0.25  # narration may take this share of the physics thread, no more
MIRROR_HZ = 20.0  # the 3D twin: fluid motion; 30 Hz of per-mesh
# messages (~275 ms/s of Python serialization) blew the loop's realtime
# budget and slowed BOTH panes (2026-09-01)
# Shadows are on until the render lane measures that it cannot keep the
# display rate with them: a shadow pass is a flat cost per frame on some
# GPUs (17 ms of a 26 ms frame on the kitting scene, Apple Silicon,
# 2026-09-09, finding studio-viewport-pipe-2026-09-09), proportional to
# geoms on others (43 vs 10.5 ms on the 20-duck flock, 2026-09-01). A
# geom count cannot tell the two apart; the frame time can. `--shadows=`
# on|off|auto overrides.
SHADOW_BUDGET_MS = 14.0  # the render lane must fit under one 60 Hz display frame
SHADOW_PROBE_FRAMES = 30  # frames averaged before shadows are judged
SHADOW_PROBE_MS = 1000.0  # or this much render time, whichever comes first
SHADOWS_MODE = "auto"
# The lane reports itself on stderr this often: frames, render and ship
# times — the same facts MuJoCo's simulate shows in its Info overlay.
LANE_STATS_EVERY_S = 5.0
RTF_WINDOW_S = 1.0  # the real-time factor the status reports, over this window
# The last slice of each physics tick is spun, not slept, for accuracy.
PACE_SPIN_S = 0.0015
SPEED_MIN, SPEED_MAX = 0.01, 100.0  # simulate's Speed slider, roughly


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
        self._defaults = defaults
        self.default_distance = defaults["distance"]
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

    def zoom_to(self, distance: float) -> None:
        """A followed world is small: the camera closes in on it; an
        unfollow returns to the rig's default distance."""
        with self._lock:
            self.distance = max(MIN_DISTANCE_M, min(MAX_DISTANCE_M, distance))

    def set_view(self, preset: str) -> None:
        """A named view from the rig's default: `reset` restores it, `front`
        looks along the rig's default azimuth, `side` a quarter turn on,
        `top` straight down; the distance is the default's."""
        d = self._defaults
        with self._lock:
            self.distance = d["distance"]
            if preset == "front":
                self.azimuth, self.elevation = d["azimuth"], -15.0
            elif preset == "side":
                self.azimuth, self.elevation = d["azimuth"] + 90.0, -15.0
            elif preset == "top":
                self.azimuth, self.elevation = d["azimuth"], -MAX_ELEVATION_DEG
            else:
                self.azimuth, self.elevation = d["azimuth"], d["elevation"]

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

    # -- render-process side: the force, computed where the scene is ----
    def force(self, local: "mujoco.MjData") -> "tuple[int, np.ndarray] | None":
        """The wrench the reference offset asks for, on the render side's
        copy of the state (one frame behind the physics — the native
        viewer's own lag, since it too resolves against the last drawn
        scene). The physics process applies exactly this wrench."""
        self._live_xpos = local.xpos
        if not self.pert.active:
            return None
        local.xfrc_applied[:] = 0.0
        mujoco.mjv_applyPerturbForce(self._model, local, self.pert)
        body = int(self.pert.select)
        return body, local.xfrc_applied[body].copy()


# The stdin protocol's tags, one home (mirrored by viewport.rs).
TAG_CAMERA, TAG_SELECT, TAG_DRAG, TAG_RELEASE, TAG_PAUSE = 1, 2, 3, 4, 5
# The simulate controls (2026-09-09, docs/76 §10.2): what MuJoCo's own
# window offers in its Simulation, Joint, Control, Visualization and
# Rendering sections, one tag each. RUN/STEP/RESET/SPEED/MANUAL/CTRL/QPOS
# reach the physics process through the ring; VIS/RND stay on the render
# side.
TAG_RUN, TAG_STEP, TAG_RESET, TAG_SPEED, TAG_MANUAL = 6, 7, 8, 9, 10
TAG_CTRL, TAG_QPOS, TAG_VIS, TAG_RND, TAG_VIEW = 11, 12, 13, 14, 15
TAG_FOLLOW = 16  # i32 world to keep the camera on, -1 for none
FOLLOW_DISTANCE_M = 0.9  # a followed world is one small robot: close in on it
VIEW_PRESETS = ("reset", "front", "side", "top")  # TAG_VIEW's u8, in order
TAG_PAYLOAD_BYTES = {
    TAG_CAMERA: 20,
    TAG_SELECT: 8,
    TAG_DRAG: 8,
    TAG_RELEASE: 0,
    TAG_PAUSE: 0,
    TAG_RUN: 1,  # u8: 1 run, 0 pause
    TAG_STEP: 4,  # u32 steps (pauses first; takes manual control)
    TAG_RESET: 4,  # i32 keyframe, -1 for the model's initial state
    TAG_SPEED: 4,  # f32 real-time factor asked for
    TAG_MANUAL: 1,  # u8: 1 the sliders drive the scene, 0 its own motion again
    TAG_CTRL: 8,  # u32 actuator, f32 value (manual)
    TAG_QPOS: 8,  # u32 qpos address, f32 value (manual)
    TAG_VIS: 5,  # u32 mjtVisFlag, u8 on
    TAG_RND: 5,  # u32 mjtRndFlag, u8 on
    TAG_VIEW: 1,  # u8 VIEW_PRESETS index: the camera to a named view
    TAG_FOLLOW: 4,  # i32 world index, -1 none (many-worlds scenes)
}
STATUS_TOKEN = b"\xf8"  # then u32 LE length, then a JSON status (module docstring)
STATUS_EVERY_S = 1.0 / 30.0  # the sliders echo the scene at this rate


def _read_control_messages(  # noqa: PLR0912 - one branch per wire tag
    camera: OrbitCamera,
    perturber,
    poke,
    exit_on_eof: bool = False,
    sim: "SimControl | None" = None,
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
        elif tag == TAG_VIEW:
            index = payload[0]
            if index < len(VIEW_PRESETS):
                camera.set_view(VIEW_PRESETS[index])
        elif tag == TAG_SELECT:
            perturber.queue_select(*struct.unpack("<ff", payload))
        elif tag == TAG_DRAG:
            perturber.queue_drag(*struct.unpack("<ff", payload))
        elif tag == TAG_RELEASE:
            perturber.queue_release()
        elif tag == TAG_PAUSE:
            perturber.toggle_pause()
            if sim is not None:
                sim.handle(TAG_RUN, bytes([int(sim.paused)]))  # toggles Run
        elif sim is not None:
            sim.handle(tag, payload)
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

    def status(self, payload: bytes) -> None:
        """A JSON status to the controller (only on the token wire: the
        raw-frame fallback has no room for a second message kind)."""
        if not self.shared:
            return
        out = sys.stdout.buffer
        out.write(STATUS_TOKEN + struct.pack("<I", len(payload)) + payload)
        out.flush()

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


STATE_MAGIC = 0x5354_4154  # "STAT": the physics -> render state ring
# The ring header, all u32 LE: magic, seq, nq, nmocap, then the
# perturbation seqlock, active, body, paused; then nworld and reserved
# words. A many-worlds scene (the RL view) publishes per-world stats —
# reward, done — after the state floats.
STATE_HEADER = 48
WORLD_STATS = 2  # floats per world: reward, done
PERTURB_FLOATS = 6  # one wrench: force xyz, torque xyz
STATE_STATS = 4  # floats before qpos in the state region: time, rtf, manual, spare
MAILBOX_SLOTS = 16  # commands queued between two physics polls; older ones are dropped
MAILBOX_SLOT_BYTES = 16  # cmd u32, arg i32, arg f64
MAILBOX_BYTES = 8 + MAILBOX_SLOTS * MAILBOX_SLOT_BYTES  # cseq u32, pad; then the slots
MAILBOX_COMMANDS = ("none", "step", "reset", "speed", "manual")


class StateRing:
    """The seam between the physics process and the render process: one
    memory-mapped file the render side creates. Physics publishes its
    state (time, qpos, mocap) under a seqlock; the render side reads the
    newest stable one. The render side publishes the perturbation wrench
    (which body, which force) under its own seqlock; physics applies it
    every step. Two processes because two threads share one
    interpreter, and MuJoCo's render holds the interpreter lock: measured
    2026-09-09, a 12 ms render took 41 ms beside a busy Python thread
    and a 109 ms flock render slowed the physics to a crawl (finding
    studio-viewport-pipe-2026-09-09). The native viewer's physics thread
    is C and shares nothing; a second process is the same thing here."""

    def __init__(
        self, path: str, model: "mujoco.MjModel", *, create: bool, nworld: int = 0
    ) -> None:
        import mmap  # noqa: PLC0415

        self.nq, self.nmocap, self.nu = int(model.nq), int(model.nmocap), int(model.nu)
        self.nv, self.na = int(model.nv), int(model.na)
        if not create:
            nworld = self._peek_nworld(path)
        self.nworld = int(nworld)
        # State: time, rtf, manual flag, spare; then qpos; then mocap pos +
        # quat; then ctrl; then per-world stats.
        # qvel and act travel too: a forward pass on the render side then
        # reproduces the physics' contact forces (at zero velocity it drew
        # the support force of a frozen pose, up to 13 % off, 2026-09-09).
        floats = (
            STATE_STATS
            + self.nq
            + 7 * self.nmocap
            + self.nu
            + self.nv
            + self.na
            + WORLD_STATS * self.nworld
        )
        self._state_off = STATE_HEADER
        self._pert_off = self._state_off + 8 * floats
        # The mailbox (render -> physics): cseq u32, cmd u32, arg i32, pad, arg f64.
        self._mail_off = self._pert_off + 8 * PERTURB_FLOATS
        # The manual arrays (render -> physics): mseq u32, pad; ctrl[nu]; qpos[nq].
        self._manual_off = self._mail_off + MAILBOX_BYTES
        size = self._manual_off + 8 + 8 * (self.nu + self.nq)
        if create:
            with open(path, "wb") as f:
                f.write(b"\0" * size)
        with open(path, "r+b") as handle:  # mmap keeps its own reference
            self._mm = mmap.mmap(handle.fileno(), size)
        if create:
            struct.pack_into("<IIII", self._mm, 0, STATE_MAGIC, 0, self.nq, self.nmocap)
            struct.pack_into("<I", self._mm, 32, self.nworld)
        else:
            magic, _, nq, nmocap = struct.unpack_from("<IIII", self._mm, 0)
            if (magic, nq, nmocap) != (STATE_MAGIC, self.nq, self.nmocap):
                raise RuntimeError(
                    f"state ring {path}: header {(magic, nq, nmocap)} does not match "
                    f"this model {(STATE_MAGIC, self.nq, self.nmocap)}"
                )
        self.world_stats = np.zeros((self.nworld, WORLD_STATS))
        self._seq = 0
        self._pseq = 0
        self._cseq = 0
        self._mseq = 0
        self._buf = np.zeros(floats)
        self._manual_ctrl = np.zeros(self.nu)
        self._manual_qpos = np.zeros(self.nq)
        self._seen_cseq = 0
        self._seen_mseq = 0

    @staticmethod
    def _peek_nworld(path: str) -> int:
        with open(path, "rb") as f:
            f.seek(32)
            return struct.unpack("<I", f.read(4))[0]

    # -- physics side --------------------------------------------------------
    def publish(
        self, data: "mujoco.MjData", rtf: float = 0.0, manual: bool = False
    ) -> None:
        self._seq += 1
        struct.pack_into("<I", self._mm, 4, self._seq * 2 - 1)  # odd: writing
        buf = self._buf
        buf[0] = data.time
        buf[1] = rtf
        buf[2] = 1.0 if manual else 0.0
        k0 = STATE_STATS
        buf[k0 : k0 + self.nq] = data.qpos
        if self.nmocap:
            k = k0 + self.nq
            buf[k : k + 3 * self.nmocap] = data.mocap_pos.ravel()
            buf[k + 3 * self.nmocap : k + 7 * self.nmocap] = data.mocap_quat.ravel()
        k1 = k0 + self.nq + 7 * self.nmocap
        if self.nu:
            buf[k1 : k1 + self.nu] = data.ctrl
        k2 = k1 + self.nu
        buf[k2 : k2 + self.nv] = data.qvel
        if self.na:
            buf[k2 + self.nv : k2 + self.nv + self.na] = data.act
        if self.nworld:
            buf[k2 + self.nv + self.na :] = self.world_stats.ravel()
        self._mm[self._state_off : self._state_off + 8 * len(buf)] = buf.tobytes()
        struct.pack_into("<I", self._mm, 4, self._seq * 2)  # even: stable

    def read_perturbation(self) -> "tuple[int, np.ndarray] | None":
        """The wrench the render side asks for, or None; also the pause."""
        for _ in range(3):
            pseq, active, body, _paused = struct.unpack_from("<IIII", self._mm, 16)
            if pseq % 2:
                continue
            wrench = np.frombuffer(
                self._mm, dtype="<f8", count=PERTURB_FLOATS, offset=self._pert_off
            ).copy()
            if struct.unpack_from("<I", self._mm, 16)[0] == pseq:
                return (int(body), wrench) if active else None
        return None

    def paused(self) -> bool:
        return bool(struct.unpack_from("<I", self._mm, 28)[0])

    def take_command(self) -> "tuple[str, int, float] | None":
        """The oldest mailbox command not yet taken: (name, int arg, float
        arg). A queue, not a slot: a reset followed at once by a speed
        change lost the reset when the slot held only the newest
        (2026-09-09); commands older than MAILBOX_SLOTS are dropped."""
        cseq = struct.unpack_from("<I", self._mm, self._mail_off)[0]
        if cseq % 2 or cseq <= self._seen_cseq:
            return None
        self._seen_cseq = max(self._seen_cseq, cseq - 2 * MAILBOX_SLOTS)
        self._seen_cseq += 2
        slot = (self._seen_cseq // 2) % MAILBOX_SLOTS
        cmd, arg_i, arg_f = struct.unpack_from(
            "<Iid", self._mm, self._mail_off + 8 + slot * MAILBOX_SLOT_BYTES
        )
        return (MAILBOX_COMMANDS[cmd], arg_i, arg_f)

    def manual_inputs(self) -> "tuple[np.ndarray, np.ndarray] | None":
        """The sliders' ctrl and qpos, when they moved since last read."""
        mseq = struct.unpack_from("<I", self._mm, self._manual_off)[0]
        if mseq == self._seen_mseq or mseq % 2:
            return None
        off = self._manual_off + 8
        ctrl = np.frombuffer(self._mm, dtype="<f8", count=self.nu, offset=off).copy()
        qpos = np.frombuffer(
            self._mm, dtype="<f8", count=self.nq, offset=off + 8 * self.nu
        ).copy()
        if struct.unpack_from("<I", self._mm, self._manual_off)[0] != mseq:
            return None
        self._seen_mseq = mseq
        return ctrl, qpos

    # -- render side ---------------------------------------------------------
    def read_into(self, local: "mujoco.MjData") -> bool:
        """The newest stable state into `local`; False when none yet."""
        for _ in range(3):
            seq = struct.unpack_from("<I", self._mm, 4)[0]
            if seq == 0 or seq % 2:
                continue
            buf = np.frombuffer(
                self._mm, dtype="<f8", count=len(self._buf), offset=self._state_off
            ).copy()
            if struct.unpack_from("<I", self._mm, 4)[0] != seq:
                continue
            local.time = buf[0]
            self.rtf, self.manual = float(buf[1]), bool(buf[2])
            k0 = STATE_STATS
            local.qpos[:] = buf[k0 : k0 + self.nq]
            if self.nmocap:
                k = k0 + self.nq
                local.mocap_pos[:] = buf[k : k + 3 * self.nmocap].reshape(-1, 3)
                local.mocap_quat[:] = buf[
                    k + 3 * self.nmocap : k + 7 * self.nmocap
                ].reshape(-1, 4)
            k1 = k0 + self.nq + 7 * self.nmocap
            if self.nu:
                local.ctrl[:] = buf[k1 : k1 + self.nu]
            k2 = k1 + self.nu
            local.qvel[:] = buf[k2 : k2 + self.nv]
            if self.na:
                local.act[:] = buf[k2 + self.nv : k2 + self.nv + self.na]
            if self.nworld:
                self.world_stats[:] = buf[k2 + self.nv + self.na :].reshape(
                    self.nworld, WORLD_STATS
                )
            return True
        return False

    rtf = 0.0
    manual = False

    def set_paused(self, paused: bool) -> None:
        struct.pack_into("<I", self._mm, 28, int(paused))

    def post_command(self, name: str, arg_i: int = 0, arg_f: float = 0.0) -> None:
        """One command into the queue, in order."""
        self._cseq += 2
        slot = (self._cseq // 2) % MAILBOX_SLOTS
        struct.pack_into("<I", self._mm, self._mail_off, self._cseq - 1)
        struct.pack_into(
            "<Iid",
            self._mm,
            self._mail_off + 8 + slot * MAILBOX_SLOT_BYTES,
            MAILBOX_COMMANDS.index(name),
            arg_i,
            arg_f,
        )
        struct.pack_into("<I", self._mm, self._mail_off, self._cseq)

    def set_manual_input(
        self, ctrl_index: int | None, qpos_index: int | None, value: float
    ) -> None:
        """One slider moved: rewrite the manual arrays under their seqlock."""
        if ctrl_index is not None and 0 <= ctrl_index < self.nu:
            self._manual_ctrl[ctrl_index] = value
        if qpos_index is not None and 0 <= qpos_index < self.nq:
            self._manual_qpos[qpos_index] = value
        self._mseq += 2
        struct.pack_into("<I", self._mm, self._manual_off, self._mseq - 1)
        off = self._manual_off + 8
        self._mm[off : off + 8 * self.nu] = self._manual_ctrl.tobytes()
        self._mm[off + 8 * self.nu : off + 8 * (self.nu + self.nq)] = (
            self._manual_qpos.tobytes()
        )
        struct.pack_into("<I", self._mm, self._manual_off, self._mseq)

    def seed_manual(self, ctrl: "np.ndarray", qpos: "np.ndarray") -> None:
        """Start the manual arrays from the live state, so taking control
        does not snap the scene to zero."""
        self._manual_ctrl[:] = ctrl
        self._manual_qpos[:] = qpos

    def write_perturbation(self, wrench: "tuple[int, np.ndarray] | None") -> None:
        """The wrench only: the pause word at offset 28 belongs to
        `set_paused` (writing it here every frame erased a pause, 2026-09-09)."""
        self._pseq += 1
        struct.pack_into("<I", self._mm, 16, self._pseq * 2 - 1)
        body, force = wrench if wrench is not None else (0, np.zeros(PERTURB_FLOATS))
        struct.pack_into("<II", self._mm, 20, int(wrench is not None), body)
        self._mm[self._pert_off : self._pert_off + 8 * PERTURB_FLOATS] = force.tobytes()
        struct.pack_into("<I", self._mm, 16, self._pseq * 2)


class TakeOver(Exception):  # noqa: N818 - a hand-over, not an error
    """Raised out of a scene loop when the human takes the controls (a
    slider, a step): the physics process continues in the manual loop
    from the state the loop had reached."""

    def __init__(self, data: "mujoco.MjData", steps: int = 0) -> None:
        super().__init__("manual control")
        self.data = data
        self.steps = steps


class ResetScene(Exception):  # noqa: N818 - a hand-over, not an error
    """Raised out of any loop on Reset: the scene restarts from its
    initial state (or a keyframe), under its own motion again."""

    def __init__(self, keyframe: int = -1) -> None:
        super().__init__("reset")
        self.keyframe = keyframe


class PhysicsPump:
    """What one observed physics step does in the PHYSICS process: apply
    the render side's perturbation, take the mailbox's command, narrate,
    publish the state, pace to real time (times the asked speed). Same
    `tick` the scene loops call; no pixels here."""

    def __init__(self, model: "mujoco.MjModel", narrator, ring: StateRing) -> None:
        self.model = model
        self.narrator = narrator
        self.ring = ring
        self.last_narrated = 0.0
        self.time_offset = 0.0
        self.last_sim_time = 0.0
        self._stats_since = time.monotonic()
        self._stats_sim = 0.0
        self._stats_ticks = 0
        # Pacing runs against an absolute deadline, not a per-tick sleep:
        # time.sleep overshoots by several ms on macOS and a per-tick sleep
        # accumulates it (measured 2026-09-09: a 20 ms tick paced at
        # 25 ms, RTF 0.79, for a policy that alone runs 22x real time).
        self._deadline: float | None = None
        self._narrate_every = 1.0 / MIRROR_HZ
        self._rtf_since = time.monotonic()
        self._rtf_sim = 0.0
        self.speed = 1.0  # the real-time factor asked for (simulate's Speed)
        self.rtf = 0.0  # the one achieved, over the last statistics window
        self.manual = False

    def take_mail(self, data: "mujoco.MjData") -> None:
        """The mailbox: speed applies here; step, reset and manual hand
        the loop over (exceptions, because the scene loops own their
        stepping and cannot be told from inside a callback)."""
        command = self.ring.take_command()
        if command is None:
            return
        name, arg_i, arg_f = command
        if name == "speed":
            self.speed = max(SPEED_MIN, min(SPEED_MAX, arg_f))
            self._deadline = None
        elif name == "reset":
            raise ResetScene(arg_i)
        elif name == "step":
            raise TakeOver(data, steps=max(1, arg_i))
        elif name == "manual":
            if arg_i and not self.manual:
                raise TakeOver(data)
            if not arg_i and self.manual:
                raise ResetScene(-1)

    def tick(
        self, data: "mujoco.MjData", pace_seconds: float, hold_when_paused: bool = True
    ) -> None:
        """`hold_when_paused=False` is the manual loop stepping through a
        pause on purpose (Step n): observe and publish, do not hold."""
        self.last_sim_time = data.time
        now = time.monotonic()
        self._stats_ticks += 1
        self._stats_sim += pace_seconds
        self.take_mail(data)
        self._rtf_sim += pace_seconds
        if now - self._rtf_since >= RTF_WINDOW_S:
            self.rtf = self._rtf_sim / (now - self._rtf_since)
            self._rtf_since, self._rtf_sim = now, 0.0
        if now - self._stats_since >= LANE_STATS_EVERY_S:
            wall = now - self._stats_since
            # The real-time factor, the number Gazebo's World Stats and
            # simulate's info overlay show: simulated seconds per wall second.
            print(
                f"physics: sim +{self._stats_sim:.2f} s in {wall:.2f} s wall "
                f"(RTF {self._stats_sim / wall:.2f}); "
                f"{self._stats_ticks / wall:.0f} ticks/s",
                file=sys.stderr,
                flush=True,
            )
            self._stats_since, self._stats_sim, self._stats_ticks = now, 0.0, 0
        data.xfrc_applied[:] = 0.0
        wrench = self.ring.read_perturbation()
        if wrench is not None:
            body, force = wrench
            if 0 < body < self.model.nbody:
                data.xfrc_applied[body] = force
        if (
            self.narrator is not None
            and now - self.last_narrated >= self._narrate_every
        ):
            self.last_narrated = now
            self.narrator.log(data, self.time_offset + data.time)
            # Narration on a budget: it may take at most NARRATE_SHARE of
            # the physics thread, so its rate falls where a log is slow
            # (the 20-duck flock: a mirror pass over 1500 geoms cost the
            # whole real-time budget at 20 Hz — RTF 0.06 with it, 1.00
            # without, measured 2026-09-09).
            took = time.monotonic() - now
            self._narrate_every = max(1.0 / MIRROR_HZ, took / NARRATE_SHARE)
        self.ring.publish(data, self.rtf, self.manual)
        self._pace(pace_seconds / self.speed)
        # Paused (Run off): hold here, still publishing so a drag on a
        # paused scene shows its connector; a step or reset gets out.
        if hold_when_paused and self.ring.paused():
            while self.ring.paused():
                time.sleep(0.02)
                self.take_mail(data)
                self.ring.publish(data, 0.0, self.manual)
            self._deadline = None  # resume from now, not from before the pause

    def _pace(self, pace_seconds: float) -> None:
        """Hold the loop to real time against a running deadline: sleep
        for all but the last slice, spin for that slice (sleep's
        granularity is coarser than a physics tick). A loop that has
        fallen behind by more than one tick does not try to catch up —
        the deadline is reset, so a stall shows as a low RTF in the
        statistics line instead of a burst of fast motion afterwards."""
        now = time.monotonic()
        if self._deadline is None or now - self._deadline > pace_seconds:
            self._deadline = now
        self._deadline += pace_seconds
        remaining = self._deadline - now
        if remaining > PACE_SPIN_S:
            time.sleep(remaining - PACE_SPIN_S)
        while time.monotonic() < self._deadline:
            pass


class SimControl:
    """The render process's half of the simulate controls: the stdin
    tags land here (reader thread); the lane applies the visualization
    toggles and forwards the rest through the ring; every STATUS_EVERY_S
    it reports the clock, the inputs and, once, the model."""

    def __init__(
        self,
        model: "mujoco.MjModel",
        ring: StateRing,
        camera: "OrbitCamera | None" = None,
    ) -> None:
        self.model = model
        self.ring = ring
        self.camera = camera
        self._lock = threading.Lock()
        self._vis: dict[int, bool] = {}
        self._rnd: dict[int, bool] = {}
        self.paused = False
        self.manual = False
        self.speed = 1.0
        self._model_sent = False
        self._last_status = 0.0
        # The live ctrl and qpos as last drawn: what the sliders start
        # from when the human takes control (without this the first
        # slider sent every OTHER actuator as zero — arms collapsed onto
        # the table in a burst of contact arrows, 2026-09-09).
        self._live_ctrl = np.zeros(model.nu)
        self._live_qpos = np.zeros(model.nq)
        # Many-worlds scenes: the world the camera keeps in view (-1 none);
        # its root is the first body named `wNN/...` for that world.
        self.follow = -1
        self._world_roots = {}
        for body in range(1, model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
            prefix, sep, _ = name.partition("/")
            if sep and prefix.startswith("w") and prefix[1:].isdigit():
                self._world_roots.setdefault(int(prefix[1:]), body)

    # -- reader thread ----------------------------------------------------
    def handle(self, tag: int, payload: bytes) -> None:
        if tag == TAG_RUN:
            self.paused = not bool(payload[0])
            self.ring.set_paused(self.paused)
        elif tag == TAG_STEP:
            (n,) = struct.unpack("<I", payload)
            self.paused = True
            self.manual = True
            self.ring.set_paused(True)
            self.ring.post_command("step", n)
        elif tag == TAG_RESET:
            (key,) = struct.unpack("<i", payload)
            self.ring.post_command("reset", key)
        elif tag == TAG_SPEED:
            (factor,) = struct.unpack("<f", payload)
            self.speed = max(SPEED_MIN, min(SPEED_MAX, factor))
            self.ring.post_command("speed", 0, self.speed)
        elif tag == TAG_MANUAL:
            on = bool(payload[0])
            if on and not self.manual:
                self.ring.seed_manual(self._live_ctrl, self._live_qpos)
            self.manual = on
            self.ring.post_command("manual", int(self.manual))
        elif tag == TAG_CTRL:
            index, value = struct.unpack("<If", payload)
            self._take_control()
            self.ring.set_manual_input(index, None, value)
        elif tag == TAG_QPOS:
            index, value = struct.unpack("<If", payload)
            self._take_control()
            self.ring.set_manual_input(None, index, value)
        elif tag == TAG_FOLLOW:
            (world,) = struct.unpack("<i", payload)
            self.follow = world if world in self._world_roots else -1
            if self.camera is not None:
                self.camera.zoom_to(
                    FOLLOW_DISTANCE_M
                    if self.follow >= 0
                    else self.camera.default_distance
                )
        elif tag == TAG_VIS:
            flag, on = struct.unpack("<IB", payload)
            with self._lock:
                self._vis[flag] = bool(on)
        elif tag == TAG_RND:
            flag, on = struct.unpack("<IB", payload)
            with self._lock:
                self._rnd[flag] = bool(on)

    def _take_control(self) -> None:
        """A slider moved: the human drives the scene from here on, from
        the pose and controls it had."""
        if not self.manual:
            self.ring.seed_manual(self._live_ctrl, self._live_qpos)
            self.manual = True
            self.ring.post_command("manual", 1)

    # -- render lane ------------------------------------------------------
    def apply_flags(self, vopt: "mujoco.MjvOption", scene: "mujoco.MjvScene") -> None:
        with self._lock:
            vis, rnd = dict(self._vis), dict(self._rnd)
        for flag, on in vis.items():
            if 0 <= flag < len(vopt.flags):
                vopt.flags[flag] = on
        for flag, on in rnd.items():
            if 0 <= flag < len(scene.flags):
                scene.flags[flag] = on

    def follow_lookat(self, local: "mujoco.MjData") -> "np.ndarray | None":
        """Where the camera should look: the followed world's root, if any."""
        body = self._world_roots.get(self.follow)
        return None if body is None else local.xpos[body].copy()

    def status(self, local: "mujoco.MjData", lane: dict, state: dict) -> bytes | None:
        """The JSON status, or None until STATUS_EVERY_S has passed."""
        import json  # noqa: PLC0415

        if not self.manual:
            self._live_ctrl[:] = local.ctrl
            self._live_qpos[:] = local.qpos
        now = time.monotonic()
        if now - self._last_status < STATUS_EVERY_S:
            return None
        self._last_status = now
        # Every flag's value as rendered — the toolbar shows the truth,
        # not the ones the human touched.
        vopt, scene = state.get("vopt"), lane.get("scene")
        vis = (
            {str(i): bool(v) for i, v in enumerate(vopt.flags)}
            if vopt is not None
            else {}
        )
        rnd = (
            {str(i): bool(v) for i, v in enumerate(scene.flags)}
            if scene is not None
            else {}
        )
        body: dict = {
            "time": float(local.time),
            "rtf": float(self.ring.rtf),
            "paused": self.paused,
            "manual": bool(self.ring.manual),
            "speed": self.speed,
            "qpos": [float(v) for v in local.qpos],
            "ctrl": [float(v) for v in local.ctrl],
            "shadows": bool(state.get("shadows", True)),
            "render_ms": float(lane.get("last_render_ms", 0.0)),
            "vis": vis,
            "rnd": rnd,
            "follow": self.follow,
            "worlds": [
                {"reward": float(r), "done": bool(d)} for r, d in self.ring.world_stats
            ],
        }
        if not self._model_sent:
            self._model_sent = True
            body["model"] = self.describe_model()
        return json.dumps(body).encode()

    def describe_model(self) -> dict:
        """What the panels need once: joints with their qpos addresses
        and ranges, actuators with their control ranges, keyframes, the
        physics facts, and the flag tables in MuJoCo's own names."""
        m = self.model
        joints = []
        for j in range(m.njnt):
            jtype = int(m.jnt_type[j])
            if jtype not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                continue  # free and ball joints have no scalar slider (simulate's rule)
            joints.append(
                {
                    "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
                    or f"joint{j}",
                    "qpos": int(m.jnt_qposadr[j]),
                    "range": [float(m.jnt_range[j][0]), float(m.jnt_range[j][1])],
                    "limited": bool(m.jnt_limited[j]),
                    "type": "hinge"
                    if jtype == mujoco.mjtJoint.mjJNT_HINGE
                    else "slide",
                }
            )
        actuators = [
            {
                "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
                or f"actuator{a}",
                "range": [
                    float(m.actuator_ctrlrange[a][0]),
                    float(m.actuator_ctrlrange[a][1]),
                ],
                "limited": bool(m.actuator_ctrllimited[a]),
            }
            for a in range(m.nu)
        ]
        keyframes = [
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_KEY, k) or f"key{k}"
            for k in range(m.nkey)
        ]
        vis_names = [
            name.removeprefix("mjVIS_").lower()
            for name, _ in sorted(
                (
                    (n, int(v))
                    for n, v in mujoco.mjtVisFlag.__members__.items()
                    if n != "mjNVISFLAG"
                ),
                key=lambda kv: kv[1],
            )
        ]
        rnd_names = [
            name.removeprefix("mjRND_").lower()
            for name, _ in sorted(
                (
                    (n, int(v))
                    for n, v in mujoco.mjtRndFlag.__members__.items()
                    if n != "mjNRNDFLAG"
                ),
                key=lambda kv: kv[1],
            )
        ]
        return {
            "joints": joints,
            "actuators": actuators,
            "keyframes": keyframes,
            "timestep": float(m.opt.timestep),
            "integrator": mujoco.mjtIntegrator(m.opt.integrator).name.removeprefix(
                "mjINT_"
            ),
            "solver": mujoco.mjtSolver(m.opt.solver).name.removeprefix("mjSOL_"),
            "iterations": int(m.opt.iterations),
            "gravity": [float(g) for g in m.opt.gravity],
            "nbody": int(m.nbody),
            "ngeom": int(m.ngeom),
            "vis_flags": vis_names,
            "rnd_flags": rnd_names,
            "nworld": self.ring.nworld,
        }


class RenderPump:
    """The RENDER process: reads the newest physics state from the ring,
    resolves camera and perturbation against the freshly drawn scene,
    renders at the display's rate, ships the frame. The perturbation
    wrench it computes goes back through the ring."""

    def __init__(  # noqa: PLR0913, PLR0917 - the lane's five collaborators, by name
        self,
        model: "mujoco.MjModel",
        orbit: OrbitCamera,
        perturber: Perturber,
        sink: "FrameSink",
        ring: StateRing,
        sim: "SimControl | None" = None,
    ) -> None:
        self.model = model
        self.orbit = orbit
        self.perturber = perturber
        self.sink = sink
        self.ring = ring
        self.sim = sim
        self.hz = TARGET_HZ if self.sink.shared else FALLBACK_HZ
        # A camera or perturb gesture re-renders NOW (the stdin reader
        # sets it), not at the next lane tick.
        self._fresh = threading.Event()

    def run_forever(self) -> None:
        """The lane, on the calling thread (GL contexts are thread-affine;
        the main thread works under every backend, GLFW included)."""
        state: dict = {}
        interval = 1.0 / self.hz
        try:
            while True:
                began = time.monotonic()
                self._render_step(state)
                # One frame per interval; a poke (camera, perturbation)
                # ends the wait early and re-renders at once.
                remaining = interval - (time.monotonic() - began)
                if remaining > 0:
                    self._fresh.wait(timeout=remaining)
                self._fresh.clear()
        except BrokenPipeError:
            # The Studio closed the frame pipe: we are done. A hard exit,
            # because the stdin reader (a daemon thread) holds stdin's
            # buffer lock and a normal shutdown trips over it.
            os._exit(0)

    def _aim(self, state: dict, local: "mujoco.MjData") -> None:
        """A followed world keeps the camera's lookat on its root."""
        if self.sim is None:
            return
        target = self.sim.follow_lookat(local)
        if target is not None:
            self.orbit.lookat = target
            state["cam"].lookat = list(target)

    def _render_step(self, state: dict) -> None:
        """One frame: ring -> forward -> (re)size -> render -> ship.
        `state` persists the renderer/camera between calls."""
        model = self.model
        if "renderer" not in state:
            state["width"], state["height"] = WIDTH, HEIGHT
            state["renderer"] = mujoco.Renderer(
                model, height=state["height"], width=state["width"]
            )
            state["shadows"] = SHADOWS_MODE != "off"
            state["render_ms"] = []  # the last SHADOW_PROBE_FRAMES render times
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat = list(self.orbit.lookat)
            state["cam"] = cam
            state["local"] = mujoco.MjData(model)
        local = state["local"]
        self.ring.read_into(local)  # before the first publish: the zero pose
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
            # MuJoCo's own defaults, as simulate starts: the Visualization
            # panel (and the agent's set_simulator_view) turns contact
            # forces and the rest on. They were on by default from
            # 2026-09-02 to 2026-09-09; at the model's force scale a
            # collapsed arm drew metre-long arrows across the whole frame.
            state["vopt"] = mujoco.MjvOption()
        self._aim(state, local)
        self.orbit.apply_to(state["cam"])
        renderer = state["renderer"]
        if self.sim is not None:
            self.sim.apply_flags(state["vopt"], renderer.scene)
        renderer.update_scene(local, camera=state["cam"], scene_option=state["vopt"])
        self.perturber.resolve(
            local, renderer.scene, state["vopt"], state["width"] / state["height"]
        )
        self.ring.write_perturbation(self.perturber.force(local))
        self.perturber.draw(renderer.scene)
        if not state["shadows"]:
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        began = time.monotonic()
        frame = renderer.render()  # HxWx3 uint8, C-contiguous
        rendered = time.monotonic()
        self._judge_shadows(state, (rendered - began) * 1000.0)
        self.sink.ship(frame, state["width"], state["height"])
        self._lane_stats(
            state, (rendered - began) * 1000.0, (time.monotonic() - rendered) * 1000.0
        )
        if self.sim is not None:
            lane = state.setdefault("lane", {})
            lane["last_render_ms"] = (rendered - began) * 1000.0
            lane["scene"] = renderer.scene  # its flags, as rendered, for the status
            status = self.sim.status(local, lane, state)
            if status is not None:
                self.sink.status(status)

    def _lane_stats(self, state: dict, render_ms: float, ship_ms: float) -> None:
        """One stderr line per LANE_STATS_EVERY_S: what the lane actually
        achieves, so a slow viewport is diagnosed from the log, not guessed."""
        stats = state.setdefault(
            "lane", {"since": time.monotonic(), "render": [], "ship": []}
        )
        stats["render"].append(render_ms)
        stats["ship"].append(ship_ms)
        elapsed = time.monotonic() - stats["since"]
        if elapsed < LANE_STATS_EVERY_S:
            return
        r = sorted(stats["render"])
        sh = sorted(stats["ship"])
        n = len(r)
        print(
            f"lane: {n} frames in {elapsed:.1f} s ({n / elapsed:.1f} fps); "
            f"render p50 {r[n // 2]:.1f} ms p90 {r[int(n * 0.9)]:.1f} ms; "
            f"ship p50 {sh[n // 2]:.2f} ms; "
            f"shadows {'on' if state['shadows'] else 'off'}; "
            f"{state['width']}x{state['height']}",
            file=sys.stderr,
            flush=True,
        )
        stats["since"] = time.monotonic()
        stats["render"].clear()
        stats["ship"].clear()

    def _judge_shadows(self, state: dict, render_ms: float) -> None:
        """Shadows stay while the measured render fits the display budget;
        past it they go, once, and the decision is logged with the number."""
        if SHADOWS_MODE != "auto" or not state["shadows"]:
            return
        times = state["render_ms"]
        times.append(render_ms)
        # Judge after the probe window, or sooner when the frames are so
        # slow that waiting for the window would itself take seconds (the
        # 20-duck flock: 274 ms/frame with shadows).
        if len(times) < SHADOW_PROBE_FRAMES and sum(times) < SHADOW_PROBE_MS:
            return
        mean = sum(times) / len(times)
        if mean > SHADOW_BUDGET_MS:
            state["shadows"] = False
            print(
                f"shadows off: render averaged {mean:.1f} ms over {len(times)} frames, "
                f"budget {SHADOW_BUDGET_MS:g} ms",
                file=sys.stderr,
                flush=True,
            )
        del times[:]  # judge again on the next window if shadows survived


def run_expert_forever(task: "object", pump: PhysicsPump) -> None:
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


def run_idle_forever(model: "mujoco.MjModel", pump: PhysicsPump) -> None:
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
    # The flock's size is a render budget, not a taste: each microduck is
    # 431,750 faces (onshape's export, 21k-face PCBs and bearings), and
    # the offscreen path draws ~7 ms per duck here (1 duck 11.5 ms, 4
    # ducks 39 ms, 20 ducks 156 ms; Apple M1 Pro, 2026-09-09). Twenty is
    # a slideshow in every viewer, MuJoCo's own included; four is a
    # parade at ~25 fps. Decimated preview meshes at onboarding would
    # give the twenty back — an asset job, not a viewer one.
    count, spacing = FLOCK_COUNT, 0.4
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


FLOCK_COUNT = 4
GAIT_HZ = 1.6  # step frequency of the parade waddle
PARADE_SPEED = 0.12  # m/s along +x, wrapping at the floor's edge
PARADE_WRAP_X = 2.4


def run_flock_parade_forever(model: "mujoco.MjModel", pump: PhysicsPump) -> None:
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


WALK = "walk"  # the RL view: N policy-driven worlds mirrored from the batched sim
WALK_SCENE_PARTS = 3  # walk:<robot>:<worlds>


def walk_scene_of(scene: str) -> tuple[str, int]:
    """`walk:<robot>:<worlds>` -> (robot, worlds); refused by name when the
    robot is missing — the RL view has no robot of its own."""
    parts = scene.split(":")
    if len(parts) != WALK_SCENE_PARTS or not parts[1] or not parts[2].isdigit():
        raise ValueError(
            f"a walk scene is {WALK}:<robot>:<worlds> (the robot's bundle name and "
            f"the number of worlds), not {scene!r}"
        )
    return parts[1], int(parts[2])


def walk_scene(
    robot: str, worlds: int, offscreen_side: int = MAX_RENDER_SIDE
) -> "mujoco.MjModel":
    """One CPU model holding `worlds` copies of the walk robot on one
    ground plane, each under a `wNN/` prefix at its grid cell (the RL
    view's mirror; rq_mjlab.walk_view fills its qpos from the batched
    sim). The batched env's world origins are already in each free
    joint's global qpos, so the copies land on their origins by the copy
    alone. Built here, not in rq_mjlab, so the render process — the
    pipeline venv, no mjlab — can build the same model. The robot's model
    is the one its bundle records (project first, then the library)."""
    from rq_pipeline.bundles.bundle import model_file_of  # noqa: PLC0415
    from rq_pipeline.bundles.locate import find_bundle  # noqa: PLC0415
    from rq_pipeline.tasks.scene import grid_of  # noqa: PLC0415

    bundle = find_bundle(robot)
    model_file = model_file_of(bundle) if bundle is not None else None
    if bundle is None or model_file is None:
        raise FileNotFoundError(
            f"no bundle {robot!r} with a model file in the project or the library"
        )
    scene, _ = grid_of(
        f"{robot}-rl-{worlds}",
        (mujoco.MjSpec.from_file(str(model_file)) for _ in range(worlds)),
        pitch=0.0,
    )
    for geom in scene.geoms:
        if geom.name == "ground":
            geom.pos[2] = 0.0  # the display grids' table offset; ducks walk at z=0
    scene.visual.global_.offwidth = offscreen_side
    scene.visual.global_.offheight = offscreen_side
    return scene.compile()


def build_scene(task_name: str) -> "tuple[object, mujoco.MjModel, str]":
    """The task (or None for the duck preview), its compiled model with
    the offscreen budget raised to the viewer's cap, and its rig name.
    Both processes build the same model from the same spec path."""
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
    return task, spec.compile(), rig


def render_on(scene: str, ring_path: str, shm_path: str | None, rig: str) -> None:
    """Render-only, for a physics process that already exists (the RL
    view's batched worlds): `scene` is `walk:<worlds>` or a model file,
    the ring was created by that process, the wire and status are the
    same."""
    if scene.startswith(f"{WALK}:"):
        model = walk_scene(*walk_scene_of(scene))
    else:
        model = mujoco.MjModel.from_xml_path(scene)
    ring = StateRing(ring_path, model, create=False)
    orbit = OrbitCamera(
        RIG_CAMERAS.get(rig or DEFAULT_CAMERA, RIG_CAMERAS[DEFAULT_CAMERA])
    )
    perturber = Perturber(model)
    sim = SimControl(model, ring, orbit)
    pump = RenderPump(model, orbit, perturber, FrameSink(shm_path), ring, sim)
    threading.Thread(
        target=_read_control_messages,
        args=(orbit, perturber, pump._fresh.set, pump.sink.shared, sim),
        daemon=True,
    ).start()
    pump.run_forever()


def stream(task_name: str, shm_path: str | None) -> None:
    """The render process: the one the Studio spawns. It creates the
    state ring, spawns the physics process on it, and renders."""
    import subprocess  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    _task, model, rig = build_scene(task_name)
    fd, ring_path = tempfile.mkstemp(prefix="studio-state-", suffix=".ring")
    os.close(fd)
    ring = StateRing(ring_path, model, create=True)
    physics_args = [
        sys.executable,
        os.path.abspath(__file__),
        task_name,
        f"--physics={ring_path}",
    ]
    if "--no-rerun" in sys.argv:
        physics_args.append("--no-rerun")
    # The child's stdin is a pipe this process never writes: when this
    # process dies, the pipe closes and the child exits on EOF — no
    # orphaned physics at 100 % of a core.
    physics = subprocess.Popen(physics_args, stdin=subprocess.PIPE)

    orbit = OrbitCamera(
        RIG_CAMERAS.get(rig or DEFAULT_CAMERA, RIG_CAMERAS[DEFAULT_CAMERA])
    )
    perturber = Perturber(model)
    sim = SimControl(model, ring, orbit)
    pump = RenderPump(model, orbit, perturber, FrameSink(shm_path), ring, sim)
    threading.Thread(
        target=_read_control_messages,
        args=(orbit, perturber, pump._fresh.set, pump.sink.shared, sim),
        daemon=True,
    ).start()
    try:
        pump.run_forever()
    finally:
        physics.terminate()
        pathlib.Path(ring_path).unlink(missing_ok=True)


def run_manual_forever(
    model: "mujoco.MjModel", data: "mujoco.MjData", pump: PhysicsPump, steps: int = 0
) -> None:
    """The human's loop (simulate's own): ctrl from the Control sliders,
    qpos edits from the Joint sliders applied with a forward pass, Run
    on or off, Step n while paused. Leaves by ResetScene (Reset, or
    manual off) — the scene's own motion resumes from its start."""
    pump.manual = True
    dt = model.opt.timestep
    pending = steps
    while True:
        inputs = pump.ring.manual_inputs()
        if inputs is not None:
            ctrl, qpos = inputs
            if not np.array_equal(qpos, data.qpos):
                data.qpos[:] = qpos
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
            data.ctrl[:] = ctrl
        if pending > 0 or not pump.ring.paused():
            mujoco.mj_step(model, data)
            pending = max(0, pending - 1)
            pump.tick(data, dt, hold_when_paused=False)
        else:
            try:
                pump.take_mail(data)
            except TakeOver as more:  # a further step while paused
                pending += more.steps
            pump.ring.publish(data, 0.0, True)
            time.sleep(0.02)


def physics_main(task_name: str, ring_path: str) -> None:
    """The physics process: the scene loop with the narrator, publishing
    into the ring the render process created; exits when its stdin
    closes (the render process is gone). Hand-overs (TakeOver, ResetScene)
    move it between the scene's own loop and the manual loop."""
    task, model, _rig = build_scene(task_name)
    ring = StateRing(ring_path, model, create=False)

    def watch_parent() -> None:
        sys.stdin.buffer.read()  # EOF when the render process dies
        os._exit(0)

    threading.Thread(target=watch_parent, daemon=True).start()
    pump = PhysicsPump(model, narrator_for(model, task_name), ring)
    manual: tuple[mujoco.MjData, int] | None = None
    while True:
        try:
            if manual is not None:
                data, steps = manual
                run_manual_forever(model, data, pump, steps)
            elif task is not None and task_name in TASKS_WITH_EXPERTS:
                run_expert_forever(task, pump)
            elif task_name == DUCK:
                run_flock_parade_forever(model, pump)
            else:
                run_idle_forever(model, pump)
        except TakeOver as hand:
            ring.seed_manual(hand.data.ctrl.copy(), hand.data.qpos.copy())
            manual = (hand.data, hand.steps)
        except ResetScene as reset:
            pump.manual = False
            pump.time_offset += pump.last_sim_time
            if manual is not None and reset.keyframe >= 0:
                data = manual[0]
                mujoco.mj_resetDataKeyframe(model, data, reset.keyframe)
                mujoco.mj_forward(model, data)
                manual = (data, 0)
            else:
                manual = None  # the scene's own loop, from its start


if __name__ == "__main__":
    arguments = [a for a in sys.argv[1:] if not a.startswith("--")]
    shm = None
    for flag in sys.argv[1:]:
        if flag.startswith("--shm="):
            shm = flag.removeprefix("--shm=")
        elif flag.startswith("--shadows="):
            SHADOWS_MODE = flag.removeprefix("--shadows=")
            if SHADOWS_MODE not in ("on", "off", "auto"):
                sys.exit(f"--shadows must be on, off or auto, not {SHADOWS_MODE!r}")
    physics_ring = None
    model_path = None
    ring_path = None
    rig: str | None = None  # the scene's own rig, or the default camera
    for flag in sys.argv[1:]:
        if flag.startswith("--physics="):
            physics_ring = flag.removeprefix("--physics=")
        elif flag.startswith("--scene="):
            model_path = flag.removeprefix("--scene=")
        elif flag.startswith("--ring="):
            ring_path = flag.removeprefix("--ring=")
        elif flag.startswith("--rig="):
            rig = flag.removeprefix("--rig=")
    if model_path and ring_path:
        render_on(model_path, ring_path, shm, rig)
        raise SystemExit(0)
    task_name = arguments[0] if arguments else DEFAULT_TASK
    if task_name != DUCK and task_name not in BUILDERS:
        sys.exit(f"unknown task {task_name!r}; one of {sorted([*BUILDERS, DUCK])}")
    if physics_ring:
        physics_main(task_name, physics_ring)
    else:
        stream(task_name, shm)
