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

Wire format, stdout, per frame, flushed immediately:

    width  : u32 little-endian
    height : u32 little-endian
    pixels : width * height * 3 raw RGB8 bytes, row-major, no padding

No length-of-message prefix beyond that — width/height give the reader
everything needed to know how many pixel bytes follow. A reader that reads
short must retry; this script does not resend a partial frame.

Wire format, stdin, per camera+size update — 3 little-endian f32 then 2
little-endian u32, no header (fixed size, nothing to frame):

    azimuth   : degrees, MuJoCo's own convention
    elevation : degrees
    distance  : metres from `LOOKAT`
    width     : pixels the renderer should produce
    height    : pixels the renderer should produce

`LOOKAT` itself is not sent — this is orbit-and-zoom, not pan, matching what
studio-shell's viewport currently offers. width/height let the render
resolution track the viewer's actual panel size instead of staying a fixed
1024x576 letterboxed into whatever shape the panel is; the render loop
recreates `mujoco.Renderer` only when they actually change, since the
`Renderer` object itself is fixed-size once constructed. A dedicated thread
reads updates off stdin so a slow/absent controller never blocks rendering;
the render loop just reads whatever the thread last decoded.
"""

import math
import os
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
from rq_pipeline.tasks.registry import tasks  # noqa: E402

BUILDERS = {entry.name: entry for entry in tasks().values()}
# Tasks whose accepted scripted expert drives the sim for real; anything
# else gets the idle sinusoid. Grows as experts land (the SO-101 ones
# live on the rl-engineering branch until its merge).
TASKS_WITH_EXPERTS = ("kitting",)
DEFAULT_TASK = "kitting"
WIDTH, HEIGHT = 1024, 576
TARGET_HZ = 30.0

# Free-camera framing per rig, seeded from each rig's own viewer tools
# (show-aloha2's frame_viewer; the SO-101 numbers tuned by eye earlier) —
# a starting pose the operator immediately corrects by dragging.
RIG_CAMERAS = {
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
NARRATE_HZ = 10.0


class PhysicsNarrator:
    """The sim's state into Rerun, per step, on the `sim` timeline."""

    def __init__(self, model: "mujoco.MjModel", task_name: str) -> None:
        import rerun as rr  # noqa: PLC0415 - viz extra
        import rerun.blueprint as rrb  # noqa: PLC0415

        from _rig3d import RigMirror  # noqa: PLC0415

        self.rr = rr
        rr.init(f"robotiq-sim-{task_name}", spawn=False)
        rr.connect_grpc()  # default 127.0.0.1:9876 — the Studio itself
        self.mirror = RigMirror(model, model_colors=True)

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
        ]
        self.actuators = [
            (model.actuator(a).name or f"actuator{a}", a) for a in range(model.nu)
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
        timeline must not."""
        rr = self.rr
        rr.set_time("sim", duration=sim_time)
        self.mirror.log(data)
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
CAMERA_UPDATE_BYTES = 20  # 3 x f32, 2 x u32


class OrbitCamera:
    """The latest azimuth/elevation/distance/width/height from stdin,
    lock-protected — read by the render loop, written by
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

    def _set(
        self, azimuth: float, elevation: float, distance: float, width: int, height: int
    ) -> None:
        with self._lock:
            self.azimuth, self.elevation, self.distance = azimuth, elevation, distance
            self.width, self.height = width, height


def _read_camera_updates(camera: OrbitCamera) -> None:
    stdin = sys.stdin.buffer
    while True:
        raw = stdin.read(CAMERA_UPDATE_BYTES)
        if len(raw) < CAMERA_UPDATE_BYTES:
            return  # controller closed stdin — keep rendering at the last pose
        camera._set(*struct.unpack("<fffII", raw))


class RenderPump:
    """Everything one observed physics step needs: resize, orbit, render,
    frame out, narration — shared by the expert's `on_control` hook and
    the no-expert idle loop, so both paths behave identically."""

    def __init__(self, model: "mujoco.MjModel", orbit: OrbitCamera, narrator) -> None:
        self.model = model
        self.orbit = orbit
        self.narrator = narrator
        self.width, self.height = WIDTH, HEIGHT
        self.renderer = mujoco.Renderer(model, height=self.height, width=self.width)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat = list(orbit.lookat)
        self.out = sys.stdout.buffer
        self.last_rendered = 0.0
        self.last_narrated = 0.0
        # Episodes reset `data.time` to zero; the narration timeline must
        # not rewind with them, so it runs on an offset the episode loop
        # advances at each boundary.
        self.time_offset = 0.0

    def tick(self, data: "mujoco.MjData", pace_seconds: float) -> None:
        """Observe one step: narrate and render (each rate-limited), then
        sleep toward real time — `pace_seconds` is how much simulated
        time this step advanced."""
        now = time.monotonic()
        if self.narrator is not None and now - self.last_narrated >= 1.0 / NARRATE_HZ:
            self.last_narrated = now
            self.narrator.log(data, self.time_offset + data.time)

        if now - self.last_rendered >= 1.0 / TARGET_HZ:
            self.last_rendered = now
            want_width, want_height = self.orbit.size()
            # Clamp to the compiled framebuffer no matter what the viewer
            # asked: `mujoco.Renderer` refuses (raises) beyond it, and an
            # exception here kills the whole stream.
            want_width = min(want_width, int(self.model.vis.global_.offwidth))
            want_height = min(want_height, int(self.model.vis.global_.offheight))
            if (want_width, want_height) != (self.width, self.height):
                # `Renderer` is fixed-size once constructed — a size
                # change means a rebuild, not a resize.
                self.renderer.close()
                self.width, self.height = want_width, want_height
                self.renderer = mujoco.Renderer(
                    self.model, height=self.height, width=self.width
                )
            self.orbit.apply_to(self.cam)
            self.renderer.update_scene(data, camera=self.cam)
            frame = self.renderer.render()  # HxWx3 uint8, C-contiguous
            self.out.write(struct.pack("<II", self.width, self.height))
            self.out.write(frame.tobytes())
            self.out.flush()

        # Pace toward real time: sleep off whatever of this step's
        # simulated duration wall time hasn't already consumed.
        remaining = pace_seconds - (time.monotonic() - now)
        if remaining > 0:
            time.sleep(remaining)


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
        pump.time_offset += protocol.steps * pump.model.opt.timestep
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


def stream(task_name: str) -> None:
    entry = BUILDERS[task_name]
    task = entry.build()
    # Raise (never lower) the offscreen budget to the viewer's cap — see
    # MAX_RENDER_SIDE's comment for the measured failure without this.
    task.spec.visual.global_.offwidth = max(
        task.spec.visual.global_.offwidth, MAX_RENDER_SIDE
    )
    task.spec.visual.global_.offheight = max(
        task.spec.visual.global_.offheight, MAX_RENDER_SIDE
    )
    model = task.spec.compile()

    orbit = OrbitCamera(RIG_CAMERAS.get(entry.rig, RIG_CAMERAS["so101"]))
    threading.Thread(target=_read_camera_updates, args=(orbit,), daemon=True).start()
    pump = RenderPump(model, orbit, narrator_for(model, task_name))

    if task_name in TASKS_WITH_EXPERTS:
        run_expert_forever(task, pump)
    else:
        run_idle_forever(model, pump)


if __name__ == "__main__":
    task_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TASK
    if task_name not in BUILDERS:
        sys.exit(f"unknown task {task_name!r}; one of {sorted(BUILDERS)}")
    stream(task_name)
