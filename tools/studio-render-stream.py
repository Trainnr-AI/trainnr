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

BUILDERS = {
    entry.name: entry.build for entry in tasks().values() if entry.rig == "so101"
}
WIDTH, HEIGHT = 1024, 576
TARGET_HZ = 30.0

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

    def log(self, data: "mujoco.MjData") -> None:
        rr = self.rr
        rr.set_time("sim", duration=data.time)
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

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.azimuth = 90.0
        self.elevation = -20.0
        self.distance = 1.0
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


def stream(task_name: str) -> None:
    task = BUILDERS[task_name]()
    # Raise (never lower) the offscreen budget to the viewer's cap — see
    # MAX_RENDER_SIDE's comment for the measured failure without this.
    task.spec.visual.global_.offwidth = max(
        task.spec.visual.global_.offwidth, MAX_RENDER_SIDE
    )
    task.spec.visual.global_.offheight = max(
        task.spec.visual.global_.offheight, MAX_RENDER_SIDE
    )
    model = task.spec.compile()
    data = mujoco.MjData(model)
    width, height = WIDTH, HEIGHT
    renderer = mujoco.Renderer(model, height=height, width=width)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat = [0.0, 0.0, 0.15]  # roughly the SO-101's own working height

    orbit = OrbitCamera()
    threading.Thread(target=_read_camera_updates, args=(orbit,), daemon=True).start()

    narrator = narrator_for(model, task_name)
    out = sys.stdout.buffer

    # A visible, harmless motion — not the task's real controller — so a
    # viewer can tell "streaming" apart from "stalled" at a glance. Every
    # actuator gets the same slow sinusoid; amplitude is small enough that
    # a small-hobby-servo model's joint limits are never in question.
    period = 0.0
    dt = model.opt.timestep
    frame_interval = 1.0 / TARGET_HZ
    narrate_interval = 1.0 / NARRATE_HZ
    last_narrated = 0.0

    while True:
        started = time.monotonic()
        period += dt
        data.ctrl[:] = 0.2 * math.sin(period)
        mujoco.mj_step(model, data)
        if narrator is not None and started - last_narrated >= narrate_interval:
            last_narrated = started
            narrator.log(data)

        want_width, want_height = orbit.size()
        # Clamp to the compiled framebuffer no matter what the viewer
        # asked: `mujoco.Renderer` refuses (raises) beyond it, and an
        # exception here kills the whole stream. Belt to the budget
        # raise's braces — even a viewer with a different cap degrades to
        # a smaller render instead of a frozen viewport.
        want_width = min(want_width, int(model.vis.global_.offwidth))
        want_height = min(want_height, int(model.vis.global_.offheight))
        if (want_width, want_height) != (width, height):
            # `Renderer` is fixed-size once constructed — a size change
            # from the viewer means throw it away and build a new one, not
            # resize it in place.
            renderer.close()
            width, height = want_width, want_height
            renderer = mujoco.Renderer(model, height=height, width=width)

        orbit.apply_to(cam)
        renderer.update_scene(data, camera=cam)
        frame = renderer.render()  # HxWx3 uint8, C-contiguous

        out.write(struct.pack("<II", width, height))
        out.write(frame.tobytes())
        out.flush()

        elapsed = time.monotonic() - started
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)


if __name__ == "__main__":
    task_name = sys.argv[1] if len(sys.argv) > 1 else "block_stack"
    if task_name not in BUILDERS:
        sys.exit(f"unknown task {task_name!r}; one of {sorted(BUILDERS)}")
    stream(task_name)
