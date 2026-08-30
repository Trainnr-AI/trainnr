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
import struct
import sys
import threading
import time

from _lab import bootstrap

bootstrap()

import mujoco  # noqa: E402
from rq_pipeline.tasks.registry import tasks  # noqa: E402

BUILDERS = {
    entry.name: entry.build for entry in tasks().values() if entry.rig == "so101"
}
WIDTH, HEIGHT = 1024, 576
TARGET_HZ = 30.0

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
    model = task.spec.compile()
    data = mujoco.MjData(model)
    width, height = WIDTH, HEIGHT
    renderer = mujoco.Renderer(model, height=height, width=width)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat = [0.0, 0.0, 0.15]  # roughly the SO-101's own working height

    orbit = OrbitCamera()
    threading.Thread(target=_read_camera_updates, args=(orbit,), daemon=True).start()

    out = sys.stdout.buffer

    # A visible, harmless motion — not the task's real controller — so a
    # viewer can tell "streaming" apart from "stalled" at a glance. Every
    # actuator gets the same slow sinusoid; amplitude is small enough that
    # a small-hobby-servo model's joint limits are never in question.
    period = 0.0
    dt = model.opt.timestep
    frame_interval = 1.0 / TARGET_HZ

    while True:
        started = time.monotonic()
        period += dt
        data.ctrl[:] = 0.2 * math.sin(period)
        mujoco.mj_step(model, data)

        want_width, want_height = orbit.size()
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
