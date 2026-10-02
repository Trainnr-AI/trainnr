# 84 — The MuJoCo viewport as its own package

*2026-10-03. Prakhar: "can we keep the mujoco viewer as a separate module
repo that can be just imported into the app and it can be built and
improved separately as a package." docs/80 §3's rule is satellites only
where the lifecycle differs; the viewport's does: a MuJoCo viewport
widget for egui apps, with a simulation streaming into it from a Python
process, is useful to any egui app and to any MuJoCo user, and it moves
on its own cadence (rendering, the drawer, the follow rules, the ring).
This is the plan; nothing is cut yet.*

## 1. What the viewport is today

Two halves and a wire between them.

| Half | Where | Size | Depends on |
|---|---|---|---|
| the widget (Rust) | `crates/trainnr-desktop/src/viewport.rs` (the shared-memory reader, the texture, the camera, the resize and scale), `simulator.rs` (the transport bar, the overlays, the Inspect drawer, the follow rules, the worlds row), `keys.rs` (WASD and the shortcuts) | 2,650 lines | egui, re_ui's icons, `spawn.rs` (process helpers: a group, kill the tree) |
| the stream (Python) | `tools/studio-render-stream.py` (MuJoCo, the policy, the camera, the frame ring, the status, the physics ring for the viewer twin) | 2,365 lines | `trainnr` (the task registry for the scenes, `viz` for the mirror into Rerun), mujoco, onnxruntime for a deployment |
| the wire | a shared-memory ring of RGB frames (`--shm`), one wake-up byte per frame on stdout (the frame token), a JSON status per tick (the status token), a second ring for the physics twin (`--physics`), commands on stdin (camera, resize, run/pause/step/reset, inputs) | docs/35 §6 | |

The app touches the widget through five calls: spawn a scene, show
(returns the picture's rect), the transport bar, the overlays and the
drawer, stop. Nothing in the widget reads the app's model; the scene
list it offers comes in as an argument.

## 2. The cut

- **`trainnr-viewport`**, one repository, two packages with one version:
  - the Rust crate `trainnr-viewport`: `Viewport` (the reader and the
    picture), `transport`, `overlays`, `drawer`, `shortcuts`, the
    `Action` enum, and a `Launcher` trait the app implements to start
    the stream (so the crate spawns nothing itself and the app keeps its
    process rules); the ring and token protocol as a documented module
    with its version constant.
  - the Python package `trainnr-viewport` (`import trainnr_viewport`):
    the stream as a library (`Stream(scene, size, shm, physics)`), the
    ring writer, the status writer, and a `__main__` that is today's
    script; scenes arrive through a small protocol (`SceneSource`: build
    a model, step, the policy if any) that `trainnr` implements, so the
    package depends on mujoco and numpy only and `trainnr` depends on it,
    not the other way round. The mirror into Rerun stays in `trainnr`
    (it is the loop's, not the viewport's).
- **The app** (`crates/trainnr-desktop`) depends on the crate by version
  and on the Python package through `trainnr`'s own dependency; the
  Simulator page keeps its layout, the empty-state card and the
  heartbeat fields. `spawn.rs` implements `Launcher`.
- **The contract that must not move without a version bump**: the ring
  layout, the two tokens, the status JSON, the stdin command lines, the
  `--shm`/`--physics`/`--project` arguments. They become
  `trainnr_viewport.protocol` on one side and `trainnr_viewport::protocol`
  on the other, with the same `PROTOCOL = 1` and a test on each side
  that reads the other's fixture.

## 3. What it buys, what it costs

Buys: the widget improves on its own cadence and pull requests (rendering
quality, the drawer, the half-size rule, a Windows build of just the
widget's demo), a demo app in the repo that is the smallest MuJoCo
viewport anyone can run, and a second consumer is possible (a notebook,
another egui app). Costs: two version pins to keep in step (the desktop's
Cargo.toml and `trainnr`'s pyproject), a release step the monorepo did
not have, and the first week of cutting, which is mostly moving files
and naming the contract that exists already.

## 4. Order of work

1. Name the contract in place: `protocol` modules on both sides inside
   the monorepo, with the cross-fixture tests; the `Launcher` trait and
   the `SceneSource` protocol; the widget stops reaching into `spawn.rs`.
2. Move: the crate under `crates/trainnr-viewport` and the package under
   `trainnr-viewport/` in the monorepo first (a path dependency), the
   desktop and `trainnr` consuming them; every gate green.
3. Extract: the repository `Trainnr-AI/trainnr-viewport` with the files'
   history (the rig recipe, `tools/export/rig/`), the monorepo switching
   to a versioned dependency; the demo app and its README.
4. Release with the first tag; the desktop pins it.

Decided by the operator; steps 1 and 2 are a day inside the monorepo and
carry no risk; step 3 is the point of no return for the file history.
