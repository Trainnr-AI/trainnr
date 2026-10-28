# 78. mujoco-rs 7.0.0: what trainnr can use it for

Research date: **2026-10-28**. Status: **current**. The question was
whether the Rust binding `mujoco-rs` can serve trainnr as an external
library now, and for what: the Studio's simulator, its robot pages, model
editing, or anything else. It updates
[74 §5](74-the-simulation-window.md#5-mujoco-rs-601-the-supplied-bundle),
which read 6.0.1 on 2026-09-09.

**Method.** github.com/davidhozic/mujoco-rs cloned at `506268a`
(2026-10-26, `version = "7.0.0+mj-3.17.0"`), about 42,000 lines of Rust.
Five readers each read one slice line by line: building and versions;
the physics API; model editing; offscreen rendering; the viewer. The
claims the conclusion rests on were checked again against the clone and
against the Studio's `Cargo.lock`. Then three small probes were built
outside the clone and run on this machine (WSL2 + WSLg, d3d12): §9 has
what they measured, §10 what is still open. Paths below are the crate's, cited as
file:line at `506268a`; trainnr's own paths say so.

## 1. The answer

| Use | Verdict | Why, in one line |
|---|---|---|
| The Studio's simulator rendered in-process | **No** | macOS and Windows cannot: winit allows one event loop per process and eframe owns it. On Linux it works, and measured the same frame cost as the stream's Python renderer (§9). |
| The crate's viewer, in or beside the Studio | **No** | Its own window, its own event loop, an egui_glow painter, a private UI. |
| Physics in-process (reading a model, kinematics for replay, stepping) | **Not now; the one worth revisiting** | Needs no GL and no fork and adds no crate, but the engine would be 3.17 against the pipeline's 3.11, and a MuJoCo error ends the app. |
| Model editing (MjSpec) in the Studio | **No** | A second implementation of what Python composes, on a different engine. |
| A read-only preview of a picked MJCF | **Not through this crate** | A Python helper on the pinned engine gives the same list without a second engine. |
| The viewer's UI as a reference design | **Yes** | Six controls worth having on the Studio's side, over the existing stream (§8). |

The 2026-09 reason for the shared-memory stream still holds, and two new
reasons join it: winit's one-loop rule, and crash containment.

## 2. The crate

| Fact | Value | Where |
|---|---|---|
| Binding | MuJoCo **3.17.0** exactly; `mjVERSION_HEADER = 3017000` | `Cargo.toml`, `src/mujoco_c.rs:46` |
| Licence | MIT OR Apache-2.0; MuJoCo is Apache-2.0 | `LICENSE:1-44` |
| Rust | MSRV 1.95, edition 2024 (the Studio pins 1.97.1) | `Cargo.toml:7,15` |
| Authors | 319 of 321 commits from one maintainer | `git shortlog -sne --all` |
| Releases | 7 majors in 14 months, 4 in the last 6; every MuJoCo bump is a major; old majors get no fixes (6.0.x and 6.1.x: no commits after their tags) | `git tag` |
| CI | Ubuntu and Windows, `cargo test --lib` only (no integration tests, doctests, viewer, clippy or Miri); no macOS runner | `.github/workflows/tests.yml` |
| Unsafe | 753 `unsafe` blocks, 341 SAFETY comments; most changelog entries since 5.0 close out-of-bounds reads reachable from safe code | changelogs 5.0–7.0 |

**Features.** With none, the crate depends on `paste`, `bytemuck` and
`log` only (`Cargo.toml:194-210`), and the Studio's lock already has all
three at the same versions: a physics-only mujoco-rs adds no crate to the
Studio but itself. `renderer` adds glutin; `viewer` adds glutin, winit
and glutin-winit; `viewer-ui` adds egui, egui-winit and egui_glow 0.36.
The Studio's lock already carries winit 0.30.13, glutin 0.32.3 and
egui_glow 0.36.2 through eframe, so cargo would unify them, and on macOS
the required `glutin` git fork would replace eframe's glutin too, which
`deny.toml`'s `unknown-git = "deny"` refuses.

**Linking.** The build script (`build.rs:285-561`) tries, in order, `MUJOCO_STATIC_LINK_DIR`,
`MUJOCO_DYNAMIC_LINK_DIR`, pkg-config (`exactly_version`, `build.rs:389`)
and, only with the `auto-download-mujoco` feature, a download from
MuJoCo's GitHub releases at build time (`build.rs:435-560`), checked
against a hash fetched from the same release rather than one pinned in
the crate. Off by default; a default build makes no network call.
Dynamic linking wants a file named exactly `libmujoco.so`
(`build.rs:342-348`) and emits no rpath. The Python wheel ships
`libmujoco.so.3.11.0` (SONAME the same), so a symlink named
`libmujoco.so` would let the Studio link the very library the pipeline
uses, if the binding matched it.

## 3. The version wall

**Checked at run time, exactly.** `assert_mujoco_version()`
(`src/util.rs:1215-1224`) asserts `mj_version() == mjVERSION_HEADER` and
panics otherwise. It runs in every model and spec constructor (ten call
sites in `src/wrappers/mj_model.rs:456` onward and
`src/wrappers/mj_editing.rs:271` onward). A call that reaches MuJoCo
before any model load (an `mju_*` utility, a scene, the log hook) is not
guarded; with struct layouts that differ between versions, that is
undefined behaviour rather than a panic.

**The releases skip 3.11.** 5.0.0 bound 3.9.0, 6.0.0 to 6.1.0 bound
3.12.0, 7.0.0 binds 3.17.0. Between them, on `main` and never tagged:
`657ebd3` (3.10.0, 2026-07-09) and **`88f241d`** ("Pre-merge MuJoCo
3.11.0 FFI bindings", 2026-08-20, `version = "6.0.0+mj-3.11.0"`,
`mjVERSION_HEADER = 3011000`). It predates every safety fix in 6.0.0,
6.1.0 and 7.0.0 and the attach API.

**What 3.11 → 3.17 changed in the binding**, counted from
the generated binding (`src/mujoco_c.rs:46` onward) at `88f241d` against `506268a`:

| Step | Functions | Struct fields | Notable |
|---|---|---|---|
| 3.11 → 3.12 | +2 (`mjs_setToPID`, `mjv_camera2GLCamera`) | +7 / −3 | `mjtDyn` and `mjtGain` renumbered; `mjsActuator` and `mjsLight` layouts changed |
| 3.12 → 3.17 | +15 / −5 | +41 / −6 | mjData +21 (`efm_*`, flex hessian); 5 enums changed |

In the editing layer alone: 5 new functions, 1 changed signature
(`mjs_addTexture`), 21 changed fields across 11 of 31 `mjs*` structs.

**A 3.11 fork is days, not weeks, and then it is ours.** Cheapest: 6.1.0
with the 3.12 FFI commit (`387ad06`) reverted and the few 3.12 wrappers
removed; from 7.0.0, also undo 3.12 → 3.17, which is tangled with safety
work. Either way trainnr would own a fork with no upstream to follow,
and regenerating (`ffi-regenerate`, `build.rs:102-127`) needs the headers
in a git checkout of the maintainer's MuJoCo fork (`.gitmodules`), or the
wheel's `include/` copied into place.

**trainnr already runs two engines.** The core pins `mujoco~=3.11.0`
(`trainnr/pyproject.toml`); the trainer overrides to **3.13.0**
(`trainnr-mjlab/pyproject.toml`, `override-dependencies`). Neither is a
version mujoco-rs binds.

## 4. Physics only

**The API is thin and sound.** Each loader is one C call returning
`Result<MjModel, MjModelError>`: `from_xml`, `from_xml_vfs`,
`from_xml_string` (through a private VFS), `from_buffer`, `from_mjb`
(`src/wrappers/mj_model.rs:437-540`). MuJoCo's message comes back in a
100-byte buffer (`ERROR_BUF_LEN`, `src/util.rs:17`), so it is cut at 99
characters. `MjData<M>` borrows or owns its model through `M` (`&`,
`Box`, `Rc`, `Arc`, a lock guard). Stepping and the pipeline stages are
one-line wrappers (`src/wrappers/mj_data.rs:239-672`); every C array is
a slice accessor (`qpos()`, `ctrl_mut()`, `sensordata()`, `xpos()`);
named access (`data.joint("hip")`, `.actuator`, `.sensor`, …) returns
an info whose view panics on a model of another layout, with `try_view`
as the `Result` form (`src/util.rs:564-594`). Reading a robot is short:

```rust
let m = MjModel::from_xml(path)?;
for j in 0..m.njnt() as usize {
    let name = m.id_to_name(MjtObjBase::mjOBJ_JOINT, j).unwrap_or("");
    let [lo, hi] = m.jnt_range()[j];
    println!("{name} {:?} [{lo}, {hi}] limited={}", m.jnt_type()[j], m.jnt_limited()[j]);
}
```

**Threads.** `MjModel` is `Send + Sync`; `MjData<M>` is `Send`/`Sync`
when `M` is (`src/wrappers/mj_model.rs:422-423`,
`src/wrappers/mj_data.rs:66-67`); mutating a model takes `&mut`, so many
`MjData` on one shared model across threads type-checks. No test runs
threads.

**Cost.** `step` and the accessors allocate nothing; Jacobians,
`state()` and some sensor reads return a `Vec`. `ffi()` and an `unsafe`
`ffi_mut()` reach the raw structs, and the whole binding module is
public.

**No sharing with Python's MuJoCo.** The pyo3 example
(`examples/standalone/pyo3_application/src/lib.rs:23-38`) loads its own
model and never touches Python's `mujoco`; adopting a foreign
`mjModel*` is `pub(crate)` (`src/wrappers/mj_model.rs:559`), and the
3.11 and 3.17 structs differ anyway. What does carry across is the
state vector: the `mjtState` bits are the same in both versions, so
`read_state_into` / `set_state` could read the physics ring the stream
already writes, for a model both versions compile to the same sizes.

**A MuJoCo error ends the process.** MuJoCo's default error handler
exits; the crate's logging hook calls `process::exit(1)` on an error
(`src/logging.rs:166-172`) because "the C code continues in a broken
state if it does" return. In a child process that is a contained
failure; in the Studio it closes the window. Without the hook MuJoCo
writes `MUJOCO_LOG.TXT` to the working directory.

**What it could do, ranked, if the engine matched:** (1) read a robot's
joints, actuators and sensors from its MJCF for the robot page and the
simulator's panels without starting the stream (note `nu` can exceed
the actuator count: an orientation servo takes three controls, which
the Studio's one-range-per-actuator `SimActuator` assumes away);
(2) kinematics for replay: write a recording's `qpos`, run
`forward_kinematics`, log body transforms into the embedded Rerun, with
no Python and no GL; (3) a pass/fail check of a picked MJCF before the
agent onboards it. Each needs the pipeline's engine to be the one the
crate binds, or must be labelled a preview that the Python record
overrides.

## 5. Model editing (MjSpec)

**Coverage is wide.** Parse from a file, a VFS or a string
(`src/wrappers/mj_editing.rs:328`, `:339`, `:366`; the string form has
no assets argument, so assets go through an `MjVfs`), create from
scratch, add every element kind on the spec and on bodies, find by name
for 24 kinds, iterate flat or by subtree, delete (`unsafe`), attach by
deep copy or by reference with a prefix and suffix
(`src/wrappers/mj_editing/traits.rs:477`, `:523`), compile with
MuJoCo's message on failure, save to XML or encode. Every Python use in
trainnr today has a counterpart: `tasks/scene.py`'s many-worlds grid,
the components' `attach_body`, `scenes/stage.py`'s edits and `to_xml`,
`robot/onboarding.py`'s census.

**Gaps.** `mj_recompile` and `mj_copyBack` are not wrapped (the nearest
is compile, then `MjData::swap_model` on a "compatible" model), nor
`mjs_makeMesh`, a parent accessor, or plugin attributes.

**Safety.** Handles are zero-sized `#[repr(C)]` overlays on MuJoCo's own
structs, borrowed from the spec, `!Send`, `!Sync`, `!Unpin`
(`src/wrappers/mj_editing/utility.rs:614-627`); `MjSpec` is neither
`Send` nor `Sync`. `compile` is safe but documented as able to crash on
invalid values (`src/wrappers/mj_editing.rs:494-500`). Names with an
interior NUL panic; a duplicate name in `with_name` panics.

**Fit.** The bundle hash and the audit stay in Python, so a Rust
composition would be a second implementation to keep equal, on another
engine. The one narrow use, an instant list of a picked MJCF's bodies,
joints and actuators in the "Add robot" form, needs no compile; a small
Python helper on the pinned engine gives the same list and keeps the
files-and-processes rule.

## 6. Offscreen rendering

**Linux: EGL device platform, headless.** `MjRenderer` enumerates EGL
devices and takes the first that gives a GL 2.0 compatibility context
on a pbuffer (`src/renderer/egl.rs:26-95`); libEGL is loaded at run time.
This is the same mechanism Python's `mujoco.egl` uses, which already
works under WSLg with `GALLIUM_DRIVER=d3d12` (`trainnr/wsl.env`); it
worked there on the first try (§9). Pixels come back by a synchronous
`mjr_readPixels` into buffers allocated once, then a CPU flip
(`src/vis_common.rs:96-103`). RGB and depth only.

**macOS and Windows: blocked in the Studio.** There the renderer's only
backend is a hidden winit window (`src/renderer/universal.rs:134`), and
winit 0.30 refuses a second event loop in a process
(`EventLoopError::RecreationAttempt`, winit 0.30.13
`src/event_loop.rs:118-119`), the same winit the Studio's eframe already
holds. This holds with or without the glutin fork.

**What it would and would not change.** It would remove the render
process, the frame ring and the wake-up byte: one copy, one file read
and one wake-up per frame. It would not remove the physics process, the
GPU→CPU→GPU round trip (no zero-copy route into wgpu exists in the
crate), or the render itself: the same `mjr_render` and `glReadPixels`
on the same driver, which measured the same in both (§9). Rendering on
3.17 while physics stays on 3.11 means
compiling the model twice, on two engines, and posing the 3.17 one from
the 3.11 state every frame.

**Limits of the API.** No `scene_mut`, so render flags (shadows,
segmentation, the Studio's `send_rnd` flags) are unreachable without
building the context by hand; the perturbation is drawn as default
(`src/renderer.rs:628`); no resize; a width above the model's
`offwidth` is documented as not allowed but not checked
(`src/renderer.rs:147-157`).

## 7. The viewer

It cannot run in the Studio's window: `build_passive` creates its own
`EventLoop` (`src/viewer.rs:2022`), it must stay on the main thread
(`src/viewer.rs:759-760`), it paints its egui with egui_glow on a GL 2.0
compatibility context it keeps (`src/winit_gl_base.rs:129-132`), and its
UI (`mod ui`, private, `src/viewer.rs:43-44`) is one closure of about
1,270 lines (`src/viewer/ui.rs:480-1746`) typed on the crate's FFI
structs. `add_ui_callback` gives a host the `egui::Ui` and the shared
state, but not the simulation data, the camera or the selection.

Noted in passing: a lost-update race between the UI's option edits and
the physics thread's `sync_model_opt` (`src/viewer/ui.rs:589-595` and
`:833-837`); held keys repeat (no `KeyEvent.repeat` check); the render
loop busy-polls with vsync off by default; MuJoCo's C++ `simulate` is
bound separately (`src/cpp_viewer.rs:21-34`) through GLFW and a static
MuJoCo built from the maintainer's fork.

## 8. What it means for trainnr (ours)

**Keep the stream.** The simulator stays a Python process on the
pipeline's engine, with frames over shared memory. The reasons are now
four: no release binds the pinned engine; macOS needs a glutin fork that
`deny.toml` refuses; macOS and Windows cannot host a second winit event
loop beside eframe; and a MuJoCo error ends the process it runs in,
which today is the stream, not the app.

**Take the viewer's controls, on our side of the wire.** Each is a
tagged stdin message plus a status field, so the Python side keeps every
camera and physics fact:

1. Mouse pan: right-drag, Shift for the horizontal plane
   (`src/viewer.rs:1852-1853`); the Studio's `viewport.rs` says "No pan
   yet".
2. Double-click to look at a point and to track a body
   (`src/viewer.rs:1899-1937`), and a camera menu: free, track a body,
   the model's own cameras.
3. Label and Frame menus (`src/viewer/ui.rs:903-931`): body names and
   site frames on the picture.
4. Free and ball joints in the Joints tab, numeric with the quaternion
   kept normalised (`src/viewer/ui.rs:1591-1621`): the floating base of a
   legged robot.
5. "Copy camera as MJCF" (`src/viewer/ui.rs:936-969`).
6. The arena memory share beside the real-time chip.

**Revisit when** a mujoco-rs release binds the engine the pipeline pins
(watch the `+mj-` suffix), or the pin itself moves to a version a release
binds, through a re-measured record as AGENTS.md requires. Then the case
to make is physics-only (§4), on all three systems, in a child process
so a MuJoCo error stays contained.

## 9. Measured, 2026-10-28

Three probe crates, each depending on the clone by path, built with the
toolchain here and no feature beyond the one named: physics only (no
features), `renderer`, and `viewer-ui`. Linked against MuJoCo's own
3.17.0 Linux release (its SHA-256 matched the `.sha256` published beside
it) through `MUJOCO_DYNAMIC_LINK_DIR`, and against the trainnr venv's
`libmujoco.so.3.11.0` through a symlink. WSL2 + WSLg, an RTX GPU through
Mesa d3d12; single runs repeated two or three times, so these are
indications for this machine, not records.

**Against the pipeline's 3.11 library it refuses, cleanly.** The first
model load panics with "linked MuJoCo version value (3011000) does not
match expected version value (3017000)", exit 101. A version string read
before it still worked.

**Physics: the binding costs nothing; the engine version is what moves.**
10,000 steps from the initial state with zero controls, µs per step,
three runs each:

| Model | Rust, 3.17 | Python, 3.11 (`trainnr`) | Python, 3.13 (`trainnr-mjlab`) |
|---|---|---|---|
| Go2 scene (nq 19, nu 12) | 23.9–24.6 | 31.6–34.1 | 25.4–25.6 |
| ALOHA 2 scene (nq 16, nu 14) | 190.8–193.3 | 160.6–162.8 | 209.5–211.5 |

The Python figures include a Python call per step. Loading took 100–140
ms in either language. The counts (nq, nv, nu, nbody, njnt, nsensor) were
the same in both versions for both models.

**The engines do not agree on ALOHA 2.** After those 20 simulated
seconds, the Go2's first seven `qpos` matched to four decimals across
3.11, 3.13 and 3.17. ALOHA 2's did not: its fifth joint, for one, ended
at 0.0481 (3.11), −0.0026 (3.13) and 0.0067 (3.17). The same file on
three engines gives three trajectories: what AGENTS.md means by the
engine being the instrument.

**Offscreen rendering: the same cost as the stream's renderer.** The
Go2 scene at 640×480, scene update plus render plus readback per frame,
after 20 warm-up frames, two runs each: mujoco-rs 67.7–73.1 ms mean,
Python `mujoco.Renderer` (3.11, the stream's) 68.2–71.6 ms. Both reached
the GPU through EGL on d3d12. Python's EGL logged an error only while
freeing its context at exit.

**The viewer runs, at about 15 frames a second.** `MjViewer` with the
egui panel opened a window on WSLg in 2.2 s, kept the Go2 at real time,
and drew 11–17 frames a second: 59–86 ms per `render()`, against
118–155 ms with `GALLIUM_DRIVER=llvmpipe`, so it was on the GPU.

**Overlays cost the same in both, and the cost is the driver's.** The
same offscreen Go2 render with MuJoCo's visualization flags on, ms per
frame, mujoco-rs against Python: none 70 / 68, contact points 105 / 102,
contact forces 145 / 141, joints 173 / 172, actuators 172 / 169, all four
395 / 393. `update_scene` is not the cost (0.06 ms on the kitting scene);
each decoration shape is a draw call, about 3 to 7 ms apiece on WSLg's
d3d12 GL, and nearly free on llvmpipe (all four on the kitting scene:
92 → 151 ms there, against 22 → 469 ms on d3d12). With all four on, the
floor's reflection and the shadows are what multiply it, since each
draws every shape again: 394 ms as is, 292 without the reflection, 271
without shadows, 136 without both; `mesa_glthread=true` made it worse
(425), and coarser `numslices`/`numstacks` barely helped (141 → 114).
The stream now leaves the reflection off while an overlay is on, unless
asked for (pull request #65; on the kitting scene one overlay renders
in about a third of the time, all of them in about half). No binding
changes any of this.

## 10. Open: still to settle

1. Whether `88f241d` builds and passes its tests against the wheel's
   3.11 library.
2. That `EventLoop::new()` inside an eframe process fails with
   `RecreationAttempt` on each system (from winit's code; not run).
3. Whether every trainnr scene compiles under 3.17 to the same element
   counts and names, and how far the trajectories drift scene by scene.
4. The macOS and Windows wheels' library names, and whether the Windows
   wheel ships `mujoco.lib`; physics-only on macOS without the glutin
   fork.
