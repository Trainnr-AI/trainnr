# 74. The simulation window: how the field lays it out

Research date: **2026-09-09**. The question was where the simulation
section goes, and how Isaac Sim, MuJoCo and the others place theirs; the
call was a primary-source pass before deciding. Four agents, one per
field, each held to the vendor's own docs and source with an access
date on every claim; a fifth source is a Repomix bundle of the
`mujoco-rs` repository supplied the same day. §1–§5 record
what each source says, per field, with its gaps. §6 is ours: what it
means for the Studio's rail. Nothing in §6 is a claim about the field.

The standing rule ([docs/76 §2](../76-the-loop.md)): no new vocabulary. Where the field has a
word for a thing, the Studio uses it; this document is where the words
were checked.

## 1. Isaac Sim (docs.isaacsim.omniverse.nvidia.com, "latest" = 6.0.1 GA)

Accessed 2026-09-09. The site serves 6.0.1 as latest while several
pages still carry 4.5-era assets; the agent flagged version state as
mixed per page.

- **The window.** The reference UI page (`gui/reference_user_interface.html`)
  names the Stage ("allows you to see all the assets in your current USD
  Scene"), the Property panel ("displays the details of selected prim")
  and the Viewport ("The primary way of viewing assets"). Console,
  Timeline, Extensions and Layer are named as openable from the Window
  menu; one-line descriptions of Content, Console, Layer, Extensions and
  Render Settings were **not found** on a primary page.
- **Play, pause, stop.** The tool bar text is "Play — Start an animation"
  and "Stop — Stop an animation"; the shortcuts page lists "Space for
  Play/Pause". A step-by-frame button was **not confirmed**. The physics
  fundamentals page gives the stepping model instead: "120 time steps per
  second, while rendering is set to 60 frames per second, results in two
  physics steps per rendered frame"; programmatic stepping is an
  OmniGraph "On Physics Step" node.
- **Physics settings.** A Physics Scene is made by Create > Physics >
  Simulation Scene and exposes "Simulation Steps per Second" and "Enable
  CCD"; the exact USD attribute names for dt, gravity, solver and
  substeps were **not confirmed** (the Omniverse Physics schema page did
  not return them).
- **Headless and streaming.** `--no-window` launches without a window;
  standalone Python uses `SimulationApp({"headless": True})`. The "Isaac
  Sim WebRTC Streaming Client" is "the recommended streaming client to
  view Isaac Sim remotely on your desktop or workstation without a
  powerful GPU"; Docker must run with `--network=host`.
- **Debug visualization.** Colliders: the viewport's eye icon, "Show by
  type > Physics > Colliders > Selected"; also Window > Simulation >
  Debug with "Solid Mesh Collision Visualization" and an Explode View
  distance. A contact-visualization toggle name was **not confirmed**.
- **Poking the scene by hand.** The Physics Inspector opens from Tools >
  Physics > Physics Inspector, with the documented caveat: "Since the
  Physics Inspector partially initializes omni.physx, it is expected for
  general simulations to not behave properly." Joint sliders, gizmos and
  mouse-drag forces were not described on the fetched page.
- **Naming.** Only "Stage = the current USD scene" was confirmed; a
  glossary page 404'd. Prim, Articulation, Rigid Body, Joint, Sensor,
  Camera, Render Product, Action Graph, Replicator: **not confirmed** as
  official definitions in this pass.
- **What is new.** Release notes show 6.0.1 and 6.0.0 only (a redesigned
  About dialog; the Content Browser gains a SimReady folder). A new
  **Newton** physics backend, "GPU-accelerated, extensible, and
  differentiable… Built on NVIDIA Warp and integrating MuJoCo Warp",
  marked "experimental. The API and features may change in future
  releases." A 4.5 → 5.x UI changelog was **not found**.

Caveat from the agent: web search was exhausted mid-pass, so several
URLs were guessed; a 404 there does not prove absence.

## 2. Isaac Lab (github.com/isaac-sim/IsaacLab, `VERSION` = 2.3.2)

Accessed 2026-09-09, read from source.

- **`SimulationCfg`** (the `isaaclab.sim` package's simulation config): `device = "cuda:0"`,
  `dt = 1/60` ("The physics simulation time-step (in seconds)"),
  `render_interval = 1` ("The number of physics simulation steps per
  rendering step"), `gravity = (0, 0, -9.81)`, `use_fabric = True`, a
  nested `PhysxCfg` (solver type, iteration counts, GPU buffer sizes)
  and `RenderCfg`. **`decimation`** lives on `ManagerBasedEnvCfg`:
  "Number of control action updates @ sim dt per policy dt… if the
  simulation dt is 0.01s and the policy dt is 0.1s, then the decimation
  is 10." Two distinct ratios: physics-per-render and physics-per-policy.
- **`InteractiveSceneCfg`**: `num_envs` ("Number of environment
  instances handled by the scene"), `env_spacing` ("distance between
  environment origins"), `replicate_physics = True` (clones USD prims
  across envs), `lazy_sensor_update`, `filter_collisions`,
  `clone_in_fabric`. A scene holds terrain, articulations, rigid
  objects, sensors and lights.
- **The managers** (`managers/__init__.py`): `ActionManager`,
  `ObservationManager`, `RewardManager`, `TerminationManager`,
  `EventManager`, `CommandManager`, `CurriculumManager`,
  `RecorderManager`. The EventManager "applies operations to the
  environment based on different simulation events… changing the masses
  of objects… or applying random pushes at a fixed interval"; modes
  `prestartup`, `startup`, `reset`, `interval`.
- **Direct vs manager-based** (`task_workflows.rst`): "Manager-based:
  The environment is decomposed into individual components (or managers)…
  Direct: The user defines a single class that implements the entire
  environment directly without the need for separate managers."
- **Running.** AppLauncher flags: `--headless`, `--livestream {0,1,2}`,
  `--enable_cameras`, `--device`, `--xr`, `--rendering_mode`.
  `--num_envs`, `--video`, `--video_length`, `--video_interval` are
  per-script. Video uses `gymnasium.wrappers.RecordVideo`, needs ffmpeg
  and `--enable_cameras` when headless.
- **Randomization** is `events`: `randomize_rigid_body_mass`,
  `randomize_actuator_gains`, `randomize_rigid_body_scale`,
  `randomize_rigid_body_material`, `randomize_joint_parameters`,
  `push_by_setting_velocity`, `apply_external_force_torque`,
  `randomize_physics_scene_gravity` (`envs/mdp/events.py`).
- **Actuators** (`ActuatorBaseCfg`): `joint_names_expr`, `effort_limit`,
  `velocity_limit`, `effort_limit_sim`, `velocity_limit_sim`,
  `stiffness`, `damping`, `armature`, `friction`, `dynamic_friction`,
  `viscous_friction`. Classes: `ImplicitActuator`, `IdealPDActuator`,
  `DCMotor`, `DelayedPDActuator`, `RemotizedPDActuator`.
- **Assets**: `ArticulationCfg(AssetBaseCfg)` with `InitialStateCfg`,
  `RigidObjectCfg`; spawners `UsdFileCfg` and `UrdfFileCfg` (the latter
  inherits the URDF converter).
- **A window of its own?** None. The play script loads a checkpoint and runs
  inference; the only GUI-adjacent code is `ViewportCameraController`
  driving Isaac Sim's viewport from **`ViewerCfg`**: `eye`, `lookat`,
  `resolution`, `origin_type` in {`world`, `env`, `asset_root`,
  `asset_body`}, `env_index`, `asset_name`, `body_name`.

## 3. MuJoCo `simulate` and the Python viewer (MuJoCo 3.12.0, 2026-08-20)

Accessed 2026-09-09 at mujoco.readthedocs.io and the `main` branch of
google-deepmind/mujoco.

- **The left panel** (`simulate/simulate.cc`), sections in order:
  File, Option, Simulation, Watch, Physics, Rendering, Visualization,
  Group, Logging. Simulation holds Run (space plays and pauses),
  Threadpool, Reset, Reload, Align, Copy key / Adjust key / Load key /
  Save key, and a history scrubber. Physics holds Integrator
  (Euler / RK4 / implicit / implicitfast / discrete), Cone, Jacobian,
  Solver (PGS / CG / Newton), Timestep, Iterations, Tolerance, LS Iter /
  Tol, Noslip Iter / Tol, CCD Iter / Tol, Sleep Tol, SDF Iter / Init,
  Gravity, Wind, Magnetic, Density, Viscosity, Imp Ratio, Disable Flags,
  Enable Flags, Contact Override.
- **The right panel**: a slider per hinge or slide joint, a "Clear all"
  button and a slider per actuator, and the Watch field (which answers
  "invalid field" / "invalid index" when a name does not resolve).
- **Mouse.** Plain drag moves the camera (left rotates, right moves,
  shift picks the axis). With a body selected and Ctrl held, right-drag
  is `mjPERT_TRANSLATE` and left-drag `mjPERT_ROTATE`. The help overlay
  begins "Play / Pause, Speed Up / Down, Step Back / Forward…"; the
  complete key table was **not captured** verbatim.
- **Overlays.** Info (Time, Size, CPU, Solver, FPS, Memory); the
  Profiler (Counts, Convergence, Dimensions, CPU time); a Sensor plot.
  Visualization flags on the docs page: `mjVIS_CONTACT`, `mjVIS_FORCE`,
  `mjVIS_JOINT`, `mjVIS_ACTUATOR`, `mjVIS_CONSTRAINT`, `mjVIS_INERTIA`,
  `mjVIS_TRANSPARENT`, `mjVIS_STATIC`, `mjVIS_SKIN`.
- **History.** A buffer of "the smaller of {2000 states, 100 MB}";
  scrubbing restores with `mj_setState(…, mjSTATE_INTEGRATION)` and
  `mj_forward`; Left and Right arrows step it. The version that added it
  was **not found** in the changelog.
- **Python viewer** (the `mujoco.viewer` module and its docs page): `launch()` "launches an
  empty visualization session, where a model can be loaded by
  drag-and-drop"; `launch_passive` "launches the interactive viewer in a
  way which does not block, allowing user code to continue execution";
  the handle offers `lock()`, `sync(state_only=False)`, `user_scn`,
  `cam` / `opt` / `pert`, `key_callback`; `launch_from_path(path)`.
- **Keyframes.** `mj_resetDataKeyframe`: "If 0 <= key < nkey, set fields
  from specified keyframe."
- **Headless.** "On Linux, MuJoCo currently supports GLX for rendering to
  an X11 window, OSMesa for headless software rendering, and EGL for
  hardware accelerated headless rendering"; `mujoco.GLContext` for
  offscreen contexts. The `MUJOCO_GL` variable's own doc text was **not
  located**.
- **Editing at runtime.** `mjSpec` "is in one-to-one correspondence with
  MJCF"; `mj_recompile` "updates an existing mjModel and mjData pair
  in-place, while preserving the simulation state. This allows model
  editing to occur during simulation"; stable since 3.2.5 (2024-11-04).

## 4. Gazebo (gz-sim 8, Harmonic) and Genesis (main branch)

Accessed 2026-09-09.

- **Gazebo's panels are plugins**, each a window you add: World Control
  (play, pause, step), World Stats (real time factor, sim time, real
  time, iterations), Entity Tree, Component Inspector, Transform Control,
  Grid Config, Image Display, Plotting, Resource Spawner, Joint Position
  Controller ("Control position of all joints in the selected model"),
  Align Tool, View Angle, Visualize Lidar, Visualize Contacts.
- **Server and GUI are separate processes**: `-s` "Run only the server
  (headless mode)", `-g` "Run only the GUI", `--headless-rendering`.
- **The world is SDF**: `<physics>` with `type`, `max_step_size` ("maximum
  time at which every system in simulation can interact with the states
  of the world"), `real_time_factor`; system plugins Physics,
  UserCommands, SceneBroadcaster.
- **Runtime control is transport**: services `/world/<name>/control`,
  `/create`, `/set_pose`; `gz topic -t /cmd_vel -m gz.msgs.Twist -p …`.
- **Genesis**: `SimOptions(dt=1e-2, substeps=1, gravity=(0,0,-9.81))`,
  `ViewerOptions(camera_pos, camera_lookat, camera_fov=40, res,
  run_in_thread, refresh_rate)`, `VisOptions(show_world_frame,
  show_link_frame, show_cameras, shadow)`. `Scene(show_viewer=False)` is
  headless; `scene.build(n_envs=…, env_spacing=…)` ("If n_envs is 0, the
  scene will not have a batching dimension"); morphs `MJCF`, `URDF`,
  `Mesh`; joint control `control_dofs_position / velocity / force`,
  `set_dofs_kp / kv`. Cameras record with `start_recording` /
  `stop_recording`. No pinned release was found; the readthedocs site
  404'd and source was read instead.

## 5. mujoco-rs 6.0.1 (the supplied bundle)

Read 2026-09-09 from a Repomix export of github.com/davidhozic/mujoco-rs
at `main` (`Cargo.toml`: `version = "6.0.1+mj-3.12.0"`, `license = "MIT
OR Apache-2.0"`, Rust 1.95+). "A high-level Rust wrapper around the
MuJoCo C library, with a native viewer (re-)written in Rust."

- **The viewer is a window of its own.** `MjViewer` "can be launched only
  in passive mode… and needs to be periodically 'synced' by the user
  application"; it owns a winit window with a glutin GL context
  (mujoco-rs's `viewer` module). Physics may run on another thread through
  `ViewerSharedState` (`sync_data`, `sync_model_opt`, …). With the
  `viewer-ui` feature it draws a side panel in **egui 0.36.1** — the
  Studio's own egui — "which tries to replicate the original C++ viewer
  as best as possible… allows control of constraints and actuators,
  inspection of joint state"; `add_ui_callback(|ctx, data| …)` lets a
  caller add its own egui panels. `X` toggles the panel; `F1` the help.
  Its viewer UI module carries the Rendering flags (Shadow, Wireframe,
  Reflection, Skybox, Fog, Haze, Depth, Segment, Cull face), the
  Visualization flags (Convex hull, Texture, Joint, Camera, Actuator,
  Activation, Light, Tendon, Range finder, Constraint, Inertia, Scale
  inertia, Perturbation force / object, Contact point, Island, Contact
  force, Contact split, Transparent, Center of mass, Select, Static,
  Skin, Flex vertex / edge / face / skin) and the Label and Frame menus.
- **Offscreen rendering** is `MjRenderer` (`renderer` feature): RGB and
  depth to memory or file, `opts_mut`, `camera_mut`, `user_scene_mut`.
- **macOS, verbatim from `installation.rst`**: "visualization (the viewer
  and the renderer) does not work on macOS with the upstream glutin
  crate. To make it work, patch glutin in your project's Cargo.toml with
  the following fork"; "On Windows and macOS the
  renderer-winit-fallback feature is required whenever the renderer
  feature is enabled"; "Automatic download for macOS is not supported."
  MuJoCo itself is a shared library the crate links to
  (`MUJOCO_DYNAMIC_LINK_DIR`, absolute), at exactly 3.12.0.
- **WebAssembly**: builds for `wasm32-unknown-emscripten`, "only headless
  (physics-only) execution is available".
- **Model editing**: `MjSpec` wrappers with examples (`model_editing`,
  `terrain_generation`, `procedural_tree`, `multi_legged_creatures`).

What this settles for the Studio: the viewport already made this call
once. `crates/trainnr-studio/src/viewport.rs` streams frames from
`tools/studio-render-stream.py` (the pipeline's own `mujoco.Renderer`,
MuJoCo 3.11) through shared memory, with camera deltas and the
Ctrl+drag perturbation going back over stdin, and its module doc gives
the reason: "the FFI route needs a patched fork of glutin on macOS plus
a second, older MuJoCo install." The bundle confirms both halves still
hold at 6.0.1 (the glutin patch; MuJoCo 3.12.0 against the pipeline's
3.11). The reason has not decayed.

## 6. What it means for the rail (ours, not the field's)

**Every simulator separates three things, and two of them we already
have.** Isaac Lab says it in code: assets (`ArticulationCfg`), the
scene plus task (`InteractiveSceneCfg` plus the managers), and the
runtime (`SimulationCfg`). Gazebo says it in processes: the SDF world,
the server, the GUI. MuJoCo's `simulate` collapses the three into one
window because a model file is the whole world. Our rail has Robots
(assets) and Environments (scene plus task). The runtime had no page
when this was written; the Simulator page absorbed the Live view the
same day ([docs/76 §10.2](../76-the-loop.md)).

**What a runtime page holds, by the sources' own lists.** The common
core across `simulate`'s Simulation section, Gazebo's World Control and
World Stats, and Isaac's Timeline: **run and pause, step, reset,
reload, a speed, and the clock** (sim time, real time, the real-time
factor). Beside it the **physics facts** (`simulate`'s Physics section:
integrator, timestep, solver, iterations, gravity — Isaac Lab's
`SimulationCfg.dt`, `gravity`, `PhysxCfg`), the **per-joint and
per-actuator sliders** (`simulate`'s right panel, Gazebo's Joint
Position Controller, Genesis's `control_dofs_position`), the
**visualization toggles** (the `mjVIS_*` list, Gazebo's Visualize
Contacts, Isaac's collider display), **perturbation by mouse**
(`simulate`'s Ctrl+drag, which the viewport already has), and
**keyframes** (Copy / Adjust / Load / Save key). `simulate` adds the
history scrubber; nobody else does.

**The two ratios have names.** Isaac Lab's `render_interval`
(physics steps per render) and `decimation` (physics steps per policy
step) are the words for what a run and an evaluation already do in
the pipeline; the runtime page shows both, by those names.

**Randomization is not a page.** Isaac Lab calls it `events` on the
environment; it is already in the environment drawer and the datasheet
as the domain randomization range. Nothing to add.

**Headless is a mode of the same runtime, not a different product.**
Isaac: `--headless` and a WebRTC stream. Gazebo: `-s` and a GUI that
attaches. MuJoCo: EGL and an offscreen context. Our equivalent exists
already: the presenter streams into the Live view, and the agent
screenshots it ([docs/76 §10.1](../76-the-loop.md)). A headless run writes the same
recording and the window opens it later ([docs/76 §10](../76-the-loop.md)).

**The words, checked.** Simulation (Isaac: "Simulation Scene",
"Simulation Steps per Second"; Gazebo: "sim time"; MuJoCo:
`simulate`), run / pause / step / reset / reload (MuJoCo, Gazebo),
timestep and integrator and solver (MuJoCo, Isaac Lab), real-time
factor (Gazebo), keyframe (MuJoCo), perturbation (MuJoCo's `mjvPerturb`),
render interval and decimation (Isaac Lab). None of these is ours.

**Not settled here, and left for the discussion** (settled the same
day: the Simulator page, [docs/76 §10.2](../76-the-loop.md)): whether "Live view"
becomes the runtime page or stays beside it; and how far the viewport's
subprocess protocol grows (run, pause, step, sliders, keyframes, flags
are each one tagged stdin message away, on the wire that already carries
camera deltas and the perturbation).
