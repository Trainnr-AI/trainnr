# The scene loop: a captured scene as a data source, with its physics honest

*Branch `scene-loop-2026-09-22`, cut from main at the A0–A8 merge.
Designed 2026-09-22 from docs/e2e-research/75, before any code. The
operator's framing (2026-09-22): data collection, augmentation,
generation, telemetry and the loop that improves the model are the
product's most important feature; a captured scene is judged by what
it adds to that loop, never by how it looks.*

## 1. What the research settled

Six things, each with its evidence in docs/e2e-research/75:

1. Capture to scene is minutes on one consumer GPU, and the chain can
   be Apache end to end: poses from a commercially licensed
   feed-forward model, gsplat, CoACD, mujoco_warp's own splat ray
   tracer, Rerun's splat archetype. The phone apps are the only
   non-permissive piece, and they are avoidable: a video is enough.
2. The splat-to-physics pairing is nobody's shipped product; the
   vendor claim did not survive a read of the model card. It is ours
   to build, and the part the field skips is the audit: the gap
   between what the eye sees and what the solver touches, measured
   and carried on the record, because altering collision geometry
   alone drops real success 61.7 points while the simulation looks
   unchanged.
3. A splat scene ranks policies with a real correlation and never
   predicts absolute real success; two assets with equal simulation
   scores gave 4/40 and 33/40 on the same robot.
4. Our simulator has the renderer since 2026-08-19; it costs a version
   step to mujoco_warp 3.13 and a twenty-line patch to mjlab's camera.
5. Every capture-driven physics method is a point estimate; nobody
   sets a randomization span from a measured interval on a scene;
   nobody re-judges scene physics over time. Our identification and
   drift machinery is shaped for exactly that.
6. One scan plus one demonstration is worth about 150 teleoperated
   demonstrations once a thousand episodes are rendered; below that,
   real wins; co-training always needs real data; no one closes the
   loop from deployed failures back into the scene.

## 2. The claim this branch can earn, and the one it cannot

**Can:** an agent turns a phone video into a stamped scene artifact
whose record says how it was made, how far its collision proxy departs
from its visible surface, and which of its physics are declared rather
than measured; the walk trains and generates data in that scene with
camera observations rendered inside our own simulator, occlusion
correct, watched in the Studio; and the scene's physics enters the
loop under the same honesty as the actuators — an interval where a
robot touched, a declared span where it did not, drift judged against
the interval.

**Cannot, in this branch:** any number about a real robot in that
scene. No Go2 hardware exists here; the Pico rig's drivetrain fit says
nothing about a floor's friction on its own. The real-to-sim
correlation the field reports comes from real trials we do not have.
Every certificate this branch produces is sim-only at declared scene
physics, and the record says so.

## 3. The artifact: a scene

The `scene` kind reserved in docs/76 §3 becomes real. One folder per
scene under the project's `scenes/`, schema `trainnr-scene/1`:

- **Provenance of the capture**: device and app if any, frame count,
  resolution, duration; the pose tool and its version; the trainer,
  its version, iterations, minutes and GPU; every tool's licence.
  Nothing invented: a field the capture did not record is
  "unrecorded".
- **The splat**: a PLY the renderer loads as it is (positions,
  rotations, scales, opacities, colour; spherical harmonics kept in
  the file, dropped at the renderer, which is degree 0 — recorded as
  such).
- **The collision proxy**: MJCF geoms in the collision group, built by
  gsplat's 2DGS depth into a TSDF mesh into CoACD, with the
  decomposition's parameters; `inertia="exact"` refused where the
  mesh is not watertight, and the refusal recorded, never worked
  around.
- **The alignment**: the transform from the capture frame to the world
  frame (gravity from the poses, scale from a known length in the
  capture, the floor plane fitted), with the residuals.
- **The gap, measured**: samples on the splat's visible surface
  against the proxy — Chamfer, the 95th-percentile distance, the
  fraction of visible surface farther than a stated tolerance from
  any collider, and the volume the colliders enclose that the eye
  cannot see. Recorded whatever it is; a threshold is the task's
  declaration, not the scene's.
- **The physics, by basis**: floor friction and restitution per proxy
  part, each with a basis word: `measured` with an interval and the
  fit record it cites, or `declared` with the span and who declared
  it. This branch declares; §6 says how it measures.
- **The version**: the folder's content hash, as every kind.

The Studio reads it: a card (the capture's still, the gap as one
number), a drawer (the provenance, the gap's four numbers, the physics
by basis), Show in viewer (the splat through Rerun's `GaussianSplats3D`
beside the proxy as `Mesh3D`, so the gap is a picture, not only a
number).

## 4. The experiments, each with its gate

| # | experiment | gate, provable here |
|---|---|---|
| E0 | **The instrument step.** mujoco-warp 3.13, mujoco 3.13, warp ≥1.15 in the walk package; the walk suite; go2-c2's evaluation re-run on the new instrument so the certificate names it (both instruments' certificates stand, each with its stamp). | the suite green; a certificate on the new instrument, its interval overlapping the old one's or the difference recorded as a finding — **met 2026-09-22 on the box (§8.2):** 38/40 on both, identical to the last digit, every trial the same (finding `e0-instrument-step-2026-09-22`) |
| E1 | **Capture to scene.** The `scene` kind and its record; `import_scene` for a Neverwhere folder (built 2026-09-22), `capture_scene(video, name)` for a phone video (next: COLMAP, Brush, the 2DGS proxy chain); the Studio reads it. | **half met 2026-09-22**: the first scene is the field's own (finding `scene-gap-neverwhere-hurdle-2026-09-22`: alignment verified, the gap measured and scoped, the card, drawer and viewer checked by capture); the capture chain built 2026-09-23 (§8.6) and measured against a known scene by a synthetic walk; the operator's phone video is the first real capture |
| E2 | **The walk in the scene.** The Go2 walk takes a scene as its stage: the proxy is the terrain, the splat is what the cameras see, rendered by mujoco_warp's ray tracer across every world; mjlab's camera passes the splat arrays through (the patch offered upstream); the Studio's Simulator shows the scene. Reward preview, then a smoke train. | camera observations from N worlds with the robot occluding the scene and the scene occluding the robot, checked by capture; the smoke train's reward terms in the Live view; frame rate with splats measured and recorded (the field has no number) — **half met 2026-09-22 (§8.1):** the stage, the cameras from the splat checked by capture on one CPU world with the frame time recorded, the contact-site gap and the assay's plumbing; the smoke train with cameras and the N-world rate wait for the box — **met 2026-09-23 (§8.5):** the Go2 trains on the scene's heightfield from the course's start, the head camera sees the splat in every world, its picture is in the actor, the Studio films it; 256 worlds at 64x64 over 393,684 gaussians: 393 steps/s against 4,054 without the camera and 4,500 on the plane |
| E3 | **Data in the scene.** `generate_walk_demos` in the captured scene with camera frames; the batch cites the scene's version, the datasheet names the gap and the physics basis; camera pose, exposure and lighting as declared spans on the batch. | a batch whose provenance names the scene, the gap and the basis; the Studio's datasheet shows the frames from the scene — **built 2026-09-23 (§8.7), the batch empty by the referee:** the press stands on the scene, films the head camera inside the env and names the scene, its gap and its floor on every manifest; the first scene-trained walker survives 38/40 and tracks 0/40 on its own scene, so no episode passes the criterion and the record says so |
| E4 | **Reproduction of the field's own number, in our stack.** Neverwhere's Go1 parkour scenes (MIT, the only legged splat benchmark with code and paired real trials) loaded as scene artifacts; a walk trained on their proxy terrain; our evaluation against their published 15/20 and 12/20 real, with the interval. | our sim rate on their scene, with the exact interval, beside their real rate; the gap between the two recorded as a finding, never explained away |
| E5 | **The physics, measured (designed, gated on hardware).** A robot on the real floor of E1's capture: friction from interaction with an interval (the field's DROPO shape through `mujoco.sysid`'s intervals), the randomization span set from the interval, `check_drift` on a later recording of the same floor. | waits on a robot; the record's fields and the door's refusal ("declared, not measured") are built in E1 |

E0 through E3 need the box's GPU and a phone. E4 needs their data and
a day. E5 needs a robot and is where the loop closes.

**The Mac order (2026-09-22, the box unavailable).** Probed the same
day, before any building: mujoco_warp 3.13's splat renderer runs on a
CPU-only Warp on this Mac and composites correctly, at about 3 frames a
second for four worlds at 320 by 240 (finding
`splat-renderer-cpu-2026-09-22`), so E2 can be smoked here at a small
resolution and measured for real on the box later. Brush ships an
Apple-Silicon binary under Apache that trains a splat from COLMAP data
on Metal and logs into Rerun; COLMAP 4.2 installs from Homebrew; CoACD
and Open3D install from PyPI. And Neverwhere's scenes (MIT) each ship
a gsplat checkpoint, a web splat, a MuJoCo collision mesh with its
alignment transform, the COLMAP poses, the dense reconstruction and
the original capture — a complete scene artifact by another author,
one of them under 200 MB. So the order on the Mac is: E4's data first
as E1's first scene (import, the record, the gap measured against
their own collision mesh, the Studio); the operator's phone video as
E1's second scene through the Brush chain; E2 smoked on the CPU; E0 the
instrument step and every GPU rate when the box returns. The scratch
venv with mujoco_warp 3.13 stays beside the repo; the walk package's
pins do not move until E0.

### 4.1 Amended by the physics pass (2026-09-22, docs/e2e-research/76)

The proxy stays under E2: for thousands of environments the field's only
working recipe is rigid physics on a proxy with gaussians as the
renderer, and every alternative (continuum physics on gaussians,
gradients through contact, learned gaussian dynamics, a pure gaussian
world model) either trains no deployed policy or does not scale. Three
amendments:

- **E2 gains a contact-site gap and a perturbation assay.** Moving a
  collision mesh by 20 mm and 5° zeroed a contact task while the image
  did not change (2608.21416, simulation, 39 scenes). So beside the
  scene's four numbers over its footprint, a task records the same four
  within its contact regions (for a walk, the ground under the paths
  the feet take), and E2 runs the assay: the proxy shifted ±20 mm and
  ±5°, the walk's evaluation re-measured, the cliff recorded. The
  cliff, never a visual metric, sets the collision tolerance the task
  declares and the geometric span it randomises over.
- **Movable objects follow Newton's design when they come.** Gaussians
  attached to bodies as shapes with their own BVH refit from forward
  kinematics, a collision proxy built from the gaussians, the gaussians
  never colliding (Newton 1.6.0, verified in source). The bounded port
  into mujoco_warp is the route; until a task needs a moving object the
  camera must see, objects stay meshes.
- **E5 is a sampling fit with intervals over a tracked-pose residual.**
  No shipped engine differentiates contact make and break; the twin
  line that tried dropped gradients for a sampling fit. A gaussian pose
  tracker is the front end, `mujoco.sysid`'s Gauss-Newton with
  intervals or a DROPO-style posterior the fitter, the span from the
  interval; RigPI's 1.5–7 % on mass is the bar.

## 5. Frictions expected, named before they are found

- The splat renderer keeps splats static: a scene is a stage, never a
  movable object. Objects a task moves stay MuJoCo meshes; a photoreal
  movable object is DISCOVERSE's per-body binding, a later branch.
- Segmentation ignores splats: a segmentation observation sees the
  robot and the objects, not the scene. Recorded on the observation
  term, the way the manifest records every source.
- Spherical harmonics drop to degree 0 at the renderer: the scene's
  colour is view-independent in the simulator and view-dependent in
  the Studio. Recorded.
- Lighting is baked at capture time (every paper says so): the scene
  card names the capture's lighting as a fact, and a batch's lighting
  span is declared over the baked one.
- The instrument step moves every walk certificate's instrument stamp;
  E0 measures what that costs before anything else is built.

## 6. What this branch refuses to claim

No real-robot number. No "the scene predicts the robot": the field's
own evidence is that two equal-scoring scenes differ eightfold on
hardware. No measured scene physics until E5. No "photoreal" as a
word: the record carries PSNR from the trainer and the gap from the
audit, and the reader decides.

## 7. Positioning, planned

Three sentences enter docs/33 marked *planned*, to be dated when their
gate passes: the Apache end-to-end capture chain against a field whose
splat-to-physics methods are non-commercial and whose vendor pairing
does not ship; the visual–collision gap measured and carried on every
scene against a field that never audits it; scene physics with an
interval and a drift check against a field of point estimates and
hand-picked spans.

## 8. E1 as built (2026-09-22, the Mac)

`rq_pipeline/scenes/`: `splat.py` (the 3DGS PLY read and written with
the activations applied at the file boundary, the web `.splat` read,
a similarity applied to centres, rotations and scales together;
MuJoCo's intrinsic x-y-z euler in radians, checked against `xmat`),
`record.py` (schema `trainnr-scene/1`: capture, tools with licences,
splat and proxy facts, alignment, the gap, physics by basis - a
declared value must carry its span, a measured one its interval),
`gap.py` (the four numbers plus coverage, scoped to the proxy's
footprint), `neverwhere.py` (the import). The kind on both sides, the
`scenes/` folder, the card, the drawer, the top-down tile, the viewer
presentation through Rerun's splat archetype beside the proxy mesh,
the doors `import_scene` and `describe_scene`, the `scene` extra
(Open3D, CoACD), eleven tests.

What the first real scene taught, in the order it was learned:

1. **The web splat shares the checkpoint's frame** (nearest
   neighbour 0.0000 m), so the cheap file is the right input for the
   audit; the checkpoint's degree-3 harmonics are for a finer export.
2. **MuJoCo's `euler` is degrees by default and radians in every
   robot file that sets the compiler so.** The scene's transform is
   radians; the first unit test used the default and failed until the
   compiler said radian. The convention itself, intrinsic x-y-z,
   matched to 1e-9.
3. **An audit of the whole capture is not an audit of the course.** A
   capture sees the corridor; the proxy covers the hurdles. The first
   pass reported a 3.2 m 95th percentile on a floor aligned to 1.4 cm.
   The audit is now scoped to the proxy's footprint (its extent grown
   by half a metre), and the share of the visible surface inside that
   footprint is reported beside the four numbers as coverage.
4. **The honest gap on the field's benchmark is not small.** Over the
   course: chamfer 4.6 cm, 95th percentile 18.5 cm, 45 % of the visible
   surface more than 2 cm from any collider, 74 % of the proxy more
   than 2 cm from any visible gaussian (32 % beyond 5 cm, 7 % beyond
   10 cm), mostly hurdle faces and low walls. The floor is tight; the
   obstacles are coarse. Whether a walk tolerates that is the task's
   declaration; the record now carries the number the task can cite.
5. **Their collider is a signed-distance field, ours would be a hull.**
   A hurdle is not convex, so a MuJoCo task on this scene decomposes
   the proxy (CoACD) or loads MuJoCo's SDF plugin, and the record's
   notes say so; the decomposition's own gap is E2's to measure.

## 8.1 E2 as built, the Mac half (2026-09-22)

The stage. `rq_pipeline/scenes/`: `obj.py` (the one OBJ parser; the
viewer's copy now delegates), `proxy.py` (CoACD convex parts with their
own two-way gap, written once beside the proxy as `proxy-parts.json`),
`terrain.py` (the registry: `heightfield`, the proxy's top surface on a
2 cm vertex grid with bilinear heights, the default; `hulls`, the
parts; each builder adds its geoms to the stage's terrain body and
returns the facts the manifest records), `stage.py` (MjSpec: the
trained scene's plane floor deleted, the terrain body added with the
scene's declared friction, the robot's keyframe moved to one metre
before the scene's first waypoint heading along its course - the
importer now reads the course from the scene's own XML, and a scene
that lays out none refuses a guessed start - a head camera on the base
and a course camera behind it, the hulls' vertices and the
heightfield's data embedded so `stage.xml` stands alone; the
perturbation as a rigid move of the terrain body about the start),
`cameras.py` (mujoco_warp's ray tracer over the model and the scene's
visible gaussians, refusing by name on an instrument older than 3.13),
`assay.py` (nine stages gated with one seed, the cliff on the nominal
stage, honest when nothing walked). The gate keeps every tick's contact
points (`GateRuntime.contact_points`; the DDS runtime answers None and
the record says unrecorded), saves them beside its record, and measures
the scene's gap within 10 cm of them; its mirror draws the splat under
the robot and, when the instrument can, the cameras' pictures at 2 Hz.
Doors `stage_deployment` (synchronous) and `assay_deployment` (a job);
the deployment card names the scene and terrain it stands on; the
drawer shows the contact-site gap and the assay's table. `deploy`
moves to tier 3 beside `scenes`. Fourteen new tests.

What the first stage taught:

1. **MuJoCo collides a mesh as its hull, and a hurdle course as hulls
   is a plateau.** CoACD's 59 parts sat a mean 9.4 cm from the proxy
   and roofed the 29 cm hurdles into a 36 cm shelf the robot would
   stand on; 230 or 512 parts did not fix it (finding
   `scene-stage-heightfield-vs-hulls-2026-09-22`). The heightfield -
   the field's own representation for legged terrain - matches the top
   surface to a median of 0.09 mm, and its 95th percentile (15.6 cm) is
   the edge cells where a vertical face becomes a 2 cm ramp. Both gaps
   are recorded on every staged deployment; the heightfield is the
   default and the hulls stay for scenes with undersides.
2. **The ray that finds the ground must ask the terrain, not the
   group.** The first start height came from the Go2's own base
   collider answering a group-3 ray; the stage now rays the terrain
   body's geoms by name.
3. **The gap where the robot touched is tighter than the course's.**
   Over the carpet the feet stood on: chamfer 2.3 cm, 95th percentile
   7.3 cm, against 4.6 / 18.5 cm course-wide; the hurdle faces the
   course-wide number is made of are where a walking policy will meet
   it (finding `scene-cameras-and-assay-smoke-2026-09-22`).
4. **The pictures cost seconds a frame on a CPU.** Both cameras at
   160x120, one world, the full 393,684 gaussians: 1.47 s; 60,000
   gaussians: 0.26 s. The mirror renders at 2 Hz for that reason; the
   gate's picture is a film strip, and the smoke train's cameras wait
   for the box. Checked by eye: the head camera sees the first hurdle
   across the carpet, the course camera sees the row with the robot in
   it, the meshes occluding the splat.
5. **The assay's plumbing runs; its number waits for a walker.** Nine
   stages, every gate 0/2 with the smoke checkpoint, the cliff recorded
   as unmeasurable by name. go2-c2 on the box is the first policy that
   can measure it.

Still E2's: the smoke train with camera observations (mjlab's
CameraSensor passing the splat arrays through, the box's GPU), the
frame rate at N worlds, the Go2's real camera pose from the bundle.

## 8.2 E0 as built (2026-09-22, the box)

The walk package's pins moved by override, not by an mjlab release:
mjlab 1.6.0 is still the newest and pins mujoco and mujoco-warp to 3.11,
so `rq_mjlab/pyproject.toml` overrides both to 3.13.0 (warp stays at
1.17); only those two lines of the lock moved. The walk suite passed on
the new instrument, and go2-c2's final checkpoint certified 38/40,
interval 0.83 to 0.99, median error ratio 0.2727 - the 3.11
certificate's numbers to the last digit, and all 40 trials with the same
outcome and length on both (finding `e0-instrument-step-2026-09-22`).

What the step taught, in the order it was found:

1. **A certificate did not know its instrument.** Its name comes from
   the device tag, seed, trials and a hash of the protocol, and the
   protocol left the instrument out: the 3.13 judgment would have taken
   the 3.11 certificate's name, and the rotated verdict file's backup
   name (policy, seed, trials) could be taken by a third judgment. The
   verdict's protocol now names the instrument and the backup name
   carries the protocol's hash; both certificates stand
   (`...-pa29326` for 3.11, `...-p37d3df` for 3.13).
2. **Both walk doors were broken for the Go2.** The trainer and the
   verdict pass every walk the duck's `head` knob; the Go2's builder
   never took it. It does now (refusing anything but "free"), and a test
   checks every registered walk against the doors' keywords.
3. **mujoco_warp 3.13 prints the line-search overflow on every step.**
   mjlab's velocity task caps the line search at 20 and 3.13 turns every
   overflow warning on: 149,000 lines, 25 MB, for a 20-iteration smoke
   train, and 3.11 had hit the same cap silently. The cap is physics and
   stays; `rq_mjlab.sim_options` extends mjlab's own warp-options hook to
   clear that one bit, and every overflow that drops physics still
   prints. 16 KB.
4. **A feed with nowhere to go blocked training.** With no Studio
   listening and no stream file (a smoke run keeps none) the recorder
   still opened a network sink; its queue filled and the next log call
   waited forever. The stream now opens switched off in that case and
   says so once; a Studio closed mid-run was measured not to hang. And
   the TERM handler's disconnect is bounded: the stuck trainer had
   ignored TERM.

## 8.3 E2 on the box: the first walker on the course (2026-09-22)

The scene came to the box as it came to the Mac: Neverwhere's
`hurdle_226_blue_carpet_v3.zip` from their Hugging Face dataset (MIT,
198,253,978 bytes, sha256 `b36604d74cf4b733...`, the same file), imported
through `import_scene` into `go2-walk`; the gap reproduced to the last
digit (chamfer 4.6 cm, 95th percentile 18.5 cm), and so did the staged
heightfield's (median 0.09 mm). go2-c2's cited deployment staged on it
by `stage_deployment`, gated with four trials, watched in the Studio
over the splat:

| same four seeded commands | successes | tracking error ratio | fell |
|---|---|---|---|
| the trained plane | 3/4 | 0.06-0.18 (and 0.72 on the near-standstill command) | 0 |
| the captured course | 0/4 | 0.63-0.86 | 0 |

The policy never falls on the course; it cannot go where it is told.
12 % of its feet's contacts are 22-30 cm up, on the hurdles' tops, and
it stays inside a 3 x 3.5 m patch where the plane would have carried it
ten metres. The gate holds a random twist for twenty seconds, which on a
course drives a flat-ground policy into its hurdles, desks and walls: on
a scene, that protocol measures collisions more than terrain. The
contact-site gap where it touched: chamfer 2.0 cm, 95th percentile
6.6 cm (the Mac's smoke policy: 2.3 / 7.3).

Two consequences:

1. **The assay needs a protocol that walks the course** — built the same
   night, §8.4. At nominal the twist gate scores near zero, so the cliff
   would be recorded as unmeasurable, as it was with the smoke
   checkpoint; the course gate commands along the course the scene's
   author laid out and judges arrival.
2. **The gate's cameras are dark on the box** (open). The gate runs in the
   pipeline's environment, which is still mujoco 3.11 without
   mujoco_warp; the splat renderer is 3.13's. E0 moved the walk package
   only. Pictures in the gate mean the pipeline's own instrument step,
   which moves every arm and ALOHA certificate's stamp, or running a
   staged-scene gate in the walk package's environment.

## 8.4 The course gate: the scene's protocol, and where the walker stops (2026-09-22, the box)

A deployment staged on a scene is now judged along the scene's course
(`pipeline/rq_pipeline/deploy/course.py`); the manifest chooses — the stage
writes the scene's course into its scene block beside the start, and
a gate reads the manifest alone, so the DDS runtime is judged the same
way. Every field that differs from the plane's protocol is in the
record's protocol block:

| | the plane's gate | the scene's gate |
|---|---|---|
| commands | held per episode, drawn in the manifest's twist ranges | forward along the course, steered to the next waypoint |
| steering | none | mjlab's heading pursuit, the law the policy trained under: yaw rate = clip(gain × wrap(heading to the waypoint − yaw), the manifest's yaw-rate range); forward = speed × max(cos(that error), 0), so it stands and turns when facing away |
| the gain | — | the manifest's `heading_gain` when the export recorded it (it does now, from `heading_control_stiffness`); else the protocol's own 0.5 (mjlab's Go2 value), and the record says which |
| speed | drawn on every axis | drawn per trial in the upper half of the forward range, seeded |
| length | the manifest's episode | the path's length at that speed × 2 |
| success | survived and err_ratio < 0.5 | survived and every waypoint reached within 0.3 m inside the budget; the tracking ratio is still recorded |

The steering has to slow when it turns: a pure yaw-rate pursuit at
0.5 rad/s per radian has a turn radius of 2 m at 1 m/s, and the
kinematic walker in the tests missed a 45° corner inside its budget
until the forward command was scaled by the cosine of the heading
error. The Studio draws the course as a line with numbered waypoints
under the robot, and the trials log names each trial's speed.

go2-c2 on the course by the new protocol, four seeded trials at
0.60–0.80 m/s:

| trial | speed m/s | reached | of | seconds | budget | fell | err ratio |
|---|---|---|---|---|---|---|---|
| 0 | 0.76 | 0 | 4 | 14.0 | 14.0 | no | 0.96 |
| 1 | 0.80 | 0 | 4 | 13.3 | 13.3 | no | 0.96 |
| 2 | 0.73 | 0 | 4 | 14.5 | 14.5 | no | 0.96 |
| 3 | 0.60 | 0 | 4 | 17.7 | 17.7 | no | 0.96 |

0/4, and the record says exactly where: the course's first waypoint
sits on top of the first hurdle (x 0.9–1.1 m, 29 cm tall by the
heightfield along the course line; the next two at 2.4–2.7 and
4.0–4.3 m, 30 cm), and every one of the 5,327 contact points above
15 cm is at x = 0.84 m — the front feet on the hurdle's face, the
robot pushing against it for the whole budget, tracking 4 % of what it
was told. A policy trained to walk on a plane does not climb a 29 cm
hurdle, and the course is a hurdle course: this is the protocol
measuring the policy, not the collision. The contact-site gap where it
stood: chamfer 1.0 cm, 95th percentile 5.6 cm (31,876 points).

So the first cliff number is the honest one — nominal 0, unmeasurable
— recorded by the assay (`assay.json` on the nominal stage; the
finding `course-gate-first-walker-2026-09-22`). The number that means
something waits for a policy that can take the hurdles: E2's train on
the scene, or E4's reproduction on Neverwhere's own terrain.

## 8.5 E2 on the box: the walk on the scene, and what the camera costs (2026-09-23)

The Go2 now trains on a captured scene (`rq_mjlab/scene_stage.py`,
`train_walk(..., scene=)`): the scene's heightfield grid — the same
surface the staged gate collides with, sampled once by the pipeline's
Open3D path and saved beside the scene as a hidden numpy cache the walk
package reads — becomes an mjlab sub-terrain placed at the scene's own
coordinates (mjlab centres its terrain grid at the origin, so the patch
places itself relative to the corner the generator adds), every world
spawned at the course's start on the surface; the stage's head camera
rides on the base as an mjlab `CameraSensor`, its picture of the splat
rendered by mujoco_warp's ray tracer for every world and flattened into
the actor's and the critic's observations (`head_rgb`, 12,288 of the
actor's 12,523 inputs); the scene's gaussians reach mjlab's render
context through a thin proxy over the module its sensor context calls,
since mjlab builds that context without the renderer's `splat_*`
arguments (the patch offered upstream). The rough recipe's rules and
sensors stay (the height scan sees the hurdles); the flat recipe's
deployable actor does not apply here yet. The recorder films the
watched world's camera sensors at the frame cadence, so the Studio's
Live page shows the policy's own picture beside the reward terms.

Three smoke trains of 20 iterations, 256 worlds, all in the Studio:

| stage | camera | env-steps/s | wall for 122,880 env-steps | log |
|---|---|---|---|---|
| the trained plane (E0's smoke) | — | ≈4,500 | 27 s | 16 KB |
| the scene, 5 cm grid | none | 4,054 | 30 s | 18 KB |
| the scene, 5 cm grid | head 64×64 rgb, 393,684 gaussians | 393 | 313 s | 19 MB |

The terrain costs nothing measurable; the camera costs everything: the
difference is 283 s for 122,880 frames, 2.3 ms per 64×64 frame across
256 worlds (434 frames/s) on the RTX 3090 Ti — the field's missing
number, at the smallest picture a policy could plausibly use. A g3 run
at 4,096 worlds would render sixteen times as much per step; the
picture's cost, not the physics, sets the scene walk's scale.

What the first scene smoke taught, before that table could be written:

1. **mujoco_warp caps the prisms one geom collides with at MuJoCo's
   own `mjMAXCONPAIR` (50) and drops the rest, printing a line per
   world per step.** At the stage's 2 cm grid a calf capsule's footprint
   alone holds 66 prisms; the first smoke printed 778,000 lines (2.3 GB
   of log) and ran at a third of its later speed. The training grid is
   the saved grid resampled to 5 cm (`TRAIN_CELL_M`, bilinear on the
   same footprint; the hurdles are four cells deep at that spacing);
   there only a trunk lying flat exceeds the cap — a fallen robot early
   in training, 74,000 lines in the second smoke — so the print is off
   (`sim_options.QUIET_OVERFLOWS`) and the fact is recorded here: a
   fallen trunk's contact with the ground is truncated at 50 prisms.
   The staged gate keeps its 2 cm grid; CPU MuJoCo has no such cap.
2. **A derived cache must not move a scene's version.** The grid file
   landed in the scene folder and the scene's stamp changed
   (`3c9ee1da4ed3` → `b70638809abd`) in the first smoke's identity;
   `bundles.hashing` skips hidden files, so the cache is hidden
   (`.heightfield.npz`) and the stamp is what it was.
3. **The smoke prints no rate.** rsl_rl's per-iteration table goes to a
   log directory the smoke never has; the trainer now prints the
   env-steps per second it measured, for every agent.

Open: a policy that takes the hurdles (this was a smoke), evaluating a
scene-trained checkpoint on the scene (the verdict door passes no scene
yet), exporting one (the manifest would carry the scene as a stage
does), and the g3 scale with pictures.

## 8.6 E1's second half: the capture chain, on the Mac (2026-09-23)

`capture_scene(source, name)` (`rq_pipeline/scenes/capture.py`, a job
through `tools/capture-scene.py`): a phone video or a folder of frames
into a scene artifact by the chain the research chose — ffmpeg for the
frames (two a second), COLMAP for the poses (one OPENCV camera,
sequential matching with loop detection for a video, exhaustive for
stills, the mapper, the model kept as text beside the binary), Brush
for the splat (headless, one export at the end, into the Studio when
one listens). Every tool is a subprocess named on the record with its
version and licence; a missing one refuses by name before anything
runs; each stage's output, when present, is kept, so a failed Brush run
does not re-run COLMAP. Three decisions the record states rather than
hides:

- **Alignment.** COLMAP's frame is arbitrary and unscaled. The floor is
  the largest plane through the visible centres (RANSAC), its normal
  turned to +z with the side most of the off-floor scene lies on as
  up, its centroid at the origin. Scale is metres per COLMAP unit when
  the caller declares one (a length measured in the capture) and 1.0,
  recorded as unrecorded, when not: the scene's metres are then
  COLMAP's units and the record says so. (The first draft let the floor
  vote on which side is up and the noise of six thousand floor points
  outvoted the box on it; the floor abstains now.)
- **The proxy.** No dense reconstruction runs on a laptop (COLMAP's
  patch-match needs CUDA; the 2DGS chain of §3 waits on the box), so
  the proxy is the visible surface itself: the visible centres' top on
  a 2 cm grid, each cell its highest centre, a hole filled from its
  neighbours only when at least three of the eight around it are seen
  (so a sparse splat yields a sparse surface, not an invented one), the
  cells seen and filled both counted on the record. Poisson was tried
  first and aborted the interpreter from C++ on a flat room. The gap
  then measures the proxy's fidelity to the splat it came from, not to
  the world — the first note on every record this makes. The world's
  number comes from the synthetic capture below.
- **The poses.** The mapper may split a walk into several models where
  the chain of matches breaks; the chain converts every model, records
  each one's frame count, and trains Brush on the largest. The first
  synthetic walk showed why: an ellipse five metres across the course
  carried the camera through the capture's fringe, where frames hold
  two or three features, and the mapper made two models of 19 and 20
  frames from 72. The walk that a phone would take — inside the
  captured volume, looking at the course — is the second run below.
- **Physics.** Declared by the caller or absent; nothing measured.

The test that does not need a phone: a synthetic walk around the
Neverwhere hurdle course, one camera carried at head height on an
ellipse around the course looking in, 72 frames at 640x480 rendered
from the scene's own splat by mujoco_warp's ray tracer (13 s a frame on
this CPU) — a capture whose ground truth is known to the millimetre.
The captured scene is put back in the true frame by the similarity
that maps COLMAP's camera centres onto the true ones (Umeyama), which
also yields the scale the record left unrecorded; then the visible
centres against the original's and the captured proxy against
Neverwhere's collision mesh. On the loose walk (19 frames registered,
4,000 Brush steps, two minutes for the whole chain) the poses were
already right — the camera centres fit the truth to 0.18 cm mean after
the similarity, the fitted floor 0.58° from level, the scale 0.435 m
per COLMAP unit — while the splat was thin (3,065 visible centres over
the course against the original's 357,903) and the proxy with it. The
tight walk — 96 frames on an ellipse inside the captured volume,
93 registered in one model with 16,477 points, 30,000 Brush steps,
26 minutes for the whole chain on this Mac — put the poses within
5 mm of the truth (camera centres 0.51 cm mean, 1.35 cm max after the
similarity; the fitted floor 0.12° from level; scale 0.770 m per COLMAP
unit) and the surface within centimetres where it was seen: the
captured visible centres lie a median 1.7 cm from the original's (95th
percentile 8.0 cm), the original's a median 3.6 cm from the captured
(13.1 cm), chamfer 4.0 cm; the first hurdle's top reads 30.1 cm against
the collision mesh's 29.1. What the chain does not give from 96 frames
at 640x480 is density: 34,785 visible gaussians against the original's
393,684, so the top-surface proxy has holes — 25,017 cells seen, 40,199
filled from neighbours, and along the course line only 4 of 29 samples
find a captured surface at all; the record's own gap (chamfer 7.8 cm,
93 % of the proxy unseen) says so without needing the truth. A phone
video at 1080p over a minute gives three times the pixels and frames;
the proxy's coverage is the number to watch on the first real capture
(finding `capture-chain-synthetic-2026-09-23`).

Gotchas met on the way, each recorded: the ray tracer faults on a
model with nothing to draw (a camera rig alone), so `scenes.cameras`
refuses such a model by name; a gate on a staged scene cited the
plane's evaluation and would have judged its course rate against it,
so the verdict now says the protocols differ and judges nothing
(`OTHER_PROTOCOL`), and the assay carries the cited evaluation through
as the gate does (the cited record's one home: `project.cited`);
COLMAP 4.2 renamed its GPU flags (`FeatureExtraction.use_gpu`) and
defaults them on even in Homebrew's CPU-only build, so the chain asks
each command's help before passing one; Open3D's Poisson aborted the
whole test process from C++ on a flat room, which is why the proxy is
a grid; and the Studio's launch door never saw a heartbeat when the
project was named by a relative path, because the Studio is spawned
with the checkout as its working directory — the override resolves
now. Reviewed in the Studio by its own doors (launch, show,
screenshot): a capture's card led with a scratch path as its source
(now the folder's name, the path in the notes) and its gap read like
a measurement against the world (the card now says "proxy from the
splat"); the viewer drew every gaussian and Brush's metres-wide faint
ones hid the room (it draws the visible set, the audit's, and says how
many of how many); a staged deployment's lineage named no environment
(it cites its scene now, and its card leads with scene and terrain).


**Installing the chain, on any machine.** It runs three tools as
subprocesses, found on PATH: `ffmpeg`/`ffprobe` (`brew install
ffmpeg`, `sudo apt install ffmpeg`, `winget install Gyan.FFmpeg`),
`colmap` (`brew install colmap`, `sudo apt install colmap`, a release
from colmap's GitHub on Windows) and Brush's `brush_app`
(`python3 tools/install-brush.py` fetches the release binary for the
running platform into a user bin directory with its SHA-256 checked,
no root; or pass the binary's path). A missing tool is refused by name
with the install line for the machine the chain runs on — never
another machine's package manager (the first draft said `brew` on a
Linux box, 2026-09-23). COLMAP's CPU build suffices: the chain runs
the sparse mapper only.

## 8.7 E3 built, and the first scene walker judged: a press that stands on the scene, a certificate that names it, an empty batch (2026-09-23, the box)

**The press on a scene.** `generate_walk_demos(checkpoint, robot=, scene=)`
(`rq_mjlab/walk_press.py`, `--robot --scene`): the walk press, built for
the microduck, now presses any registered walk, and on a captured scene
the rollouts stand on its heightfield from the course's start; the
frames are the head camera's picture of the splat, rendered inside the
batched env by mujoco_warp's ray tracer and captured per tick from the
camera sensor (`walk_verdict.rollout_episodes(capture_cameras=)`) —
no CPU replay; every episode manifest carries a visual basis naming
the scene's version, its renderer and its gap ("no visual draws"), and
a dynamics basis naming the floor's friction as the scene declares it;
the export descriptor's notes carry the scene; the datasheet prints a
visual basis even when nothing was drawn (a splat is a basis nothing
is drawn from). The loader (`walk_view.load_walk`) gates robot,
actuator and scene between checkpoint and env, and shows the actor a
camera exactly when its identity says it trained with one.

**The certificate on a scene.** `evaluate_walk(checkpoint, robot=,
scene=)`: the verdict judges a scene-trained checkpoint on its scene,
refuses the plane for it and it for the plane by name
(`require_same_scene`), and its protocol names the scene and the
terrain, so the certificate's name (the protocol hash) never collides
with a plane's.

**The first scene walker.** `go2-scene-c1`: the g3 recipe on the
hurdle scene — 4,096 worlds, the height scan, no camera, 51,600
env-steps/s at scale (the plane's 60,000). It died at iteration 1,017
of 1,500 with NaN in the actor's observations. The consistent cause:
mjlab's bounds termination judges its origin-centred grid, which the
scene patch is not, and it had been dropped; a world that walks off
the heightfield's edge (10 m along the course at up to 1 m/s inside a
20 s episode) falls forever, and its numbers stop being numbers. The
scene stage now truncates a world that leaves the scene's footprint
(`scene_stage.out_of_scene_bounds`, the grid's extent less 0.3 m, a
time-out like mjlab's own). The last checkpoint, `model_1000`, judged
on its scene by the new door:

| | plane (go2-c2, 1500 it) | scene (go2-scene-c1, 1000 it) |
|---|---|---|
| tracking reward at the end (of 2.0) | 1.69 | 0.67 |
| certificate on its own ground | 38/40 [0.831, 0.994] | **0/40** [0, 0.088], survived 38/40, error ratios 0.79–1.19 |
| certificate name | `go2-c2-model_1499-cuda-seed1000-n40-p37d3df` | `go2-scene-c1-model_1000-cuda-seed1000-n40-pdf8c16` |

The certificate and the training reward agree: the scene walker learned
to stay upright on the hurdle course and not to go where it is told.
Every world starts at the course's start facing the first hurdle and is
commanded up to 1 m/s in any direction; a thousand iterations of that
is not a walker. So the press pressed nothing: 40 attempts, 38
survived, 0 tracked, `failures.jsonl` names each one with its command
and error, `export.json` stands, no datasheet (the press writes one
for kept episodes). E3's plumbing and provenance are built and pinned;
its batch waits for a policy that walks its scene — the honest empty
result, recorded (finding `scene-walk-first-certificate-2026-09-23`).

Two more things fixed on the way: the press rolled out forever on the
Go2 (its play config's episode is 10⁹ s and only a fall ends one; the
first scene press sat fifty minutes in its first batch) — every rollout
is capped at the training episode now, said so in the batch's notes;
and the verdict showed a scene walker a camera it never trained with
(12,523 inputs against 235) — the loader's rule is the verdict's too.

What a walker for this scene needs, for the record: a run long enough
(the plane took 1,500 iterations to 38/40; this is harder), commands
that do not send it into the hurdles from the first tick (the course
protocol's own, or mjlab's terrain curriculum over difficulty rows the
scene does not have), and a reference for the jump the velocity reward
alone has not found (mocap and retargeting are the data engine's D5
lane, designed, not built). None fits the box's one-hour rule; the
cloud is empty; the operator's call.

## 9. How a scene's physics gets certified for a task

Assembled from what exists (docs/e2e-research/76 §7), every step but
the last runnable here today:

1. Identify the floor's or object's friction, mass and centre of mass
   by a scripted interaction; report the spread over at least five
   repeats and the interval (E5).
2. Replay at least ten recorded real trajectories open-loop; report the
   pose after interaction (the field's good pipelines: 0.4–1.5 cm) and
   the joint-tracking error.
3. Run the perturbation assay on the task's contact sites (±20 mm,
   ±5°); report the success cliff; the cliff sets the collision
   tolerance and the geometric span.
4. Evaluate at least three policies with paired real and simulated
   trials, at least twenty a cell; report the correlation, the
   rank-violation rate, a prediction-powered or conformal interval, and
   the bias in points.
5. Fix and publish the seeds.

Steps 1, 2 and 4 need a robot; step 3 and the seeds are E2's. A scene
record that has not been through this is a scene, not a certified one,
and its card says which.
