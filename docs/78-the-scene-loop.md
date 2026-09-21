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
| E0 | **The instrument step.** mujoco-warp 3.13, mujoco 3.13, warp ≥1.15 in the walk package; the walk suite; go2-c2's evaluation re-run on the new instrument so the certificate names it (both instruments' certificates stand, each with its stamp). | the suite green; a certificate on the new instrument, its interval overlapping the old one's or the difference recorded as a finding |
| E1 | **Capture to scene.** `capture_scene(video, name)`: poses, splat, proxy, alignment, the gap, the declared physics, the record; the Studio reads it. First capture: a real floor the operator walks with a phone. | a stamped scene whose record carries every field of §3; the gap measured on the first capture and recorded as a finding; the card, drawer and viewer checked by capture |
| E2 | **The walk in the scene.** The Go2 walk takes a scene as its stage: the proxy is the terrain, the splat is what the cameras see, rendered by mujoco_warp's ray tracer across every world; mjlab's camera passes the splat arrays through (the patch offered upstream); the Studio's Simulator shows the scene. Reward preview, then a smoke train. | camera observations from N worlds with the robot occluding the scene and the scene occluding the robot, checked by capture; the smoke train's reward terms in the Live view; frame rate with splats measured and recorded (the field has no number) |
| E3 | **Data in the scene.** `generate_walk_demos` in the captured scene with camera frames; the batch cites the scene's version, the datasheet names the gap and the physics basis; camera pose, exposure and lighting as declared spans on the batch. | a batch whose provenance names the scene, the gap and the basis; the Studio's datasheet shows the frames from the scene |
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
