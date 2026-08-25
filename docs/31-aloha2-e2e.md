# The end-to-end test: ALOHA 2 through every stage

*Started 2026-08-26. The operator's call: run the whole pipeline, stage
by stage, on one robot — first a ViperX 300S, then (once the camera
question was answered) the full ALOHA 2 rig, because it has cameras.
This doc is the plan and the running record; each rung links its
evidence. The critical path (Paper 2 on the WSL card) is unchanged —
this is the platform proving it can swallow a second stranger, and the
first with cameras and someone else's measurement.*

## 0. Why ALOHA 2

- **It sits inside the wedge.** ViperX 300S arms run Dynamixel
  servos that report position, velocity, current and temperature —
  feedback the SG90 arm never had. Identification, drift telemetry and
  the HIL loop all have something to read.
- **It has cameras — the arm alone does not.** Menagerie's
  `trossen_vx300s/vx300s.xml` carries zero `<camera>` elements (checked
  in the file, pinned by `test_the_arm_alone_has_no_cameras`). The
  ALOHA rig carries six: two D405 wrist cameras on the gripper bodies,
  D405 overhead and worm's-eye cameras on the frame, and two viewpoint
  cameras. A bare arm is proprioception only — it knows where it is,
  never what is in front of it — so any task where the object moves
  needs the rig, not the arm.
- **Someone else measured it.** The ALOHA 2 team system-identified the
  model (11 real trajectories, box-constrained NLS on follower
  position/velocity). The standalone `vx300s` Menagerie file carries
  DIFFERENT, un-identified numbers for the same servos — 3–5× apart on
  shoulder damping and gains (table in `robots/aloha2-nominal/README.md`).
  Paper 1 on this robot is a replication with intervals they did not
  report, against a reference that exists.
- **The ecosystem is ready-made.** ACT was born on this rig; LeRobot
  drives ViperX/ALOHA natively for collection, ships ALOHA ACT
  configs and pretrained sim checkpoints, and `gym-aloha` has MuJoCo
  eval tasks (transfer cube, insertion).

## 1. Stage by stage

| Stage | On ALOHA 2 | Hardware? | Status |
|---|---|---|---|
| ② ONBOARD — bundle | `robots/aloha2-nominal/`: Menagerie `aloha` byte-identical (commit `da76818e`, BSD-3) + `aloha2.xml` sensor wrapper (28 sensors: 14 jointpos + 14 jointvel in actuator order); census pins 14 actuators, 28 sensors, **6 cameras**, 95 geoms; wrapper-purity and hold/travel tests | No | **DONE 2026-08-26** — `pipeline/tests/test_aloha2_bundle.py`, 6/6 |
| ② ONBOARD — identification | Excitation on the real arms → `mujoco.sysid` → intervals + identifiability verdicts, compared against ALOHA 2's published values (Paper 1 as replication). Rehearse first in sim exactly as Paper 0 did: true model → sweep → degraded telemetry → `identify()` recovers | Real: yes. Rehearsal: no | queued |
| ① SCAN | The scene IS the bundle's `scene.xml` (frame, table, cameras); task objects as cousins with measured mass/μ | No | queued |
| ③ VALIDATE | Gate A: paired sim/real rank correlation, certificate on the lower bound | Yes | after hardware |
| ④ COLLECT | LeRobot's ALOHA recorder (Dynamixel bus + cameras → LeRobotDataset) — not the Pico `.wire` | Yes (or scripted sim demos) | queued |
| ⑤⑥⑦ TRAIN | ACT via LeRobot's ALOHA config on the 3090 Ti; released sim checkpoints first | No | queued |
| ⑧ EVALUATE | Harness tasks on the bundle (rq_pipeline/tasks/aloha2.py (next rung)): transfer cube first, matching gym-aloha's protocol; paired trials → certificate dry run | No | next |
| ⑨ ENVELOPE | Dynamixel current/torque limits + software watchdog — no Pico Tier 0 here; the safety story is different and must be written | Yes | queued |
| ⑩⑪ OPERATE / TELEMETRY | Leader-arm interventions; present-current/temperature registers as drift telemetry | Yes | after hardware |

## 2. Decisions and findings, dated

**2026-08-26 — episodes start where the protocol says, not where the
model resets.** The first hold test drove both arms from qpos=0 (arms
straight up) to `neutral_pose` and found the shoulders stuck 1 rad
short: the elbows fold inward faster than the shoulders lean back, the
grippers meet at the top centre and jam at **over 1 kN** of contact
force (`left/gripper_base` ↔ `right/gripper_base`), both arms
symmetric. The real rig never traverses that pose. First fix tried —
make the backend's home the first keyframe — broke four SO-101 task
tests whose scripted picks were tuned from the zero start. Final fix:
`EpisodeProtocol.home` names the keyframe an episode starts from
(`None` = reset); the harness resolves it through
`backend.keyframe_state(name)`. SO-101 keeps its tuned start byte for
byte; ALOHA protocols declare `neutral_pose`. Where an episode starts
is a protocol fact, hash-stamped with the rest, never inherited from
the simulator's zero.

**2026-08-26 — the census counts cameras.** `ModelCounts.cameras`
(from `model.ncam`) joins actuators/sensors/geoms, because a vision
policy on a camera-less model fails silently — black frames score 0%
with no error, the failure class the census gate exists for.

**2026-08-26 — verified in the viewer.** All six cameras rendered at
`neutral_pose` through the offscreen renderer (contact sheet in the
session), each wrist camera looking down its own fingers at the
opposite arm; the jam pose rendered from the teleoperator viewpoint.

## 3. Next rung

rq_pipeline/tasks/aloha2.py (next rung) — transfer cube: cube on the table
between the arms, right arm picks, hands to left, success = cube held
by the left gripper clear of the table. Scripted reference policy,
graded ladder (pick / no-close / limp), paired trials, through the
harness to a certificate dry run — the same shape that proved the
SO-101 tasks, one rig up. Then released ACT checkpoints through the
vision path, with camera matching against gym-aloha's `top` view.
