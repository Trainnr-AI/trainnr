# aloha2-nominal — the first stranger with cameras, and with someone else's measurement

This bundle is the **end-to-end test robot** (docs/31-aloha2-e2e.md):
Menagerie's `aloha` model — two ViperX 300S arms on the ALOHA 2 frame
— fetched byte-identical from `google-deepmind/mujoco_menagerie`
(branch `main`, commit `da76818e`, 2026-08-09, BSD-3-Clause; upstream
`LICENSE`, `CHANGELOG.md`, the actuator/keyframe variants, the MJX
patches and their README — preserved as `MENAGERIE-README.md` — travel
with it; only the two preview PNGs were left behind). The harness
loads `aloha2.xml`, which includes upstream `scene.xml` unmodified and
adds only a `<sensor>` block, because policies observe sensors by
contract and the upstream model ships none.

## What the model carries

| | Count | Notes |
|---|---|---|
| Joints | 16 | 8 per arm: waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate, left_finger, right_finger |
| Actuators | 14 | 7 per arm, position-controlled; the gripper drives `left_finger`, `right_finger` follows by equality constraint — exactly the real robot's seven Dynamixels per arm |
| Sensors | 28 (ours) | jointpos ×14 then jointvel ×14, left arm first, actuator order |
| Cameras | 6 | `wrist_cam_left`, `wrist_cam_right` (on the gripper bodies), `overhead_cam`, `worms_eye_cam` (on the frame) — all four with Intel RealSense D405 intrinsics (focal 1.93 mm, 1280×720); plus two viewpoint cameras `teleoperator_pov`, `collaborator_pov` |
| Keyframe | `neutral_pose` | qpos ×16, ctrl ×14 — the hold pose the contract test drives to |

The arm itself has **no cameras**: `trossen_vx300s/vx300s.xml` upstream
contains zero `<camera>` elements. Cameras are rig content, and this
rig is the canonical one.

## The dynamics are NOT nominal — and that is the point of choosing it

Unlike `so101-nominal`, whose numbers are one default repeated across
joints, the ALOHA 2 team **system-identified** this model (upstream
README step 3): eleven real trajectories driven by the leader arms,
sinusoids targeting the motors' control limits, follower joint
positions and velocities recorded, a box-constrained nonlinear least
squares fit of actuator gain, torque limits, joint damping, armature
and friction. Their values are in `aloha.xml`'s defaults. They report
no intervals, no per-run spread, no identifiability verdicts — which
is precisely what Paper 1's replication adds.

Same arm, two Menagerie models, different dynamics — the standalone
`trossen_vx300s` file is the un-identified one:

| Joint | `vx300s.xml` damping / kp | `aloha.xml` damping / kp (identified) |
|---|---|---|
| waist | 2.86 / 25 | 5.76 / 43 |
| shoulder | 6.25 / 76 (armature 0.004, friction 0.06) | 20.0 / 265 (armature 0.395, friction 2.0) |
| elbow | 8.15 / 106 (armature 0.072, friction 1.74) | 18.49 / 227 (armature 0.383, friction 1.15) |
| forearm_roll | 3.07 / 35 | 6.78 / 78 |
| wrist_angle | 1.18 / 8 | 6.28 / 37 |
| wrist_rotate | 0.78 / 7 | 1.2 / 10.4 |

Factors of 3–5× between the two files for the same servos. Whoever
downloads "the ViperX model" gets one or the other depending on the
folder — the field's nominal-parameter problem, visible inside a single
repository.

No `profile.json` yet, for the reason `so101-nominal` gives: the
profile schema is still drivetrain-shaped. Generalising it is queued;
inventing values for the wrong fields is not.

Addressed as `aloha2-nominal@hash` via `rq_pipeline.bundles.stamp`,
like every bundle.

Onboarded before the importer audit existed (2026-09-24), so its record carries none and the robot card says "not audited"; `tools/audit-bundle.py` against its model file finds nothing changed (an MJCF bundle is a copy of its source).
