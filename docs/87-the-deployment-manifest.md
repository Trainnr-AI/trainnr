# 87 The deployment manifest, `trainnr-deploy/1`

*Written 2026-11-15 (issue #106), for runtimes outside trainnr that run
an exported policy.*

`export_deployment` writes a folder: `policy.onnx`, `scene.xml` and
`deploy.json` (and, for a tracking policy, `motion.npz` and `assets/`;
§8). This page is `deploy.json`, the contract between the
policy and whatever runs it. Every rule here is what the reference
runtime does (`trainnr/trainnr/deploy/runtime.py`, plain MuJoCo and
onnxruntime, nothing from the training stack), and what the exporter
writes (`trainnr-mjlab/src/trainnr_mjlab/walk_export.py`). The loader
(`trainnr/trainnr/deploy/manifest.py`, `load_manifest`) refuses a key or
an observation source it does not know, by name; a third-party runtime
should do the same rather than guess. A change to any rule below is a new
schema version, not an edit of this one.

Units are SI throughout: metres, seconds, radians, newtons, newton
metres. Every per-joint list is in **policy order** (`joints.policy_order`)
unless it says otherwise.

## One control tick

```
obs     = concat(term(t) for t in observations)          # float32, width onnx.input_width
action  = onnx(obs)                                      # width onnx.output_width
action  = clip(action, action.clip) if action.clip       # rsl_rl's clip_actions
target  = joints.default_pos                              # every joint at home, then
target[acting] += action.scale * action                  # the acting joints moved
ctrl[joints.action_to_ctrl[i]] = target[i]               # then decimation physics steps
last_action = action                                     # clipped, unscaled
```

`acting` are the indices in `policy_order` of `joints.action_joints`, or
every joint when the key is absent: a policy that drives only some joints
(a legged walk with its head pinned, docs/88) holds the rest at
`default_pos`, and its output is `len(action_joints)` wide.
`action.offset` is always equal to `joints.default_pos`; it is written so
`action` reads on its own (`kind` spells the rule with `default_pos`). The
runtime ticks at `control.control_hz`, that is every
`control.physics_timestep_s * control.decimation` seconds.

## The keys

| Key | What | Required |
|---|---|---|
| `schema` | `trainnr-deploy/1` | yes |
| `stamp_of` | a label: the robot and the checkpoint | |
| `policy` | the policy's stamp, `name@hash` | |
| `checkpoint` | the checkpoint file the ONNX was exported from | |
| `run` | the training run's stamp | yes |
| `robot` | the robot bundle's stamp; its `assets/` holds the meshes `scene.xml` names | yes |
| `actuator` | the actuator model's stamp (a declared PD set, or a BAM actuator bundle) | |
| `task` | the declared task's stamp; `unrecorded` when the run was started by robot name instead of a declared task | |
| `dr_basis` | in words, what was randomized during training and on what basis | |
| `fit`, `fit_basis` | the identified fit the policy trained under, when there was one | |
| `seed` | the training seed | |
| `certificate` | the evaluation the deployment cites; the gate compares against it | |
| `control` | `physics_timestep_s`, `decimation`, `control_hz` (rounded, for printing), `episode_length_s` | yes |
| `joints` | below | yes |
| `action` | `kind`, `scale` (one per acting joint), `offset`, `clip` (`[lo, hi]` or null) | yes |
| `observations` | below, in concatenation order | yes |
| `onnx` | `file`, `input` and `output` tensor names, `input_width`, `output_width`, `normalization` (always "folded into the graph": feed raw terms), `export_check_max_abs_diff` | yes |
| `scene` | `file`, `meshes`, `terrain`, `floor` (the trained floor's size, condim, priority, friction) | yes |
| `commands` | `twist`: the ranges the policy trained under for `lin_vel_x`, `lin_vel_y`, `ang_vel_z` (and `heading` when it was trained with one); `heading_gain` when it steered by heading | a walk |
| `tracking` | a tracking policy's reference and bounds, in place of `commands` (§8) | a tracking policy |
| `command_basis` | in words, which command curriculum stage the checkpoint had reached | |
| `termination` | `fell_over_deg`: the tilt at which the evaluation counted a fall | |
| `unitree` | for a Unitree robot only: the names their simulator and controller use | |
| `limits` | the robot's limits over several joints the policy trained against (docs/88 §3): `file` (beside the policy, `trainnr-joint-limits/1`), `stamp`, `mode` (`terminate` or `cost`); a runtime should refuse or hold a target that breaks one, as the gate fails a trial that does | |

## `joints`

| Field | What |
|---|---|
| `policy_order` | joint names, in the order of the action and of every per-joint observation |
| `ctrl_order` | the simulator's actuator names, in `scene.xml`'s order |
| `action_to_ctrl` | for policy joint `i`, the index of its actuator in `ctrl_order` |
| `default_pos` | the home pose (rad), the zero of `joint_pos_rel` and of the action |
| `stiffness`, `damping` | the PD gains the policy trained under (N m/rad, N m s/rad); **null for a joint a position servo drives** |
| `effort_limit` | the actuator's torque limit (N m); null for a servo's joint |
| `sdk_order_map`, `sdk_order_source` | for a robot with a vendor SDK: the SDK's joint index of each policy joint, and where that map was read |
| `servos` | groups of joints a position servo drives (below); absent or empty for a PD robot |
| `action_joints` | the joints the action drives, in its order, when not all of `policy_order`; absent otherwise |

A PD robot's runtime sets `stiffness` and `damping` on each motor and
sends `target` as its position set point. A servo closes its own loop,
so a simulated motor's gains would mean nothing on it; each `servos`
group instead says what the servo itself must be set to:

| Field | What |
|---|---|
| `law` | `bam`: the servo modelled by BAM's law (Duclusaud et al., ICRA 2025) |
| `actuator` | the actuator bundle's stamp |
| `joints` | the joints of the group |
| `vin_v` | the supply voltage the law assumes (V) |
| `kp` | the servo's position-gain register value the law assumes |
| `error_gain`, `max_pwm` | the firmware's constants the law uses |

The servo on the robot must run at `vin_v` with `kp` in its register;
the policy's targets are sent as the servo's goal positions.

## `observations`

Each term: `name`, `width`, `source`, `params`, `scale`, `clip`,
`history_length`. A term's value is computed from its source, **then
multiplied by `scale`, then clipped** to `clip` when it is set. `scale`
is a number or a list of `width` numbers. `history_length` is 0 in every
export: the current value only (mjlab counts past frames, so 0 is no
history); the reference runtime implements no longer history. The terms
are concatenated in the listed order.

| `source` | Width | Value |
|---|---|---|
| `sensor robot/imu_ang_vel` | 3 | the IMU's gyro (rad/s), in the IMU site's frame: the MuJoCo sensor named `imu_ang_vel` in `scene.xml` |
| `sensor robot/imu_lin_vel` | 3 | the IMU site's linear velocity (m/s) in its own frame: the `velocimeter` named `imu_lin_vel`; on a robot, a state estimator's |
| `projected_gravity` | 3 | the unit vector (0, 0, -1) of world gravity, rotated into the floating base's frame (the root body, not the IMU site); (0, 0, -1) standing upright |
| `joint_pos_rel` | joints | joint position minus `default_pos` (rad). `params.biased: true` means training added a per-episode encoder offset (mjlab's `joint_pos_biased`); a robot reads its encoders as they are |
| `joint_vel_rel` | joints | joint velocity (rad/s); relative to a default velocity of zero, so the plain velocity |
| `last_action` | joints | the previous tick's action after `action.clip`, before `scale` and `default_pos`; zeros on the first tick |
| `command twist` | 3 | the commanded `[lin_vel_x, lin_vel_y, ang_vel_z]` (m/s, m/s, rad/s) in the base's frame |
| `gait_phase` | 2 | `[sin 2πφ, cos 2πφ]` with φ the time since reset modulo `params.period` over the period; zeros while the command's norm is below 0.1 |

`sensor <entity>/<name>` reads the sensor named `<name>`; other sensor
names follow the same rule, and the loader refuses any source not in
this table.

## 8. A tracking policy

*Added 2026-11-23 (docs/86 phase 3): an optional `tracking` section and
three observation sources. A walk's manifest is unchanged; a runtime
that predates them refuses a tracking manifest by name, as the loader
does.*

A policy trained on mjlab's tracking task follows a reference motion
instead of a commanded twist. Its folder holds, beside `deploy.json`:

- `policy.onnx`, mjlab's tracking export as it is: two inputs, `obs`
  (`onnx.input`) and `time_step` (`onnx.time_step_input`, one float, the
  reference frame), and seven outputs, the action first and then the
  reference at that frame (`joint_pos`, `joint_vel`, `body_pos_w`,
  `body_quat_w`, `body_lin_vel_w`, `body_ang_vel_w`), with mjlab's
  metadata in the model's properties. The action depends on `obs` alone;
- `motion.npz`, the reference as the runtime reads it: `joint_pos` and
  `joint_vel` (frames × joints, policy order), and `body_pos_w`,
  `body_quat_w` (w x y z), `body_lin_vel_w`, `body_ang_vel_w` (frames ×
  `tracking.body_names` × 3 or 4, world frame), one frame per control tick;
- `assets/`, the robot's meshes, when the robot is not a project bundle
  (the G1 comes from mjlab's asset zoo).

**The clock.** The robot starts at frame 0 and the reference's clock
leads the control tick by one: at tick k (0 on the first) the policy
reads frame k + 1, the physics steps, and the bounds are checked against
frame k + 1 (mjlab advances its clock once at reset). Past the last frame
the frame stays at the last.

| `tracking` field | What |
|---|---|
| `motion`, `frames` | the reference file and its frame count |
| `anchor_body` | the torso the reference terms are taken against |
| `body_names` | the bodies `motion.npz` holds, in its order; the first is the floating base |
| `terminations` | `anchor_z_m`, `anchor_tilt`, `end_effector_z_m`, `end_effectors` (below) |
| `start` | `pose_range` and `velocity_range` (`x y z roll pitch yaw`, added to frame 0's root pose and velocity), `joint_position_range` (added to every joint), `soft_limit_factor` (joints clipped to their range narrowed about its middle by this factor) |
| `pushes` | `interval_s` (`[lo, hi]` between pushes) and `velocity_range` (added to the root's world velocity), or null |
| `noise` | per observation term, `[lo, hi]` of uniform noise added in training |
| `randomization` | training's start-up draws: `encoder_bias` (`[lo, hi]` per joint, read by a `biased` joint term), `body_com` (`body`, per-axis offsets of its centre of mass), `foot_friction` (`geoms`, one shared `range` value) |
| `lag_ms` | the command lag of the task the policy is judged under (whole control ticks, uniform in `[0, lag_ms]`) |
| `criterion` | what success is: the clip played to its end with no bound broken |

`start`, `pushes`, `noise`, `randomization` and `lag_ms` are the
conditions an evaluation judges under; a robot brings its own.

| `source` | Width | Value |
|---|---|---|
| `command motion` | 2 × joints | the reference's `joint_pos` then `joint_vel` at the frame |
| `motion_anchor_pos_b` | 3 | the reference anchor's position minus the robot anchor's, rotated into the robot anchor's frame (m) |
| `motion_anchor_ori_b` | 6 | the rotation from the robot anchor to the reference anchor (R = R_robotᵀ R_ref), its first two columns read row by row: R00, R01, R10, R11, R20, R21 |

The robot anchor is `anchor_body`'s pose in the world (MuJoCo's `xpos`,
`xquat`). A trial fails, at the first tick it happens, when:

- the robot anchor's height differs from the reference anchor's by more
  than `anchor_z_m`;
- the z of gravity projected into the robot anchor's frame differs from
  the reference anchor's by more than `anchor_tilt`;
- any end effector's height differs from the reference body's by more
  than `end_effector_z_m` (mjlab compares the reference carried to the
  robot's position and heading; a turn about the vertical leaves heights
  as they are).

The reference runtime's definitions are `trainnr/trainnr/deploy/tracking.py`;
they were checked against mjlab's own terms on mjlab's rollout states,
and the bounds against its terminations
([record](findings/g1-court-dance-gated-2026-11-23.json)). The gate judges
a tracking deployment through its clip under the cited evaluation's
conditions; pre-flight, attribution, the assay, staging on a scene and the
DDS gate are built for a walk's twist and refuse one by name.

## Reading the files

`scene.xml` is the trained model as MJCF, actuators and home keyframe
injected; its meshes are the robot bundle's, by name. `policy.onnx` takes
one observation of `onnx.input_width` float32 values (a batch of one)
and returns `onnx.output_width` actions.

What a runtime may rely on is this page and `load_manifest`; the
reference runtime is the executable definition. A runtime that wants to
check itself can run trainnr's gate against its own build of the same
deployment and compare the two records.
