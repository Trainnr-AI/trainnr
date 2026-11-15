# 87 The deployment manifest, `trainnr-deploy/1`

*Written 2026-11-15 (issue #106), for runtimes outside trainnr that run
an exported policy.*

`export_deployment` writes a folder: `policy.onnx`, `scene.xml` and
`deploy.json`. This page is `deploy.json`, the contract between the
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
target  = joints.default_pos + action.scale * action     # joint position targets
ctrl[joints.action_to_ctrl[i]] = target[i]               # then decimation physics steps
last_action = action                                     # clipped, unscaled
```

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
| `action` | `kind`, `scale` (one per joint), `offset`, `clip` (`[lo, hi]` or null) | yes |
| `observations` | below, in concatenation order | yes |
| `onnx` | `file`, `input` and `output` tensor names, `input_width`, `output_width`, `normalization` (always "folded into the graph": feed raw terms), `export_check_max_abs_diff` | yes |
| `scene` | `file`, `meshes`, `terrain`, `floor` (the trained floor's size, condim, priority, friction) | yes |
| `commands` | `twist`: the ranges the policy trained under for `lin_vel_x`, `lin_vel_y`, `ang_vel_z` (and `heading` when it was trained with one); `heading_gain` when it steered by heading | yes |
| `command_basis` | in words, which command curriculum stage the checkpoint had reached | |
| `termination` | `fell_over_deg`: the tilt at which the evaluation counted a fall | |
| `unitree` | for a Unitree robot only: the names their simulator and controller use | |

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

## Reading the files

`scene.xml` is the trained model as MJCF, actuators and home keyframe
injected; its meshes are the robot bundle's, by name. `policy.onnx` takes
one observation of `onnx.input_width` float32 values (a batch of one)
and returns `onnx.output_width` actions.

What a runtime may rely on is this page and `load_manifest`; the
reference runtime is the executable definition. A runtime that wants to
check itself can run trainnr's gate against its own build of the same
deployment and compare the two records.
