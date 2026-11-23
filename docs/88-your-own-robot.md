# 88 Your own robot

trainnr ships no robot you must use. Any legged robot with an MJCF model
can be brought in through the MCP tools and walked, without code: onboard
the model, measure its servo on a bench if you can, declare its limits
over several joints if it has them, then declare a walk and train it.
This is the reference for each step (issues #100, #101 and #121).

## 1. The model

`onboard_robot("<file>.xml", "<name>")` copies it into the project as a
stamped bundle. A walk reads these off the model, by name or by shape:

| What | How it is found | Declare instead |
|---|---|---|
| the trunk | the body of the model's one free joint | `trunk` |
| the feet | sites, and collision geoms, whose name contains `foot` | `foot_sites`, `foot_geoms` |
| the IMU | sensors `imu_lin_vel` (a velocimeter) and `imu_ang_vel` (a gyro) | add them to the model |
| the stand pose | the keyframe named `home` or `stand` (any case), else the first | `stand_keyframe` |
| the driven joints | every hinge | `pinned_joints` holds some at the stand pose |
| the actuators | the model's own, when the walk says `model` | an actuator bundle (§2) |

A scene file is fine: its floor plane is left to the simulator's terrain.
Anything missing is refused by name with the setting that would supply it;
so is a stand pose outside its joints' ranges or one that breaks a declared
limit (§3). Bring the same robot again under its name and it is a new
revision of it, the old one kept.

## 2. The actuator

A walk names its actuator:

- **`model`**: the model's own actuators, as written. Recorded in the run as
  declared, not identified (`dr_basis: none: the model's own actuators,
  declared, not identified`), so no result reads as measured dynamics.
- **a library bundle** by name: `xl330.m6`, `sts3215.m6` and the other
  certified fits in `robots/actuator-bundles`.
- **a bundle fitted to your servo**: the `servo-bench` identification method
  measures it from a bench recording and writes one.

### The servo bench

Onboard the bench as a bundle of its own: a model with one hinge (the
servo's output) driving a pendulum arm whose mass and length are known,
and beside it `bench.json`:

```json
{"servo": "scs0009", "joint": "servo", "vin": 6.0, "kp": 32, "max_pwm": 1.0}
```

`servo` names the family (the bundle's `actuator`), `joint` the hinge, `vin`
the supply voltage, `kp` the P-gain register the servo ran at, `max_pwm` its
duty ceiling (1.0 unless known). Record the servo's commanded position
(`joint.command`) and its measured one (`joint.position`) at a known rate,
for example a ROS 2 MCAP's `/joint_commands` and `/joint_states`, through
`ingest_recording`: the MCAP reader takes a `sensor_msgs/JointState` topic
named `/joint_commands`, `/joint_command` or `/commands` (the first of
these the bag holds) as the command, and keeps any other JointState topic
under its own name (`joint_targets.joint.position`). The protocol is BAM's: steps and chirps, a few seconds
each, starting at rest; 12 s or more per recording; two or three arm masses
in separate recordings.

`identify_system(robot="<bench>", recording="<recording>")` then fits BAM's
first-tier servo law (Duclusaud et al., ICRA 2025): the firmware's duty
`clip(G (target - q))` driving a DC motor, `tau = S duty - B qd`, against
Coulomb friction, the armature and the arm, with the command delay. The
arm's gravity torque and inertia, read off the bench model, anchor the
torque scale and the armature. From positions alone the back-EMF and the
gearbox's viscous friction are one term, fitted as the motor's; the fit
says so. It writes a fit record (each parameter's interval and pinned
verdict, the held-out error) and `fits/actuators/<servo>.bundle.json` in
the library's format, with the firmware it measured in its `context`. A
walk names it as `robots/<bench>/fits/actuators/<servo>.bundle.json`.

One recording's interval is about that recording: an encoder's quantization
biases a run systematically (a synthetic SCS0009 bench recovers each
parameter it pins within 7 percent, the intervals narrower than that bias:
[record](findings/servo-bench-synthetic-scs0009-2026-11-22.json)), so
fit two or more recordings and read the cross-run spread `identify_system`
writes before trusting a number.

## 3. Limits over several joints

A robot's parts often cannot take every combination of in-range joint
angles: a head that rolls only while it faces forward, an ankle whose floor
rises as the knee bends, a hip whose collision map spans five joints. If the
robot's firmware blocks such poses, a policy trained without them is free to
use poses the robot refuses. Declare them in `limits.json` beside the model;
it is part of the bundle and stamped with it.

```json
{
  "schema": "trainnr-joint-limits/1",
  "source": "where the numbers come from",
  "limits": [ ... ]
}
```

Angles are in degrees. Each limit has a `name`, a `form` and an optional
`note`, and may carry `signs` (joint to +1 or -1, its angle multiplied
before the limit is read: a right leg written in the left leg's signs).

**`rule`**: forbidden when every condition holds. A condition names a
`joint` and one of `above`, `below`, `outside` [lo, hi] or `within` [lo, hi],
with an optional `sign`.

```json
{"name": "head roll and yaw interlock", "form": "rule",
 "when": [{"joint": "head_yaw", "outside": [-1, 1]},
          {"joint": "head_roll", "outside": [-1, 1]}]}
```

**`table`**: a joint's `minimum_by` (and or `maximum_by`) as a step table
over another joint (`over`): past `from`, the bound is the one of the first
row whose value is at or above the other joint's, the last row's beyond.

```json
{"name": "left knee and ankle", "form": "table",
 "joint": "left_ankle", "over": "left_knee", "from": 30,
 "minimum_by": [[40, -60], [45, -45], [55, -35], [60, -25], [68, 0]]}
```

**`grid`**: forbidden cells of a grid over several joints (`axes`, each
with its `values`); a cell is its indices. Between grid values the stricter
neighbour holds (a point is forbidden when any grid point around it is);
beyond an axis's end, its end value.

```json
{"name": "left hip map", "form": "grid",
 "axes": ["left_hip_pitch", "left_hip_yaw", "left_knee", "left_ankle", "left_hip_roll"],
 "values": {"left_hip_pitch": [-90, -75, ...], "left_hip_roll": [-12, -8, ...], ...},
 "forbidden": [[0, 1, 3, 1, 7], ...]}
```

Microduck Nano's firmware limits (its head interlock, knee and ankle table,
two hip maps and the both-legs rule) are nine limits in these three forms;
checked against a port of its firmware's `safety.cpp` on 200,000 random
poses, the two agree on every one
([record](findings/nano-limits-match-firmware-2026-11-22.json)).

Where the limits act:

- **the robot's page** lists each in words and the keyframes that break it;
- **the walk**: `limits` `terminate` (the default) ends an episode on a
  broken limit, `cost` pays for each, `off` ignores them; the run's identity
  names the limits' stamp;
- **the evaluation** records each trial's steps past a limit (`limit_steps`);
- **the deployment** carries `limits.json` beside the policy (`limits` in
  `deploy.json`, docs/87), and **the gate** fails a trial that breaks one,
  naming it.

Holding the limb where it is, as a firmware does when a command breaks a
limit, is not yet a training mode.

## 4. The walk

```
create_task(family="legged-walk", name="my-walk",
            settings={"robot": "my-robot", "actuator": "model",
                      "pinned_joints": [".*head.*"]})
check_task("my-walk")                                   # builds and learns
train_walk(task="my-walk", recipe="full")               # hours on a GPU
play_walk(experiment=..., checkpoint="all", viewer="film")  # a video per checkpoint
evaluate_walk(checkpoint="<experiment>")                # paired trials, intervals
export_deployment(experiment=..., checkpoint=..., name=...)
gate_deployment(deployment=...)                         # plain MuJoCo, every tick
```

`viewer="film"` films without a window: each checkpoint (one, several
comma-separated, or `all`) walks one world from the same seed under one
held command, forward at the top of the task's range, for 10 s or until
it falls, drawn from the chase camera at the control rate. Each lands
beside the run as `play-model_N/play.mp4` with `play.json` (the command,
the seconds walked, whether and when it fell, the distance covered);
several are joined in iteration order as `film.mp4`, the iteration on
the picture, and the run's page in the Studio plays them: on the
iteration timeline each checkpoint's film, on the `video` timeline the
whole run. One world and one seed: a look, not an evaluation. The
module, `trainnr_mjlab.walk_film`, also takes `--command`, `--seconds`
and `--seed`.

Other settings: `reset_z` (the trunk's spread above the stand pose at
reset), `lin_vel_x`, `lin_vel_y`, `ang_vel_z` (the commands, microduck's by
default), `dr_span`, `episode_s`, `trials`. The recipe is microduck's (its
rewards, events and PPO settings); the pose reward's widths come from
microduck's joints by name, a middle width for names it does not know. A
deployment whose policy drives only some joints holds the rest at the
stand pose (`action_joints` in `deploy.json`).
