"""ALOHA 2 task builders — the sim side of the end-to-end test.

Each builder loads the `aloha2-nominal` bundle as an `MjSpec` (the
scene IS the rig: frame, table, four D405 cameras), adds the task's
objects, referee sensors and any extra cameras, and returns the spec
with the protocol that judges it. Conventions inherited from so101.py:
the referee's sensors ride AFTER the robot's (sensordata[0:28] is the
arm's fourteen jointpos + fourteen jointvel), a task never leaks its
target through a sensor, and success reads privileged STATE — no
sensor carries the cube's pose.

Two conventions that are new here:

- **Episodes start at the bundle's keyframe.** `EpisodeProtocol.home`
  is `neutral_pose`; the reset state (both arms straight up) jams the
  grippers together on the way down — measured, docs/31.
- **The gym-aloha frame is one shift away.** The public ALOHA data and
  the released ACT checkpoints were made in the original ACT
  simulator, whose arm bases sit at y=+0.5 with its table at y=+0.6;
  Menagerie's bases sit at y=-0.019 with the table centred at the
  origin, same x. The transfer-cube spawn box and the `top` camera are
  translated by that shift so a checkpoint sees the scene it was
  trained on — minus the dynamics, which is the experiment.

`transfer_cube` follows gym-aloha's protocol: the cube spawns on the
right arm's side, the right arm picks it up, hands it to the left, and
success (their reward 4) is the LEFT gripper holding the cube clear of
the table at the end.

The package: `rig.py` (the rig and the scene helpers), `transfer_cube.py`
and `kitting.py` (the tasks), `expert.py` (the scripted kitting expert).
Every public name is re-exported here, so `from trainnr.tasks.aloha2
import build_kitting` reads as it did when this was one module, and the
`trainnr.tasks` entry point still names the package.
"""

from __future__ import annotations

from trainnr.tasks.aloha2.expert import (
    KITTING_EXPERT,
    ArmSession,
    Episode,
    KittingChoreography,
    KittingStats,
    Waypoint,
    expert_stamp,
    grasp_axis,
    kitting_waypoints,
    scripted_kitting_episode,
)
from trainnr.tasks.aloha2.kitting import (
    KITTING_INSTRUCTION,
    KITTING_SPEC,
    PART_HOME,
    PART_RGBA,
    PART_STATE_SLICE,
    TRAY_RGBA,
    KittingSpec,
    build_kitting,
    part_body,
    part_qpos_slice,
)
from trainnr.tasks.aloha2.rig import (
    ACT_SIM_LOOK,
    ACT_SIM_Y_SHIFT,
    ALOHA2_GRIPPER_CTRL_CLOSE,
    ALOHA2_GRIPPER_CTRL_OPEN,
    ALOHA2_GRIPPER_JOINT_CLOSED,
    ALOHA2_GRIPPER_JOINT_OPEN,
    ALOHA2_LOOK,
    ALOHA_TOP_CAMERAS,
    ARM_CTRL_SLICES,
    ARM_IK_JOINTS,
    ARM_NAMES,
    ARM_PREFIXES,
    ARM_SENSOR_WIDTH,
    ARMS,
    BUNDLE_XML,
    CUBE_BODY,
    CUBE_HALF,
    CUBE_HOME,
    CUBE_SPAWN_INSET,
    CUBE_SPAWN_X,
    CUBE_SPAWN_Y,
    CUBE_STATE_SLICE,
    CUBE_Z_STATE_INDEX,
    FINGERS,
    HOME_KEYFRAME,
    KITTING,
    LEFT_GRIPPER_POS_SLICE,
    LOOKS,
    MOVED_M,
    NEUTRAL_CTRL,
    PART_ORDER,
    PUBLIC_TRANSFER_CUBE_DEMOS,
    RIG,
    SERVOS,
    SERVOS_PER_ARM,
    TOP_CAMERA_FOVY,
    TOP_CAMERA_POS,
    TRANSFER_CUBE,
    ActionSpace,
    act_sim_state,
    ctrl_from_act_sim_action,
    gripper_ctrl_from_normalized,
    gripper_normalized_from_joint,
    scale_dynamics,
)
from trainnr.tasks.aloha2.transfer_cube import (
    TRANSFER_CUBE_INSTRUCTION,
    build_transfer_cube,
)

__all__ = [
    "ACT_SIM_LOOK",
    "ACT_SIM_Y_SHIFT",
    "ALOHA2_GRIPPER_CTRL_CLOSE",
    "ALOHA2_GRIPPER_CTRL_OPEN",
    "ALOHA2_GRIPPER_JOINT_CLOSED",
    "ALOHA2_GRIPPER_JOINT_OPEN",
    "ALOHA2_LOOK",
    "ALOHA_TOP_CAMERAS",
    "ARMS",
    "ARM_CTRL_SLICES",
    "ARM_IK_JOINTS",
    "ARM_NAMES",
    "ARM_PREFIXES",
    "ARM_SENSOR_WIDTH",
    "BUNDLE_XML",
    "CUBE_BODY",
    "CUBE_HALF",
    "CUBE_HOME",
    "CUBE_SPAWN_INSET",
    "CUBE_SPAWN_X",
    "CUBE_SPAWN_Y",
    "CUBE_STATE_SLICE",
    "CUBE_Z_STATE_INDEX",
    "FINGERS",
    "HOME_KEYFRAME",
    "KITTING",
    "KITTING_EXPERT",
    "KITTING_INSTRUCTION",
    "KITTING_SPEC",
    "LEFT_GRIPPER_POS_SLICE",
    "LOOKS",
    "MOVED_M",
    "NEUTRAL_CTRL",
    "PART_HOME",
    "PART_ORDER",
    "PART_RGBA",
    "PART_STATE_SLICE",
    "PUBLIC_TRANSFER_CUBE_DEMOS",
    "RIG",
    "SERVOS",
    "SERVOS_PER_ARM",
    "TOP_CAMERA_FOVY",
    "TOP_CAMERA_POS",
    "TRANSFER_CUBE",
    "TRANSFER_CUBE_INSTRUCTION",
    "TRAY_RGBA",
    "ActionSpace",
    "ArmSession",
    "Episode",
    "KittingChoreography",
    "KittingSpec",
    "KittingStats",
    "Waypoint",
    "act_sim_state",
    "build_kitting",
    "build_transfer_cube",
    "ctrl_from_act_sim_action",
    "expert_stamp",
    "grasp_axis",
    "gripper_ctrl_from_normalized",
    "gripper_normalized_from_joint",
    "kitting_waypoints",
    "part_body",
    "part_qpos_slice",
    "scale_dynamics",
    "scripted_kitting_episode",
]
