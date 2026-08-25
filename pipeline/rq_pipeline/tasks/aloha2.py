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
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rq_pipeline.evaluate.harness import EpisodeProtocol
from rq_pipeline.evaluate.vision import CameraSpec

BUNDLE_XML = (
    Path(__file__).resolve().parents[3] / "robots" / "aloha2-nominal" / "aloha2.xml"
)
HOME_KEYFRAME = "neutral_pose"

ARMS = 2
SERVOS_PER_ARM = 7
SERVOS = ARMS * SERVOS_PER_ARM
ARM_SENSOR_WIDTH = 2 * SERVOS  # jointpos x14 then jointvel x14
LEFT_GRIPPER_POS_SLICE = slice(ARM_SENSOR_WIDTH, ARM_SENSOR_WIDTH + 3)  # referee
NEUTRAL_CTRL = [0.0, -0.96, 1.16, 0.0, -0.3, 0.0, 0.0084] * ARMS

# gym-aloha frame -> bundle frame (module docstring).
ACT_SIM_Y_SHIFT = -0.519
# gym-aloha's sample_box_pose: x in [0, 0.2], y in [0.4, 0.6], dropped from
# z=0.05. Translated, and resting on the table instead of dropped.
CUBE_HALF = 0.02
CUBE_SPAWN_X = (0.0, 0.2)
CUBE_SPAWN_Y = (0.4 + ACT_SIM_Y_SHIFT, 0.6 + ACT_SIM_Y_SHIFT)
CUBE_HOME = (0.1, 0.5 + ACT_SIM_Y_SHIFT, CUBE_HALF)
# gym-aloha's `top`: pos (0, 0.6, 0.8), fovy 78, aimed at the table's
# origin directly beneath it. Translated; orientation fixed to look
# straight down with image-up toward +y (the arms at the bottom of the
# frame is verified by eye against a dataset frame — docs/31).
TOP_CAMERA_POS = (0.0, 0.6 + ACT_SIM_Y_SHIFT, 0.8)
TOP_CAMERA_FOVY = 78
ALOHA_TOP_CAMERAS: tuple[CameraSpec, ...] = (
    CameraSpec("top", "top", width=640, height=480),
)

# Gripper channel: gym-aloha normalises the ACT sim's finger POSITION
# (close 0.01844 m, open 0.05800 m) to [0, 1] in both state and action.
# ALOHA 2's gripper drive spans ctrlrange 0.002..0.037; commands map
# onto that. The STATE side uses the finger's MEASURED travel: commanded
# to 0.002 the fingers meet pad-on-pad at 0.0078 m and stop (eight
# contacts, 2026-08-26), so 0.0078 is "closed" — normalising from the
# ctrl floor instead left a closed gripper reading 0.18 where the
# dataset reads 0.00 (found by replaying a human demo: every arm joint
# tracked at r >= 0.98, the gripper at 0.72).
ALOHA2_GRIPPER_CTRL_CLOSE = 0.002
ALOHA2_GRIPPER_CTRL_OPEN = 0.037
ALOHA2_GRIPPER_JOINT_CLOSED = 0.0078  # measured
ALOHA2_GRIPPER_JOINT_OPEN = 0.037  # measured, equals the ctrl ceiling
_GRIPPER_INDICES = (SERVOS_PER_ARM - 1, SERVOS - 1)  # 6 and 13

# FULLPHYSICS layout: [time, qpos(16 arm + 7 cube), qvel(...)]. The cube's
# free joint is declared after the arms, so its position is state 17..19.
# Pinned by test.
CUBE_STATE_SLICE = slice(17, 20)
CUBE_Z_STATE_INDEX = 19

# Episode design: 500 Hz physics, policies at 50 Hz (gym-aloha's DT=0.02),
# 400 policy steps = 8 s like the public episodes, judged over the last
# 0.5 s.
_STEPS = 4000
_CONTROL_INTERVAL = 10
_TRIALS = 4
_HOLD_STEPS = 250
_TRANSFERRED_HEIGHT_M = 0.06  # cube centre 4 cm above resting
_HELD_RADIUS_M = 0.05  # cube centre within this of the left gripper referee


@dataclass(frozen=True)
class ALOHA2Task:
    """A composed scene and the protocol that scores episodes in it."""

    name: str
    spec: Any
    protocol: EpisodeProtocol
    cameras: tuple[CameraSpec, ...]


def gripper_ctrl_from_normalized(value: float) -> float:
    """gym-aloha's [0, 1] gripper command -> ALOHA 2 gripper ctrl (m)."""
    clipped = min(1.0, max(0.0, float(value)))
    span = ALOHA2_GRIPPER_CTRL_OPEN - ALOHA2_GRIPPER_CTRL_CLOSE
    return ALOHA2_GRIPPER_CTRL_CLOSE + clipped * span


def gripper_normalized_from_joint(position: float) -> float:
    """ALOHA 2 finger jointpos (m) -> gym-aloha's [0, 1] state, over
    the finger's measured travel (closed pads to full open)."""
    span = ALOHA2_GRIPPER_JOINT_OPEN - ALOHA2_GRIPPER_JOINT_CLOSED
    value = (float(position) - ALOHA2_GRIPPER_JOINT_CLOSED) / span
    return min(1.0, max(0.0, value))


def act_sim_state(sensordata: Any) -> Any:
    """The 14-d `observation.state` a gym-aloha checkpoint expects, from
    the bundle's sensors: joints pass through, grippers normalised."""
    import numpy as np  # noqa: PLC0415

    state = np.asarray(sensordata[:SERVOS], dtype=np.float32).copy()
    for index in _GRIPPER_INDICES:
        state[index] = gripper_normalized_from_joint(state[index])
    return state


def ctrl_from_act_sim_action(action: Any) -> Any:
    """A gym-aloha 14-d action -> the bundle's 14 actuator commands."""
    import numpy as np  # noqa: PLC0415

    ctrl = np.asarray(action, dtype=np.float64).copy()
    for index in _GRIPPER_INDICES:
        ctrl[index] = gripper_ctrl_from_normalized(ctrl[index])
    return ctrl


def act_sim_vision_policy(policy: Any) -> Any:
    """Wrap a gym-aloha-trained VisionPolicy for the bundle.

    The harness hands the policy `observation.state` as the bundle's
    raw fourteen jointpos values and expects fourteen actuator commands
    back; a checkpoint trained on the public data speaks gym-aloha's
    convention on both sides (normalised grippers). This wrapper
    translates both directions and nothing else — the images pass
    through untouched, because the `top` camera was placed to match.
    """
    from rq_pipeline.evaluate.vision import VisionPolicy  # noqa: PLC0415

    def act(step: int, observation: dict[str, Any]) -> Any:
        translated = dict(observation)
        translated["observation.state"] = act_sim_state(
            observation["observation.state"]
        )
        return ctrl_from_act_sim_action(policy.act(step, translated))

    return VisionPolicy(name=policy.name, act=act, reset=policy.reset)


def _rig_scene(name: str, bundle_xml: Path) -> Any:
    import mujoco  # noqa: PLC0415 - sim extra

    scene = mujoco.MjSpec.from_file(str(bundle_xml))
    scene.modelname = name
    # The top camera renders 640x480; the D405 cameras 1280x720. The
    # offscreen framebuffer must cover the largest.
    scene.visual.global_.offwidth = 1280
    scene.visual.global_.offheight = 720
    return scene


def _extend_keyframes(scene: Any, extra_qpos: list[float]) -> None:
    """Append a new free joint's pose to every bundle keyframe.

    The bundle's keyframes carry sixteen arm values; MuJoCo pads a
    shorter keyframe with ZEROS, not with the body's declared pose — so
    a cube added after the arms would start at the origin, half-buried
    in the table, with a null quaternion (measured 2026-08-26). Every
    task that adds a free body extends the keyframes explicitly.
    """
    for key in scene.keys:
        key.qpos = [*key.qpos, *extra_qpos]


ACT_SIM_LOOK = "act_sim"
ALOHA2_LOOK = "aloha2"
_ACT_SIM_GREY = [0.2, 0.2, 0.2, 1.0]  # their tabletop rgba
_ACT_SIM_ARM = [0.5, 0.5, 0.5, 1.0]  # their meshes carry no material: MuJoCo's default
_HIDDEN_GROUP = 4  # renderers show geom groups 0-2 by default


def _act_sim_cosmetics(scene: Any) -> None:
    """Make the bundle LOOK like the original ACT simulator.

    Same geometry, same dynamics, their appearance: a flat grey table
    on black, grey arms, no frame, no floor, flat lighting without
    shadows. A checkpoint trained on gym-aloha's renders then sees the
    scene it was trained on, and whatever it scores differently here
    is the DYNAMICS gap — not the wood grain (measured side by side
    2026-08-26: the viewpoint matched, the cosmetics did not).
    """
    for texture in scene.textures:
        # The gradient skybox becomes black; the floor is hidden anyway.
        if hasattr(texture, "rgb1"):
            texture.rgb1 = [0, 0, 0]
            texture.rgb2 = [0, 0, 0]
    for material in scene.materials:
        if material.name == "black":
            material.rgba = _ACT_SIM_ARM
    for geom in scene.geoms:
        if geom.classname is not None and geom.classname.name == "frame":
            geom.group = _HIDDEN_GROUP
        elif geom.meshname in ("tabletop", "tablelegs"):
            geom.material = ""
            geom.rgba = _ACT_SIM_GREY
        elif geom.name == "floor":
            geom.group = _HIDDEN_GROUP
    # Their lighting: headlight ambient 0.4, three dim directional lights,
    # one of which casts shadows. Side by side (2026-08-26) our table and
    # arms rendered brighter than theirs; the dimmer headlight closes
    # most of that, and the bundle's one light keeps casting shadows as
    # their third light does.
    scene.visual.headlight.ambient = [0.4, 0.4, 0.4]
    scene.visual.headlight.diffuse = [0.3, 0.3, 0.3]


def build_transfer_cube(
    bundle_xml: Path = BUNDLE_XML, look: str = ALOHA2_LOOK
) -> ALOHA2Task:
    """Transfer cube, gym-aloha's protocol on the identified rig.

    `look` selects the appearance: the bundle's own (`aloha2`) or the
    ACT simulator's (`act_sim`) for checkpoints trained on its renders.
    """
    import mujoco  # noqa: PLC0415 - sim extra
    import numpy as np  # noqa: PLC0415

    if look not in (ALOHA2_LOOK, ACT_SIM_LOOK):
        raise ValueError(
            f"look must be {ALOHA2_LOOK!r} or {ACT_SIM_LOOK!r}, got {look!r}"
        )
    scene = _rig_scene(f"aloha2-transfer-cube-{look}", bundle_xml)
    if look == ACT_SIM_LOOK:
        _act_sim_cosmetics(scene)
    cube = scene.worldbody.add_body(name="cube", pos=list(CUBE_HOME))
    cube.add_freejoint()
    # gym-aloha's red_box, verbatim: 2 cm half-size, its contact params
    # (their XML gives solimp three values; MjSpec wants all five, so the
    # parser's defaults for midpoint and power are written out).
    cube.add_geom(
        name="cube_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[CUBE_HALF] * 3,
        condim=4,
        solimp=[2, 1, 0.01, 0.5, 2],
        solref=[0.01, 1],
        friction=[1, 0.005, 0.0001],
        rgba=[1, 0, 0, 1],
    )
    _extend_keyframes(scene, [*CUBE_HOME, 1.0, 0.0, 0.0, 0.0])
    scene.worldbody.add_camera(
        name="top",
        pos=list(TOP_CAMERA_POS),
        xyaxes=[1, 0, 0, 0, 1, 0],
        fovy=TOP_CAMERA_FOVY,
    )
    # Referee: where the left gripper is, so success can ask "is the cube
    # held by the LEFT arm" without a contact query. A real arm computes
    # its own forward kinematics, so policies may see this too.
    scene.add_sensor(
        name="referee/left_gripper_pos",
        type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
        objtype=mujoco.mjtObj.mjOBJ_BODY,
        objname="left/gripper_link",
    )

    def perturb(trial: int, home: Any) -> Any:
        # Deterministic paired starts across the spawn box: four corners
        # pulled in by 2 cm, matching gym-aloha's per-episode cube draw
        # in spirit while keeping trials identical across policies.
        initial = home.copy()
        fx, fy = ((0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9))[trial % 4]
        initial[CUBE_STATE_SLICE.start] = CUBE_SPAWN_X[0] + fx * (
            CUBE_SPAWN_X[1] - CUBE_SPAWN_X[0]
        )
        initial[CUBE_STATE_SLICE.start + 1] = CUBE_SPAWN_Y[0] + fy * (
            CUBE_SPAWN_Y[1] - CUBE_SPAWN_Y[0]
        )
        return initial

    def success(states: Any, sensors: Any) -> bool:
        tail_cube = states[-_HOLD_STEPS:, CUBE_STATE_SLICE]
        tail_left = sensors[-_HOLD_STEPS:, LEFT_GRIPPER_POS_SLICE]
        lifted = bool(np.min(tail_cube[:, 2]) > _TRANSFERRED_HEIGHT_M)
        held = bool(
            np.max(np.linalg.norm(tail_cube - tail_left, axis=1)) < _HELD_RADIUS_M
        )
        return lifted and held

    return ALOHA2Task(
        name="transfer_cube",
        spec=scene,
        protocol=EpisodeProtocol(
            trials=_TRIALS,
            steps=_STEPS,
            control_interval=_CONTROL_INTERVAL,
            perturb=perturb,
            success=success,
            home=HOME_KEYFRAME,
        ),
        cameras=ALOHA_TOP_CAMERAS,
    )
