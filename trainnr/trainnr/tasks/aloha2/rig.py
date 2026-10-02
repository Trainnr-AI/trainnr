"""The ALOHA 2 rig, as every task here sees it: the bundle, the arms'
names and slices, the gripper's units, the ACT-simulator frame and
looks, the scene helpers (bundle scene, free box with keyframes, the top
camera and the referee sensors, the paired corners). Nothing task-
specific: `transfer_cube.py` and `kitting.py` compose scenes from this.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from trainnr.bundles.locate import bundle_file, require_bundle_file
from trainnr.physics.servo_dr import named_joints, scale_servo_dynamics
from trainnr.protocol import CameraSpec
from trainnr.tasks.scene import (
    FLOOR_GEOM,
    GeomGroup,
    add_free_box,
    pin_nominal_options,
    set_render_budget,
)

BUNDLE_XML = bundle_file("aloha2-nominal", "aloha2.xml")
HOME_KEYFRAME = "neutral_pose"

# The bundle names the arms and the pad geoms; the right arm's part is
# declared first, which pins the state slices below.
ARM_NAMES = ("left", "right")
PART_ORDER = ("right", "left")
ARM_PREFIXES = tuple(f"{arm}/" for arm in ARM_NAMES)
FINGERS = ("left", "right")  # the pad geoms: <arm>/<finger>_g1
ARMS = len(ARM_NAMES)
SERVOS_PER_ARM = 7
SERVOS = ARMS * SERVOS_PER_ARM
ARM_SENSOR_WIDTH = 2 * SERVOS  # jointpos x14 then jointvel x14
# Referee framepos sensors, after the arm's block: left gripper first.
# (Both grippers' referees are DECLARED in the scenes for tools and
# future predicates; only the left one is consumed today, so only its
# slice exists — an unused right slice implied a check nobody wrote.)
LEFT_GRIPPER_POS_SLICE = slice(ARM_SENSOR_WIDTH, ARM_SENSOR_WIDTH + 3)
NEUTRAL_CTRL = [0.0, -0.96, 1.16, 0.0, -0.3, 0.0, 0.0084] * ARMS

# gym-aloha frame -> bundle frame (module docstring).
ACT_SIM_Y_SHIFT = -0.519
# gym-aloha's sample_box_pose: x in [0, 0.2], y in [0.4, 0.6], dropped from
# z=0.05. Translated, and resting on the table instead of dropped.
CUBE_BODY = "cube"  # the transfer task's object
CUBE_HALF = 0.02
CUBE_SPAWN_X = (0.0, 0.2)
CUBE_SPAWN_Y = (0.4 + ACT_SIM_Y_SHIFT, 0.6 + ACT_SIM_Y_SHIFT)
CUBE_HOME = (0.1, 0.5 + ACT_SIM_Y_SHIFT, CUBE_HALF)
CUBE_SPAWN_INSET = 0.1  # the paired corners' pull-in, as a fraction of each side
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

# Episode design: 500 Hz physics (scene.NominalOptions.TIMESTEP), policies
# at 50 Hz (gym-aloha's DT=0.02),
# 400 policy steps = 8 s like the public episodes, judged over the last
# 0.5 s.
_STEPS = 4000
_HOLD_STEPS = 250
# Task names: the registry key, `Task.name`, and the gym id's last part.
TRANSFER_CUBE = "transfer_cube"
KITTING = "kitting"
RIG = "aloha2"
_TRANSFERRED_HEIGHT_M = 0.06  # cube centre 4 cm above resting
_HELD_RADIUS_M = 0.05  # cube centre within this of the left gripper referee
# Milestone zero on both tasks: an object displaced this far from its
# start was touched. Below the pad-jitter a settling box shows (<1 mm),
# far below a nudge (1-2 cm measured when a close missed the part).
MOVED_M = 0.01


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
    """A gym-aloha 14-d action -> the bundle's 14 actuator commands.

    With `act_sim_state`, the two halves of the gym-aloha convention a
    checkpoint trained on the public data speaks (normalised grippers);
    a tool playing such a checkpoint applies them around the model
    call. Images pass through untouched, because the `top` camera was
    placed to match.
    """
    import numpy as np  # noqa: PLC0415

    ctrl = np.asarray(action, dtype=np.float64).copy()
    for index in _GRIPPER_INDICES:
        ctrl[index] = gripper_ctrl_from_normalized(ctrl[index])
    return ctrl


def scale_dynamics(spec: Any, *, damping_scale: float, gain_scale: float) -> None:
    """Domain randomisation on an UNCOMPILED spec: every ARM joint's
    damping and every actuator's stiffness, scaled in place.

    The both-terms gain rule and the story behind it live once, in
    `physics/servo_dr.py`; this wrapper is the ALOHA rig's own joint
    predicate (arm joints by name — the scene's free objects carry
    damping too and must not be randomised) and the name every caller
    here already uses.
    """
    scale_servo_dynamics(
        spec,
        damping_scale=damping_scale,
        gain_scale=gain_scale,
        joints=named_joints(ARM_PREFIXES),
    )


def _rig_scene(name: str, bundle_xml: Path) -> Any:
    import mujoco  # noqa: PLC0415 - sim extra

    scene = mujoco.MjSpec.from_file(str(require_bundle_file(bundle_xml)))
    scene.modelname = name
    pin_nominal_options(scene)
    # The top camera renders 640x480; the D405 cameras 1280x720. The
    # offscreen framebuffer must cover the largest.
    # Menagerie's scene asks for an 8192x8192 shadow map — a screenshot
    # setting. Every harness rollout and demo frame renders through this
    # scene, so it is a per-frame cost (measured 2026-08-26: 31 ms per
    # 640x480 frame on the RTX at 8192). Physics is untouched by it.
    set_render_budget(scene)
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
LOOKS = (ALOHA2_LOOK, ACT_SIM_LOOK)
# The public demos the T0-T2 rungs trained on (gym-aloha's sim, docs/31).
PUBLIC_TRANSFER_CUBE_DEMOS = "lerobot/aloha_sim_transfer_cube_human"


class ActionSpace:
    """What a checkpoint speaks: gym-aloha's normalised grippers (the
    public-demo checkpoints) or the bundle's own ctrl (our demos, T5)."""

    ACT_SIM = "act_sim"
    BUNDLE = "bundle"


_ACT_SIM_GREY = [0.2, 0.2, 0.2, 1.0]  # their tabletop rgba
_ACT_SIM_ARM = [0.5, 0.5, 0.5, 1.0]  # their meshes carry no material: MuJoCo's default


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
            geom.group = GeomGroup.HIDDEN
        elif geom.meshname in ("tabletop", "tablelegs"):
            geom.material = ""
            geom.rgba = _ACT_SIM_GREY
        elif geom.name == FLOOR_GEOM:
            geom.group = GeomGroup.HIDDEN
    # Their lighting: headlight ambient 0.4, three dim directional lights,
    # one of which casts shadows. Side by side (2026-08-26) our table and
    # arms rendered brighter than theirs; the dimmer headlight closes
    # most of that, and the bundle's one light keeps casting shadows as
    # their third light does.
    scene.visual.headlight.ambient = [0.4, 0.4, 0.4]
    scene.visual.headlight.diffuse = [0.3, 0.3, 0.3]


def _task_scene(task: str, bundle_xml: Path, look: str) -> Any:
    """Load the rig, validate + apply the look — every builder's opening."""
    if look not in LOOKS:
        raise ValueError(
            f"look must be {ALOHA2_LOOK!r} or {ACT_SIM_LOOK!r}, got {look!r}"
        )
    scene = _rig_scene(f"aloha2-{task}-{look}", bundle_xml)
    if look == ACT_SIM_LOOK:
        _act_sim_cosmetics(scene)
    return scene


def _add_free_box(scene: Any, name: str, pos: Any, half: float, rgba: Any) -> None:
    """A free-floating box with gym-aloha's red_box contact params,
    keyframes extended (see `_extend_keyframes` for why that is not
    optional). The transfer cube and both kitting parts are this box —
    same physics, different colour — so it exists once.

    Contact params are their XML verbatim; it gives solimp three values
    and MjSpec wants all five, so the parser's defaults for midpoint
    and power are written out.
    """
    add_free_box(
        scene,
        name,
        pos,
        half,
        rgba=rgba,
        condim=4,
        solimp=[2, 1, 0.01, 0.5, 2],
        solref=[0.01, 1],
        friction=[1, 0.005, 0.0001],
    )
    _extend_keyframes(scene, [*pos, 1.0, 0.0, 0.0, 0.0])


def _add_top_camera_and_referees(scene: Any) -> None:
    """The task instrumentation both scenes share: the translated
    gym-aloha `top` camera, and a framepos referee per gripper so
    success can ask "is the object at the LEFT/RIGHT pads" without a
    contact query. A real arm computes its own forward kinematics, so
    policies may see the referees too. Each referee reads the finger
    PADS (the inner collision sphere of the arm's left finger), not the
    wrist body: "held" means at the pads, which sit ~10 cm past the
    wrist.
    """
    import mujoco  # noqa: PLC0415 - sim extra

    scene.worldbody.add_camera(
        name="top",
        pos=list(TOP_CAMERA_POS),
        xyaxes=[1, 0, 0, 0, 1, 0],
        fovy=TOP_CAMERA_FOVY,
    )
    for arm in ARM_NAMES:
        scene.add_sensor(
            name=f"referee/{arm}_gripper_pos",
            type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
            objtype=mujoco.mjtObj.mjOBJ_GEOM,
            objname=f"{arm}/left_g1",
        )


ARM_IK_JOINTS = {
    arm: tuple(
        f"{arm}/{name}"
        for name in (
            "waist",
            "shoulder",
            "elbow",
            "forearm_roll",
            "wrist_angle",
            "wrist_rotate",
        )
    )
    for arm in ARM_NAMES
}
ARM_CTRL_SLICES = {
    arm: slice(i * SERVOS_PER_ARM, (i + 1) * SERVOS_PER_ARM)
    for i, arm in enumerate(ARM_NAMES)
}
