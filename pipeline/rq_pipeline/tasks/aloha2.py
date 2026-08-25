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
from typing import Any, NamedTuple

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
    # The referee reads the finger PADS (the inner collision spheres of
    # each arm's left finger), not the wrist body: "held" means the
    # cube is at the pads, and the pads sit ~10 cm beyond the wrist.
    for arm in ("left", "right"):
        scene.add_sensor(
            name=f"referee/{arm}_gripper_pos",
            type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
            objtype=mujoco.mjtObj.mjOBJ_GEOM,
            objname=f"{arm}/left_g1",
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


# ------------------------------------------------------------- kitting --
# T5's industrial task (docs/31): parts into a tray's slots, one part
# per arm so the task is bimanual by reach. Everything below serves two
# consumers: `build_kitting` gives the harness the scene and the judge;
# `kitting_waypoints` gives the DEMO GENERATOR (tools/kitting-demos.py)
# the Cartesian choreography it converts to joint-space commands via
# IK — chained solves, each warm-starting the next, because a cold IK
# jump to a low target can stall in a local minimum (measured: a lone
# 6 cm-height target failed at 69 mm where the same target approached
# from 10 cm above converges).

TRAY_CENTER = (0.0, -0.02)
SLOT_HALF = 0.045
SLOT_WALL = 0.008
SLOT_WALL_HEIGHT = 0.015
_SLOT_OFFSET_X = 0.09
# One slot per arm, mirrored about the tray centre; the part that
# starts on the RIGHT goes into the right slot. Derived, not restated —
# the tray centre existed beside hand-copied slot y values once.
SLOT_CENTERS = {
    "right": (TRAY_CENTER[0] + _SLOT_OFFSET_X, TRAY_CENTER[1]),
    "left": (TRAY_CENTER[0] - _SLOT_OFFSET_X, TRAY_CENTER[1]),
}
PART_HALF = 0.02
# Spawn bands, one per arm side, inside the proven reach envelope
# (transfer_cube's spawn box, mirrored for the left arm).
PART_SPAWN = {
    "right": ((0.14, 0.24), (0.28 + ACT_SIM_Y_SHIFT, 0.44 + ACT_SIM_Y_SHIFT)),
    "left": ((-0.24, -0.14), (0.28 + ACT_SIM_Y_SHIFT, 0.44 + ACT_SIM_Y_SHIFT)),
}
PART_HOME = {
    "right": (0.19, 0.36 + ACT_SIM_Y_SHIFT, PART_HALF),
    "left": (-0.19, 0.36 + ACT_SIM_Y_SHIFT, PART_HALF),
}
# FULLPHYSICS: qpos = 16 arm + 7 right part + 7 left part; positions at
# 17..19 and 24..26. Pinned by test.
PART_STATE_SLICE = {"right": slice(17, 20), "left": slice(24, 27)}
# 28 s at 500 Hz: two sequential pick-places PLUS the closed-loop
# corrections and one grasp retry per arm. The budget was 18 s and
# every robustness fix shifted which trial's endgame got truncated —
# the whack-a-mole was the clock, not the choreography.
_KITTING_STEPS = 14000
# The choreographer's judgement thresholds, named for the lint and the
# reader alike: a lift that left the part below this never lifted it,
# and closed-loop grip corrections stop inside this radius.
_LIFT_CHECK_Z_M = 0.05
_CORRECTION_DONE_M = 0.008
_PART_IN_SLOT_XY_M = 0.035
_PART_IN_SLOT_Z_M = 0.045

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
    for arm in ("left", "right")
}
ARM_CTRL_SLICES = {"left": slice(0, 7), "right": slice(7, 14)}
# The grasp approach: near-horizontal, tilted 45 degrees down and
# pointing inward (away from the arm's own base). A straight-down
# approach is IMPOSSIBLE on this gripper: the base housing's collision
# meshes reach the table before the pads reach a 4 cm part (measured
# via contacts: 1-8 cm of penetration in every vertical solve; tilted
# solves are clean). An adaptive base-toward-target axis was tried and
# REVERTED — its y-tilt degraded the wrist pose and every trial failed.


def grasp_axis(arm: str, target_xy: Any) -> tuple[float, float, float]:
    """Fixed inward tilt — the adaptive base-to-target version made
    every trial fail (the y-tilt degrades the wrist pose); the fixed
    axis carried 2/4. target_xy stays in the signature for the day a
    smarter axis earns its way back with evidence."""
    del target_xy
    return (-0.7, 0.0, -0.71) if arm == "right" else (0.7, 0.0, -0.71)


def build_kitting(bundle_xml: Path = BUNDLE_XML, look: str = ALOHA2_LOOK) -> ALOHA2Task:
    """Kitting: each arm places its side's part into its slot."""
    import mujoco  # noqa: PLC0415 - sim extra
    import numpy as np  # noqa: PLC0415

    if look not in (ALOHA2_LOOK, ACT_SIM_LOOK):
        raise ValueError(
            f"look must be {ALOHA2_LOOK!r} or {ACT_SIM_LOOK!r}, got {look!r}"
        )
    scene = _rig_scene(f"aloha2-kitting-{look}", bundle_xml)
    if look == ACT_SIM_LOOK:
        _act_sim_cosmetics(scene)

    # The tray: two shallow square wells built from wall boxes, static.
    for arm, (cx, cy) in SLOT_CENTERS.items():
        for index, (dx, dy, sx, sy) in enumerate(
            (
                (0.0, SLOT_HALF, SLOT_HALF + SLOT_WALL, SLOT_WALL),
                (0.0, -SLOT_HALF, SLOT_HALF + SLOT_WALL, SLOT_WALL),
                (SLOT_HALF, 0.0, SLOT_WALL, SLOT_HALF),
                (-SLOT_HALF, 0.0, SLOT_WALL, SLOT_HALF),
            )
        ):
            scene.worldbody.add_geom(
                name=f"slot_{arm}_wall{index}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[sx, sy, SLOT_WALL_HEIGHT],
                pos=[cx + dx, cy + dy, SLOT_WALL_HEIGHT],
                rgba=[0.35, 0.25, 0.15, 1.0],
            )

    colors = {"right": [1, 0, 0, 1], "left": [0, 0.55, 1, 1]}
    for arm in ("right", "left"):  # declaration order pins the state slices
        part = scene.worldbody.add_body(name=f"part_{arm}", pos=list(PART_HOME[arm]))
        part.add_freejoint()
        part.add_geom(
            name=f"part_{arm}_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[PART_HALF] * 3,
            condim=4,
            solimp=[2, 1, 0.01, 0.5, 2],
            solref=[0.01, 1],
            friction=[1, 0.005, 0.0001],
            rgba=colors[arm],
        )
    _extend_keyframes(scene, [*PART_HOME["right"], 1.0, 0.0, 0.0, 0.0])
    _extend_keyframes(scene, [*PART_HOME["left"], 1.0, 0.0, 0.0, 0.0])

    scene.worldbody.add_camera(
        name="top",
        pos=list(TOP_CAMERA_POS),
        xyaxes=[1, 0, 0, 0, 1, 0],
        fovy=TOP_CAMERA_FOVY,
    )
    for arm in ("left", "right"):
        scene.add_sensor(
            name=f"referee/{arm}_gripper_pos",
            type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
            objtype=mujoco.mjtObj.mjOBJ_GEOM,
            objname=f"{arm}/left_g1",
        )

    def perturb(trial: int, home: Any) -> Any:
        initial = home.copy()
        fx, fy = ((0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8))[trial % 4]
        for arm in ("right", "left"):
            (x_low, x_high), (y_low, y_high) = PART_SPAWN[arm]
            part = PART_STATE_SLICE[arm]
            initial[part.start] = x_low + fx * (x_high - x_low)
            initial[part.start + 1] = y_low + fy * (y_high - y_low)
        return initial

    def success(states: Any, sensors: Any) -> bool:
        del sensors
        tail = states[-_HOLD_STEPS:]
        for arm, (cx, cy) in SLOT_CENTERS.items():
            part = tail[:, PART_STATE_SLICE[arm]]
            xy_off = np.hypot(part[:, 0] - cx, part[:, 1] - cy)
            if not (
                bool(np.max(xy_off) < _PART_IN_SLOT_XY_M)
                and bool(np.max(part[:, 2]) < _PART_IN_SLOT_Z_M)
            ):
                return False
        return True

    return ALOHA2Task(
        name="kitting",
        spec=scene,
        protocol=EpisodeProtocol(
            trials=_TRIALS,
            steps=_KITTING_STEPS,
            control_interval=_CONTROL_INTERVAL,
            perturb=perturb,
            success=success,
            home=HOME_KEYFRAME,
        ),
        cameras=ALOHA_TOP_CAMERAS,
    )


# Per-segment (dz above part/slot, gripper normalized, seconds).
_PICK_PLACE_SEGMENTS = (
    ("above_part", 0.10, 1.0, 1.2),
    ("descend", 0.005, 1.0, 1.0),
    ("close", 0.005, 0.0, 0.6),
    ("lift", 0.12, 0.0, 1.0),
    ("above_slot", 0.12, 0.0, 1.4),
    ("lower", 0.045, 0.0, 1.0),
    ("open", 0.045, 1.0, 0.6),
    ("retreat", 0.14, 1.0, 0.8),
)


class Waypoint(NamedTuple):
    """One choreography beat — typed because the tuple's arity already
    lied once (an annotation described a removed 3-tuple shape while
    the code appended and unpacked four)."""

    name: str
    target: list[float]
    grip: float
    seconds: float


def kitting_waypoints(arm: str, part_xy: Any) -> list[Waypoint]:
    """The Cartesian choreography for one arm.

    Targets before `above_slot` track the PART's spawn position;
    from `above_slot` on they track the slot. The demo generator turns
    each into a joint waypoint via chained IK.
    """
    slot = SLOT_CENTERS[arm]
    plan = []
    for name, dz, grip, secs in _PICK_PLACE_SEGMENTS:
        anchor = slot if name in ("above_slot", "lower", "open", "retreat") else part_xy
        plan.append(Waypoint(name, [anchor[0], anchor[1], PART_HALF + dz], grip, secs))
    return plan


def scripted_kitting_episode(  # noqa: PLR0915 - a choreographer narrates
    model: Any,
    initial_state: Any,
    *,
    on_control: Any = None,
    stats: dict | None = None,
) -> tuple[Any, Any, Any]:
    """One scripted kitting demonstration, privileged, R7-correct.

    Arms act sequentially (right, then left), each running the
    choreography over ITS part's actual position read from state —
    which is why this is the demo GENERATOR's path, not a harness
    policy. Returns (states, sensors, actions): per-physics-step
    FULLPHYSICS states and sensors shaped exactly like the harness
    rollout (so the task's `success` judges them unchanged), and the
    50 Hz commanded-position actions a dataset records.

    `on_control(step, data)` is the generator's hook (render frames).
    """
    import mujoco  # noqa: PLC0415 - sim extra
    import numpy as np  # noqa: PLC0415

    from rq_pipeline.robot.arm_ik import solve_arm_ik  # noqa: PLC0415

    size = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data = mujoco.MjData(model)
    mujoco.mj_setState(
        model,
        data,
        np.asarray(initial_state, dtype=float),
        mujoco.mjtState.mjSTATE_FULLPHYSICS,
    )
    mujoco.mj_forward(model, data)
    scratch = mujoco.MjData(model)

    steps_total = _KITTING_STEPS
    states = np.empty((steps_total, size))
    sensors = np.empty((steps_total, model.nsensordata))
    actions: list[Any] = []
    ctrl = np.array(NEUTRAL_CTRL, dtype=float)
    physics_step = 0

    def advance(seconds: float, target_ctrl: Any) -> None:
        nonlocal physics_step, ctrl
        controls = max(1, round(seconds * 50))
        start = ctrl.copy()
        target = np.asarray(target_ctrl, dtype=float)
        for tick in range(controls):
            if physics_step >= steps_total:
                return
            blend = (tick + 1) / controls
            ctrl = start + blend * (target - start)
            actions.append(ctrl.copy())
            if on_control is not None:
                on_control(physics_step, data)
            data.ctrl[:] = ctrl
            for _ in range(_CONTROL_INTERVAL):
                if physics_step >= steps_total:
                    return
                mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)  # R7: one instant per row
                sensors[physics_step] = data.sensordata
                mujoco.mj_getState(
                    model,
                    data,
                    states[physics_step],
                    mujoco.mjtState.mjSTATE_FULLPHYSICS,
                )
                physics_step += 1

    for arm in ("right", "left"):
        part = data.qpos[
            PART_STATE_SLICE[arm].start - 1 : PART_STATE_SLICE[arm].stop - 1
        ].copy()
        ctrl_slice = ARM_CTRL_SLICES[arm]
        gripper_index = ctrl_slice.start + SERVOS_PER_ARM - 1
        pad_names = tuple(f"{arm}/{finger}_g1" for finger in ("left", "right"))
        pad_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in pad_names
        ]

        def command_reach(  # noqa: PLR0913 - loop bindings, not an interface
            target_xyz: Any,
            grip: float,
            secs: float,
            *,
            arm: str = arm,
            ctrl_slice: Any = ctrl_slice,
            gripper_index: int = gripper_index,
            pad_names: Any = pad_names,
        ) -> None:
            # Chained IK on the GRIP CENTRE itself (the pad midpoint) —
            # each solve warm-starts from the live pose. The site+offset
            # approach died twice; the postmortems live in arm_ik.py.
            scratch.qpos[:] = data.qpos
            reached = solve_arm_ik(
                model,
                scratch,
                site=f"{arm}/gripper",
                joints=ARM_IK_JOINTS[arm],
                grip_geoms=pad_names,
                target_pos=target_xyz,
                approach_axis=grasp_axis(arm, target_xyz[:2]),
                down_weight=0.5,
                pos_tol=0.008,
                max_iters=250,
                damping=5e-3,
            )
            if not reached:
                raise RuntimeError(
                    f"IK failed: {arm} arm to {target_xyz} — the choreography "
                    "must not pretend a reach happened"
                )
            target_ctrl = ctrl.copy()
            for joint, name in enumerate(ARM_IK_JOINTS[arm]):
                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                target_ctrl[ctrl_slice.start + joint] = scratch.qpos[
                    model.jnt_qposadr[joint_id]
                ]
            target_ctrl[gripper_index] = gripper_ctrl_from_normalized(grip)
            advance(secs, target_ctrl)

        def live_grip_error(target_xyz: Any, *, pad_ids: Any = pad_ids) -> Any:
            grip_center = 0.5 * (
                data.geom_xpos[pad_ids[0]] + data.geom_xpos[pad_ids[1]]
            )
            return grip_center - np.asarray(target_xyz)

        def part_z(*, arm: str = arm) -> float:
            return float(data.qpos[PART_STATE_SLICE[arm].start - 1 + 2])

        plan = kitting_waypoints(arm, part[:2])
        segment_index = 0
        grasp_retried = False
        while segment_index < len(plan):
            if physics_step >= steps_total:
                # Out of clock mid-choreography: solving IK against a
                # frozen sim and returning states that end mid-reach
                # with no signal was the silent-truncation gap the
                # review named. Record it and stop pretending.
                if stats is not None:
                    stats["truncated"] = True
                break
            name, target_xyz, grip, secs = plan[segment_index]
            segment_index += 1
            command_reach(target_xyz, grip, secs)
            if name == "lift" and part_z() < _LIFT_CHECK_Z_M and not grasp_retried:
                if stats is not None:
                    stats.setdefault("retries", []).append(
                        (arm, physics_step, round(part_z(), 3))
                    )
                # The referee's cheapest service: a lift that lifted
                # nothing restarts the grasp once (open, re-descend on
                # the part's CURRENT position — the failed close may
                # have nudged it).
                grasp_retried = True
                here = data.qpos[
                    PART_STATE_SLICE[arm].start - 1 : PART_STATE_SLICE[arm].stop - 1
                ]
                plan = kitting_waypoints(arm, here[:2])
                segment_index = 0
                continue
            if name in ("descend", "lower"):
                # Closed-loop correction: kinematic IK is exact but the
                # position servos carry ~3 cm of steady-state error
                # (waist friction), which put one pad inside the part's
                # footprint — closing then NUDGES the part instead of
                # grasping it (measured: parts ended 1-2 cm from spawn,
                # never lifted). Measure the LIVE grip error and lean
                # the command the other way before closing/releasing.
                # All THREE axes: correcting xy only re-IK'd into
                # postures whose z steady-state error reached +4 cm and
                # the pads closed above the cube (left arm, trial 2 —
                # grip rose 3.2 cm during close, pinched air).
                for _ in range(3):
                    error = live_grip_error(target_xyz)
                    if float(np.linalg.norm(error)) < _CORRECTION_DONE_M:
                        break
                    # CLAMPED compensation: linear correction only
                    # holds locally — mirroring a 14 cm miss once aimed
                    # the left arm across the table into impossible IK,
                    # but SKIPPING big errors regressed a working trial
                    # whose place drifted 6 cm. Correct in 5 cm bites;
                    # three rounds converge either way.
                    step_err = np.clip(error, -0.05, 0.05)
                    corrected = [
                        target_xyz[0] - step_err[0],
                        target_xyz[1] - step_err[1],
                        # Floored: a +5 cm z error once corrected the
                        # target to BELOW the table and IK rightly
                        # refused. The pads never need to go under 2 cm.
                        max(0.02, target_xyz[2] - step_err[2]),
                    ]
                    command_reach(corrected, grip, 0.5)
    if stats is not None:
        stats["steps_used_before_hold"] = physics_step
    # Hold the final pose for the rest of the protocol window.
    while physics_step < steps_total:
        advance(1.0, ctrl)
    return states, sensors, np.asarray(actions)
