"""The walk on a Unitree Go2 the project onboarded (docs/77): mjlab
1.6's own velocity task wired to the Go2's names, the reference's
actuator constants as a DECLARED basis, and the study's actuator
randomization around them.

The robot is the project's bundle `go2` — `unitree_rl_mjlab`'s
`go2.xml` (Apache 2.0), which is already in the shape mjlab wants:
collision geoms named `*_collision`, foot sites FL/FR/RL/RR, an `imu`
site, the trunk body `base_link`, and no actuators or keyframe of its
own (mjlab injects position actuators and the home pose from Python).
The bundle is found through the project-first locator, so the walk
tools take `--project` and never hardcode a path.

The actuator numbers are the reference's (`go2_constants.py`, read
2026-09-10): hip and thigh kp 20 kd 1 effort 23.5 armature 0.01; calf
kp 40 kd 2 effort 45 armature 0.02. Nothing here measured them — the
identity string says `declared-pd`, and the loop's "system identified"
state stays unproven until a recording of a real Go2 is fitted.

Every Go2 fact a deployment needs beyond the built environment lives
here too (`DEPLOY`): the SDK's joint order and the reference's stack
(their controller, their simulator's scene), each with its source.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import EventTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.spec_config import CollisionCfg
from rq_pipeline.bundles.hashing import fields_hash, stamp
from rq_pipeline.deploy.manifest import UnitreeFacts
from rq_pipeline.deploy.runtime import STANDING_COMMAND
from rq_pipeline.deploy.unitree_yaml import deployable_actor_terms

from rq_mjlab.envelope import COMMAND_TERM
from rq_mjlab.go1_walk import SCALED, GainsBasis, actuator_dr_events
from rq_mjlab.linter import lint
from rq_mjlab.scene_stage import TERRAIN_BODY
from rq_mjlab.walks import (
    ACTOR_DEPLOYABLE,
    ACTOR_LEGACY,
    ACTOR_ROUGH,
    NO_CAMERAS,
    DeployFacts,
    Identity,
)

ROBOT = "unitree-go2"
BUNDLE = "go2"
MODEL_FILE = "go2.xml"
# The study's default span, the C1 arms' middle value.
ACTUATOR_DR_SPAN = 0.10
TRUNK = "base_link"
LEGS = ("FR", "FL", "RR", "RL")
FOOT_SITES = LEGS
FOOT_GEOMS = tuple(f"{leg}_foot_collision" for leg in LEGS)
THIGH_GEOMS = tuple(f"{leg}_thigh_collision" for leg in LEGS)
CALF_GEOMS = tuple(f"{leg}_calf{i}_collision" for leg in LEGS for i in (1, 2))
TRUNK_GEOMS = ("base1_collision", "base2_collision", "base3_collision")
# The reference trains every joint with one action scale (g1 alone has
# a per-joint rule); declared, like the gains.
ACTION_SCALE = 0.25
FELL_OVER_DEG = 70.0
FAMILY = "go2"  # the robot family the head registry and the bundle name

# The reference's constants, verbatim: what the identity hashes.
DECLARED: dict[str, Any] = {
    "model": "mjlab BuiltinPositionActuator (PD baked into the MJCF at build)",
    "hip": {"stiffness": 20.0, "damping": 1.0, "effort_limit": 23.5, "armature": 0.01},
    "thigh": {
        "stiffness": 20.0,
        "damping": 1.0,
        "effort_limit": 23.5,
        "armature": 0.01,
    },
    "calf": {"stiffness": 40.0, "damping": 2.0, "effort_limit": 45.0, "armature": 0.02},
    "home": {"z": 0.32, "thigh": 0.9, "calf": -1.8, "hip_abs": 0.1},
    "foot_friction": 0.6,
    "source": "unitreerobotics/unitree_rl_mjlab src/assets/robots/unitree_go2/"
    "go2_constants.py (commit 1425b15, read 2026-09-10)",
}
# The Go1's events word their basis around mjlab's DERIVED gains; the
# Go2's gains are the reference's DECLARED constants.
DECLARED_GAINS = GainsBasis(
    around="the reference's declared constants",
    at="declared",
    exact="the reference's declared PD gains",
)

# What a deployment of this robot carries beyond the built environment.
# The SDK joint order: the policy's MJCF order FL, FR, RL, RR against the
# SDK's FR, FL, RR, RL (the reference's deploy.yaml, docs/77 §1).
DEPLOY = DeployFacts(
    sdk_joint_map=(3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8),
    sdk_joint_map_source=(
        "unitree_rl_mjlab deploy/robots/go2 deploy.yaml joint_ids_map (declared)"
    ),
    unitree=UnitreeFacts(
        robot="go2",
        controller="go2_ctrl",
        scene="src/assets/robots/unitree_go2/xmls/scene_go2.xml",
        source="unitree_rl_mjlab simulate/config.yaml and deploy/robots/go2 "
        "(read 2026-09-11)",
    ),
)


def bundle_dir() -> Path:
    """The project's (or the library's) `go2` bundle, by the locator."""
    from rq_pipeline.bundles.locate import find_bundle  # noqa: PLC0415

    found = find_bundle(BUNDLE)
    if found is None:
        raise FileNotFoundError(
            f"no bundle {BUNDLE!r} in the project or the library: onboard the Go2 "
            "(onboard_robot) and pass --project so the walk can find it"
        )
    return found


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(bundle_dir() / MODEL_FILE))


def declared_actuator_constants() -> dict[str, Any]:
    return dict(DECLARED)


def robot_stamp() -> str:
    """The bundle's own version — the same `go2@…` the project index shows."""
    return stamp(BUNDLE, bundle_dir())


def actuator_stamp() -> str:
    return f"{ROBOT}-declared-pd@{fields_hash(declared_actuator_constants())}"


def _actuator(
    group: str, targets: tuple[str, ...], *, declared_armature: bool = True
) -> BuiltinPositionActuatorCfg:
    """One group's PD. Under a fit the armature is the fit's, per joint,
    already on the model's joints: the actuator leaves it alone (mjlab's
    None preserves the XML value)."""
    numbers = DECLARED[group]
    return BuiltinPositionActuatorCfg(
        target_names_expr=targets,
        stiffness=numbers["stiffness"],
        damping=numbers["damping"],
        effort_limit=numbers["effort_limit"],
        armature=numbers["armature"] if declared_armature else None,
    )


GROUPS = (("hip", (".*hip_.*",)), ("thigh", (".*thigh_.*",)), ("calf", (".*calf_.*",)))


def articulation(*, declared_armature: bool = True) -> EntityArticulationInfoCfg:
    return EntityArticulationInfoCfg(
        actuators=tuple(
            _actuator(group, targets, declared_armature=declared_armature)
            for group, targets in GROUPS
        ),
        soft_joint_pos_limit_factor=0.9,
    )


ARTICULATION = articulation()

INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, DECLARED["home"]["z"]),
    joint_pos={
        ".*thigh_joint": DECLARED["home"]["thigh"],
        ".*calf_joint": DECLARED["home"]["calf"],
        ".*R_hip_joint": DECLARED["home"]["hip_abs"],
        ".*L_hip_joint": -DECLARED["home"]["hip_abs"],
    },
    joint_vel={".*": 0.0},
)

_FOOT_REGEX = "^[FR][LR]_foot_collision$"
FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    condim={_FOOT_REGEX: 3, ".*_collision": 1},
    priority={_FOOT_REGEX: 1, ".*": 0},
    friction={_FOOT_REGEX: (DECLARED["foot_friction"],)},
    solimp={_FOOT_REGEX: (0.9, 0.95, 0.023)},
    contype=1,
    conaffinity=0,
)


def go2_robot_cfg(fit: Any = None) -> EntityCfg:
    """A fresh entity config each time: mjlab mutates configs in place.
    `fit`: a fit record whose estimates the model's joints take."""
    if fit is None:
        return EntityCfg(
            init_state=INIT_STATE,
            collisions=(FULL_COLLISION,),
            spec_fn=get_spec,
            articulation=ARTICULATION,
        )
    from rq_mjlab.fit_walk import apply_to_spec  # noqa: PLC0415

    return EntityCfg(
        init_state=INIT_STATE,
        collisions=(FULL_COLLISION,),
        spec_fn=lambda: apply_to_spec(get_spec(), fit),
        articulation=articulation(declared_armature=False),
    )


def _rough_env_cfg(play: bool) -> ManagerBasedRlEnvCfg:
    """mjlab 1.6's velocity task on the Go2 — the Go1 config's wiring,
    the Go2's names (trunk `base_link`, single thigh capsule, two calf
    capsules, three trunk boxes)."""
    cfg = make_velocity_env_cfg()
    cfg.sim.mujoco.ccd_iterations = 500
    cfg.sim.contact_sensor_maxmatch = 500
    cfg.scene.entities = {"robot": go2_robot_cfg()}
    for sensor in cfg.scene.sensors or ():
        if sensor.name == "terrain_scan":
            assert isinstance(sensor, RayCastSensorCfg)
            assert isinstance(sensor.frame, ObjRef)
            sensor.frame.name = TRUNK
    for sensor in cfg.scene.sensors or ():
        if sensor.name == "foot_height_scan":
            assert isinstance(sensor, RayCastSensorCfg)  # TerrainHeightSensorCfg
            sensor.frame = tuple(
                ObjRef(type="site", name=s, entity="robot") for s in FOOT_SITES
            )
    feet_ground = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=FOOT_GEOMS, entity="robot"),
        secondary=ContactMatch(mode="body", pattern=TERRAIN_BODY),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    thigh_ground = ContactSensorCfg(
        name="thigh_ground_touch",
        primary=ContactMatch(mode="geom", entity="robot", pattern=THIGH_GEOMS),
        secondary=ContactMatch(mode="body", pattern=TERRAIN_BODY),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    shank_ground = ContactSensorCfg(
        name="shank_ground_touch",
        primary=ContactMatch(mode="geom", entity="robot", pattern=CALF_GEOMS),
        secondary=ContactMatch(mode="body", pattern=TERRAIN_BODY),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    trunk_ground = ContactSensorCfg(
        name="trunk_ground_touch",
        primary=ContactMatch(mode="geom", entity="robot", pattern=TRUNK_GEOMS),
        secondary=ContactMatch(mode="body", pattern=TERRAIN_BODY),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        feet_ground,
        thigh_ground,
        shank_ground,
        trunk_ground,
    )
    if (
        cfg.scene.terrain is not None
        and cfg.scene.terrain.terrain_generator is not None
    ):
        cfg.scene.terrain.terrain_generator.curriculum = True
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = ACTION_SCALE
    cfg.viewer.body_name = TRUNK
    cfg.viewer.distance = 1.5
    cfg.viewer.elevation = -10.0
    cfg.events["foot_friction"].params["asset_cfg"] = SceneEntityCfg(
        "robot", geom_names=FOOT_GEOMS
    )
    cfg.events["base_com"].params["asset_cfg"].body_names = (TRUNK,)
    cfg.rewards["pose"].params["std_standing"] = {
        r".*(FR|FL|RR|RL)_hip_joint.*": 0.05,
        r".*(FR|FL|RR|RL)_thigh_joint.*": 0.1,
        r".*(FR|FL|RR|RL)_calf_joint.*": 0.15,
    }
    for mode in ("std_walking", "std_running"):
        cfg.rewards["pose"].params[mode] = {
            r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
            r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
            r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
        }
    cfg.rewards["upright"].params["asset_cfg"].body_names = (TRUNK,)
    cfg.rewards["upright"].params["terrain_sensor_names"] = ("terrain_scan",)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (TRUNK,)
    for reward_name in ("foot_clearance", "foot_slip"):
        cfg.rewards[reward_name].params["asset_cfg"].site_names = FOOT_SITES
    cfg.terminations.pop("fell_over", None)
    cfg.terminations["illegal_contact"] = TerminationTermCfg(
        func=mdp.illegal_contact, params={"sensor_name": thigh_ground.name}
    )
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.terminations.pop("out_of_terrain_bounds", None)
        cfg.curriculum = {}
        cfg.events["randomize_terrain"] = EventTermCfg(
            func=envs_mdp.randomize_terrain, mode="reset", params={}
        )
    return cfg


GAIT_PHASE_PERIOD_S = 0.6  # the reference's clock (velocity_env_cfg.py, `phase`)
# The actor's terms, in their order: what their deploy runtime provides
# (`rq_pipeline.deploy.unitree_yaml.UNITREE_TERMS`, one truth).
DEPLOYABLE_ACTOR = deployable_actor_terms()


def gait_phase(env: Any, period: float, command_name: str) -> Any:
    """The reference's gait clock (unitree_rl_mjlab `mdp.phase`, transcribed):
    sine and cosine of the episode time modulo `period`, zero while the
    command is under the standing threshold - what their deploy runtime
    computes as `gait_phase`, and what `rq_pipeline.deploy.runtime`
    computes in numpy, so a policy trained on it runs in either stack."""
    import torch  # noqa: PLC0415

    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    standing = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    still = standing < STANDING_COMMAND
    return torch.where(still.unsqueeze(1), torch.zeros_like(phase), phase)


def deployable_actor(cfg: ManagerBasedRlEnvCfg) -> None:
    """The actor sees what the real Go2 can measure, in the reference's
    order: no base linear velocity (a critic-only term in the reference;
    no sensor on the robot, no term in Unitree's deploy runtime - found
    2026-09-11 when the certified policy could not be written as their
    deploy.yaml) and the reference's gait phase, which their runtime
    provides. The critic keeps mjlab's full set."""
    from mjlab.managers import ObservationTermCfg  # noqa: PLC0415

    actor = cfg.observations["actor"]
    terms = dict(actor.terms)
    terms.pop("base_lin_vel", None)
    terms["phase"] = ObservationTermCfg(
        func=gait_phase,
        params={"period": GAIT_PHASE_PERIOD_S, "command_name": COMMAND_TERM},
    )
    actor.terms = {name: terms[name] for name in DEPLOYABLE_ACTOR}


def go2_flat_env_cfg(
    play: bool = False, *, legacy_actor: bool = False
) -> ManagerBasedRlEnvCfg:
    """The flat-ground variant, the Go1's flat rules on the Go2."""
    cfg = _rough_env_cfg(play=play)
    cfg.sim.njmax = 300
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 64
    cfg.sim.nconmax = None
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    remove = {
        "terrain_scan",
        "thigh_ground_touch",
        "shank_ground_touch",
        "trunk_ground_touch",
    }
    cfg.scene.sensors = tuple(
        s for s in (cfg.scene.sensors or ()) if s.name not in remove
    )
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]
    if not legacy_actor:  # the recipe before 2026-09-11: mjlab's actor, 48 terms
        deployable_actor(cfg)
    cfg.rewards["upright"].params.pop("terrain_sensor_names", None)
    cfg.terminations.pop("illegal_contact", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    # A plane has no terrain to re-draw at reset (the play-mode event the
    # rough variant adds; never exercised on the Go2 until play_walk, 2026-09-11).
    cfg.events.pop("randomize_terrain", None)
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=mdp.bad_orientation, params={"limit_angle": math.radians(FELL_OVER_DEG)}
    )
    cfg.curriculum.pop("terrain_levels", None)
    return cfg


def _with_actuator_dr(
    cfg: ManagerBasedRlEnvCfg,
    *,
    dr_span: float | None,
    pin_scale: float | None,
    pin_only: tuple[str, ...] | None,
    fit: str | None = None,
) -> dict[str, str]:
    """The study's actuator randomization around the declared gains,
    added to the config; returns the identity it gives the run. `fit`:
    a fit's stamp (or its recording's) in the bundle - the joints take
    its estimates, its intervals replace the declared armature scaling,
    and the gains keep the declared span (a fit measures the joints, not
    the controller)."""
    events, dr_basis = actuator_dr_events(
        dr_span=dr_span,
        pin_scale=pin_scale,
        pin_only=pin_only,
        gains=DECLARED_GAINS,
        scaled=GAINS_ONLY if fit is not None else SCALED,
    )
    identity: dict[str, str] = {}
    if fit is not None:
        from rq_pipeline.robot.fit_record import fit_stamp  # noqa: PLC0415

        from rq_mjlab.fit_walk import basis_text, dr_events, resolve  # noqa: PLC0415

        record = resolve(bundle_dir(), fit)
        cfg.scene.entities["robot"] = go2_robot_cfg(record)
        events.pop(ARMATURE_EVENT, None)  # the fit's own draw replaces it
        events.update(dr_events(record))
        dr_basis = f"{basis_text(record)}; gains: {dr_basis}"
        identity = {
            Identity.FIT: fit_stamp(record),
            Identity.FIT_BASIS: record.basis or "unrecorded",
        }
    for name in events:
        if name in cfg.events:
            raise ValueError(f"the Go2 cfg already carries an event named {name!r}")
    cfg.events.update(events)
    lint(cfg.events, ())
    return {
        Identity.ROBOT: robot_stamp(),
        Identity.ACTUATOR: actuator_stamp(),
        Identity.DR_BASIS: dr_basis,
        **identity,
    }


ARMATURE_EVENT = "actuator_armature"  # go1_walk.actuator_dr_events' name for it
# Under a fit the declared span moves the gains only; the armature is the fit's.
GAINS_ONLY = tuple(name for name in SCALED if name != "armature")


def go2_walk_env_cfg(  # noqa: PLR0913 - the walk's knobs, each named
    *,
    play: bool = False,
    dr_span: float | None = ACTUATOR_DR_SPAN,
    pin_scale: float | None = None,
    pin_only: tuple[str, ...] | None = None,
    legacy_actor: bool = False,
    fit: str | None = None,
) -> tuple[ManagerBasedRlEnvCfg, dict[str, str]]:
    """The Go2 flat walk with the study's actuator randomization around
    the declared gains, and its identity; `fit` (a fit's stamp in the
    bundle): the joints at the fit, randomized over its intervals."""
    from rq_mjlab.scene_stage import unrender_splats  # noqa: PLC0415

    cfg = go2_flat_env_cfg(play=play, legacy_actor=legacy_actor)
    unrender_splats()  # a plane after a scene in one process sees no splats
    identity = _with_actuator_dr(
        cfg, dr_span=dr_span, pin_scale=pin_scale, pin_only=pin_only, fit=fit
    )
    identity[Identity.ACTOR] = ACTOR_LEGACY if legacy_actor else ACTOR_DEPLOYABLE
    return cfg, identity


# A captured scene has things in it. The plane's first stage (every way at
# up to 1 m/s from the first tick) drove go2-scene-c1 into the hurdles
# before it could walk: survived 38/40, tracked 0/40 (2026-09-23). A scene
# starts slow and widens to the plane's first stage; each row is (the
# iteration it starts at, forward/back m/s, sideways m/s, turn rad/s).
SCENE_COMMAND_STAGES = (
    (0, (-0.5, 0.5), (-0.3, 0.3), (-0.5, 0.5)),
    (500, (-0.8, 0.8), (-0.6, 0.6), (-0.5, 0.5)),
    (1000, (-1.0, 1.0), (-1.0, 1.0), (-0.5, 0.5)),
)


def scene_command_stages(steps_per_iteration: int) -> list[dict[str, Any]]:
    """`SCENE_COMMAND_STAGES` in mjlab's `commands_vel` shape (stages keyed
    on the environment's step counter)."""
    return [
        {
            "step": iteration * steps_per_iteration,
            "lin_vel_x": forward,
            "lin_vel_y": sideways,
            "ang_vel_z": turn,
        }
        for iteration, forward, sideways, turn in SCENE_COMMAND_STAGES
    ]


def gentle_scene_commands(cfg: ManagerBasedRlEnvCfg) -> None:
    """The scene's command schedule in place of the plane's, its first
    stage also the ranges before the curriculum's first call."""
    from rq_mjlab.envelope import (  # noqa: PLC0415
        COMMAND_TERM,
        CURRICULUM_TERM,
        STAGES_KEY,
    )

    stages = scene_command_stages(go2_agent(1).num_steps_per_env)
    cfg.curriculum[CURRICULUM_TERM].params[STAGES_KEY] = stages
    ranges = cfg.commands[COMMAND_TERM].ranges
    ranges.lin_vel_x = stages[0]["lin_vel_x"]
    ranges.lin_vel_y = stages[0]["lin_vel_y"]
    ranges.ang_vel_z = stages[0]["ang_vel_z"]


def seated_at(
    spec_fn: Callable[[], mujoco.MjSpec], ground: tuple[float, float, float]
) -> Callable[[], mujoco.MjSpec]:
    """`spec_fn` with the free-floating root body moved over `ground` (x, y,
    surface z), keeping its own height above the floor. A free joint's
    resting pose is its body's position, so the model's qpos0 stands there;
    `init_state` only writes a keyframe."""

    def build() -> mujoco.MjSpec:
        spec = spec_fn()
        root = spec.worldbody.first_body()
        if not any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in root.joints):
            raise ValueError(f"the root body {root.name!r} has no free joint to seat")
        x, y, z = ground
        root.pos = [x, y, z + float(root.pos[2])]
        return spec

    return build


def go2_scene_env_cfg(  # noqa: PLR0913 - the walk's knobs, each named
    scene_dir: Path,
    *,
    play: bool = False,
    dr_span: float | None = ACTUATOR_DR_SPAN,
    pin_scale: float | None = None,
    pin_only: tuple[str, ...] | None = None,
    cameras: bool = True,
    camera_in_actor: bool = True,
    camera_size: tuple[int, int] | None = None,
    fit: str | None = None,
) -> tuple[ManagerBasedRlEnvCfg, dict[str, str]]:
    """The Go2 walk on a captured scene (docs/78 §4 E2): the rough
    recipe's rules and sensors (the height scan sees the hurdles) on the
    scene's heightfield at the scene's coordinates, every world at the
    course's start; with `cameras`, the head camera renders the splat
    for every world - into the actor's and the critic's observations
    when `camera_in_actor` (training), as frames alone when not (a press
    pressing a policy that never saw it); `camera_size` overrides the
    training picture's width and height."""
    from rq_mjlab.scene_stage import (  # noqa: PLC0415
        CAMERA_TERM,
        TRAIN_CELL_M,
        camera_term,
        head_camera_cfg,
        out_of_scene_bounds,
        render_splats,
        scans_see_the_scene,
        scene_stamp,
        scene_start,
        scene_terrain_cfg,
    )

    cfg = _rough_env_cfg(play=play)
    cfg.scene.terrain = scene_terrain_cfg(scene_dir)
    scans_see_the_scene(cfg.scene.sensors)
    # one patch: no difficulty rows to climb, no other patch to move to;
    # the bounds are the scene's footprint, not the grid mjlab centres at
    # the origin (a world off the heightfield falls forever: NaN, 2026-09-23)
    cfg.curriculum.pop("terrain_levels", None)
    cfg.events.pop("randomize_terrain", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.terminations["out_of_scene_bounds"] = TerminationTermCfg(
        func=out_of_scene_bounds, time_out=True
    )
    gentle_scene_commands(cfg)
    identity = _with_actuator_dr(
        cfg, dr_span=dr_span, pin_scale=pin_scale, pin_only=pin_only, fit=fit
    )
    # the model's resting pose (qpos0, what mjlab allocates contacts from
    # before the first reset) at the course's start, not inside the scene
    robot = cfg.scene.entities["robot"]
    robot.spec_fn = seated_at(robot.spec_fn, scene_start(scene_dir))
    identity[Identity.SCENE] = scene_stamp(scene_dir)
    identity[Identity.TERRAIN] = f"the scene's heightfield at {TRAIN_CELL_M} m"
    identity[Identity.ACTOR] = ACTOR_ROUGH
    identity[Identity.CAMERAS] = NO_CAMERAS
    if cameras:
        head = (
            head_camera_cfg(FAMILY, width=camera_size[0], height=camera_size[1])
            if camera_size is not None
            else head_camera_cfg(FAMILY)
        )
        cfg.scene.sensors = (*(cfg.scene.sensors or ()), head)
        # the scene's gaussians into every camera this env renders: the
        # one place, so a trainer, a verdict and a press all see the splat
        gaussians = render_splats(scene_dir)
        if camera_in_actor:
            for group in ("actor", "critic"):
                cfg.observations[group].terms[CAMERA_TERM] = camera_term(head.name)
            identity[Identity.CAMERAS] = (
                f"{head.name} {head.width}x{head.height} rgb over {gaussians} splats"
            )
    return cfg, identity


def go2_agent(iterations: int) -> Any:
    """The reference's PPO numbers are mjlab's Go1 numbers; one runner
    config, the experiment named for the Go2, tensorboard like every
    study run."""
    from dataclasses import replace  # noqa: PLC0415

    from mjlab.tasks.velocity.config.go1.rl_cfg import (  # noqa: PLC0415
        unitree_go1_ppo_runner_cfg,
    )

    return replace(
        unitree_go1_ppo_runner_cfg(),
        max_iterations=iterations,
        logger="tensorboard",
        experiment_name="go2_velocity",
    )
