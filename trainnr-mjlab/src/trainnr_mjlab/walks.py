"""The walks the study tools know: one spec per robot, so `walk_train`,
`walk_verdict` and `walk_view` take `--robot` instead of a copy each.

A spec is the two factories the tools need and the span they default
to: the env cfg with its identity (`env_cfg(play=, dr_span=,
pin_scale=)` → `(cfg, {"robot", "actuator", "dr_basis"})`) and the
rsl-rl runner cfg (`agent(iterations)`). The microduck's factory names
its knobs `law_dr_span` / `law_pin_scale` (a BAM law); the spec maps
the study's words onto them.
"""

from __future__ import annotations

import dataclasses
import functools
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from trainnr.deploy.manifest import UnitreeFacts

EnvFactory = Callable[..., tuple[Any, dict[str, str]]]


@dataclass(frozen=True)
class CameraFraming:
    """How a viewer frames this walk: the orbit camera's azimuth and
    elevation (degrees), its distance (m), the height it looks at (m),
    and how close it sits when following one world. Declared by the walk
    (its size decides), read by the Studio's render stream for the RL
    view and by the walk press's chase camera - never keyed by a robot's
    name in a generic tool."""

    azimuth: float
    elevation: float
    distance: float
    lookat_height: float
    follow: float

    def as_dict(self) -> dict[str, float]:
        return {
            "azimuth": self.azimuth,
            "elevation": self.elevation,
            "distance": self.distance,
            "lookat_height": self.lookat_height,
            "follow": self.follow,
        }


# The 15 cm microduck: nine worlds on a grid seen from above the corner
# (the Studio's RL view until 2026-09-13), and one world chased from
# 0.6 m (the walk press's camera, the picture its students train on).
MICRODUCK_VIEW = CameraFraming(120.0, -20.0, 3.0, 0.1, 0.9)
MICRODUCK_CHASE = CameraFraming(135.0, -18.0, 0.6, 0.0, 0.6)
# A 70 cm quadruped (the Go2; the Go1 is the same class and had no
# framing of its own before 2026-09-13): the grid from above, one world
# followed from 2.6 m; the chase framing is that follow distance at the
# same angles - tuned on the Go2's RL view, not on a pressed picture.
QUADRUPED_VIEW = CameraFraming(135.0, -28.0, 4.2, 0.25, 2.6)
QUADRUPED_CHASE = CameraFraming(135.0, -28.0, 2.6, 0.0, 2.6)


@dataclass(frozen=True)
class DeployFacts:
    """What a deployment manifest carries beyond the built environment:
    the vendor SDK's joint order and where it was read, and the Unitree
    stack that runs the robot, when there is one. Declared by the walk
    (its robot module), never guessed by the exporter."""

    sdk_joint_map: tuple[int, ...] | None = None
    sdk_joint_map_source: str | None = None
    unitree: UnitreeFacts | None = None


@dataclass(frozen=True)
class WalkSpec:
    name: str
    env_cfg: EnvFactory
    agent: Callable[[int], Any]
    default_span: float
    source_prefix: str  # the record's task source: "<prefix>@<identity hash>"
    deploy: DeployFacts = DeployFacts()
    view: CameraFraming = MICRODUCK_VIEW  # the RL view's many-worlds framing
    chase: CameraFraming = MICRODUCK_CHASE  # one world, followed (the press)


def _no_scene(  # noqa: PLR0913, PLR0917 - the door keywords every walk takes, refused by name
    robot: str,
    scene: Path | None,
    cameras: bool,
    legacy_actor: bool,
    camera_in_actor: bool,
    camera_size: tuple[int, int] | None,
    fit: str | None = None,
) -> None:
    """The scene stage and its camera are the Go2's so far (docs/78 E2),
    and only the Go2 has an earlier actor recipe; the other walks take
    every keyword the doors pass and refuse what they lack by name."""
    del cameras, camera_in_actor, camera_size
    if scene is not None:
        raise ValueError(
            f"the {robot} walk has no scene stage yet (docs/78 E2: the Go2)"
        )
    if legacy_actor:
        raise TypeError(f"the {robot} walk has no earlier actor recipe")
    if fit is not None:
        raise ValueError(
            f"the {robot} walk takes no joint fit yet (trainnr_mjlab.fit_walk: the Go2)"
        )


def _microduck_env(  # noqa: PLR0913 - the walk's knobs, named
    *,
    play: bool = False,
    dr_span: float | str | None,
    pin_scale: float | None,
    pin_axis: str = "all",
    bundle: Any = None,
    head: str = "free",
    scene: Path | None = None,
    cameras: bool = True,
    legacy_actor: bool = False,
    camera_in_actor: bool = True,
    camera_size: tuple[int, int] | None = None,
    fit: str | None = None,
) -> tuple[Any, dict[str, str]]:
    from trainnr_mjlab.microduck_walk import (  # noqa: PLC0415
        PIN_AXES,
        microduck_walk_env_cfg,
    )

    _no_scene(
        "microduck", scene, cameras, legacy_actor, camera_in_actor, camera_size, fit
    )

    return microduck_walk_env_cfg(
        play=play,
        law_dr_span=dr_span,
        law_pin_scale=pin_scale,
        law_pin_only=_axis(PIN_AXES, pin_axis),
        bundle=bundle,
        head=head,
    )


# A declared legged walk (#100): any onboarded legged robot, its spec
# resolved by the trainnr side. The walk's name in the registry is LEGGED;
# its robot comes from the spec a tool adopts (`adopt_legged`: the train
# door's --legged-spec, or the trained run's identity for a judge, a play
# or an export), never from the name.
LEGGED = "legged"
_LEGGED_SPEC: dict[str, Any] = {}


def adopt_legged(spec: dict[str, Any] | None) -> None:
    """The legged walk this process builds from now on (None: none)."""
    _LEGGED_SPEC.clear()
    if spec:
        _LEGGED_SPEC.update(spec)


def adopt_legged_of(robot: str, checkpoint: Path) -> None:
    """For a legged walk's checkpoint, adopt the spec its run recorded, so
    a judge, a play or an export builds the robot it trained on."""
    if robot != LEGGED:
        return
    import json  # noqa: PLC0415

    from trainnr.project.kinds import IDENTITY_FILE  # noqa: PLC0415

    identity = Path(checkpoint).resolve().parent / IDENTITY_FILE
    recorded = json.loads(identity.read_text()) if identity.is_file() else {}
    adopt_legged(recorded.get(Identity.LEGGED))


def bundle_of(robot: str) -> str:
    """The bundle a walk is drawn with: the robot's own name, or for the
    legged walk the adopted spec's robot (the Studio's mirror, a still)."""
    if robot != LEGGED:
        return robot
    return str(adopted_legged()["robot"]).split("@", 1)[0]


def adopted_legged() -> dict[str, Any]:
    """The adopted spec, refused by name when none was adopted."""
    if not _LEGGED_SPEC:
        raise ValueError(
            "the legged walk builds from a declared task's spec: pass "
            "--legged-spec, or judge a run whose identity carries it"
        )
    return dict(_LEGGED_SPEC)


def _legged_env(  # noqa: PLR0913 - the walk's knobs, named
    *,
    play: bool = False,
    dr_span: float | str | None,
    pin_scale: float | None,
    pin_axis: str = "all",
    bundle: Any = None,
    head: str = "free",
    scene: Path | None = None,
    cameras: bool = True,
    legacy_actor: bool = False,
    camera_in_actor: bool = True,
    camera_size: tuple[int, int] | None = None,
    fit: str | None = None,
) -> tuple[Any, dict[str, str]]:
    from trainnr_mjlab.legged_walk import (  # noqa: PLC0415
        legged_robot,
        legged_walk_env_cfg,
    )
    from trainnr_mjlab.microduck_walk import PIN_AXES  # noqa: PLC0415

    spec = adopted_legged()
    _no_scene(
        spec["robot"], scene, cameras, legacy_actor, camera_in_actor, camera_size, fit
    )
    held = "pinned" if spec.get("pinned_joints") else "free"
    if bundle is not None or head not in ("free", held):
        # a judge passes the head its run recorded: the spec's own, accepted
        raise ValueError(
            "a legged walk's actuator and held joints come from its spec "
            "(actuator, pinned_joints), not from --bundle or --head"
        )
    cfg, identity = legged_walk_env_cfg(
        legged_robot(spec),
        play=play,
        law_dr_span=dr_span,
        law_pin_scale=pin_scale,
        law_pin_only=_axis(PIN_AXES, pin_axis),
    )
    # the spec rides in the identity: a judge, a play or an export of the
    # run rebuilds the same robot from it
    recorded: dict[str, Any] = {**identity, Identity.LEGGED: spec}
    return cfg, recorded


def _legged_spec() -> WalkSpec:
    spec = _LEGGED_SPEC
    return WalkSpec(
        LEGGED,
        _legged_env,
        _microduck_agent,  # microduck's PPO recipe, the legged walk's own
        float(spec.get("dr_span", _microduck_span())) if spec else _microduck_span(),
        str(spec.get("name", "legged-walk")).split("@", 1)[0]
        if spec
        else "legged-walk",
    )


def _axis(axes: dict[str, tuple[str, ...] | None], name: str) -> tuple[str, ...] | None:
    if name not in axes:
        raise KeyError(f"no mismatch axis {name!r}; this walk has {sorted(axes)}")
    return axes[name]


def _microduck_agent(iterations: int) -> Any:
    from trainnr_mjlab.walk_train import g3_agent  # noqa: PLC0415

    return g3_agent(iterations)


def _go1_env(  # noqa: PLR0913 - the walk's knobs, named
    *,
    play: bool = False,
    dr_span: float | str | None,
    pin_scale: float | None,
    pin_axis: str = "all",
    bundle: Any = None,
    head: str = "free",
    scene: Path | None = None,
    cameras: bool = True,
    legacy_actor: bool = False,
    camera_in_actor: bool = True,
    camera_size: tuple[int, int] | None = None,
    fit: str | None = None,
) -> tuple[Any, dict[str, str]]:
    from trainnr_mjlab.go1_walk import PIN_AXES, go1_walk_env_cfg  # noqa: PLC0415

    _no_scene("go1", scene, cameras, legacy_actor, camera_in_actor, camera_size, fit)
    if bundle is not None:
        raise ValueError("the Go1 walk has no actuator bundle to swap (derived PD)")
    if isinstance(dr_span, str):  # "identified" names the bundle's interval
        raise ValueError("the Go1 walk has no identified interval (derived PD)")
    if head != "free":
        raise ValueError("the Go1 has no head to pin")

    return go1_walk_env_cfg(
        play=play,
        dr_span=dr_span,
        pin_scale=pin_scale,
        pin_only=_axis(PIN_AXES, pin_axis),
    )


def _go1_agent(iterations: int) -> Any:
    from trainnr_mjlab.go1_walk import go1_agent  # noqa: PLC0415

    return go1_agent(iterations)


def _microduck_span() -> float:
    from trainnr_mjlab.microduck_walk import LAW_DR_SPAN  # noqa: PLC0415

    return LAW_DR_SPAN


def _go1_span() -> float:
    from trainnr_mjlab.go1_walk import ACTUATOR_DR_SPAN  # noqa: PLC0415

    return ACTUATOR_DR_SPAN


def _go2_env(  # noqa: PLR0913 - the walk's knobs, each named
    *,
    play: bool = False,
    dr_span: float | str | None,
    pin_scale: float | None,
    pin_axis: str = "all",
    bundle: Any = None,
    head: str = "free",
    legacy_actor: bool = False,
    scene: Path | None = None,
    cameras: bool = True,
    camera_in_actor: bool = True,
    camera_size: tuple[int, int] | None = None,
    fit: str | None = None,
) -> tuple[Any, dict[str, str]]:
    from trainnr_mjlab.go1_walk import PIN_AXES  # noqa: PLC0415
    from trainnr_mjlab.go2_walk import (  # noqa: PLC0415
        go2_scene_env_cfg,
        go2_walk_env_cfg,
    )

    if bundle is not None:
        raise ValueError("the Go2 walk has no actuator bundle to swap (declared PD)")
    if isinstance(dr_span, str):  # "identified" names the bundle's interval
        raise ValueError("the Go2 walk has no identified interval (declared PD)")
    # The head knob is the duck's; the trainer and the verdict pass it to
    # every walk, and without it here both doors failed on the Go2
    # (found by E0's re-certification, 2026-09-22).
    if head != "free":
        raise ValueError("the Go2 has no head to pin")
    if scene is not None:  # the walk on a captured scene (docs/78 E2)
        if legacy_actor:
            raise TypeError("the scene walk has no earlier actor recipe")
        return go2_scene_env_cfg(
            scene,
            play=play,
            dr_span=dr_span,
            pin_scale=pin_scale,
            pin_only=_axis(PIN_AXES, pin_axis),
            cameras=cameras,
            camera_in_actor=camera_in_actor,
            camera_size=camera_size,
            fit=fit,
        )
    return go2_walk_env_cfg(
        play=play,
        dr_span=dr_span,
        pin_scale=pin_scale,
        pin_only=_axis(PIN_AXES, pin_axis),
        legacy_actor=legacy_actor,
        fit=fit,
    )


def _go2_agent(iterations: int) -> Any:
    from trainnr_mjlab.go2_walk import go2_agent  # noqa: PLC0415

    return go2_agent(iterations)


def _go2_span() -> float:
    from trainnr_mjlab.go2_walk import ACTUATOR_DR_SPAN  # noqa: PLC0415

    return ACTUATOR_DR_SPAN


def _go2_deploy() -> DeployFacts:
    from trainnr_mjlab.go2_walk import DEPLOY  # noqa: PLC0415

    return DEPLOY


class ScanSensor:
    """mjlab's rough recipe's two height scans, by the names it gives them."""

    TERRAIN = "terrain_scan"
    FOOT_HEIGHT = "foot_height_scan"
    ALL = (TERRAIN, FOOT_HEIGHT)


# A captured scene's training ground sits in a geometry group of its own:
# mjlab's height scans see only the groups they name, the head camera draws
# groups 0-2, and the robot's colliders are group 3. In group 3 the scans
# saw nothing: go2-scene-c1 trained blind to its hurdles (0/40 tracked,
# 2026-09-23), and without a camera mujoco_warp refit an empty ray structure
# and crashed (2026-09-25). Every viewer draws only 0-2 by default, so a
# scene played or recorded without its splat turns this group on.
TERRAIN_SCAN_GROUP = 4

DEFAULT_ROBOT = "microduck"
# mjlab's name for the walking entity in every walk's scene: what the
# sensors, the cameras and the export address bodies through.
ROBOT_ENTITY = "robot"


class Identity:
    """The keys of a run's identity (the trainer writes it beside the
    checkpoints; the doors read it): one spelling each."""

    ROBOT = "robot"
    ACTUATOR = "actuator"
    DR_BASIS = "dr_basis"
    SCENE = "scene"
    TERRAIN = "terrain"
    CAMERAS = "cameras"
    ACTOR = "actor"
    HEAD = "head"
    SEED = "seed"
    TASK = "task"
    # 2026-09-25: the fit a run trained under (its own stamp, apart from
    # the robot's) and whose robot the fit measured (`bundles.basis`).
    FIT = "fit"
    FIT_BASIS = "fit_basis"
    # the command lag a run trained under (trainnr_mjlab.lag_dr): training
    # randomization, recorded, never gated - the verdict judges without it
    LAG_DR = "lag_dr"
    # a cross-evaluation's judged world (walk_verdict --judge-in-fit): the
    # fit it ran in (or "declared") and that fit's basis, apart from the
    # trained `fit` / `fit_basis`
    JUDGED_IN_FIT = "judged_in_fit"
    JUDGED_FIT_BASIS = "judged_fit_basis"
    # the command schedule a run trained under, when it is not the recipe's
    # own (a captured scene's, go2_walk.SCENE_COMMAND_STAGES)
    COMMANDS = "commands"
    # a declared legged walk's resolved spec (#100): the robot, its
    # actuator, its stand pose and the rest; a judge rebuilds from it
    LEGGED = "legged"


NO_CAMERAS = "none"  # the identity's word for an actor that saw no camera
# The actor recipes a run's identity names.
ACTOR_DEPLOYABLE = "deployable"  # the flat recipe's 48 terms (docs/77)
ACTOR_LEGACY = "legacy"  # the recipe before 2026-09-11, 47 terms
ACTOR_ROUGH = "rough"  # mjlab's rough recipe: the height scan sees the terrain


def training_episode_s(spec: WalkSpec) -> float:
    """The training episode's length, from the walk's own config (a play
    config runs forever): what a press caps its rollouts at."""
    cfg, _ = spec.env_cfg(play=False, dr_span=None, pin_scale=None)
    return float(cfg.episode_length_s)


def _quieted(builder: EnvFactory) -> EnvFactory:
    """A walk's env builder whose simulation config keeps mujoco_warp's
    line-search warning off (`trainnr_mjlab.sim_options`); the builder's own
    signature stays what callers and tests inspect."""

    @functools.wraps(builder)
    def build(**kwargs: Any) -> tuple[Any, dict[str, str]]:
        from trainnr_mjlab.sim_options import quiet  # noqa: PLC0415

        cfg, identity = builder(**kwargs)
        return quiet(cfg), identity

    return build


def walk_spec(robot: str = DEFAULT_ROBOT) -> WalkSpec:
    """The spec for a robot name; imports lazily so a tool's --help
    never builds an env."""
    spec = _walk_spec(robot)
    return dataclasses.replace(spec, env_cfg=_quieted(spec.env_cfg))


def _microduck_spec() -> WalkSpec:
    return WalkSpec(
        "microduck",
        _microduck_env,
        _microduck_agent,
        _microduck_span(),
        "microduck-walk",
    )


def _go1_spec() -> WalkSpec:
    return WalkSpec(
        "go1",
        _go1_env,
        _go1_agent,
        _go1_span(),
        "go1-walk",
        view=QUADRUPED_VIEW,
        chase=QUADRUPED_CHASE,
    )


def _go2_spec() -> WalkSpec:
    return WalkSpec(
        "go2",
        _go2_env,
        _go2_agent,
        _go2_span(),
        "go2-walk",
        _go2_deploy(),
        view=QUADRUPED_VIEW,
        chase=QUADRUPED_CHASE,
    )


# The registry: a walk by its robot's name. A third party's walk lands
# here through `register_walk`, never by editing a chain.
WALKS: dict[str, Callable[[], WalkSpec]] = {
    "microduck": _microduck_spec,
    "go1": _go1_spec,
    "go2": _go2_spec,
    LEGGED: _legged_spec,
}


def register_walk(robot: str, spec: Callable[[], WalkSpec]) -> None:
    if robot in WALKS:
        raise ValueError(f"a walk for {robot!r} is already registered")
    WALKS[robot] = spec


def _walk_spec(robot: str) -> WalkSpec:
    try:
        return WALKS[robot]()
    except KeyError as unknown:
        raise KeyError(
            f"no walk for robot {robot!r}; known: {sorted(WALKS)}"
        ) from unknown


# The registered walks' names (the doors' choices); read, never written.
ROBOTS = tuple(WALKS)


def use_project(root: Path | None) -> None:
    """Search a project's robots first (the walk tools' `--project`), so
    a robot onboarded there — the Go2 — is found by name."""
    if root is not None:
        from trainnr.project.locate import Project  # noqa: PLC0415

        Project(Path(root).expanduser().resolve()).use()
