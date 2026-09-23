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
    from rq_pipeline.deploy.manifest import UnitreeFacts

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


def _no_scene(robot: str, scene: Path | None, cameras: bool) -> None:
    """The scene stage is the Go2's so far (docs/78 E2); the other walks
    take the keyword the doors pass and refuse it by name."""
    del cameras
    if scene is not None:
        raise ValueError(
            f"the {robot} walk has no scene stage yet (docs/78 E2: the Go2)"
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
) -> tuple[Any, dict[str, str]]:
    from rq_mjlab.microduck_walk import (  # noqa: PLC0415
        PIN_AXES,
        microduck_walk_env_cfg,
    )

    _no_scene("microduck", scene, cameras)

    return microduck_walk_env_cfg(
        play=play,
        law_dr_span=dr_span,
        law_pin_scale=pin_scale,
        law_pin_only=_axis(PIN_AXES, pin_axis),
        bundle=bundle,
        head=head,
    )


def _axis(axes: dict[str, tuple[str, ...] | None], name: str) -> tuple[str, ...] | None:
    if name not in axes:
        raise KeyError(f"no mismatch axis {name!r}; this walk has {sorted(axes)}")
    return axes[name]


def _microduck_agent(iterations: int) -> Any:
    from rq_mjlab.walk_train import g3_agent  # noqa: PLC0415

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
) -> tuple[Any, dict[str, str]]:
    from rq_mjlab.go1_walk import PIN_AXES, go1_walk_env_cfg  # noqa: PLC0415

    _no_scene("go1", scene, cameras)
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
    from rq_mjlab.go1_walk import go1_agent  # noqa: PLC0415

    return go1_agent(iterations)


def _microduck_span() -> float:
    from rq_mjlab.microduck_walk import LAW_DR_SPAN  # noqa: PLC0415

    return LAW_DR_SPAN


def _go1_span() -> float:
    from rq_mjlab.go1_walk import ACTUATOR_DR_SPAN  # noqa: PLC0415

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
) -> tuple[Any, dict[str, str]]:
    from rq_mjlab.go1_walk import PIN_AXES  # noqa: PLC0415
    from rq_mjlab.go2_walk import go2_scene_env_cfg, go2_walk_env_cfg  # noqa: PLC0415

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
        return go2_scene_env_cfg(
            scene,
            play=play,
            dr_span=dr_span,
            pin_scale=pin_scale,
            pin_only=_axis(PIN_AXES, pin_axis),
            cameras=cameras,
            camera_in_actor=camera_in_actor,
            camera_size=camera_size,
        )
    return go2_walk_env_cfg(
        play=play,
        dr_span=dr_span,
        pin_scale=pin_scale,
        pin_only=_axis(PIN_AXES, pin_axis),
        legacy_actor=legacy_actor,
    )


def _go2_agent(iterations: int) -> Any:
    from rq_mjlab.go2_walk import go2_agent  # noqa: PLC0415

    return go2_agent(iterations)


def _go2_span() -> float:
    from rq_mjlab.go2_walk import ACTUATOR_DR_SPAN  # noqa: PLC0415

    return ACTUATOR_DR_SPAN


def _go2_deploy() -> DeployFacts:
    from rq_mjlab.go2_walk import DEPLOY  # noqa: PLC0415

    return DEPLOY


DEFAULT_ROBOT = "microduck"


def _quieted(builder: EnvFactory) -> EnvFactory:
    """A walk's env builder whose simulation config keeps mujoco_warp's
    line-search warning off (`rq_mjlab.sim_options`); the builder's own
    signature stays what callers and tests inspect."""

    @functools.wraps(builder)
    def build(**kwargs: Any) -> tuple[Any, dict[str, str]]:
        from rq_mjlab.sim_options import quiet  # noqa: PLC0415

        cfg, identity = builder(**kwargs)
        return quiet(cfg), identity

    return build


def walk_spec(robot: str = DEFAULT_ROBOT) -> WalkSpec:
    """The spec for a robot name; imports lazily so a tool's --help
    never builds an env."""
    spec = _walk_spec(robot)
    return dataclasses.replace(spec, env_cfg=_quieted(spec.env_cfg))


def _walk_spec(robot: str) -> WalkSpec:
    if robot == "microduck":
        return WalkSpec(
            "microduck",
            _microduck_env,
            _microduck_agent,
            _microduck_span(),
            "microduck-walk",
        )
    if robot == "go1":
        return WalkSpec(
            "go1",
            _go1_env,
            _go1_agent,
            _go1_span(),
            "go1-walk",
            view=QUADRUPED_VIEW,
            chase=QUADRUPED_CHASE,
        )
    if robot == "go2":
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
    raise KeyError(f"no walk for robot {robot!r}; known: {sorted(ROBOTS)}")


ROBOTS = ("microduck", "go1", "go2")


def use_project(root: Path | None) -> None:
    """Search a project's robots first (the walk tools' `--project`), so
    a robot onboarded there — the Go2 — is found by name."""
    if root is not None:
        from rq_pipeline.project.locate import Project  # noqa: PLC0415

        Project(Path(root).expanduser().resolve()).use()
