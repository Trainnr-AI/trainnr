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

from dataclasses import dataclass
from typing import Any, Callable

EnvFactory = Callable[..., tuple[Any, dict[str, str]]]


@dataclass(frozen=True)
class WalkSpec:
    name: str
    env_cfg: EnvFactory
    agent: Callable[[int], Any]
    default_span: float
    source_prefix: str  # the record's task source: "<prefix>@<identity hash>"


def _microduck_env(
    *,
    play: bool = False,
    dr_span: float | str | None,
    pin_scale: float | None,
    pin_axis: str = "all",
    bundle: Any = None,
) -> tuple[Any, dict[str, str]]:
    from rq_mjlab.microduck_walk import (  # noqa: PLC0415
        PIN_AXES,
        microduck_walk_env_cfg,
    )

    return microduck_walk_env_cfg(
        play=play,
        law_dr_span=dr_span,
        law_pin_scale=pin_scale,
        law_pin_only=_axis(PIN_AXES, pin_axis),
        bundle=bundle,
    )


def _axis(axes: dict[str, tuple[str, ...] | None], name: str) -> tuple[str, ...] | None:
    if name not in axes:
        raise KeyError(f"no mismatch axis {name!r}; this walk has {sorted(axes)}")
    return axes[name]


def _microduck_agent(iterations: int) -> Any:
    from rq_mjlab.walk_train import g3_agent  # noqa: PLC0415

    return g3_agent(iterations)


def _go1_env(
    *,
    play: bool = False,
    dr_span: float | str | None,
    pin_scale: float | None,
    pin_axis: str = "all",
    bundle: Any = None,
) -> tuple[Any, dict[str, str]]:
    from rq_mjlab.go1_walk import PIN_AXES, go1_walk_env_cfg  # noqa: PLC0415

    if bundle is not None:
        raise ValueError("the Go1 walk has no actuator bundle to swap (derived PD)")
    if dr_span == "identified":
        raise ValueError("the Go1 walk has no identified interval (derived PD)")

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


DEFAULT_ROBOT = "microduck"


def walk_spec(robot: str = DEFAULT_ROBOT) -> WalkSpec:
    """The spec for a robot name; imports lazily so a tool's --help
    never builds an env."""
    if robot == "microduck":
        return WalkSpec(
            "microduck",
            _microduck_env,
            _microduck_agent,
            _microduck_span(),
            "microduck-walk",
        )
    if robot == "go1":
        return WalkSpec("go1", _go1_env, _go1_agent, _go1_span(), "go1-walk")
    raise KeyError(f"no walk for robot {robot!r}; known: {sorted(ROBOTS)}")


ROBOTS = ("microduck", "go1")
