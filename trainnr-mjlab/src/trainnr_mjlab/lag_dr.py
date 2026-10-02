"""Command lag as a training randomization, on any walk.

The cliff grid of 2026-09-25 (docs/07) found no walk trained so far keeps
walking when its action arrives 2 control ticks late, except the one
trained on a measured fit, and that one only 18/40: no run had ever SEEN
a late command. This draws one: every actuator's target reaches the
motor 0..`max_ms` late, the lag resampled every `LAG_RESAMPLE_S` per
world (staggered), on top of any delay the robot's bundle identified.

It is mjlab's own delayed actuator (`delay_min_lag`/`delay_max_lag`, in
physics steps, `mjlab.utils.buffers.DelayBuffer`), set after the walk's
config is built, so every robot takes it the same way. It is training
randomization, not the world: the verdict judges without it and measures
a delay only as a named rung (`walk_verdict --delay`).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from trainnr_mjlab.actuator import seconds_to_steps
from trainnr_mjlab.walks import ROBOT_ENTITY, Identity

# A bus latency drifts; it does not jump every 5 ms. One second per draw,
# each world on its own phase.
LAG_RESAMPLE_S = 1.0


def lag_steps(max_ms: float, physics_dt: float) -> int:
    """The widest lag in physics steps, rounded up so the draw covers
    `max_ms` - the bundle's identified delay's own rule
    (`actuator.seconds_to_steps`)."""
    if max_ms < 0:
        raise ValueError(f"a command lag is not negative: {max_ms} ms")
    return seconds_to_steps(max_ms / 1000.0, physics_dt)


def lag_basis(max_ms: float, steps: int, physics_dt: float) -> str:
    """The identity's sentence for the draw, every unit named."""
    return (
        f"command lag 0-{max_ms:g} ms (0-{steps} physics steps of "
        f"{physics_dt * 1000:g} ms), redrawn every {LAG_RESAMPLE_S:g} s per world, "
        "on top of the bundle's identified delay"
    )


def with_command_lag(
    cfg: Any, identity: dict[str, str], max_ms: float
) -> dict[str, str]:
    """Widen every actuator's command lag by 0..`max_ms` in `cfg` (in
    place) and return the identity that says so; `max_ms` 0 changes
    nothing."""
    if max_ms == 0:
        return identity
    physics_dt = float(cfg.sim.mujoco.timestep)
    steps = lag_steps(max_ms, physics_dt)
    # a NEW articulation on this config's entity: a walk may share one
    # articulation object across configs (mjlab mutates configs in place),
    # and editing it would put the lag into every later config silently
    entity = cfg.scene.entities[ROBOT_ENTITY]
    entity.articulation = replace(
        entity.articulation,
        actuators=tuple(
            replace(
                actuator,
                delay_max_lag=actuator.delay_max_lag + steps,
                delay_update_period=max(1, round(LAG_RESAMPLE_S / physics_dt)),
                delay_per_env_phase=True,
            )
            for actuator in entity.articulation.actuators
        ),
    )
    basis = lag_basis(max_ms, steps, physics_dt)
    return {
        **identity,
        Identity.DR_BASIS: f"{identity[Identity.DR_BASIS]}; {basis}",
        Identity.LAG_DR: basis,
    }
