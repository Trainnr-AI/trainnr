"""The field-expansion event, made unforgettable.

The BAM actuator writes `dof_frictionloss` every physics step, per
world. mjlab allocates real per-world memory for a model field only
when some event function declares it via `@requires_model_fields`
(`Simulation.expand_model_fields` reads the event manager's collected
list). microduck carried a no-op startup event for exactly this, and
every env had to REMEMBER to register it — forget, and per-world
friction silently collapses to one shared value (docs/e2e-research/57
§5, the class of bug this package exists to kill).

Here the event exists once, `bam_expansion_event()` hands it to any env
cfg in one line, and the actuator itself REFUSES at initialize when the
field arrives unexpanded — the forgetting fails loudly, at the right
moment, with the fix in the message.
"""

from __future__ import annotations

from typing import Any

from mjlab.managers.event_manager import EventTermCfg, requires_model_fields

EXPANDED_FIELDS = ("dof_frictionloss",)


@requires_model_fields(*EXPANDED_FIELDS)
def expand_bam_fields(env: Any, env_ids: Any) -> None:
    """A deliberate no-op: its only job is the decorator above, which
    registers `dof_frictionloss` for per-world expansion. The BAM
    actuator does the writing; the solver does the stick/slip."""
    del env, env_ids


def bam_expansion_event() -> EventTermCfg:
    """The one-line registration an env cfg adds to its events:

    events.expand_bam_fields = bam_expansion_event()
    """
    return EventTermCfg(func=expand_bam_fields, mode="startup")


MISSING_EXPANSION_MESSAGE = (
    "dof_frictionloss is not expanded per world: the BAM actuator writes "
    "it every step and every world would share one value. Add "
    "`rq_mjlab.events.bam_expansion_event()` to the env cfg's events "
    "(any field name), or run `rq_mjlab.lint` on the cfg to catch this "
    "before the simulation is built."
)
