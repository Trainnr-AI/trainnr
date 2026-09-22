"""mujoco_warp's overflow warnings, as the walks want them.

mujoco_warp 3.13 turns every overflow warning on by default
(`opt.warn_overflow = OverflowType.ALL`, set by `put_model`), and prints
the one it hit from the device on every step. mjlab's velocity task caps
the solver's line search at 20 iterations, the cap is reached, and a
20-iteration smoke train wrote about 149,000 lines of

    linesearch iterations limit reached - please increase ls_iterations beyond 20

- 25 MB of log - where 3.11 had hit the same cap silently (E0,
docs/78, 2026-09-22; the certificate on both instruments identical to the
last digit, so the physics is the same). Raising the cap would change the
physics every walk trained under; the warning is information. So the
LINE-SEARCH warning alone is switched off; every other overflow (too many
constraints or contacts, a broadphase or collision buffer) still prints,
because those drop physics.

mjlab 1.6.0 predates the flag and has no knob for it. It does apply
warp-only options through one method, `SimulationCfg.apply_wp_opt`,
after `put_model` and before any graph is captured; `QuietSimulationCfg`
is mjlab's config with that method extended - mjlab's own work first,
then the one bit cleared. On an instrument without the flag (< 3.13) it
is mjlab's config unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from mjlab.sim.sim import SimulationCfg

# Overflows that are information, not lost physics: the line search
# stopping at the cap the task declared.
# HFIELD: mujoco_warp caps the prisms one geom collides with at mujoco's
# MJ_MAXCONPAIR (50) and prints a line per world per step past it. On a
# scene at the 5 cm training grid only a trunk lying flat exceeds the cap
# (a fallen robot early in training: 74,000 lines in a 20-iteration smoke,
# 2026-09-23); the contacts past 50 are dropped with or without the print.
QUIET_OVERFLOWS: tuple[str, ...] = ("LS_ITERATIONS", "HFIELD")


@dataclass(kw_only=True)
class QuietSimulationCfg(SimulationCfg):
    quiet_overflows: tuple[str, ...] = QUIET_OVERFLOWS

    def apply_wp_opt(self, wp_opt: Any) -> None:
        super().apply_wp_opt(wp_opt)
        import mujoco_warp as mjwarp  # noqa: PLC0415

        overflow = getattr(mjwarp, "OverflowType", None)
        if overflow is None or not hasattr(wp_opt, "warn_overflow"):
            return  # an instrument older than the flag: nothing to quiet
        bits = int(wp_opt.warn_overflow)
        for name in self.quiet_overflows:
            bits &= ~int(overflow[name])
        wp_opt.warn_overflow = bits


def quiet(cfg: Any) -> Any:
    """The env config with its simulation config as `QuietSimulationCfg`,
    every field of mjlab's kept."""
    sim = cfg.sim
    if not isinstance(sim, QuietSimulationCfg):
        cfg.sim = QuietSimulationCfg(
            **{f.name: getattr(sim, f.name) for f in fields(sim) if f.init}
        )
    return cfg
