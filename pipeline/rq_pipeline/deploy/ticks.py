"""One control tick of a gate trial, whatever commands it (docs/76 A6):
the policy acts, the base velocity is read against the command, the
mirror and the contacts see the tick, a fall ends the trial. The
held-twist trial (`deploy.gate`) and the course trial (`deploy.course`)
step through this and nothing else, so the judged quantity has one
definition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from rq_pipeline.deploy.mirror import GateMirror
    from rq_pipeline.deploy.runtimes import GateRuntime


@dataclass
class Ticks:
    """One control tick of a gate trial, whatever commands it: the policy
    acts, the base velocity is read against the command, the mirror and
    the contacts see the tick, a fall ends the trial. The held-twist and
    the course trials step through this and nothing else, so the judged
    quantity has one definition."""

    dt: float
    mirror: GateMirror | None = None
    contacts: list[np.ndarray] | None = None
    err_sum: float = 0.0
    cmd_sum: float = 0.0
    steps: int = 0
    fell: bool = False

    def tick(self, runtime: GateRuntime, command: np.ndarray) -> bool:
        """Advance one tick under `command`; True when the robot fell."""
        runtime.command = command.astype(np.float32)
        runtime.apply(runtime.act(runtime.observe()))
        v = runtime.base_velocity_b()
        if self.mirror is not None:
            self.mirror.tick(self.dt, runtime.pose(), command, v)
        if self.contacts is not None:
            touched = runtime.contact_points()
            if touched is not None and len(touched):
                self.contacts.append(touched)
        self.err_sum += float(np.linalg.norm(v[:2] - command[:2]))
        self.cmd_sum += float(np.linalg.norm(command[:2]))
        self.steps += 1
        self.fell = bool(runtime.fell_over())
        return self.fell

    def outcome(self) -> dict[str, Any]:
        """The `TrackingOutcome` fields this trial measured."""
        return {
            "steps": self.steps,
            "fell": self.fell,
            "mean_err": self.err_sum / max(self.steps, 1),
            "mean_cmd": self.cmd_sum / max(self.steps, 1),
        }
