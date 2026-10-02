"""The locomotion judgment, declared once: what "survived" and "tracked"
mean for an episode at a held velocity command.

The walk evaluation (`trainnr_mjlab.walk_verdict`) and the sim-to-sim gate
(`trainnr.deploy.gate`) judge under this rule; a policy that passes
one and fails the other did so under the same criterion, never under
two transcriptions of it.

- survived: the episode ended by time-out, never by a fall.
- tracked: the episode-mean planar velocity error, in the base frame
  against the commanded twist, closes at least half the gap standing
  still would leave — err_ratio = mean‖v - v*‖ / max(mean‖v*‖, floor)
  below the bound. Dimensionless, so a gentle command is not an easier
  exam than a brisk one; the floor keeps a near-zero command from
  dividing by nothing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

ERR_RATIO_BOUND = 0.5  # tracked = closes at least half the standing-still gap
ERR_FLOOR_MPS = 0.1  # below this commanded speed the ratio's denominator floors


def criterion_text() -> str:
    """The rule as a record states it."""
    return f"survived and err_ratio<{ERR_RATIO_BOUND}"


@dataclass(frozen=True)
class TrackingOutcome:
    """One episode's measured facts, before any threshold is applied."""

    steps: int
    fell: bool
    mean_err: float  # mean ‖v_xy - v*_xy‖ over the episode, m/s
    mean_cmd: float  # mean ‖v*_xy‖ over the episode, m/s

    @property
    def err_ratio(self) -> float:
        return self.mean_err / max(self.mean_cmd, ERR_FLOOR_MPS)

    @property
    def survived(self) -> bool:
        return not self.fell

    @property
    def tracked(self) -> bool:
        return self.err_ratio < ERR_RATIO_BOUND

    @property
    def success(self) -> bool:
        return self.survived and self.tracked

    def row(self) -> dict[str, Any]:
        """The record's row: the facts and what they judged."""
        return asdict(self) | {"err_ratio": self.err_ratio, "success": self.success}
