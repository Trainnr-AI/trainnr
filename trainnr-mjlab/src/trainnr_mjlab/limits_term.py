"""A robot's limits over several joints (#121) as mjlab terms: the same
rules, tables and grids `trainnr.robot.joint_limits` reads (as the robot's
firmware does), checked on every world at once in torch, so a walk can end
an episode on a broken limit or pay for it in reward.

`TorchLimits` mirrors `JointLimits.violated` exactly (its test draws poses
and compares the two); the terms cache one per environment and limits file.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from trainnr.robot.joint_limits import GRID_EPS_DEG, JointLimits

ROBOT = "robot"


class TorchLimits:
    """Declared limits on a batch of poses: `violated(q)` with `q` (N, J)
    radians in `joint_names` order gives (N, L) booleans."""

    def __init__(
        self, limits: JointLimits, joint_names: list[str], device: Any
    ) -> None:
        missing = limits.missing(joint_names)
        if missing:
            raise KeyError(
                f"the limits read joints the robot lacks: {', '.join(missing)}"
            )
        self.limits = limits
        self.column = {name: i for i, name in enumerate(joint_names)}
        self.device = device
        self._tables: dict[str, dict[str, torch.Tensor]] = {}
        self._grids: dict[str, tuple[torch.Tensor, list[torch.Tensor]]] = {}
        for limit in limits.limits:
            d = limit.data
            if limit.form == "table":
                self._tables[limit.name] = {
                    key: torch.tensor(d[key], dtype=torch.float32, device=device)
                    for key in ("minimum_by", "maximum_by")
                    if d.get(key)
                }
            elif limit.form == "grid":
                shape = [len(d["values"][a]) for a in d["axes"]]
                dense = torch.zeros(shape, dtype=torch.bool, device=device)
                for cell in d["forbidden"]:
                    dense[tuple(cell)] = True
                axes = [
                    torch.tensor(d["values"][a], dtype=torch.float32, device=device)
                    for a in d["axes"]
                ]
                self._grids[limit.name] = (dense, axes)

    @property
    def names(self) -> list[str]:
        return [limit.name for limit in self.limits.limits]

    def violated(self, q: torch.Tensor) -> torch.Tensor:
        deg = q * (180.0 / math.pi)
        out = torch.zeros(
            (q.shape[0], len(self.limits.limits)), dtype=torch.bool, device=q.device
        )
        for k, limit in enumerate(self.limits.limits):
            signs = limit.data.get("signs") or {}

            def angle(joint: str, signs: dict[str, Any] = signs) -> torch.Tensor:
                return deg[:, self.column[joint]] * float(signs.get(joint, 1.0))

            if limit.form == "rule":
                hit = torch.ones(q.shape[0], dtype=torch.bool, device=q.device)
                for c in limit.data["when"]:
                    hit &= _holds(c, angle(c["joint"]) * float(c.get("sign", 1.0)))
            elif limit.form == "table":
                hit = self._table(
                    limit, angle(limit.data["joint"]), angle(limit.data["over"])
                )
            else:
                points = [angle(a) for a in limit.data["axes"]]
                hit = self._grid(limit.name, points)
            out[:, k] = hit
        return out

    def _table(
        self, limit: Any, joint: torch.Tensor, over: torch.Tensor
    ) -> torch.Tensor:
        d = limit.data
        active = (
            over > d["from"] if "from" in d else torch.ones_like(over, dtype=torch.bool)
        )
        hit = torch.zeros_like(joint, dtype=torch.bool)
        for key, table in self._tables[limit.name].items():
            at = torch.searchsorted(
                table[:, 0].contiguous(), over.contiguous(), right=False
            )
            bound = table[at.clamp(0, len(table) - 1), 1]
            hit |= active & (joint < bound if key == "minimum_by" else joint > bound)
        return hit

    def _grid(self, name: str, points: list[torch.Tensor]) -> torch.Tensor:
        dense, axes = self._grids[name]
        brackets = []
        for point, grid in zip(points, axes, strict=True):
            if len(grid) == 1:
                zero = torch.zeros_like(point, dtype=torch.long)
                brackets.append((zero, zero))
                continue
            v = point.clamp(float(grid[0]), float(grid[-1]))
            hi = torch.searchsorted(grid, v.contiguous()).clamp(1, len(grid) - 1)
            lo = hi - 1
            on_lo = (v - grid[lo]).abs() <= GRID_EPS_DEG
            on_hi = (v - grid[hi]).abs() <= GRID_EPS_DEG
            lo, hi = torch.where(on_hi & ~on_lo, hi, lo), torch.where(on_lo, lo, hi)
            brackets.append((lo, hi))
        hit = torch.zeros_like(points[0], dtype=torch.bool)
        n = len(brackets)
        for corner in range(1 << n):
            index = tuple(brackets[a][(corner >> a) & 1] for a in range(n))
            hit |= dense[index]
        return hit


def _holds(condition: dict[str, Any], v: torch.Tensor) -> torch.Tensor:
    if "above" in condition:
        return v > condition["above"]
    if "below" in condition:
        return v < condition["below"]
    lo, hi = condition["outside"] if "outside" in condition else condition["within"]
    inside = (v >= lo) & (v <= hi)
    return ~inside if "outside" in condition else inside


_CACHE: dict[tuple[int, str], TorchLimits] = {}


def limits_of(env: Any, limits_file: str) -> TorchLimits:
    """The limits for this environment's robot, built once."""
    key = (id(env), limits_file)
    if key not in _CACHE:
        robot = env.scene[ROBOT]
        _CACHE[key] = TorchLimits(
            JointLimits.load(Path(limits_file)), list(robot.joint_names), env.device
        )
    return _CACHE[key]


def limits_broken(env: Any, limits_file: str) -> torch.Tensor:
    """A termination: the worlds whose pose breaks a declared limit."""
    q = env.scene[ROBOT].data.joint_pos
    return limits_of(env, limits_file).violated(q).any(dim=1)


def limits_count(env: Any, limits_file: str) -> torch.Tensor:
    """A cost: how many declared limits each world's pose breaks."""
    q = env.scene[ROBOT].data.joint_pos
    return limits_of(env, limits_file).violated(q).sum(dim=1).float()
