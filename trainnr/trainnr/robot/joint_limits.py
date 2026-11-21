"""A robot's limits over several joints (#121): poses its parts cannot take
although every joint is inside its own range, declared with the robot as
data in its bundle (`limits.json`, stamped with it) and checked here.

Three forms, each a general shape rather than one robot's code:

- `rule`: the pose is forbidden when every condition holds. A condition
  names a joint and `above`, `below`, `outside` [lo, hi] or `within`
  [lo, hi]. A head that rolls only while its yaw is centred is two rules;
  both legs forward while both hips are toed out is one.
- `table`: a joint's minimum (and or maximum) as a step table over another
  joint: past `from`, the bound is the one of the first row whose value is
  at or above the other joint's (the last row's beyond the table). A knee
  that lifts the ankle's floor as it bends.
- `grid`: forbidden cells of a grid over several joints; between grid
  values the stricter neighbour holds (a point is forbidden when any grid
  point around it is), beyond an axis's end its end value is used. A hip's
  collision map over pitch, yaw, knee, ankle and roll.

Every condition, table and grid may carry `signs` (joint -> +1 or -1): the
joint's angle is multiplied by it before the limit is read, so a right
leg's limits can be written in the left leg's signs. Angles are in
degrees in the file, radians at the checker.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

LIMITS_FILE = "limits.json"
SCHEMA = "trainnr-joint-limits/1"
FORMS = ("rule", "table", "grid")
OPERATORS = ("above", "below", "outside", "within")
# A value this close to a grid value reads as on it (degrees).
GRID_EPS_DEG = 1e-3


@dataclass(frozen=True)
class Limit:
    """One declared limit: its name, form, the joints it reads, and its
    data as declared (degrees)."""

    name: str
    form: str
    joints: tuple[str, ...]
    data: Mapping[str, Any]
    note: str = ""

    def describe(self) -> str:
        """The limit in words, as a robot's page shows it."""
        d = self.data
        if self.form == "rule":
            return "forbidden when " + " and ".join(
                _condition_text(c) for c in d["when"]
            )
        if self.form == "table":
            bounds = " and ".join(b for b in ("minimum", "maximum") if f"{b}_by" in d)
            start = f" past {d['from']:g}°" if "from" in d else ""
            rows = len(d.get("minimum_by") or d.get("maximum_by") or [])
            return f"{d['joint']} {bounds} by {d['over']}{start}, a {rows}-row table"
        cells = len(d["forbidden"])
        sizes = "x".join(str(len(d["values"][a])) for a in d["axes"])
        return (
            f"{cells} forbidden cells of a {sizes} grid over {', '.join(d['axes'])}, "
            "the stricter neighbour between grid values"
        )


@dataclass(frozen=True)
class JointLimits:
    """A robot's declared limits, ready to check poses against."""

    limits: tuple[Limit, ...]
    source: str = ""
    _grids: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def load(cls, path: Path) -> JointLimits:
        """Read and check a limits file; refuses a malformed one by name."""
        raw = json.loads(Path(path).read_text())
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> JointLimits:
        if raw.get("schema") != SCHEMA:
            raise ValueError(
                f"a limits file says schema {SCHEMA!r}, got {raw.get('schema')!r}"
            )
        limits = []
        names: set[str] = set()
        for i, item in enumerate(raw.get("limits") or []):
            name = str(item.get("name") or f"limit {i + 1}")
            if name in names:
                raise ValueError(f"two limits are named {name!r}")
            names.add(name)
            limits.append(_parse(name, item))
        return cls(limits=tuple(limits), source=str(raw.get("source", "")))

    @property
    def joints(self) -> tuple[str, ...]:
        """Every joint any limit reads, in first-use order."""
        seen: list[str] = []
        for limit in self.limits:
            seen += [j for j in limit.joints if j not in seen]
        return tuple(seen)

    def missing(self, model_joints: Sequence[str]) -> list[str]:
        """The joints the limits read that the model does not have."""
        return [j for j in self.joints if j not in model_joints]

    def violated(self, q: np.ndarray, joint_names: Sequence[str]) -> np.ndarray:
        """Which limits each pose breaks: `q` is (N, J) radians in
        `joint_names` order; returns (N, L) booleans in `limits` order."""
        q = np.atleast_2d(np.asarray(q, dtype=float))
        column = {name: i for i, name in enumerate(joint_names)}
        missing = self.missing(list(joint_names))
        if missing:
            raise KeyError(
                f"the limits read joints the poses lack: {', '.join(missing)}"
            )
        degrees = np.degrees(q)
        out = np.zeros((q.shape[0], len(self.limits)), dtype=bool)
        for k, limit in enumerate(self.limits):
            out[:, k] = self._check(limit, degrees, column)
        return out

    def violations(self, pose: Mapping[str, float]) -> list[str]:
        """The names of the limits one pose (joint -> radians) breaks."""
        names = list(self.joints)
        row = np.array([[pose[j] for j in names]])
        hit = self.violated(row, names)[0]
        return [limit.name for limit, bad in zip(self.limits, hit, strict=True) if bad]

    def _check(
        self, limit: Limit, deg: np.ndarray, column: Mapping[str, int]
    ) -> np.ndarray:
        signs = limit.data.get("signs") or {}

        def angle(joint: str) -> np.ndarray:
            return deg[:, column[joint]] * float(signs.get(joint, 1.0))

        if limit.form == "rule":
            hit = np.ones(deg.shape[0], dtype=bool)
            for c in limit.data["when"]:
                hit &= _holds(c, angle(c["joint"]) * float(c.get("sign", 1.0)))
            return hit
        if limit.form == "table":
            return _table_hit(
                limit.data, angle(limit.data["joint"]), angle(limit.data["over"])
            )
        return self._grid_hit(
            limit, np.stack([angle(a) for a in limit.data["axes"]], axis=1)
        )

    def _grid_hit(self, limit: Limit, points: np.ndarray) -> np.ndarray:
        d = limit.data
        dense = self._grids.get(limit.name)
        if dense is None:
            dense = np.zeros([len(d["values"][a]) for a in d["axes"]], dtype=bool)
            for cell in d["forbidden"]:
                dense[tuple(cell)] = True
            self._grids[limit.name] = dense
        # each axis: the grid indices around the value (one on a grid value
        # or beyond an end, two between values)
        brackets = []
        for a, axis in enumerate(d["axes"]):
            grid = np.asarray(d["values"][axis], dtype=float)
            if len(grid) == 1:
                zero = np.zeros(points.shape[0], dtype=int)
                brackets.append((zero, zero))
                continue
            v = np.clip(points[:, a], grid[0], grid[-1])
            hi = np.clip(np.searchsorted(grid, v, side="left"), 1, len(grid) - 1)
            lo = hi - 1
            # on a grid value (within GRID_EPS_DEG, as firmware compares)
            # the point reads that value alone
            on_lo = np.abs(v - grid[lo]) <= GRID_EPS_DEG
            on_hi = np.abs(v - grid[hi]) <= GRID_EPS_DEG
            lo, hi = np.where(on_hi & ~on_lo, hi, lo), np.where(on_lo, lo, hi)
            brackets.append((lo, hi))
        hit = np.zeros(points.shape[0], dtype=bool)
        for corner in np.ndindex(*(2,) * len(brackets)):
            index = tuple(brackets[a][side] for a, side in enumerate(corner))
            hit |= dense[index]
        return hit


def _parse(name: str, item: Mapping[str, Any]) -> Limit:  # noqa: PLR0912 - one refusal per malformation
    form = item.get("form")
    if form not in FORMS:
        raise ValueError(
            f"limit {name!r}: form is one of {', '.join(FORMS)}, got {form!r}"
        )
    note = str(item.get("note", ""))
    if form == "rule":
        when = item.get("when") or []
        if not when:
            raise ValueError(f"limit {name!r}: a rule needs at least one condition")
        for c in when:
            ops = [op for op in OPERATORS if op in c]
            if "joint" not in c or len(ops) != 1:
                raise ValueError(
                    f"limit {name!r}: a condition names a joint and one of "
                    f"{', '.join(OPERATORS)}, got {dict(c)}"
                )
        return Limit(name, form, tuple(c["joint"] for c in when), dict(item), note)
    if form == "table":
        for key in ("joint", "over"):
            if key not in item:
                raise ValueError(f"limit {name!r}: a table names its {key!r}")
        if "minimum_by" not in item and "maximum_by" not in item:
            raise ValueError(
                f"limit {name!r}: a table has minimum_by or maximum_by rows"
            )
        for key in ("minimum_by", "maximum_by"):
            rows = [row[0] for row in item.get(key) or []]
            if rows != sorted(rows) or len(set(rows)) != len(rows):
                raise ValueError(
                    f"limit {name!r}: {key}'s first column rises (the lookup reads "
                    f"the first row at or above), got {rows}"
                )
        return Limit(name, form, (item["over"], item["joint"]), dict(item), note)
    axes = list(item.get("axes") or [])
    values = item.get("values") or {}
    if not axes or any(a not in values for a in axes):
        raise ValueError(f"limit {name!r}: a grid has axes, each with its values")
    for a in axes:
        grid = list(values[a])
        if grid != sorted(grid) or len(set(grid)) != len(grid):
            raise ValueError(
                f"limit {name!r}: axis {a}'s values rise (write a right leg's in "
                f"rising order, with signs), got {grid}"
            )
    shape = [len(values[a]) for a in axes]
    for cell in item.get("forbidden") or []:
        if len(cell) != len(axes) or any(
            not 0 <= c < n for c, n in zip(cell, shape, strict=True)
        ):
            raise ValueError(f"limit {name!r}: cell {cell} is outside the {shape} grid")
    return Limit(name, form, tuple(axes), dict(item), note)


def _holds(condition: Mapping[str, Any], v: np.ndarray) -> np.ndarray:
    if "above" in condition:
        return v > condition["above"]
    if "below" in condition:
        return v < condition["below"]
    lo, hi = condition["outside"] if "outside" in condition else condition["within"]
    inside = (v >= lo) & (v <= hi)
    return ~inside if "outside" in condition else inside


def _table_hit(d: Mapping[str, Any], joint: np.ndarray, over: np.ndarray) -> np.ndarray:
    active = over > d["from"] if "from" in d else np.ones_like(over, dtype=bool)
    hit = np.zeros_like(joint, dtype=bool)
    for key, breaks in (("minimum_by", np.less), ("maximum_by", np.greater)):
        rows = d.get(key)
        if not rows:
            continue
        table = np.asarray(rows, dtype=float)
        at = np.clip(np.searchsorted(table[:, 0], over, side="left"), 0, len(table) - 1)
        hit |= active & breaks(joint, table[at, 1])
    return hit


def _condition_text(c: Mapping[str, Any]) -> str:
    sign = "-" if float(c.get("sign", 1.0)) < 0 else ""
    joint = f"{sign}{c['joint']}"
    if "above" in c:
        return f"{joint} > {c['above']:g}°"
    if "below" in c:
        return f"{joint} < {c['below']:g}°"
    lo, hi = c["outside"] if "outside" in c else c["within"]
    word = "outside" if "outside" in c else "within"
    return f"{joint} {word} [{lo:g}°, {hi:g}°]"


def check_keyframes(limits: JointLimits, model: Any) -> dict[str, list[str]]:
    """Every keyframe of a compiled model that breaks a limit, with the
    limits it breaks (a model missing a limit's joint is refused by name)."""
    import mujoco  # noqa: PLC0415 - sim extra

    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        for j in range(model.njnt)
    ]
    missing = limits.missing(names)
    if missing:
        raise KeyError(f"the limits read joints the model lacks: {', '.join(missing)}")
    wanted = list(limits.joints)
    adr = [int(model.jnt_qposadr[names.index(j)]) for j in wanted]
    out: dict[str, list[str]] = {}
    for k in range(model.nkey):
        pose = {
            j: float(model.key_qpos[k][a]) for j, a in zip(wanted, adr, strict=True)
        }
        broken = limits.violations(pose)
        if broken:
            out[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, k) or f"#{k}"] = (
                broken
            )
    return out
