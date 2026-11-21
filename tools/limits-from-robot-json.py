"""A robot's limits over several joints, from Microduck Nano's firmware
configuration, as trainnr's limits file (#121):

    python3 tools/limits-from-robot-json.py <firmware/config/robot.json> \\
        --out <bundle>/limits.json

The four limits of that firmware (`head_interlock`, `knee_ankle`,
`hip_map`, `both_legs`, read as `firmware/.../duck/safety.cpp` reads them)
written in the three general forms of `trainnr.robot.joint_limits`: rules,
step tables and grids. Its right leg is checked in the left leg's signs,
all five joints flipped, as the firmware's `leg_pose` does. A worked
example of the forms; another robot writes its limits file directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trainnr"))

from trainnr.robot.joint_limits import SCHEMA, JointLimits

LEG = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")
FLIP = -1.0  # the right leg's sign, every joint (safety.cpp, leg_pose)


def _right_signs(*names: str) -> dict[str, float]:
    return {f"right_{n}": FLIP for n in names}


def head(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Roll only while the yaw is centred, yaw only while the roll is:
    one rule; and the roll's own stop, its margin subtracted."""
    yaw = cfg["yaw_window_deg"] - cfg.get("yaw_margin_deg", 0.0)
    roll_free = cfg["roll_free_deg"]
    roll_max = cfg["roll_max_deg"] - cfg.get("roll_margin_deg", 0.0)
    return [
        {
            "name": "head roll and yaw interlock",
            "form": "rule",
            "when": [
                {"joint": "head_yaw", "outside": [-yaw, yaw]},
                {"joint": "head_roll", "outside": [-roll_free, roll_free]},
            ],
            "note": "roll only while the head's yaw is centred, and the reverse",
        },
        {
            "name": "head roll stop",
            "form": "rule",
            "when": [{"joint": "head_roll", "outside": [-roll_max, roll_max]}],
            "note": "the roll's stop with the firmware's margin taken off",
        },
    ]


def knee_ankle(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for side in ("left", "right"):
        item: dict[str, Any] = {
            "name": f"{side} knee and ankle",
            "form": "table",
            "joint": f"{side}_ankle",
            "over": f"{side}_knee",
            "from": cfg["knee_over_deg"],
            "minimum_by": cfg["ankle_min_by_knee_deg"],
            "note": "past the knee angle, the ankle's minimum of the first row "
            "at or above it: else the foot meets the hip bracket or the thigh",
        }
        if side == "right":
            item["signs"] = _right_signs("knee", "ankle")
        out.append(item)
    return out


def hip(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Each leg's collision map: rows by (pitch, yaw, knee), a group of
    characters per ankle value, character i = roll value i, '1' = not
    allowed; the review's extra collisions joined in, as the firmware does."""
    grid = cfg["grid"]
    axes = ("hip_pitch", "hip_yaw", "knee", "ankle", "hip_roll")
    index = {a: {float(v): i for i, v in enumerate(grid[a])} for a in axes}
    cells: dict[str, set[tuple[int, ...]]] = {"left": set(), "right": set()}
    for row in cfg["rows"]:
        p, y, k = (index[a][float(row[a])] for a in ("hip_pitch", "hip_yaw", "knee"))
        for side in ("left", "right"):
            for a, group in enumerate(row[side].split()):
                for r, char in enumerate(group):
                    if char == "1":
                        cells[side].add((p, y, k, a, r))
    ankle_zero = index["ankle"][0.0]
    for extra in cfg.get("review_extra_collisions", []):
        cell = (
            index["hip_pitch"][float(extra["hip_pitch"])],
            index["hip_yaw"][float(extra["hip_yaw"])],
            index["knee"][float(extra["knee"])],
            ankle_zero,  # the review swept with the ankle at 0
            index["hip_roll"][float(extra["hip_roll"])],
        )
        for side in ("left", "right") if extra["leg"] == "both" else (extra["leg"],):
            cells[side].add(cell)
    out = []
    for side in ("left", "right"):
        joints = [f"{side}_{a}" for a in axes]
        item: dict[str, Any] = {
            "name": f"{side} hip map",
            "form": "grid",
            "axes": joints,
            "values": {j: grid[a] for j, a in zip(joints, axes, strict=True)},
            "forbidden": sorted(list(c) for c in cells[side]),
            "note": "poses where the leg's parts cross or pass closer than 0.3 mm",
        }
        if side == "right":
            item["signs"] = _right_signs(*LEG)
        out.append(item)
    return out


def both_legs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Both hips toed out: not both legs forward of a pitch; nor, with a
    hip rolled in, both forward of a nearer pitch."""
    toed = [
        {"joint": "left_hip_yaw", "above": cfg["yaw_out_over_deg"]},
        {"joint": "right_hip_yaw", "above": cfg["yaw_out_over_deg"], "sign": FLIP},
    ]

    def forward(deg: float) -> list[dict[str, Any]]:
        return [
            {"joint": "left_hip_pitch", "below": deg},
            {"joint": "right_hip_pitch", "below": deg, "sign": FLIP},
        ]

    out = [
        {
            "name": "both legs forward, toed out",
            "form": "rule",
            "when": toed + forward(cfg["pitch_forward_of_deg"]),
            "note": "the feet meet",
        }
    ]
    for side, sign in (("left", 1.0), ("right", FLIP)):
        out.append(
            {
                "name": f"both legs forward, toed out, {side} hip rolled in",
                "form": "rule",
                "when": toed
                + forward(cfg["band_pitch_forward_of_deg"])
                + [
                    {
                        "joint": f"{side}_hip_roll",
                        "below": cfg["band_roll_in_past_deg"],
                        "sign": sign,
                    }
                ],
                "note": "the band the design's sweep added",
            }
        )
    return out


def convert(robot: dict[str, Any], source: str) -> dict[str, Any]:
    limits = (
        head(robot["head_interlock"])
        + knee_ankle(robot["knee_ankle"])
        + hip(robot["hip_map"])
        + both_legs(robot["both_legs"])
    )
    return {"schema": SCHEMA, "source": source, "limits": limits}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("robot_json", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    robot = json.loads(args.robot_json.read_text())
    limits = convert(
        robot, f"Microduck Nano firmware configuration ({args.robot_json.name})"
    )
    JointLimits.from_dict(limits)  # refuses a file it would write wrong
    args.out.write_text(json.dumps(limits, indent=1) + "\n")
    print(f"{len(limits['limits'])} limits -> {args.out}")


if __name__ == "__main__":
    main()
