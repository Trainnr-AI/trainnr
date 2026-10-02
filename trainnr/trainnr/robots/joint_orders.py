"""The joint orders the Go2's recordings come in — one table per order,
defined once and read by every adapter that meets it.

Two orders exist and neither is wrong. Unitree's SDK and its messages
(`LowState.motor_state`, `LowCmd.motor_cmd`) number the legs front-right,
front-left, rear-right, rear-left, with hip (abduction), thigh, calf per
leg, and name them without a suffix. Menagerie's `go2.xml` and the public
logs that follow it (DFKI's field bags, IIT's chirp) number them
front-left, front-right, rear-left, rear-right with the model's own
joint names. A recording carries the order its source used; the
identification maps components to the bundle by name, never by index.
"""

from __future__ import annotations

# Unitree's leg order (`unitree_go` messages, the SDK's `LegID`).
GO2_MOTORS = (
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
)  # fmt: skip

# Menagerie's `go2.xml` joint names in its body order.
GO2_MENAGERIE_JOINTS = (
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
)  # fmt: skip
GO2_MENAGERIE_FEET = ("FL", "FR", "RL", "RR")

# Unitree's foot order in `LowState.foot_force` (their leg order).
GO2_UNITREE_FEET = ("FR", "FL", "RR", "RL")

# What a Go2 joint is, by name, in either spelling: its leg and its class.
# Read by the identification (which leg a contact channel masks) and the
# synthetic study (which truth and amplitude a joint gets), never parsed
# out of the name.
GO2_JOINT_CLASSES = ("hip", "thigh", "calf")
MODEL_JOINT_SUFFIX = "_joint"
GO2_ANATOMY: dict[str, tuple[str, str]] = {
    f"{leg}_{kind}{suffix}": (leg, kind)
    for leg in GO2_MENAGERIE_FEET
    for kind in GO2_JOINT_CLASSES
    for suffix in ("", MODEL_JOINT_SUFFIX)
}

# Declared renames from a recording's joint names to a model's: Unitree's
# bus names a motor `FR_hip`, the Go2 MJCFs name the joint `FR_hip_joint`.
# The identification applies the one table that maps every unknown name
# onto a hinge of the model, and records which (a live capture off the
# vendor's bus reached no model joint before this table existed; review
# 2026-09-24).
JOINT_RENAMES: dict[str, dict[str, str]] = {
    "unitree-motors-to-model": {m: f"{m}{MODEL_JOINT_SUFFIX}" for m in GO2_MOTORS},
}


def anatomy_of(joint: str) -> tuple[str, str]:
    """(leg, class) of a Go2 joint by its name in either spelling; refused
    by name for a joint the table does not know."""
    try:
        return GO2_ANATOMY[joint]
    except KeyError:
        raise ValueError(
            f"joint {joint!r} is not a Go2 hip, thigh or calf in "
            "trainnr.robots.joint_orders (add its robot's table)"
        ) from None


def rename_to(
    names: tuple[str, ...], targets: set[str]
) -> tuple[tuple[str, ...], str | None]:
    """`names` onto `targets` (a model's hinges): unchanged when every name
    is already a target, else through the one declared table that maps
    every name onto a target. Returns the names and the table used (None
    when none was needed); refused by name when no table maps them."""
    if all(n in targets for n in names):
        return names, None
    for table_name, table in JOINT_RENAMES.items():
        mapped = tuple(table.get(n, n) for n in names)
        if all(m in targets for m in mapped):
            return mapped, table_name
    unknown = [n for n in names if n not in targets]
    raise ValueError(
        f"joints {unknown} are not hinges of the model, and no table in "
        f"JOINT_RENAMES ({sorted(JOINT_RENAMES)}) maps them onto one"
    )
