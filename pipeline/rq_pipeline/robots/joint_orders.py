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
