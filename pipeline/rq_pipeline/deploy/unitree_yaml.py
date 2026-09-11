"""Our deploy manifest as Unitree's `deploy.yaml`: the file their C++
controller (unitree_rl_mjlab `deploy/`) reads to run a policy against
their DDS simulator and the real robot.

Their file is our manifest reordered - the manifest was shaped on its
field set (docs/77 §6) - with their names for the observation terms.
Their runtime implements a fixed set of terms (`REGISTER_OBSERVATION`
in their `deploy/include`), so a manifest whose policy observes
something else is REFUSED here, by name: the writer is the
compatibility check (the first certified Go2 policy observed the base
linear velocity, a quantity the robot cannot measure, 2026-09-11).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rq_pipeline.deploy.manifest import Manifest

UNITREE_DEPLOY_FILE = "deploy.yaml"
# Our observation term -> theirs (their deploy runtime's registered name)
# and the params their term takes from ours.
TERMS: dict[str, tuple[str, tuple[str, ...]]] = {
    "base_ang_vel": ("base_ang_vel", ()),
    "projected_gravity": ("projected_gravity", ()),
    "command": ("velocity_commands", ()),
    "phase": ("gait_phase", ("period",)),
    "joint_pos": ("joint_pos_rel", ()),
    "joint_vel": ("joint_vel_rel", ()),
    "actions": ("last_action", ()),
}
COMMAND_NAME = "base_velocity"  # their name for the twist command


class NotDeployableError(ValueError):
    """The policy observes something their runtime cannot provide."""


def unitree_observations(manifest: Manifest) -> dict[str, dict[str, Any]]:
    """The manifest's observation list in their order-preserving form;
    refuses a term their runtime does not implement."""
    out: dict[str, dict[str, Any]] = {}
    for term in manifest.observations:
        name = term["name"]
        if name not in TERMS:
            raise NotDeployableError(
                f"observation {name!r} has no term in Unitree's deploy runtime "
                f"(it implements {', '.join(t for t, _ in TERMS.values())}); "
                "train the actor on what the robot measures (go2_walk.deployable_actor)"
            )
        theirs, keys = TERMS[name]
        params = {k: term.get("params", {}).get(k) for k in keys}
        if theirs == "velocity_commands":
            params = {"command_name": COMMAND_NAME}
        scale = term.get("scale", 1.0)
        width = int(term["width"])
        out[theirs] = {
            "params": params,
            "clip": term.get("clip"),
            "scale": scale if isinstance(scale, list) else [float(scale)] * width,
            "history_length": int(term.get("history_length", 1)),
        }
    return out


def unitree_deploy(manifest: Manifest) -> dict[str, Any]:
    """The whole file as a mapping, ready for YAML."""
    joints = manifest.joints
    action = dict(manifest.raw["action"])
    control = manifest.control
    twist = manifest.raw.get("commands", {}).get("twist", {})
    return {
        "joint_ids_map": list(joints["sdk_order_map"]),
        "step_dt": round(1.0 / float(control["control_hz"]), 6),
        "stiffness": list(joints["stiffness"]),
        "damping": list(joints["damping"]),
        "default_joint_pos": list(joints["default_pos"]),
        "commands": {
            COMMAND_NAME: {
                "ranges": {
                    k: (list(twist[k]) if twist.get(k) is not None else None)
                    for k in ("lin_vel_x", "lin_vel_y", "ang_vel_z", "heading")
                }
            }
        },
        "actions": {
            "JointPositionAction": {
                "clip": action.get("clip"),
                "joint_names": [".*"],
                "scale": list(action["scale"]),
                "offset": list(action["offset"]),
                "joint_ids": None,
            }
        },
        "observations": unitree_observations(manifest),
    }


def write_unitree_deploy(manifest: Manifest, out: Path) -> Path:
    import yaml  # noqa: PLC0415

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(unitree_deploy(manifest), sort_keys=False))
    return out
