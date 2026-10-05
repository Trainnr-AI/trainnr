"""A hash-stamped robot bundle as an mjlab entity.

`entity_from_bundle` turns a directory under `robots/` — an MJCF, its
assets, a README, the bundle discipline the pipeline stamps — into an
mjlab `EntityCfg` whose spec loads the bundle's own model file, with the
actuator groups the caller built from certified actuator bundles. The
robot's `name@hash` comes back beside the cfg: mjlab has no metadata
slot for it, and a policy trained on an entity whose bundle cannot be
named is exactly the untraceability this stack exists to end — put the
stamp on the run's record.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from trainnr.bundles.hashing import stamp

from trainnr_mjlab.actuator import BamActuatorCfg


def entity_from_bundle(
    robot_dir: str | Path,
    model_file: str,
    actuators: tuple[BamActuatorCfg, ...],
    *,
    init_state: EntityCfg.InitialStateCfg | None = None,
    soft_joint_pos_limit_factor: float = 1.0,
) -> tuple[EntityCfg, str]:
    """The bundle's model as an `EntityCfg`, and the bundle's stamp.

    `model_file` is named explicitly because the repo's bundles spell it
    per rig today; the stamp covers every byte in `robot_dir`, so an
    edited mesh or MJCF changes the identity the caller records.
    """
    from trainnr.bundles.fetch import ensure_fetched  # noqa: PLC0415

    robot_dir = Path(robot_dir)
    # mjlab's actuators field is an ORDERED tuple of cfgs, not a mapping.
    # A dict here silently becomes a tuple of its KEY STRINGS — an entity
    # with no actuator law at all. Caught 2026-09-01 by the walk cfg's
    # linter showcase test (the entity brick's own test only compiled
    # the raw spec, whose XML actuators masked it). Refused by name now.
    from mjlab.actuator.actuator import ActuatorCfg  # noqa: PLC0415

    if not isinstance(actuators, tuple):
        # A GENERATOR drains in the check below and leaves an empty
        # tuple — an entity with no law, the very bug this refusal
        # exists to stop (second review, 2026-09-01).
        raise TypeError(
            f"actuators must be a tuple of actuator cfgs, got "
            f"{type(actuators).__name__}"
        )
    bad = [type(a).__name__ for a in actuators if not isinstance(a, ActuatorCfg)]
    if bad:
        raise TypeError(
            f"actuators must be actuator CFGS in a tuple, got {bad} — "
            "mjlab actuators are ordered, not named; pass (cfg,) not "
            "{'name': cfg}"
        )
    # The arguments are refused before anything is downloaded.
    ensure_fetched(robot_dir)  # meshes a bundle does not carry (its licence)
    model_path = robot_dir / model_file
    if not model_path.is_file():
        raise FileNotFoundError(
            f"{robot_dir.name} has no {model_file!r} — the bundle's model "
            "file is named explicitly, never guessed"
        )
    bundle_stamp = stamp(robot_dir.name, robot_dir)
    # Only a given init_state replaces EntityCfg's own default.
    optional: dict[str, Any] = {} if init_state is None else {"init_state": init_state}
    cfg = EntityCfg(
        spec_fn=lambda: mujoco.MjSpec.from_file(str(model_path)),
        articulation=EntityArticulationInfoCfg(
            actuators=tuple(actuators),
            soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
        ),
        **optional,
    )
    return cfg, bundle_stamp
