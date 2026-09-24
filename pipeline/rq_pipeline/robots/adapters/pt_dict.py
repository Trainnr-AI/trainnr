"""A torch-saved dict of arrays (`.pt`) as a recording — the shape the
published identification chirps come in.

The first layout is IIT's sim2real-robot-identification (BSD-3-Clause,
2026): `time`, `dof_pos`, `dof_vel`, `des_dof_pos`, `des_dof_vel` at
200 Hz, `kp` and `kd` per joint, the robot's base fixed in the air, no
torque recorded — the motor torque is the PD law the joint ran under,
which the fitter reconstructs from the command and the gains. Its
column order for the Go2 is Menagerie's (FL, FR, RL, RR; hip, thigh,
calf), verified from the alternating hip-abduction signs of the
recorded poses. A file whose keys match no layout is refused by name
with the keys it has.

2026-09-24: moved here from `robot/pt_dict.py` (it was written beside
the identification that needed it first) and named `pt-dict`; the
reader of torch's zip under it is `robots/torch_pickle.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rq_pipeline.robots.adapter import adapter
from rq_pipeline.robots.joint_orders import GO2_MENAGERIE_JOINTS
from rq_pipeline.robots.recording import (
    BASIS_PUBLIC,
    COLLECTION_SCRIPTED,
    JOINT_COMMAND,
    JOINT_COMMAND_VELOCITY,
    JOINT_KD,
    JOINT_KP,
    JOINT_POSITION,
    JOINT_VELOCITY,
    Channel,
    Recording,
    monotone,
)
from rq_pipeline.robots.torch_pickle import load_tensors

NAME = "pt-dict"
SUFFIX = ".pt"


@dataclass(frozen=True)
class Layout:
    """Which keys a saved dict carries and what each one is."""

    name: str
    keys: tuple[str, ...]
    joints: tuple[str, ...]
    notes: tuple[str, ...]
    basis: str


LAYOUTS: dict[str, Layout] = {
    "iit-chirp": Layout(
        name="iit-chirp",
        keys=("time", "dof_pos", "dof_vel", "des_dof_pos", "des_dof_vel", "kp", "kd"),
        joints=GO2_MENAGERIE_JOINTS,
        basis=BASIS_PUBLIC,
        notes=(
            "public log: IIT DLS lab's Go2 chirp with the base fixed in the air "
            "(iit-DLSLab/sim2real-robot-identification, BSD-3-Clause); the "
            "collection script has a simulation flag, so the real-robot origin "
            "is the maintainers' word",
            "no torque recorded: the motor torque is the PD law kp (q_des - q) + "
            "kd (dq_des - dq) the joint ran under",
        ),
    ),
}


def layout_for(keys: set[str]) -> Layout:
    for layout in LAYOUTS.values():
        if set(layout.keys) <= keys:
            return layout
    raise ValueError(
        f"no .pt layout matches keys {sorted(keys)}; known: "
        + ", ".join(f"{name} {list(layout.keys)}" for name, layout in LAYOUTS.items())
    )


def read_pt(path: Path) -> Recording:
    saved = load_tensors(path)
    if not isinstance(saved, dict):
        raise ValueError(f"{path} holds a {type(saved).__name__}, not a dict of arrays")
    layout = layout_for(set(saved))
    arrays: dict[str, np.ndarray] = {
        k: np.asarray(saved[k], dtype=np.float64) for k in layout.keys
    }
    times, mask = monotone(arrays["time"])
    joints = layout.joints
    n = len(times)
    width = arrays["dof_pos"].shape[1]
    if width != len(joints):
        raise ValueError(f"{path}: {width} joint columns, layout names {len(joints)}")

    def per_joint(name: str, key: str, unit: str) -> Channel:
        return Channel(name, times, arrays[key][mask], unit, joints)

    gains = {
        JOINT_KP: np.tile(arrays["kp"].reshape(1, -1), (n, 1)),
        JOINT_KD: np.tile(arrays["kd"].reshape(1, -1), (n, 1)),
    }
    channels = {
        JOINT_POSITION: per_joint(JOINT_POSITION, "dof_pos", "rad"),
        JOINT_VELOCITY: per_joint(JOINT_VELOCITY, "dof_vel", "rad/s"),
        JOINT_COMMAND: per_joint(JOINT_COMMAND, "des_dof_pos", "rad"),
        JOINT_COMMAND_VELOCITY: per_joint(
            JOINT_COMMAND_VELOCITY, "des_dof_vel", "rad/s"
        ),
        JOINT_KP: Channel(JOINT_KP, times, gains[JOINT_KP], "N*m/rad", joints),
        JOINT_KD: Channel(JOINT_KD, times, gains[JOINT_KD], "N*m*s/rad", joints),
    }
    census: dict[str, Any] = {
        "layout": layout.name,
        "joints": list(joints),
        "keys": sorted(saved),
        "base": "fixed in the air",
    }
    return Recording(
        source=Path(path).name,
        adapter=NAME,
        channels=channels,
        census=census,
        notes=list(layout.notes),
        collection=COLLECTION_SCRIPTED,
        basis=layout.basis,
    )


@adapter(
    NAME, doc="A torch-saved dict of arrays (.pt), by declared layout, no torch needed"
)
class PtDict:
    name = NAME

    def accepts(self, source: Path) -> bool:
        return Path(source).suffix == SUFFIX

    def read(self, source: Path) -> Recording:
        return read_pt(Path(source))
