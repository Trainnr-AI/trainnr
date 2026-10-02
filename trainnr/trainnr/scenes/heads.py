"""Where a robot's head camera rides (docs/78 §4 E2): the base body it
is attached to, its pose on that body, its field of view - declared
per robot family in one registry, read by the stage that puts the
camera on a staged deployment and by the walk package that renders it
for every training world. A robot the registry does not know is
refused by name, never given another robot's head.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trainnr.bundles.hashing import STAMP_SEPARATOR

HEAD_CAMERA = "head"  # the camera's name in every model it is added to


@dataclass(frozen=True)
class HeadMount:
    """A head camera's mount: `body` names the base body in the robot's
    own MJCF (None: the model's one floating base, found by the stage),
    `pos` is the camera's place on it, `forward` and `up` its look, `fovy`
    its vertical field of view in degrees."""

    pos: tuple[float, float, float]
    fovy: float
    body: str | None = None
    forward: tuple[float, float, float] = (1.0, 0.0, 0.0)
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)

    @property
    def quat(self) -> np.ndarray:
        """The camera frame as a w-x-y-z quaternion (`look_quat`)."""
        return look_quat(np.asarray(self.forward, float), np.asarray(self.up, float))


def look_quat(forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    """A camera frame (MuJoCo: looks along its -z, +y up) looking along
    `forward`, as a w-x-y-z quaternion."""
    import mujoco  # noqa: PLC0415

    z = -forward / np.linalg.norm(forward)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.column_stack([x, y, z]).reshape(-1))
    return quat


# The registry, by robot family (the stamp's name before its hash).
HEAD_MOUNTS: dict[str, HeadMount] = {
    # ahead of the Go2's head mesh (the first render looked at it from
    # inside, 2026-09-22), on the trunk mjlab's entity names base_link
    "go2": HeadMount(pos=(0.38, 0.0, 0.06), fovy=90.0, body="base_link"),
}


def register_head_mount(family: str, mount: HeadMount) -> None:
    """A third party's robot declares its head here, once."""
    if family in HEAD_MOUNTS:
        raise ValueError(f"head mount {family!r} is already registered")
    HEAD_MOUNTS[family] = mount


def robot_family(stamp: str) -> str:
    """`go2@5003bf617b5f` -> `go2`; a bare name is its own family."""
    return stamp.split(STAMP_SEPARATOR, 1)[0]


def head_mount(robot: str) -> HeadMount:
    """The head mount of a robot, by its stamp or family; refuses by name."""
    family = robot_family(robot)
    try:
        return HEAD_MOUNTS[family]
    except KeyError as unknown:
        raise KeyError(
            f"no head mount registered for robot {family!r}: "
            f"scenes.heads.register_head_mount names one "
            f"(known: {', '.join(sorted(HEAD_MOUNTS)) or 'none'})"
        ) from unknown
