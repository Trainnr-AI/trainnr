"""A COLMAP sparse model in its text form (`cameras.txt`, `images.txt`,
`points3D.txt`), read into arrays: what a splat trainer starts from and
what the capture record counts. numpy only, so the chain's own process
(no torch) and the trainer's (torch) read the same file the same way.

COLMAP's conventions, kept as they are: a camera's intrinsics by model
name; an image's pose as the rotation (w, x, y, z) and translation that
take a WORLD point into the CAMERA frame (`x_cam = R x_world + t`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CAMERAS_FILE = "cameras.txt"
IMAGES_FILE = "images.txt"
POINTS_FILE = "points3D.txt"

# Each model's parameter order, as COLMAP writes it (src/colmap/sensor/models.h):
# focal lengths, principal point, then distortion. `fx fy cx cy k1 k2 p1 p2`
# is the phone's model the chain asks COLMAP for (`COLMAP_CAMERA`).
CAMERA_MODELS: dict[str, tuple[str, ...]] = {
    "SIMPLE_PINHOLE": ("f", "cx", "cy"),
    "PINHOLE": ("fx", "fy", "cx", "cy"),
    "SIMPLE_RADIAL": ("f", "cx", "cy", "k1"),
    "RADIAL": ("f", "cx", "cy", "k1", "k2"),
    "OPENCV": ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"),
    "OPENCV_FISHEYE": ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"),
    "FULL_OPENCV": (
        "fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6",
    ),
}  # fmt: skip
RADIAL_KEYS = ("k1", "k2", "k3", "k4", "k5", "k6")
TANGENTIAL_KEYS = ("p1", "p2")


@dataclass(frozen=True)
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: dict[str, float]

    @property
    def K(self) -> np.ndarray:  # noqa: N802 - the intrinsic matrix's own name
        p = self.params
        fx, fy = p.get("fx", p.get("f", 0.0)), p.get("fy", p.get("f", 0.0))
        return np.array([[fx, 0, p["cx"]], [0, fy, p["cy"]], [0, 0, 1]], np.float64)

    @property
    def radial(self) -> tuple[float, ...]:
        return tuple(self.params.get(k, 0.0) for k in RADIAL_KEYS)

    @property
    def tangential(self) -> tuple[float, ...]:
        return tuple(self.params.get(k, 0.0) for k in TANGENTIAL_KEYS)

    @property
    def distorted(self) -> bool:
        return any(v != 0.0 for v in (*self.radial, *self.tangential))


@dataclass(frozen=True)
class Image:
    id: int
    name: str
    camera_id: int
    quat_wxyz: np.ndarray  # (4,) world -> camera
    translation: np.ndarray  # (3,) world -> camera

    @property
    def world_to_camera(self) -> np.ndarray:
        """The 4x4 view matrix COLMAP's pose spells."""
        w, x, y, z = (float(v) for v in self.quat_wxyz)
        rot = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ],
            np.float64,
        )
        out = np.eye(4)
        out[:3, :3] = rot
        out[:3, 3] = self.translation
        return out


@dataclass(frozen=True)
class TextModel:
    cameras: dict[int, Camera]
    images: tuple[Image, ...]
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # (n, 3)
    colors: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.uint8))

    @property
    def camera_model(self) -> str:
        first = next(iter(self.cameras.values()), None)
        return first.model if first else ""


def _rows(path: Path) -> list[list[str]]:
    return [
        line.split()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def read_cameras(path: Path) -> dict[int, Camera]:
    cameras: dict[int, Camera] = {}
    for row in _rows(path):
        cam_id, model, width, height, *values = row
        names = CAMERA_MODELS.get(model)
        if names is None:
            raise ValueError(f"camera model {model!r} is not one this reader knows")
        if len(values) < len(names):
            raise ValueError(
                f"camera {cam_id} ({model}) carries {len(values)} parameters, "
                f"the model has {len(names)}"
            )
        cameras[int(cam_id)] = Camera(
            id=int(cam_id),
            model=model,
            width=int(width),
            height=int(height),
            params=dict(
                zip(names, (float(v) for v in values[: len(names)]), strict=True)
            ),
        )
    return cameras


def read_images(path: Path) -> tuple[Image, ...]:
    """Two lines per image, the pose line then its 2D points - a line COLMAP
    leaves EMPTY for an image with none, so the pairs are taken from the
    raw lines (comments dropped, blanks kept); skipping blank lines shifted
    every image after the first empty one by a line (2026-09-24)."""
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if not line.startswith("#")
    ]
    out: list[Image] = []
    for pose in lines[::2]:
        row = pose.split()
        if not row:
            continue
        image_id, qw, qx, qy, qz, tx, ty, tz, cam_id, name = row[:10]
        out.append(
            Image(
                id=int(image_id),
                name=name,
                camera_id=int(cam_id),
                quat_wxyz=np.array([qw, qx, qy, qz], np.float64),
                translation=np.array([tx, ty, tz], np.float64),
            )
        )
    return tuple(out)


def read_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = _rows(path)
    if not rows:
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
    xyz = np.array([[float(v) for v in r[1:4]] for r in rows], np.float64)
    rgb = np.array([[int(v) for v in r[4:7]] for r in rows], np.uint8)
    return xyz, rgb


def read_text_model(sparse: Path) -> TextModel:
    """The three files of a text model in `sparse`."""
    sparse = Path(sparse)
    xyz, rgb = read_points(sparse / POINTS_FILE)
    return TextModel(
        cameras=read_cameras(sparse / CAMERAS_FILE),
        images=read_images(sparse / IMAGES_FILE),
        points=xyz,
        colors=rgb,
    )
