"""A Gaussian splat as arrays, and the two files it travels in.

The canonical file is the 3DGS PLY (binary little-endian; the fields
Inria's trainer wrote and every reader since expects): positions, the
degree-0 colour as `f_dc_*`, higher harmonics as `f_rest_*`, opacity
and scales in their pre-activation form, the rotation as a quaternion
`rot_0..3` in (w, x, y, z). This module keeps the ACTIVATED values in
memory (scales as standard deviations, opacity in 0..1, colour in
0..1) - what a renderer consumes - and applies the activations at the
file boundary in both directions, so a file written here reads back as
the same arrays. The web `.splat` (32 bytes a record: position, scale,
RGBA bytes, rotation bytes) is read only: it has already quantised
opacity and rotation and dropped every harmonic beyond degree 0.

numpy only: the pipeline stays importable without torch; a trainer's
checkpoint (gsplat's `model.pt`) is converted by the walk package,
where torch lives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

PLY_SUFFIX = ".ply"
WEB_SPLAT_SUFFIX = ".splat"
WEB_RECORD_BYTES = 32
# The zeroth spherical-harmonic band's constant: colour = SH_C0 * f_dc + 0.5.
SH_C0 = 0.28209479177387814
# Below this opacity a gaussian is invisible for the purposes of a
# surface audit (the gap): it contributes no surface the eye would see.
VISIBLE_OPACITY = 0.5
QUAT_WXYZ = ("rot_0", "rot_1", "rot_2", "rot_3")


@dataclass
class Splats:
    """One splat scene, activated: what a renderer consumes."""

    means: np.ndarray  # (n, 3) float32, metres
    quats: np.ndarray  # (n, 4) float32, (w, x, y, z), unit
    scales: np.ndarray  # (n, 3) float32, standard deviations in metres
    opacities: np.ndarray  # (n,) float32 in 0..1
    colors: np.ndarray  # (n, 3) float32 in 0..1, the degree-0 colour
    # Higher harmonics as the file carried them, (n, k, 3) or empty;
    # kept so a round trip loses nothing, dropped by the renderer.
    sh_rest: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 3), np.float32))

    @property
    def count(self) -> int:
        return int(self.means.shape[0])

    @property
    def sh_degree(self) -> int:
        """The harmonic degree the file carried: 0 without `sh_rest`."""
        k = int(self.sh_rest.shape[1]) if self.sh_rest.size else 0
        return round(math.sqrt(k + 1)) - 1

    def visible(self, opacity: float = VISIBLE_OPACITY) -> Splats:
        keep = self.opacities >= opacity
        return Splats(
            self.means[keep],
            self.quats[keep],
            self.scales[keep],
            self.opacities[keep],
            self.colors[keep],
            self.sh_rest[keep] if self.sh_rest.size else self.sh_rest,
        )

    def transformed(
        self, *, scale: float, rotation: np.ndarray, translation: np.ndarray
    ) -> Splats:
        """The scene moved into another frame: p' = s·R·p + t, the
        gaussians' rotations composed with R, their scales multiplied
        by s (a similarity keeps them axis-aligned in their own frame)."""
        rot = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        t = np.asarray(translation, dtype=np.float64).reshape(3)
        means = (scale * (self.means.astype(np.float64) @ rot.T) + t).astype(np.float32)
        q_rot = quat_from_matrix(rot)
        quats = quat_multiply(np.broadcast_to(q_rot, self.quats.shape), self.quats)
        return Splats(
            means,
            quats.astype(np.float32),
            (self.scales * np.float32(scale)).astype(np.float32),
            self.opacities,
            self.colors,
            self.sh_rest,
        )


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of (w, x, y, z) quaternions, row-wise."""
    aw, ax, ay, az = (a[..., i] for i in range(4))
    bw, bx, by, bz = (b[..., i] for i in range(4))
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def quat_from_matrix(rot: np.ndarray) -> np.ndarray:
    """(w, x, y, z) of a rotation matrix (Shepperd's method)."""
    m = np.asarray(rot, dtype=np.float64)
    tr = np.trace(m)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = (
            0.25 * s,
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
        )
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (
            (m[2, 1] - m[1, 2]) / s,
            0.25 * s,
            (m[0, 1] + m[1, 0]) / s,
            (m[0, 2] + m[2, 0]) / s,
        )
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (
            (m[0, 2] - m[2, 0]) / s,
            (m[0, 1] + m[1, 0]) / s,
            0.25 * s,
            (m[1, 2] + m[2, 1]) / s,
        )
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (
            (m[1, 0] - m[0, 1]) / s,
            (m[0, 2] + m[2, 0]) / s,
            (m[1, 2] + m[2, 1]) / s,
            0.25 * s,
        )
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def euler_xyz_matrix(euler: np.ndarray) -> np.ndarray:
    """MuJoCo's default `euler` convention: intrinsic x-y-z in radians,
    R = Rx · Ry · Rz (checked against `mujoco` in the tests)."""
    a, b, c = (float(v) for v in euler)
    ca, sa, cb, sb, cc, sc = (
        math.cos(a),
        math.sin(a),
        math.cos(b),
        math.sin(b),
        math.cos(c),
        math.sin(c),
    )
    rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    rz = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    return rx @ ry @ rz


# -- the web .splat ------------------------------------------------------------


def read_web_splat(path: Path) -> Splats:
    """The 32-byte-record web format: float32 position and scale, RGBA
    bytes, rotation bytes as (w, x, y, z) mapped from 0..255 to -1..1."""
    raw = np.fromfile(Path(path), dtype=np.uint8)
    if raw.size % WEB_RECORD_BYTES:
        raise ValueError(
            f"{path}: {raw.size} bytes is not a whole number of "
            f"{WEB_RECORD_BYTES}-byte splat records"
        )
    rows = raw.reshape(-1, WEB_RECORD_BYTES)
    means = rows[:, 0:12].copy().view(np.float32).reshape(-1, 3)
    scales = rows[:, 12:24].copy().view(np.float32).reshape(-1, 3)
    rgba = rows[:, 24:28].astype(np.float32) / 255.0
    quats = (rows[:, 28:32].astype(np.float32) - 128.0) / 128.0
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = quats / np.where(norms > 0, norms, 1.0)
    return Splats(
        means=means,
        quats=quats.astype(np.float32),
        scales=scales,
        opacities=rgba[:, 3].astype(np.float32),
        colors=rgba[:, :3].astype(np.float32),
    )


# -- the 3DGS PLY --------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def write_ply(splats: Splats, path: Path) -> Path:
    """The 3DGS PLY, activations undone at the boundary."""
    n = splats.count
    k = int(splats.sh_rest.shape[1]) if splats.sh_rest.size else 0
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    names += [f"f_rest_{i}" for i in range(3 * k)]
    names += ["opacity", "scale_0", "scale_1", "scale_2", *QUAT_WXYZ]
    table = np.zeros((n, len(names)), dtype=np.float32)
    table[:, 0:3] = splats.means
    table[:, 6:9] = (splats.colors - 0.5) / SH_C0
    if k:
        # f_rest is stored channel-major: all k coefficients of R, then G, then B.
        table[:, 9 : 9 + 3 * k] = splats.sh_rest.transpose(0, 2, 1).reshape(n, 3 * k)
    base = 9 + 3 * k
    table[:, base] = _logit(splats.opacities)
    table[:, base + 1 : base + 4] = np.log(np.maximum(splats.scales, 1e-9))
    table[:, base + 4 : base + 8] = splats.quats
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {n}",
            *[f"property float {name}" for name in names],
            "end_header",
            "",
        ]
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as out:
        out.write(header.encode("ascii"))
        out.write(table.astype("<f4").tobytes())
    return path


def read_ply(path: Path) -> Splats:
    """A 3DGS PLY back as activated arrays; refuses by name a PLY that
    is not one (no `f_dc_0`, no `rot_0`)."""
    path = Path(path)
    with path.open("rb") as src:
        names: list[str] = []
        count = 0
        while True:
            line = src.readline()
            if not line:
                raise ValueError(f"{path}: no end_header")
            text = line.decode("ascii", errors="replace").strip()
            if text.startswith("format") and "binary_little_endian" not in text:
                raise ValueError(f"{path}: only binary little-endian PLY is read")
            if text.startswith("element vertex"):
                count = int(text.split()[-1])
            elif text.startswith("property"):
                parts = text.split()
                if parts[1] != "float":
                    raise ValueError(
                        f"{path}: property {parts[-1]} is {parts[1]}, not float"
                    )
                names.append(parts[-1])
            elif text == "end_header":
                break
        data = np.frombuffer(src.read(count * len(names) * 4), dtype="<f4")
    if data.size != count * len(names):
        raise ValueError(f"{path}: truncated: {data.size} floats for {count} vertices")
    table = data.reshape(count, len(names))
    col = {name: i for i, name in enumerate(names)}
    for needed in ("x", "f_dc_0", "opacity", "scale_0", "rot_0"):
        if needed not in col:
            raise ValueError(f"{path}: not a 3DGS PLY (no {needed})")
    rest = sorted(
        (n for n in names if n.startswith("f_rest_")), key=lambda s: int(s[7:])
    )
    k = len(rest) // 3
    sh_rest = (
        table[:, [col[n] for n in rest]].reshape(count, 3, k).transpose(0, 2, 1)
        if k
        else np.zeros((0, 0, 3), np.float32)
    )
    quats = table[:, [col[n] for n in QUAT_WXYZ]]
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    return Splats(
        means=table[:, [col["x"], col["y"], col["z"]]].astype(np.float32),
        quats=(quats / np.where(norms > 0, norms, 1.0)).astype(np.float32),
        scales=np.exp(
            table[:, [col["scale_0"], col["scale_1"], col["scale_2"]]]
        ).astype(np.float32),
        opacities=_sigmoid(table[:, col["opacity"]]).astype(np.float32),
        colors=np.clip(
            SH_C0 * table[:, [col["f_dc_0"], col["f_dc_1"], col["f_dc_2"]]] + 0.5, 0, 1
        ).astype(np.float32),
        sh_rest=sh_rest.astype(np.float32),
    )


def read_any(path: Path) -> Splats:
    path = Path(path)
    if path.suffix == WEB_SPLAT_SUFFIX:
        return read_web_splat(path)
    if path.suffix == PLY_SUFFIX:
        return read_ply(path)
    raise ValueError(f"{path}: a splat is {PLY_SUFFIX} or {WEB_SPLAT_SUFFIX}")


def describe(splats: Splats) -> dict[str, Any]:
    """The facts a record carries about a splat."""
    lo, hi = splats.means.min(0), splats.means.max(0)
    return {
        "count": splats.count,
        "sh_degree": splats.sh_degree,
        "visible_count": int((splats.opacities >= VISIBLE_OPACITY).sum()),
        "extent_m": [
            [round(float(v), 3) for v in lo],
            [round(float(v), 3) for v in hi],
        ],
        "scale_median_m": round(float(np.median(splats.scales)), 5),
    }
