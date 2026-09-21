"""Wavefront OBJ, the exchange format every mesh tool in the chain
speaks (CoACD, Open3D, Neverwhere's collision meshes): read triangles
in, write triangles out, numpy only. One home for the parser the viewer,
the proxy audit and the stage all need.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

VERTEX = "v "
FACE = "f "


def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (n, 3) float64 and triangle indices (m, 3) int64, quads
    and larger faces fanned into triangles; refuses a file with no faces."""
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as src:
        for line in src:
            if line.startswith(VERTEX):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith(FACE):
                idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
                for i in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])
    if not faces:
        raise ValueError(f"{path}: no faces")
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> Path:
    """Triangles to an OBJ (1-based faces), returns the path written."""
    path = Path(path)
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in np.asarray(vertices)]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in np.asarray(faces)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
