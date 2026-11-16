"""Where NVIDIA's video-to-motion programs are installed, and whether they
are ready: GEM-X (video -> SOMA human motion) and the SOMA Retargeter
(SOMA motion -> a humanoid's joints). Each lives in its own environment,
installed by `tools/video-setup.sh`; trainnr runs them as external
programs and never imports them (docs/86 §5.2).
"""

from __future__ import annotations

import os
from pathlib import Path

GEMX_ENV = "TRAINNR_GEMX"
SOMA_ENV = "TRAINNR_SOMA_RETARGETER"
SETUP = "tools/video-setup.sh"
# What each install must hold before a job is started on it.
GEMX_NEEDS = (
    ".venv/bin/gem-infer",
    "weights/body_video_native77.yaml",
    "weights/body_s240000.ckpt",
    "weights/vitpose.pth",
    "weights/motion_native77_e774_s292175.ckpt",
    "weights/yolox_humanart.onnx",
    "assets/soma/SOMA_neutral.npz",
)
SOMA_NEEDS = (
    ".venv/bin/python",
    "app/bvh_to_csv_converter.py",
    "soma_retargeter/assets/soma/soma_zero_frame0.bvh",
)


def gemx_dir() -> Path:
    return Path(os.environ.get(GEMX_ENV) or Path.home() / ".cache/trainnr/GEM-X")


def soma_dir() -> Path:
    return Path(
        os.environ.get(SOMA_ENV) or Path.home() / ".cache/trainnr/soma-retargeter"
    )


def missing() -> str | None:
    """Why the video-to-motion programs cannot run yet, or None when ready:
    the first file missing from either install, and the line that installs it."""
    for name, root, needs, env in (
        ("GEM-X", gemx_dir(), GEMX_NEEDS, GEMX_ENV),
        ("the SOMA Retargeter", soma_dir(), SOMA_NEEDS, SOMA_ENV),
    ):
        for need in needs:
            if not (root / need).exists():
                return (
                    f"{name} is not installed at {root} (no {need}): run {SETUP} "
                    f"from the checkout (about 15 GB with its weights), or set ${env}"
                )
    return None
