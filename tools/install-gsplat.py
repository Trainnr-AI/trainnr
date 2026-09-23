#!/usr/bin/env python3
"""Install gsplat, the capture chain's CUDA splat trainer, into the train
environment and build its kernels once.

    python3 tools/install-gsplat.py [--python pipeline/.venv-train/bin/python]

gsplat (nerfstudio-project/gsplat, Apache-2.0) ships no kernels for this
torch; it compiles them with nvcc at first use. pip carries CUDA's
compiler and headers as wheels, so no system CUDA toolkit is needed:
this installs gsplat, ninja and CUDA 13's compiler wheels, every one
pinned to the minor torch was built for (an nvvm newer than nvcc emitted
PTX the assembler refused; runtime headers newer than nvcc lacked its
launch stub, 2026-09-23), then runs the trainer's own `--build`. Linux and
Windows (MSVC on PATH); on a Mac the chain's trainer is Brush.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

GSPLAT_VERSION = "1.5.3"
REPO = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = (
    REPO
    / "pipeline"
    / ".venv-train"
    / ("Scripts/python.exe" if sys.platform.startswith("win") else "bin/python")
)
TRAINER = "rq_pipeline.scenes.gsplat_train"
# The compiler's wheels for CUDA 13, which pip lays out as one tree
# (`nvidia/cu13`: bin, include, lib, nvvm) the trainer names as CUDA_HOME.
# Every piece at torch's own minor: nvcc, its frontend (nvvm), the runtime
# headers (crt) and the CCCL headers (cub) the kernels include.
COMPILER_WHEELS = {
    "13": ("nvidia-cuda-nvcc", "nvidia-nvvm", "nvidia-cuda-crt", "nvidia-cuda-cccl"),
}
SPLIT_LAYOUT = (
    "torch was built for CUDA {cuda}, whose pip wheels lay CUDA out in pieces; "
    "install a system CUDA toolkit of that version, set CUDA_HOME, and run "
    "`{python} -m {trainer} --build`"
)


def torch_cuda(python: Path) -> str:
    out = subprocess.run(
        [str(python), "-c", "import torch; print(torch.version.cuda or '')"],
        capture_output=True,
        text=True,
        check=True,
    )
    version = out.stdout.strip()
    if not version:
        raise SystemExit(f"{python}'s torch is not a CUDA build; gsplat needs one")
    return version


def pip_install(python: Path, packages: list[str]) -> None:
    uv = shutil.which("uv")
    argv = (
        [uv, "pip", "install", "--python", str(python), *packages]
        if uv
        else [str(python), "-m", "pip", "install", *packages]
    )
    print("$", " ".join(argv), flush=True)
    subprocess.run(argv, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--version", default=GSPLAT_VERSION)
    args = parser.parse_args()
    if platform.system() == "Darwin":
        raise SystemExit("gsplat needs CUDA; on a Mac the chain's trainer is Brush")
    if not args.python.is_file():
        raise SystemExit(
            f"no interpreter at {args.python}: build the train environment first"
        )
    cuda = torch_cuda(args.python)
    major, minor = cuda.split(".")[:2]
    wheels = COMPILER_WHEELS.get(major)
    if wheels is None:
        raise SystemExit(
            SPLIT_LAYOUT.format(cuda=cuda, python=args.python, trainer=TRAINER)
        )
    pip_install(
        args.python,
        [
            f"gsplat=={args.version}",
            "ninja",
            *(f"{w}=={major}.{minor}.*" for w in wheels),
        ],
    )
    print("[gsplat] building the kernels (minutes, once)", flush=True)
    subprocess.run([str(args.python), "-m", TRAINER, "--build"], check=True)
    print(f"[gsplat] ready in {args.python}", flush=True)


if __name__ == "__main__":
    main()
