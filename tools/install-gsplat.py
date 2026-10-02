#!/usr/bin/env python3
"""Install gsplat, the capture chain's CUDA splat trainer, into the train
environment and build its kernels once.

    python3 tools/install-gsplat.py [--python trainnr/.venv-train/bin/python]

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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lab import bootstrap  # noqa: E402

bootstrap()

from trainnr.paths import train_python  # noqa: E402

TRAINER = "trainnr.scenes.gsplat_train"
# The compiler's wheels for CUDA 13, which pip lays out as one tree
# (`nvidia/cu13`: bin, include, lib, nvvm) the trainer names as CUDA_HOME.
# Every piece at torch's own minor: nvcc, its frontend (nvvm), the runtime
# headers (crt) and the CCCL headers (cub) the kernels include.
COMPILER_WHEELS = {
    "13": ("nvidia-cuda-nvcc", "nvidia-nvvm", "nvidia-cuda-crt", "nvidia-cuda-cccl"),
}
OTHER_CUDA = (
    "torch was built for CUDA {cuda}; this tool knows the pip wheels of CUDA 13 "
    "only (its `nvidia-*` wheels lay CUDA out as one tree). For {cuda}: install "
    "that CUDA toolkit, set CUDA_HOME, and run `{python} -m {trainer} --build`"
)
NO_TORCH = "{python} cannot import torch: build the train environment first"
NO_INSTALLER = (
    "neither `uv` on PATH nor pip in {python}: install uv (astral.sh/uv) or run "
    "`{python} -m ensurepip`"
)
NO_MSVC = (
    "cl.exe is not on PATH: gsplat's kernels need MSVC on Windows; run this from "
    "an x64 Native Tools Command Prompt for Visual Studio"
)


def torch_cuda(python: Path) -> str:
    """The CUDA version torch was built for, or a refusal naming the fix."""
    try:
        out = subprocess.run(
            [str(python), "-c", "import torch; print(torch.version.cuda or '')"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as why:
        raise SystemExit(NO_TORCH.format(python=python)) from why
    version = out.stdout.strip()
    if not version:
        raise SystemExit(f"{python}'s torch is not a CUDA build; gsplat needs one")
    return version


def installer(python: Path) -> list[str]:
    """`uv pip install --python <python>` when uv is here, else that
    interpreter's own pip when it has one; refused by name otherwise (a
    uv-made environment carries no pip)."""
    uv = shutil.which("uv")
    if uv:
        return [uv, "pip", "install", "--python", str(python)]
    has_pip = subprocess.run(
        [str(python), "-m", "pip", "--version"], capture_output=True, check=False
    )
    if has_pip.returncode == 0:
        return [str(python), "-m", "pip", "install"]
    raise SystemExit(NO_INSTALLER.format(python=python))


def pip_install(python: Path, packages: list[str]) -> None:
    argv = [*installer(python), *packages]
    print("$", " ".join(argv), flush=True)
    subprocess.run(argv, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--python", type=Path, default=train_python())
    parser.add_argument("--version", default=GSPLAT_VERSION)
    args = parser.parse_args()
    system = platform.system()
    if system == "Darwin":
        raise SystemExit("gsplat needs CUDA; on a Mac the chain's trainer is Brush")
    if system == "Windows" and shutil.which("cl") is None:
        raise SystemExit(NO_MSVC)
    if not args.python.is_file():
        raise SystemExit(
            f"no interpreter at {args.python}: build the train environment first"
        )
    cuda = torch_cuda(args.python)
    major, minor = cuda.split(".")[:2]
    wheels = COMPILER_WHEELS.get(major)
    if wheels is None:
        raise SystemExit(
            OTHER_CUDA.format(cuda=cuda, python=args.python, trainer=TRAINER)
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
