"""Where the checkout is — one answer for every module that reaches
outside the package (the robot library, the findings ledger, the
Studio's build, the projects home).

Five modules each derived it as `Path(__file__).resolve().parents[3]`
before 2026-09-12; an installed wheel has no such ancestor, and one
truth is easier to override than five. `$TRAINNR_REPO` names the
checkout when the package does not live inside one (the Studio reads
the same variable); otherwise the ancestor that carries the pipeline's
own `pyproject.toml`; otherwise the package's grandparent, so a
checkout-less install still gets a deterministic answer and the
callers refuse by name when the file they want is not there.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

CHECKOUT_ENV = "TRAINNR_REPO"
HOME_ENV = "TRAINNR_HOME"  # the user's own data: projects, caches
HOME_DIR = "trainnr"  # ~/trainnr
# The file that marks the checkout, relative to its root.
CHECKOUT_MARKER = Path("trainnr") / "pyproject.toml"
_PACKAGE = Path(__file__).resolve().parent


def checkout() -> Path:
    """The repository root: `$TRAINNR_REPO`, else the nearest ancestor of
    this package that carries `trainnr/pyproject.toml`, else the
    package's grandparent."""
    named = os.environ.get(CHECKOUT_ENV, "").strip()
    if named:
        return Path(named).expanduser().resolve()
    for ancestor in _PACKAGE.parents:
        if (ancestor / CHECKOUT_MARKER).is_file():
            return ancestor
    return _PACKAGE.parents[1]


# The WSL launch environment (GL through Mesa's D3D12 driver, CUDA's
# library path, one BLAS thread). It is applied on WSL only, and only to
# variables the user has not set: on a native Linux machine it would point
# the renderer at a driver that is not there.
WSL_ENV_FILE = Path("trainnr") / "wsl.env"
WSL_INTEROP = Path("/proc/sys/fs/binfmt_misc/WSLInterop")


def on_wsl() -> bool:
    """Running under Windows' Subsystem for Linux: the rule the Studio
    uses too (crates/trainnr-studio/src/spawn.rs `on_wsl`)."""
    return bool(os.environ.get("WSL_DISTRO_NAME")) or WSL_INTEROP.exists()


def wsl_env_file() -> Path | None:
    """The WSL launch environment's file, on WSL when the checkout has it."""
    if not on_wsl():
        return None
    path = checkout() / WSL_ENV_FILE
    return path if path.is_file() else None


def user_home() -> Path:
    """Where the user's own data lives (projects, download caches):
    `$TRAINNR_HOME`, else `~/trainnr`. Never the checkout or a plugin's
    install folder, which an update replaces; read-only material (the
    robot library, the tools) still comes from `checkout()`."""
    named = os.environ.get(HOME_ENV, "").strip()
    if named:
        return Path(named).expanduser().absolute()
    return Path.home() / HOME_DIR


def user_cache(name: str, legacy: tuple[str, ...]) -> Path:
    """`<user home>/cache/<name>`, unless a checkout already holds the
    cache at its older place (`<checkout>/<legacy...>`), which is kept so
    nothing downloaded before is fetched again."""
    old = checkout().joinpath(*legacy)
    if old.is_dir():
        return old
    return user_home() / "cache" / name


def venv_bin(venv: Path, name: str) -> Path:
    """An executable inside a virtual environment, by the platform's own
    layout (`bin/` on POSIX, `Scripts/` with `.exe` on Windows) — the one
    resolver every tool that names a venv interpreter goes through."""
    if sys.platform.startswith("win"):
        return Path(venv) / "Scripts" / f"{name}.exe"
    return Path(venv) / "bin" / name


def venv_python(venv: Path) -> Path:
    return venv_bin(venv, "python")


# The pipeline's second environment: the trainer's (LeRobot, torch), beside
# the base one; every tool that drives a student through it names it here.
TRAIN_VENV = Path("trainnr") / ".venv-train"


def train_python() -> Path:
    """The train environment's interpreter in this checkout."""
    return venv_python(checkout() / TRAIN_VENV)
