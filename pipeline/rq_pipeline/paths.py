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
# The file that marks the checkout, relative to its root.
CHECKOUT_MARKER = Path("pipeline") / "pyproject.toml"
_PACKAGE = Path(__file__).resolve().parent


def checkout() -> Path:
    """The repository root: `$TRAINNR_REPO`, else the nearest ancestor of
    this package that carries `pipeline/pyproject.toml`, else the
    package's grandparent."""
    named = os.environ.get(CHECKOUT_ENV, "").strip()
    if named:
        return Path(named).expanduser().resolve()
    for ancestor in _PACKAGE.parents:
        if (ancestor / CHECKOUT_MARKER).is_file():
            return ancestor
    return _PACKAGE.parents[1]


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
TRAIN_VENV = Path("pipeline") / ".venv-train"


def train_python() -> Path:
    """The train environment's interpreter in this checkout."""
    return venv_python(checkout() / TRAIN_VENV)
