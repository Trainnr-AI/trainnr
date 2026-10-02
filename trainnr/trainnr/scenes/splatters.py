"""The splat trainers the capture chain can run, one registry: Brush
(a release binary, every OS, its own renderer) and gsplat (CUDA, in the
train environment, `scenes.gsplat_train`). Both read the same COLMAP
dataset folder and leave the same file - a 3DGS PLY in COLMAP's frame -
so everything after the splat is one code path. `auto` takes gsplat
where it is installed and CUDA answers, Brush otherwise; the record
names the one that ran, with its version and licence.

Why two: Brush is the trainer any machine can install; under WSL it
reaches the GPU through a translation layer that costs fifteen cores
for one iteration a second (2026-09-23), where gsplat drives the same
GPU directly.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from trainnr.paths import train_python
from trainnr.scenes.record import UNRECORDED
from trainnr.scenes.tooling import MissingToolError, install_hint, tool_version

SPLAT_EXPORT = "splat-colmap-frame.ply"  # the trainer's output, before alignment
RENDERS_DIR = "renders"  # a trainer's own renders of its splat, beside the export
MAX_RESOLUTION = 1920  # the longest image side a trainer sees
DEFAULT_STEPS = 30_000
AUTO = "auto"
TRAINER_MODULE = "trainnr.scenes.gsplat_train"
GSPLAT_PROBE = (
    "import importlib.util as u, sys; sys.exit(0 if u.find_spec('gsplat') else 3)"
)
CUDA_PROBE = "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 4)"
PROBE_TIMEOUT_S = 60


@dataclass(frozen=True)
class Splatter:
    """One trainer: how it is found, how it is run, what it is."""

    name: str
    title: str  # the record's name for it
    license: str
    missing: str  # what a refusal names: the binary, or the environment
    takes_binary: bool  # whether a caller's path (`--brush`) means this trainer
    # `platform.system()` names it runs on; None for any.
    platforms: tuple[str, ...] | None
    argv: Callable[..., list[str | Path]]  # (tool, dataset, out, steps, narrate)
    version: Callable[[Path], str]
    folder: str  # the work folder under the scene, the trainer's name

    def find(self, binary: Path | None = None) -> Path:
        """Where it runs from, or a refusal naming this machine's install line."""
        system = platform.system()
        if self.platforms is not None and system not in self.platforms:
            raise MissingToolError(
                f"{self.name} does not run on {system}: {install_hint(self.name)}"
            )
        tool = LOCATORS[self.name](binary if self.takes_binary else None)
        if tool is None:
            raise MissingToolError(f"{self.missing} ({install_hint(self.name)})")
        return tool


# -- Brush --------------------------------------------------------------------

BRUSH_BINARY = "brush_app"
BRUSH_LICENSE = "Apache-2.0 OR MIT"  # ArthurBrussee/brush: dual, the user's choice


def _locate_brush(binary: Path | None) -> Path | None:
    if binary is not None:
        return binary if binary.is_file() else None
    found = shutil.which(BRUSH_BINARY)
    return Path(found) if found else None


def _brush_argv(
    tool: Path, dataset: Path, out: Path, *, steps: int, narrate: bool
) -> list[str | Path]:
    """Brush on the COLMAP dataset, headless, exporting once at the end;
    `--rerun-enabled` streams its training into the Studio (law 0)."""
    argv: list[str | Path] = [
        tool,
        dataset,
        "--total-steps",
        str(steps),
        "--export-every",
        str(steps),
        "--export-path",
        out,
        "--export-name",
        SPLAT_EXPORT,
        "--max-resolution",
        str(MAX_RESOLUTION),
        "--eval-every",
        str(steps + 1),  # no held-out split: every frame trains
    ]
    if narrate:
        argv.append("--rerun-enabled")
    return argv


# -- gsplat -------------------------------------------------------------------

GSPLAT_LICENSE = "Apache-2.0"  # nerfstudio-project/gsplat


def _probe(python: Path, code: str) -> bool:
    try:
        return (
            subprocess.run(
                [str(python), "-c", code],
                capture_output=True,
                timeout=PROBE_TIMEOUT_S,
                check=False,
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


def _locate_gsplat(binary: Path | None) -> Path | None:
    """The train environment's interpreter, when gsplat is installed
    there and a CUDA device answers; `binary` names another interpreter."""
    python = binary or train_python()
    if not python.is_file():
        return None
    if not _probe(python, GSPLAT_PROBE) or not _probe(python, CUDA_PROBE):
        return None
    return python


def _gsplat_argv(
    tool: Path, dataset: Path, out: Path, *, steps: int, narrate: bool
) -> list[str | Path]:
    argv: list[str | Path] = [
        tool,
        "-m",
        TRAINER_MODULE,
        dataset,
        "--out",
        out,
        "--steps",
        str(steps),
        "--export-name",
        SPLAT_EXPORT,
        "--max-resolution",
        str(MAX_RESOLUTION),
    ]
    if narrate:
        argv.append("--rerun")
    return argv


def _gsplat_version(python: Path) -> str:
    line = tool_version(python, "-c", "import gsplat; print(gsplat.__version__)")
    return line if line and line[0].isdigit() else UNRECORDED


# Where each trainer runs from: a binary by name on PATH (or the one
# passed), or the train environment's interpreter; None when not here.
LOCATORS: dict[str, Callable[[Path | None], Path | None]] = {
    "brush": _locate_brush,
    "gsplat": _locate_gsplat,
}
SPLATTERS: dict[str, Splatter] = {
    "brush": Splatter(
        name="brush",
        title="Brush",
        license=BRUSH_LICENSE,
        missing=BRUSH_BINARY,
        takes_binary=True,
        platforms=None,
        argv=_brush_argv,
        version=lambda tool: tool_version(tool, "--version"),
        folder="brush",
    ),
    "gsplat": Splatter(
        name="gsplat",
        title="gsplat",
        license=GSPLAT_LICENSE,
        missing="gsplat in the train environment",
        takes_binary=False,
        platforms=("Linux", "Windows"),
        argv=_gsplat_argv,
        version=_gsplat_version,
        folder="gsplat",
    ),
}
# `auto`: the first of these that is here runs.
AUTO_ORDER: Sequence[str] = ("gsplat", "brush")
DEFAULT_SPLATTER = AUTO


def choose_splatter(
    name: str = AUTO, binary: Path | None = None
) -> tuple[Splatter, Path]:
    """The trainer by name and where it runs from; `auto` takes the first
    of `AUTO_ORDER` this machine has. Refuses by name, with every install
    line, when none is here."""
    if name != AUTO:
        try:
            spec = SPLATTERS[name]
        except KeyError as unknown:
            raise MissingToolError(
                f"no splat trainer {name!r}; one of {', '.join(SPLATTERS)} or {AUTO}"
            ) from unknown
        return spec, spec.find(binary)
    reasons = []
    for candidate in AUTO_ORDER:
        spec = SPLATTERS[candidate]
        try:
            return spec, spec.find(binary)
        except MissingToolError as why:
            reasons.append(str(why))
    raise MissingToolError("no splat trainer here: " + "; ".join(reasons))
