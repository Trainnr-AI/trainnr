"""The capture chain's tools as this machine has them: found by name,
refused with the install line for the running platform (never another
machine's package manager), their versions read for the record.
"""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path

from rq_pipeline.scenes.record import UNRECORDED

# How each tool is installed, by platform: the refusal names the line for
# the machine it runs on. `{pkg}` is the Linux package manager's install
# line (`linux_installer`), `{python}` the interpreter word of the OS.
INSTALL_HINTS: dict[str, dict[str, str]] = {
    "ffmpeg": {
        "Darwin": "brew install ffmpeg",
        "Linux": "{pkg} ffmpeg",
        "Windows": "winget install Gyan.FFmpeg",
    },
    "colmap": {
        "Darwin": "brew install colmap",
        "Linux": "{pkg} colmap",
        "Windows": "a release from https://github.com/colmap/colmap/releases on PATH",
    },
    "brush": {
        "*": "{python} tools/install-brush.py fetches the release binary for this "
        "machine into a user bin directory; or pass its path",
    },
    "gsplat": {
        "Darwin": "gsplat needs CUDA; on a Mac the chain's trainer is Brush",
        "*": "{python} tools/install-gsplat.py installs gsplat and CUDA's compiler "
        "into the train environment and builds its kernels (an NVIDIA GPU)",
    },
}
INSTALL_HINTS["ffprobe"] = INSTALL_HINTS["ffmpeg"]
# The Linux package managers by the distro family /etc/os-release names.
LINUX_INSTALLERS: dict[str, str] = {
    "debian": "sudo apt install",
    "ubuntu": "sudo apt install",
    "fedora": "sudo dnf install",
    "rhel": "sudo dnf install",
    "arch": "sudo pacman -S",
    "suse": "sudo zypper install",
}
OS_RELEASE = Path("/etc/os-release")


def linux_installer(os_release: Path = OS_RELEASE) -> str:
    """This Linux's package manager's install line, from its os-release
    (ID and ID_LIKE); a distro the table does not know gets a neutral
    line rather than another distro's."""
    try:
        text = os_release.read_text(encoding="utf-8")
    except OSError:
        return "your distribution's package manager: install"
    ids: list[str] = []
    for line in text.splitlines():
        key, _, value = line.partition("=")
        if key in ("ID", "ID_LIKE"):
            ids += value.strip().strip('"').split()
    for name in ids:
        if name in LINUX_INSTALLERS:
            return LINUX_INSTALLERS[name]
    return "your distribution's package manager: install"


def install_hint(tool: str, system: str | None = None) -> str:
    """The install line for `tool` on `system` (this machine's by default)."""
    hints = INSTALL_HINTS[tool]
    system = system or platform.system()
    line = hints.get(system) or hints.get("*") or "see the tool's own site"
    return line.format(
        pkg=linux_installer() if system == "Linux" else "",
        python="python" if system == "Windows" else "python3",
    )


class MissingToolError(FileNotFoundError):
    """A tool the chain needs is not on this machine; named."""


def tool_version(binary: Path, *flag: str) -> str:
    """The first line a binary prints for its version flag; unrecorded
    when it prints nothing usable."""
    try:
        out = subprocess.run(
            [str(binary), *flag],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return UNRECORDED
    text = (out.stdout or out.stderr).strip().splitlines()
    return text[0].strip() if text else UNRECORDED
