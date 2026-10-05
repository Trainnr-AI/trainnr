"""Moving a repo and its runs to and from a rented machine: the ssh and
rsync command lines, built once, tested.

The rules are subtle enough to deserve a test: rsync reads include
patterns before the final `--exclude=*`, a directory needs both
`name/` and `name/**` to travel, `-a` is avoided because the machine's
volume refuses `chown` (exit 23, measured 2026-08-27), and what must
never leave this machine (`.env`) is named here and nowhere else. Standard
library only.
"""

from __future__ import annotations

import shlex
from collections.abc import Sequence
from pathlib import Path

from trainnr.cloud.provider import SshEndpoint


class Rsync:
    """What `push` ships, what `follow` mirrors, what `pull` brings back."""

    # -a minus owner/group: the machine is root's, and its volume refuses chown.
    ARCHIVE = "-rlptDz"
    PROGRESS = "--info=progress2"
    EXCLUDES = (
        ".env",  # the credentials — never
        ".git",
        ".venv*",  # every venv, every package (trainnr_mjlab's alone is gigabytes)
        "runs",  # datasets and checkpoints live on the machine's volume, not the push
        "target",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".coverage",
    )
    # What `follow` mirrors while a run is going — never the weights;
    # `pull` brings those when the run is done.
    FOLLOW_EXCLUDES = ("model.safetensors", "training_state", "*.pt", "*.pth")
    RUNS_SUBDIR = "trainnr/runs"


class Ssh:
    OPTIONS = (
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ConnectTimeout=15",
    )


def ssh_argv(door: SshEndpoint, key: Path | None = None) -> list[str]:
    """`ssh` to the machine's own sshd; `-i` only when a key is given
    (else `~/.ssh/config` and the agent decide)."""
    identity = ["-i", str(key)] if key else []
    return [
        "ssh",
        *identity,
        "-p",
        str(door.port),
        *Ssh.OPTIONS,
        f"{door.username}@{door.host}",
    ]


def remote_path(door: SshEndpoint, path: str) -> str:
    return f"{door.username}@{door.host}:{path}"


def rsync_argv(  # noqa: PLR0913 - every part of one command line, named
    door: SshEndpoint,
    key: Path | None,
    source: str,
    target: str,
    *,
    filters: Sequence[str],
    progress: bool = True,
) -> list[str]:
    transport = " ".join(shlex.quote(part) for part in ssh_argv(door, key)[:-1])
    argv = ["rsync", Rsync.ARCHIVE]
    if progress:
        argv.append(Rsync.PROGRESS)
    return [*argv, "-e", transport, *filters, source, target]


def push_filters() -> list[str]:
    return [f"--exclude={pattern}" for pattern in Rsync.EXCLUDES]


def pull_filters(name: str) -> list[str]:
    """Only this run's directories: rsync reads includes before the final
    exclude, and a directory needs itself and its contents."""
    return [f"--include={name}-*/", f"--include={name}-*/**", "--exclude=*"]


def follow_filters(name: str) -> list[str]:
    """This run's directories minus the weights."""
    return [
        *(f"--exclude={pattern}" for pattern in Rsync.FOLLOW_EXCLUDES),
        *pull_filters(name),
    ]
