"""Sample projects: finished projects a new user opens to see the whole
loop before running it on their own robot.

A sample is a curated copy of a real project (`tools/make-sample.py`
builds it, nothing private in it), published on the project's Hugging
Face dataset at a pinned revision. Opening one downloads it (its size and
SHA-256 checked), unpacks it into the projects home as an ordinary
project marked `sample`, indexes it, and makes it current. Opening it
again opens what is there. A fresh-install review (2026-10-09) found a
new user's only way to see a finished project was to do the README's
quickstart; the Go2 walk is that, ready to look at.
"""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# One downloader and one unpacker: the Studio installer's, which the
# security review hardened (HTTPS only across redirects, no links or
# paths out of the target when unpacking).
from trainnr.studio_install import (
    _OPENER,
    TIMEOUT_S,
    _request,
    _safe_extract_tar,
)

DATASET = "trainnr/trainnr-artifacts"
# The dataset commit the samples are read at; a new sample is a new commit
# and a new pin, so a published one never changes under a user.
REVISION = "9f61659a2f912a0298c48e699fbdb08c6acc6203"
MARKER = "sample"  # project.json's flag for a project that came from here


class SampleError(RuntimeError):
    """A sample could not be fetched or opened; the message says why."""


@dataclass(frozen=True)
class Sample:
    name: str
    title: str
    summary: str
    path: str  # in the dataset
    project: str  # the folder it unpacks to, in the projects home
    sha256: str
    bytes: int

    @property
    def url(self) -> str:
        return (
            f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{self.path}"
        )


SAMPLES: dict[str, Sample] = {
    "go2-walk": Sample(
        name="go2-walk",
        title="Go2 walk",
        summary=(
            "A Unitree Go2 through the whole loop: identified from a public log, a "
            "walk declared and accepted, trained by reinforcement, evaluated in "
            "seven conditions, exported and gated against Unitree's own runtime "
            "in simulation, and checked for drift."
        ),
        path="samples/go2-walk-sample.tar.gz",
        project="go2-walk-sample",
        sha256="b4dad73faccde07ce14bdc23bc85af769587fdb0cdda157cc9e4414cedb7a804",
        bytes=11_973_917,
    ),
}


def resolve(name: str) -> Sample:
    if name not in SAMPLES:
        raise KeyError(f"no sample {name!r}; the samples are {sorted(SAMPLES)}")
    return SAMPLES[name]


def installed(sample: Sample, home: Path) -> Path | None:
    """The sample's folder in the projects home, when it is there."""
    root = home / sample.project
    return root if (root / "project.json").is_file() else None


def _download(sample: Sample, out: Path) -> None:
    digest = hashlib.sha256()
    size = 0
    try:
        with (
            _OPENER.open(_request(sample.url), timeout=TIMEOUT_S) as reply,
            out.open("wb") as sink,
        ):
            while chunk := reply.read(1 << 20):
                size += len(chunk)
                if size > sample.bytes:
                    raise SampleError(
                        f"{sample.url}: larger than the {sample.bytes} bytes listed"
                    )
                digest.update(chunk)
                sink.write(chunk)
    except OSError as why:
        raise SampleError(f"download failed: {sample.url} ({why})") from why
    if size != sample.bytes or digest.hexdigest() != sample.sha256:
        raise SampleError(
            f"{sample.url}: {size} bytes, sha256 {digest.hexdigest()[:12]}; "
            f"listed {sample.bytes} bytes, sha256 {sample.sha256[:12]}"
        )


def install(name: str, home: Path) -> Path:
    """The sample in the projects home: downloaded, checked and unpacked
    the first time, the folder that is there after that."""
    sample = resolve(name)
    found = installed(sample, home)
    if found is not None:
        return found
    target = home / sample.project
    if target.exists():
        raise SampleError(
            f"{target} exists and is not the sample; move it to open the sample"
        )
    home.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=home) as tmp:
        archive = Path(tmp) / "sample.tar.gz"
        _download(sample, archive)
        unpacked = Path(tmp) / "unpacked"
        unpacked.mkdir()
        with tarfile.open(archive, "r:gz") as tar:
            _safe_extract_tar(tar, unpacked)
        inside = unpacked / sample.project
        if not (inside / "project.json").is_file():
            raise SampleError(f"{sample.url}: no {sample.project}/project.json inside")
        shutil.move(str(inside), target)
    return target


def open_sample(name: str, home: Path | None = None) -> Path:
    """Install the sample (`install`), make it the current project and
    index it: what `open_sample`, `trainnr sample open` and the Studio's
    Open button do. The Studio on its Welcome page follows `.current`
    to it."""
    from trainnr.project import index_project, write_index  # noqa: PLC0415
    from trainnr.project import use_project as choose  # noqa: PLC0415
    from trainnr.project.locate import projects_home  # noqa: PLC0415

    root = install(name, home or projects_home())
    project = choose(str(root))
    write_index(project, index_project(project))
    return project.root


def describe(home: Path) -> list[dict[str, Any]]:
    """Every sample, with where it is installed (None until opened)."""
    rows = []
    for sample in SAMPLES.values():
        found = installed(sample, home)
        rows.append(
            {
                "name": sample.name,
                "title": sample.title,
                "summary": sample.summary,
                "download_mb": round(sample.bytes / 1e6, 1),
                "installed": str(found) if found else None,
            }
        )
    return rows
