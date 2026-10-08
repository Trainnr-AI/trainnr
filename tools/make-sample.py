#!/usr/bin/env python3
"""Build a sample project's archive from a real project: the files a
sample shows, nothing private, deterministic, with its SHA-256.

    python3 tools/make-sample.py go2-walk --source <project> --out dist/samples

A sample is a finished project a new user opens to see the whole loop
(`open_sample`, the Studio's Projects page). Only the paths SAMPLES
names are copied, less the parts each excludes (logs, viewer streams,
checkpoints, a run's git state, a deployment's staged runtime); every
text file is then refused if it still holds a private marker (a home
path, the machine's name); the curves file is renamed to drop the host
name tensorboard writes into it. The archive is a tar.gz with sorted
members and zero ids; each file keeps its source's time (the index dates
artifacts by them; zero read as 1970) and a folder takes its newest
file's, so the same source gives the same bytes.
Stdlib only.
"""

from __future__ import annotations

import argparse
import fnmatch
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# Never shipped, whatever a sample includes: a run's commit state, logs
# (they carry the machine's paths), viewer streams (hundreds of MB),
# Python caches.
ALWAYS_EXCLUDED = ("git", "*.log", ".viewer", "__pycache__")
# A file still holding one of these is refused, never scrubbed silently.
PRIVATE = re.compile(rb"/home/|/Users/|prakhar|robotiq|qolaba", re.IGNORECASE)
TEXT_SUFFIXES = (".json", ".jsonl", ".xml", ".yaml", ".yml", ".txt", ".md", ".csv")
# tensorboard names its file events.out.tfevents.<time>.<host>.<pid>.<n>.
TFEVENTS = re.compile(r"^(events\.out\.tfevents\.\d+)\..*$")
SAMPLE_ROOT_NAME = "{name}"


@dataclass(frozen=True)
class Part:
    """A folder of the source project, and what of it stays behind."""

    path: str
    exclude: tuple[str, ...] = ()


@dataclass(frozen=True)
class Sample:
    name: str
    title: str
    description: str
    parts: tuple[Part, ...]
    notice: str
    licences: dict[str, str] = field(default_factory=dict)  # file in sample -> URL


GO2_CERTIFICATES = (
    "go2-c2-model_1499-cuda-seed1000-n40-p37d3df",
    "go2-c2-model_1499-cuda-seed1000-n40-pa29326",
    "go2-c2-model_1499-at-x0.8-kp-cuda-seed1000-n40-pe170b0",
    "go2-c2-model_1499-delay-2-cuda-seed1000-n40-p5a0239",
    "go2-c2-model_1499-in-fit-0ad6202797c5-cuda-seed1000-n40-p57544d",
    "go2-c2-model_1499-in-fit-0ad6202797c5-at-x0.8-kp-cuda-seed1000-n40-p27b287",
    "go2-c2-model_1499-in-fit-0ad6202797c5-delay-2-cuda-seed1000-n40-p031141",
)

SAMPLES: dict[str, Sample] = {
    "go2-walk": Sample(
        name="go2-walk-sample",
        title="Go2 walk",
        description=(
            "A Unitree Go2 through the whole loop: identified from a public log, "
            "a walk declared and accepted, trained by reinforcement, evaluated in "
            "seven conditions with 40 trials each, exported and gated against "
            "Unitree's own runtime in simulation, and checked for drift."
        ),
        parts=(
            Part("robots/go2"),
            Part("recordings/iit-go2-chirp"),
            Part("tasks/go2-flat"),
            Part("runs/go2-c2", exclude=("model_*.pt", "verdict")),
            Part("policies/go2-c2-model_1499"),
            *(Part(f"certificates/{c}") for c in GO2_CERTIFICATES),
            Part("deploy/go2-c2-deploy-cited", exclude=("unitree",)),
            Part("monitoring/go2-leg-odometry-check"),
        ),
        notice=(
            "This sample project is trainnr's own results (the fit, the walk, the run, "
            "the policy, the evaluations, the deployment and the drift check), "
            "licensed as trainnr is (FSL-1.1-ALv2), with two third-party parts under "
            "their own "
            "licences:\n\n"
            "- robots/go2: Unitree's Go2 model from unitreerobotics/unitree_rl_mjlab, "
            "Apache License 2.0 (LICENCE-unitree_rl_mjlab beside this file).\n"
            "- recordings/iit-go2-chirp: a Go2 chirp trajectory from "
            "iit-DLSLab/sim2real-robot-identification (datasets/go2/traj_0.pt), "
            "BSD 3-Clause License, Copyright (c) 2025, DLS Lab at Istituto Italiano "
            "di Tecnologia (LICENSE-sim2real-robot-identification beside this "
            "file).\n\n"
            "The drift check was computed from a public Go2 log "
            "(github.com/YibinWu/leg-odometry) that states no licence of its own; "
            "the log itself is not included, only the check's result."
        ),
        licences={
            "LICENCE-unitree_rl_mjlab": (
                "https://raw.githubusercontent.com/unitreerobotics/unitree_rl_mjlab/"
                "main/LICENCE"
            ),
            "LICENSE-sim2real-robot-identification": (
                "https://raw.githubusercontent.com/iit-DLSLab/"
                "sim2real-robot-identification/"
                "b73d6a2c4988dc9b3453369a2cdd29a307cb418e/LICENSE"
            ),
        },
    ),
}


def _excluded(relative: Path, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(part, p) for part in relative.parts for p in patterns)


def copy_parts(sample: Sample, source: Path, root: Path) -> list[Path]:
    """The listed parts into `root`, less the exclusions; the copied files."""
    copied: list[Path] = []
    for part in sample.parts:
        origin = source / part.path
        if not origin.is_dir():
            raise SystemExit(f"{origin}: not in the source project")
        patterns = ALWAYS_EXCLUDED + part.exclude
        for path in sorted(origin.rglob("*")):
            if path.is_symlink():
                raise SystemExit(f"{path}: a link; a sample carries files only")
            relative = path.relative_to(origin)
            if not path.is_file() or _excluded(relative, patterns):
                continue
            name = TFEVENTS.sub(r"\1.trainnr", path.name)
            target = root / part.path / relative.parent / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)  # its time too: the index dates by it
            copied.append(target)
    return copied


def refuse_private(files: list[Path], root: Path) -> None:
    """Every text file, and every name, free of private markers."""
    found = []
    for path in files:
        relative = path.relative_to(root)
        if PRIVATE.search(str(relative).encode()):
            found.append(f"{relative}: in its name")
        elif path.suffix in TEXT_SUFFIXES and PRIVATE.search(path.read_bytes()):
            found.append(f"{relative}: {PRIVATE.search(path.read_bytes()).group(0)!r}")
    if found:
        raise SystemExit("private markers, nothing written:\n  " + "\n  ".join(found))


def write_manifest(sample: Sample, source: Path, root: Path) -> None:
    project = json.loads((source / "project.json").read_text(encoding="utf-8"))
    manifest = {
        "schema": project["schema"],
        "name": sample.name,
        "created": project.get("created"),
        "description": sample.description,
        "library": {},
        "loop": project.get("loop", ""),
        "sample": True,
    }
    (root / "project.json").write_text(json.dumps(manifest, indent=1) + "\n")
    (root / "NOTICE").write_text(sample.notice + "\n")
    (root / "README.md").write_text(
        f"# {sample.title} (a trainnr sample)\n\n{sample.description}\n\n"
        "Open it with `open_sample` (or the Studio's Projects page); it is a "
        "project like any other, so your agent can evaluate, export or train "
        "from it. See NOTICE for what is whose.\n\n"
        "What a sample leaves out of the project it was made from: logs, "
        "viewer streams, all checkpoints but the policy's, a run's commit "
        "state, and a deployment's staged runtime; a folder that lost files "
        "has a new version (the deployment's).\n"
    )


def fetch_licences(sample: Sample, root: Path) -> None:
    import urllib.request  # noqa: PLC0415

    for name, url in sample.licences.items():
        with urllib.request.urlopen(url, timeout=30) as response:
            (root / name).write_bytes(response.read())


def _newest(path: Path) -> float:
    """A file's own time; a folder's newest file's."""
    if path.is_file():
        return path.stat().st_mtime
    times = [p.stat().st_mtime for p in path.rglob("*") if p.is_file()]
    return max(times, default=0.0)


def deterministic_tar(root: Path, out: Path) -> None:
    """tar.gz of `root` under its own name: sorted, zero ids, source times."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for path in sorted(root.rglob("*")):
            info = tar.gettarinfo(
                str(path), arcname=f"{root.name}/{path.relative_to(root)}"
            )
            info.mtime = int(_newest(path))
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o755 if path.is_dir() else 0o644
            if path.is_file():
                with path.open("rb") as handle:
                    tar.addfile(info, handle)
            else:
                tar.addfile(info)
    with (
        out.open("wb") as raw,
        gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9
        ) as gz,
    ):
        gz.write(buffer.getvalue())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sample", choices=sorted(SAMPLES))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    sample = SAMPLES[args.sample]
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / sample.name
        copied = copy_parts(sample, args.source.expanduser(), root)
        refuse_private(copied, root)
        write_manifest(sample, args.source.expanduser(), root)
        fetch_licences(sample, root)
        # The files written here take the newest copied file's time, so a
        # rebuild of the same source gives the same bytes.
        newest = max(p.stat().st_mtime for p in copied)
        for written in (p for p in root.iterdir() if p.is_file()):
            os.utime(written, (newest, newest))
        out = args.out / f"{sample.name}.tar.gz"
        deterministic_tar(root, out)
        unpacked = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(
        f"{out}  {out.stat().st_size} bytes  ({unpacked} unpacked, {len(copied)} files)"
    )
    print(f"sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
