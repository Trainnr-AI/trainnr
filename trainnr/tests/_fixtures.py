"""Fixtures several suites share: the rig's bundle inside a fresh project,
and where the Go2's MJCF lives for the legged-fit tests."""

from __future__ import annotations

import os
import shutil
import unittest
from pathlib import Path

from trainnr.project import create_project
from trainnr.project.locate import Project

REPO = Path(__file__).resolve().parents[2]
RIG_BUNDLE = REPO / "robots" / "rig-drivetrain"
RIG_FILES = ("model.xml", "profile.json", "README.md")


def rig_project(tmp: Path, name: str = "p") -> Project:
    """A project holding the committed rig bundle's model and profile."""
    project = create_project(tmp / name, name)
    bundle = project.folder("robots") / RIG_BUNDLE.name
    bundle.mkdir(parents=True)
    for file in RIG_FILES:
        shutil.copy2(RIG_BUNDLE / file, bundle / file)
    return project


# The Go2's MJCF for the legged-fit tests: not a tracked file (the bundle
# lives in a project or in the walk package's cache), so the tests find
# it where the walk does and skip by name where it is absent.
GO2_ENV = "TRAINNR_GO2_MJCF"
GO2_XML = "go2.xml"
GO2_CACHE_RELATIVE = (
    Path("unitree_rl_mjlab/src/assets/robots/unitree_go2/xmls") / GO2_XML
)


def go2_model() -> Path | None:
    """The first Go2 MJCF found: the env var, the project's bundle, the
    library's, then the walk package's cache."""
    from trainnr.bundles.locate import find_bundle  # noqa: PLC0415

    named = os.environ.get(GO2_ENV, "").strip()
    if named:
        return Path(named)
    bundle = find_bundle("go2")
    if bundle is not None and (bundle / GO2_XML).is_file():
        return bundle / GO2_XML
    cached = Path.home() / ".cache" / "trainnr" / GO2_CACHE_RELATIVE
    return cached if cached.is_file() else None


def go2_model_or_skip() -> Path:
    found = go2_model()
    if found is None:
        raise unittest.SkipTest(
            f"no Go2 MJCF: set {GO2_ENV}, onboard the Go2 into a project, or run "
            "the walk once so its cache holds unitree_rl_mjlab"
        )
    return found
