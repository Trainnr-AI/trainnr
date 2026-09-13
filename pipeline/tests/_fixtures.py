"""Fixtures several suites share: the rig's bundle inside a fresh project."""

from __future__ import annotations

import shutil
from pathlib import Path

from rq_pipeline.project import create_project
from rq_pipeline.project.locate import Project

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
