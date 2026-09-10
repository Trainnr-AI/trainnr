"""A live run in the Studio: its console log becomes its training record
as it grows, so the experiment card shows the reward curve so far.

A run folder that carries its own `train.log` (rq_mjlab's walk_train
tees its console there when it archives a run) gets a `training.json`
parsed from it whenever the log is newer than the record; the record
says whether the run is still going (no "done" line yet). The importer
does the same once for an imported study arm; this does it on every
presenter tick for whatever trains inside the project.
"""

from __future__ import annotations

import json
from pathlib import Path

from rq_pipeline.envs.rsl_rl_log import parse_rsl_rl_log
from rq_pipeline.project.locate import Project

TRAIN_LOG = "train.log"
TRAINING_FILE = "training.json"
DONE_MARK = "] done"  # walk_train's last line: "[g3] done - checkpoints in ..."
STATUS_RUNNING, STATUS_DONE = "running", "done"


def run_folders(project: Project) -> list[Path]:
    """Every run folder with its own console log, newest first."""
    runs = project.runs
    if not runs.is_dir():
        return []
    found = [p.parent for p in runs.rglob(TRAIN_LOG)]
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def stale(folder: Path) -> bool:
    """Whether the log has grown since the record was written."""
    log, record = folder / TRAIN_LOG, folder / TRAINING_FILE
    if not record.is_file():
        return True
    return log.stat().st_mtime > record.stat().st_mtime


def refresh_training(folder: Path) -> dict | None:
    """Parse the folder's console log into `training.json`; returns the
    record written, or None when the log holds no iteration yet."""
    text = (folder / TRAIN_LOG).read_text(errors="replace")
    parsed = parse_rsl_rl_log(text)
    if parsed is None:
        return None
    record = parsed.to_json()
    record["status"] = STATUS_DONE if DONE_MARK in text[-2000:] else STATUS_RUNNING
    (folder / TRAINING_FILE).write_text(json.dumps(record, indent=1) + "\n")
    return record


def refresh_project(project: Project) -> list[Path]:
    """Refresh every stale run in the project; the folders refreshed."""
    return [f for f in run_folders(project) if stale(f) and refresh_training(f)]
