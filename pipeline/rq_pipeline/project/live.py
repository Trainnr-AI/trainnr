"""A live run in the Studio: its console log becomes its training record
as it grows, so the experiment card shows the reward curve so far.

A run folder that carries its own `train.log` (rq_mjlab's walk_train
tees its console there when it archives a run) gets a `training.json`
parsed from it whenever the log is newer than the record; the record
says whether the run is still going (no "done" line yet). The importer
does the same once for an imported study arm; this does it on every
presenter tick for whatever trains inside the project.

A verdict judged inside a run folder (`certify_walk` on a checkpoint of
a run training in the project writes `verdict/walk-verdict-*.json`
beside the weights) becomes, the same tick, a policy (the checkpoint it
judged) and an evaluation citing it — the Evaluations page stayed at
zero after a 40/40 certificate until this existed (2026-09-10).
"""

from __future__ import annotations

import json
from pathlib import Path

from rq_pipeline.envs.rsl_rl_log import parse_rsl_rl_log
from rq_pipeline.envs.tfevents import EVENTS_GLOB, events_file, record_from_events
from rq_pipeline.project.importer import (
    VERDICT_GLOB,
    evaluation_suffix,
    write_certificate,
    write_policy,
)
from rq_pipeline.project.kinds import IDENTITY_FILE, stamp_run
from rq_pipeline.project.locate import Project

TRAIN_LOG = "train.log"
VERDICT_DIR = "verdict"  # walk_verdict writes beside the checkpoint it judged
TRAINING_FILE = "training.json"
DONE_MARK = "] done"  # walk_train's last line: "[g3] done - checkpoints in ..."
STATUS_RUNNING, STATUS_DONE = "running", "done"


def run_folders(project: Project) -> list[Path]:
    """Every run folder with its own console log or event file, newest first."""
    runs = project.runs
    if not runs.is_dir():
        return []
    found = {p.parent for p in runs.rglob(TRAIN_LOG)}
    found |= {p.parent for p in runs.rglob(EVENTS_GLOB)}
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def _sources(folder: Path) -> list[Path]:
    """What the record is read from: the trainer's event file when it
    exists, the console log when it exists."""
    events = events_file(folder)
    log = folder / TRAIN_LOG
    return [p for p in (events, log if log.is_file() else None) if p is not None]


def stale(folder: Path) -> bool:
    """Whether a source has grown since the record was written."""
    record = folder / TRAINING_FILE
    if not record.is_file():
        return True
    written = record.stat().st_mtime
    return any(p.stat().st_mtime > written for p in _sources(folder))


def refresh_training(folder: Path) -> dict | None:
    """The folder's `training.json` from the trainer's own event file
    (every series it logs) with the console log's facts, or from the
    console log alone; returns the record written, or None when neither
    holds an iteration yet."""
    log = folder / TRAIN_LOG
    text = log.read_text(errors="replace") if log.is_file() else ""
    parsed = parse_rsl_rl_log(text) if text else None
    events = events_file(folder)
    record = record_from_events(events, facts=parsed) if events else None
    record = record or parsed
    if record is None:
        return None
    out = record.to_json()
    out["status"] = STATUS_DONE if DONE_MARK in text[-2000:] else STATUS_RUNNING
    (folder / TRAINING_FILE).write_text(json.dumps(out, indent=1) + "\n")
    return out


def refresh_project(project: Project) -> list[Path]:
    """Refresh every stale run in the project; the folders refreshed."""
    return [f for f in run_folders(project) if stale(f) and refresh_training(f)]


def verdict_files(project: Project) -> list[Path]:
    """Every walk verdict judged inside a run folder of the project."""
    if not project.runs.is_dir():
        return []
    return sorted(project.runs.rglob(f"{VERDICT_DIR}/{VERDICT_GLOB}"))


def certificate_label(verdict_file: Path, verdict: dict) -> str:
    """The evaluation's name: the run and the checkpoint it judged
    (`go2-c1-model_1400`); the writer appends what tells one judgment
    of that checkpoint from another (instrument, seed, trials)."""
    return f"{verdict_file.parent.parent.name}-{verdict.get('policy', '')}"


def refresh_verdicts(project: Project) -> list[Path]:
    """Every verdict not yet in the project becomes a policy and an
    evaluation; the evaluation folders written. Idempotent: a verdict
    already imported costs one existence check, so the run folder's
    checkpoints are hashed only when something new landed."""
    written: list[Path] = []
    for verdict_file in verdict_files(project):
        run_dir = verdict_file.parent.parent
        verdict = json.loads(verdict_file.read_text())
        label = certificate_label(verdict_file, verdict)
        name = f"{label}-{evaluation_suffix(verdict_file, verdict)}"
        out = project.folder("certificates") / name
        identity_file = run_dir / IDENTITY_FILE
        checkpoint = run_dir / f"{verdict.get('policy', '')}.pt"
        if out.exists() or not identity_file.is_file() or not checkpoint.is_file():
            continue
        identity = json.loads(identity_file.read_text())
        run_stamp = stamp_run(run_dir)
        policy_stamp = write_policy(project, label, checkpoint, identity, run_stamp)
        if write_certificate(
            project, verdict_file, label, policy_stamp, identity, run_stamp
        ):
            written.append(out)
    return written


def newest_mtime(project: Project) -> float:
    """The newest modification time under the project's artifact
    folders - a new certificate, checkpoint or task since the index was
    written means the Studio's picture is stale."""
    newest = 0.0
    for kind_folder in (
        "runs",
        "tasks",
        "robots",
        "findings",
        "datasets",
        "recordings",
        "policies",
        "certificates",
    ):
        folder = project.root / kind_folder
        if not folder.is_dir():
            continue
        for path in folder.rglob("*"):
            try:
                newest = max(newest, path.stat().st_mtime)
            except OSError:
                continue
    return newest


def index_stale(project: Project) -> bool:
    """Whether anything under the project moved since the index was written."""
    index = project.index_path
    if not index.is_file():
        return True
    return newest_mtime(project) > index.stat().st_mtime
