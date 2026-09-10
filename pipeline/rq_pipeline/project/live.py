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
from rq_pipeline.project.importer import (
    VERDICT_GLOB,
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


def verdict_files(project: Project) -> list[Path]:
    """Every walk verdict judged inside a run folder of the project."""
    if not project.runs.is_dir():
        return []
    return sorted(project.runs.rglob(f"{VERDICT_DIR}/{VERDICT_GLOB}"))


def certificate_label(verdict_file: Path) -> str:
    """The evaluation's name: the run and the checkpoint it judged
    (`go2-c1-model_1400`); the verdict's own suffix is appended by the
    writer, so the same checkpoint judged on another instrument keeps
    its own card."""
    run_dir = verdict_file.parent.parent
    policy = json.loads(verdict_file.read_text()).get("policy", "")
    return f"{run_dir.name}-{policy}"


def refresh_verdicts(project: Project) -> list[Path]:
    """Every verdict not yet in the project becomes a policy and an
    evaluation; the evaluation folders written. Idempotent: a verdict
    already imported costs one existence check, so the run folder's
    checkpoints are hashed only when something new landed."""
    written: list[Path] = []
    for verdict_file in verdict_files(project):
        run_dir = verdict_file.parent.parent
        label = certificate_label(verdict_file)
        suffix = verdict_file.stem.removeprefix("walk-verdict-")
        out = project.folder("certificates") / f"{label}-{suffix}"
        identity_file = run_dir / IDENTITY_FILE
        checkpoint = run_dir / f"{label.removeprefix(run_dir.name + '-')}.pt"
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
