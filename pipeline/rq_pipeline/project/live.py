"""A live run in the Studio: its console log becomes its training record
as it grows, so the experiment card shows the reward curve so far.

A run folder that carries its own `train.log` (rq_mjlab's walk_train
tees its console there when it archives a run) gets a `training.json`
parsed from it whenever the log is newer than the record; the record
says what it can about whether the run is still going: `done` when the
trainer wrote its last line, `running` while a source is still being
written to, and `unrecorded` when neither is true - a trainer that
stopped writing without a last line was killed or paused, and the
record does not guess which. The importer does the same once for an
imported study arm; this does it on every presenter tick for whatever
trains inside the project.

A verdict judged inside a run folder (`evaluate_walk` on a checkpoint of
a run training in the project writes `verdict/walk-verdict-*.json`
beside the weights) becomes, the same tick, a policy (the checkpoint it
judged) and an evaluation citing it — the Evaluations page stayed at
zero after a 40/40 certificate until this existed (2026-09-10).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from rq_pipeline.envs.rsl_rl_log import (
    DONE_MARK,
    STATUS_DONE,
    STATUS_RUNNING,
    TRAIN_LOG,
    TRAINING_FILE,
    VERDICT_DIR,
    VERDICT_GLOB,
    parse_rsl_rl_log,
)
from rq_pipeline.envs.tfevents import EVENTS_GLOB, events_file, record_from_events
from rq_pipeline.project.files import read_json, read_text, write_json
from rq_pipeline.project.importer import (
    evaluation_suffix,
    write_certificate,
    write_policy,
)
from rq_pipeline.project.index import UNRECORDED
from rq_pipeline.project.kinds import IDENTITY_FILE, stamp_run
from rq_pipeline.project.locate import FOLDERS, Project

DONE_TAIL_CHARS = 2000  # the done line is among the log's last lines
# A source written to within this long is a trainer still writing.
WRITING_S = 120.0
CHECKPOINT_GLOB = "*.pt"


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


def reached_its_last_iteration(record: dict[str, Any] | None) -> bool:
    """Whether the record logged every iteration it was asked for: a run
    whose trainer crashed at teardown (an EGL error after the last
    iteration, go2-c2 on 2026-09-12) never printed its done line, yet it
    finished."""
    if not record:
        return False
    asked, logged = record.get("iterations"), record.get("iterations_logged")
    return (
        isinstance(asked, int)
        and isinstance(logged, int)
        and asked > 0
        and logged >= asked
    )


def run_status(
    folder: Path,
    text: str,
    *,
    now: float | None = None,
    record: dict[str, Any] | None = None,
) -> str:
    """What the folder says about the run: `done` on the trainer's last
    line or on its last iteration logged, `running` while a source was
    written to within `WRITING_S`, else `unrecorded`."""
    if DONE_MARK in text[-DONE_TAIL_CHARS:] or reached_its_last_iteration(record):
        return STATUS_DONE
    now = time.time() if now is None else now
    if any(now - p.stat().st_mtime < WRITING_S for p in _sources(folder)):
        return STATUS_RUNNING
    return UNRECORDED


def refresh_training(folder: Path) -> dict[str, Any] | None:
    """The folder's `training.json` from the trainer's own event file
    (every series it logs) with the console log's facts, or from the
    console log alone; returns the record written, or None when neither
    holds an iteration yet."""
    log = folder / TRAIN_LOG
    text = read_text(log, errors="replace") if log.is_file() else ""
    parsed = parse_rsl_rl_log(text) if text else None
    events = events_file(folder)
    record = record_from_events(events, facts=parsed) if events else None
    record = record or parsed
    if record is None:
        return None
    out = record.to_json()
    out["status"] = run_status(folder, text, record=out)
    write_json(folder / TRAINING_FILE, out)
    return out


def left_running(folder: Path, *, now: float | None = None) -> bool:
    """A record that says `running` while no source has been written to
    for `WRITING_S`: the run ended without its done line, and the record
    would say `running` forever (go2-c2, seen 2026-09-27)."""
    path = folder / TRAINING_FILE
    if not path.is_file():
        return False
    try:
        status = read_json(path).get("status")
    except (OSError, ValueError):
        return False
    if status != STATUS_RUNNING:
        return False
    now = time.time() if now is None else now
    return all(now - p.stat().st_mtime >= WRITING_S for p in _sources(folder))


def refresh_project(project: Project) -> list[Path]:
    """Refresh every stale run in the project; the folders refreshed."""
    return [
        f
        for f in run_folders(project)
        if (stale(f) or left_running(f)) and refresh_training(f)
    ]


def verdict_files(project: Project) -> list[Path]:
    """Every walk verdict judged inside a run folder of the project."""
    if not project.runs.is_dir():
        return []
    return sorted(project.runs.rglob(f"{VERDICT_DIR}/{VERDICT_GLOB}"))


def certificate_label(verdict_file: Path, verdict: dict[str, Any]) -> str:
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
        verdict = read_json(verdict_file)
        label = certificate_label(verdict_file, verdict)
        name = f"{label}-{evaluation_suffix(verdict_file, verdict)}"
        out = project.certificates / name
        identity_file = run_dir / IDENTITY_FILE
        checkpoint = run_dir / f"{verdict.get('policy', '')}{CHECKPOINT_GLOB[1:]}"
        if out.exists() or not identity_file.is_file() or not checkpoint.is_file():
            continue
        identity = read_json(identity_file)
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
    for kind_folder in FOLDERS:
        folder = project.folder(kind_folder)
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
