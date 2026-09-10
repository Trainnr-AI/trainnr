"""Import what exists into a project: an rq_mjlab experiment (its run,
its checkpoint as a policy, its verdicts as evaluations) and a finding
from the repo's ledger. The flagship study (docs/artifacts/walk-c1)
holds fifteen trained arms with verdicts, and the ledger holds thirty
records; none of it was reachable from a project before 2026-09-09.

Every imported artifact cites what it came from by version, in the
field's words: a policy cites its run, robot and actuator; an
evaluation cites its policy, robot and environment; nothing is renamed.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from rq_pipeline.project.kinds import IDENTITY_FILE, Kind, stamp_kind, stamp_run
from rq_pipeline.project.locate import Project, plain_name

POLICY_FILE = "policy.json"
POLICY_SCHEMA = "trainnr-policy/1"
CERTIFICATE_SCHEMA = "trainnr-evaluation/1"
VERDICT_GLOB = "walk-verdict-*.json"
REPO = Path(__file__).resolve().parents[3]
LEDGER = REPO / "docs" / "findings"


def _checkpoint(train: Path) -> Path:
    """The last checkpoint by its iteration number in the name (a copied
    tree's modification times mean nothing), mtime as the tie-break."""
    weights = sorted(
        train.glob("model_*.pt"),
        key=lambda p: (_iterations(p) or -1, p.stat().st_mtime),
    )
    if not weights:
        raise FileNotFoundError(f"{train}: no model_*.pt checkpoint")
    return weights[-1]


def _iterations(checkpoint: Path) -> int | None:
    match = re.search(r"model_(\d+)", checkpoint.stem)
    return int(match.group(1)) + 1 if match else None


def import_experiment(
    project: Project, source: Path, name: str | None = None
) -> dict[str, Any]:
    """An rq_mjlab run directory (`<arm>/train/{identity.json,
    model_*.pt, verdict/}`, or the `train/` itself) into the project as a
    run, a policy and one evaluation per verdict file. Refuses to
    overwrite a name already in the project."""
    source = Path(source)
    train = source / "train" if (source / "train").is_dir() else source
    arm = train.parent  # the run's own directory, whichever form was given
    identity_file = train / IDENTITY_FILE
    if not identity_file.is_file():
        raise FileNotFoundError(f"{train}: no {IDENTITY_FILE} (not an rq_mjlab run)")
    identity = json.loads(identity_file.read_text())
    label = plain_name(name or arm.name, "run name")
    run_dir = project.folder("runs") / label
    policy_dir = project.folder("policies") / label
    if run_dir.exists() or policy_dir.exists():
        raise FileExistsError(f"{label!r} is already in the project")
    # Everything that can refuse does so before anything is written, so
    # a refused import leaves the project exactly as it was.
    checkpoint = _checkpoint(train)
    text = _training_log(arm, train, label)

    # The run: its identity and, when the study kept one, its log.
    run_dir.mkdir(parents=True)
    shutil.copy2(identity_file, run_dir / IDENTITY_FILE)
    if text is not None:
        (run_dir / "train.log").write_text(text)
        _write_training_record(run_dir / "train.log")
    run_stamp = stamp_run(run_dir)

    policy_stamp = write_policy(project, label, checkpoint, identity, run_stamp)
    evaluations = [
        stamp
        for verdict_file in sorted((train / "verdict").glob(VERDICT_GLOB))
        if (
            stamp := write_certificate(
                project, verdict_file, label, policy_stamp, identity, run_stamp
            )
        )
    ]
    return {
        "run": run_stamp,
        "policy": policy_stamp,
        "evaluations": evaluations,
        "name": label,
    }


def write_policy(
    project: Project, label: str, checkpoint: Path, identity: dict, run_stamp: str
) -> str:
    """A checkpoint as a policy artifact: the weights copied, a manifest
    citing the run, robot and actuator it came from. Returns its stamp;
    an existing policy of that name is returned as it is."""
    policy_dir = project.folder("policies") / label
    if policy_dir.is_dir():
        return stamp_kind(Kind.POLICY, policy_dir)
    policy_dir.mkdir(parents=True)
    shutil.copy2(checkpoint, policy_dir / checkpoint.name)
    # No identity.json here: that file marks a RUN to the kind detector;
    # the policy's manifest carries the same facts.
    manifest = {
        "schema": POLICY_SCHEMA,
        "name": label,
        "checkpoint": checkpoint.name,
        "format": "rsl_rl (torch)",
        "iterations": _iterations(checkpoint),
        "run": run_stamp,
        "robot": identity.get("robot"),
        "actuator": identity.get("actuator"),
        "dr_basis": identity.get("dr_basis"),
        "seed": identity.get("seed"),
    }
    (policy_dir / POLICY_FILE).write_text(json.dumps(manifest, indent=1) + "\n")
    return stamp_kind(Kind.POLICY, policy_dir)


def write_certificate(  # noqa: PLR0913, PLR0917 - what a certificate cites, each named
    project: Project,
    verdict_file: Path,
    label: str,
    policy_stamp: str,
    identity: dict,
    run_stamp: str,
) -> str | None:
    """One walk verdict (`walk-verdict-<suffix>.json` with its
    `records-<suffix>.jsonl`) as an evaluation `<label>-<suffix>` citing
    the policy, robot, environment and run. Returns the stamp, or None
    when that evaluation is already in the project."""
    suffix = verdict_file.stem.removeprefix("walk-verdict-")
    out = project.folder("certificates") / f"{label}-{suffix}"
    if out.exists():
        return None
    out.mkdir(parents=True)
    verdict = json.loads(verdict_file.read_text())
    certificate = {
        "schema": CERTIFICATE_SCHEMA,
        **verdict,
        "policy": policy_stamp,
        "robot": identity.get("robot"),
        "task": verdict.get("source"),
        "run": run_stamp,
    }
    (out / "certificate.json").write_text(json.dumps(certificate, indent=1) + "\n")
    # Exactly this verdict's records — "at-fit" is a substring of
    # "at-x0.7-at-fit", so a glob would hand a sibling's trials over.
    records = verdict_file.parent / f"records-{suffix}.jsonl"
    if records.is_file():
        shutil.copy2(records, out / records.name)
    return stamp_kind(Kind.CERTIFICATE, out)


TRAINING_FILE = "training.json"
LOG_DIR_LINE = "log_dir:"


def _training_log(arm: Path, train: Path, label: str) -> str | None:
    """The run's console log: its own `train.log` (beside or inside
    `train/`), else the segment of a study log — a `*.log` beside the arm
    — whose `log_dir:` banner names this run (a study launches several
    runs into one log, and a restarted run leaves two segments: the
    longest wins)."""
    for own in (train / "train.log", arm / "train.log"):
        if own.is_file():
            return own.read_text(errors="replace")
    best: str | None = None
    for log in sorted(arm.parent.glob("*.log")):
        lines = log.read_text(errors="replace").splitlines(keepends=True)
        starts = [i for i, line in enumerate(lines) if LOG_DIR_LINE in line]
        for n, i in enumerate(starts):
            if not lines[i].rstrip().endswith(f"/{label}/train"):
                continue
            end = starts[n + 1] if n + 1 < len(starts) else len(lines)
            segment = "".join(lines[i:end])
            if best is None or segment.count("Learning iteration") > best.count(
                "Learning iteration"
            ):
                best = segment
    return best


def _write_training_record(log: Path) -> None:
    """The run's training facts and reward curve, read once from the
    console log so the index never re-reads hundreds of thousands of lines."""
    from rq_pipeline.envs.rsl_rl_log import parse_rsl_rl_log  # noqa: PLC0415

    record = parse_rsl_rl_log(log.read_text(errors="replace"))
    if record is not None:
        (log.parent / TRAINING_FILE).write_text(
            json.dumps(record.to_json(), indent=1) + "\n"
        )


def import_finding(project: Project, record: str | Path) -> dict[str, Any]:
    """A record from the repo's findings ledger (by id, or a path to a
    record file) into the project's findings."""
    path = Path(record)
    if not path.is_file():
        path = LEDGER / f"{record}.json"
    if not path.is_file():
        raise FileNotFoundError(f"no finding {record!r} (looked in {LEDGER})")
    body = json.loads(path.read_text())
    if "claim" not in body or "id" not in body:
        raise ValueError(f"{path} is not a finding record (no id/claim)")
    out = project.folder("findings") / path.name
    if out.exists():
        raise FileExistsError(f"finding {path.name} is already in the project")
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, out)
    return {
        "finding": stamp_kind(Kind.FINDING, out),
        "id": body["id"],
        "claim": body["claim"],
    }


def ledger_findings(prefix: str = "") -> list[dict[str, str]]:
    """The ledger's records, id and claim, optionally by id prefix."""
    out = []
    for path in sorted(LEDGER.glob("*.json")):
        if prefix and not path.stem.startswith(prefix):
            continue
        try:
            body = json.loads(path.read_text())
        except ValueError:
            continue
        out.append(
            {
                "id": body.get("id", path.stem),
                "date": body.get("date", ""),
                "claim": body.get("claim", ""),
            }
        )
    return out
