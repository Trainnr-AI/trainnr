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

import re
import shutil
from pathlib import Path
from typing import Any

from rq_pipeline.bundles.hashing import fields_hash
from rq_pipeline.envs.rsl_rl_log import (
    ITERATION_LINE,
    TRAIN_LOG,
    TRAINING_FILE,
    VERDICT_DIR,
    VERDICT_GLOB,
    VERDICT_PREFIX,
    parse_rsl_rl_log,
)
from rq_pipeline.paths import checkout
from rq_pipeline.project.files import read_json, read_text, write_json, write_text
from rq_pipeline.project.kinds import (
    CERTIFICATE_FILE,
    IDENTITY_FILE,
    POLICY_FILE,
    Kind,
    stamp_kind,
    stamp_run,
)
from rq_pipeline.project.locate import Project, plain_name

POLICY_SCHEMA = "trainnr-policy/1"
CERTIFICATE_SCHEMA = "trainnr-evaluation/1"
LEDGER_DIR = Path("docs") / "findings"  # the repo's findings ledger


def ledger() -> Path:
    return checkout() / LEDGER_DIR


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
    identity = read_json(identity_file)
    label = plain_name(name or arm.name, "run name")
    run_dir = project.runs / label
    policy_dir = project.policies / label
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
        write_text(run_dir / TRAIN_LOG, text)
        _write_training_record(run_dir / TRAIN_LOG)
    run_stamp = stamp_run(run_dir)

    policy_stamp = write_policy(project, label, checkpoint, identity, run_stamp)
    evaluations = [
        stamp
        for verdict_file in sorted((train / VERDICT_DIR).glob(VERDICT_GLOB))
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
    project: Project,
    label: str,
    checkpoint: Path,
    identity: dict[str, Any],
    run_stamp: str,
) -> str:
    """A checkpoint as a policy artifact: the weights copied, a manifest
    citing the run, robot and actuator it came from. Returns its stamp;
    an existing policy of that name is returned as it is."""
    policy_dir = project.policies / label
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
    write_json(policy_dir / POLICY_FILE, manifest)
    return stamp_kind(Kind.POLICY, policy_dir)


# walk_verdict rotates a replaced verdict to <name>[.<policy>].seed<n>.n<k>.json;
# the suffix is what stands before that tail (a suffix itself may hold a
# dot: "at-x0.7-at-fit").
BACKUP_TAIL = re.compile(r"(\.model_\d+)?\.seed\d+\.n\d+$")


PROTOCOL_HASH_CHARS = 6


def evaluation_suffix(verdict_file: Path, verdict: dict[str, Any]) -> str:
    """What tells one evaluation of a policy from another: the verdict
    file's instrument suffix, the seed, the trial count and a hash of
    the whole protocol (the criterion, the DR basis, the command
    envelope). A verdict rq_mjlab rotated to its backup name (the same
    judgment, renamed when a later one took the primary name) maps to
    the same evaluation, so it is imported once; a re-judge under
    another seed or another protocol - the Go2 checkpoints judged
    again at their trained command envelope, 2026-09-11 - is its own."""
    suffix = BACKUP_TAIL.sub("", verdict_file.stem.removeprefix(VERDICT_PREFIX))
    protocol = verdict.get("protocol") or {}
    seed, trials = protocol.get("seed"), verdict.get("trials")
    if seed is None or trials is None:
        return suffix
    return (
        f"{suffix}-seed{seed}-n{trials}-p{fields_hash(protocol)[:PROTOCOL_HASH_CHARS]}"
    )


def write_certificate(  # noqa: PLR0913, PLR0917 - what a certificate cites, each named
    project: Project,
    verdict_file: Path,
    label: str,
    policy_stamp: str,
    identity: dict[str, Any],
    run_stamp: str,
) -> str | None:
    """One walk verdict (`walk-verdict-<suffix>.json` with its
    `records-<suffix>.jsonl`) as an evaluation `<label>-<suffix>` citing
    the policy, robot, environment and run. Returns the stamp, or None
    when that evaluation is already in the project."""
    verdict = read_json(verdict_file)
    name = f"{label}-{evaluation_suffix(verdict_file, verdict)}"
    out = project.certificates / name
    if out.exists():
        return None
    out.mkdir(parents=True)
    suffix = verdict_file.stem.removeprefix(VERDICT_PREFIX)
    certificate = {
        "schema": CERTIFICATE_SCHEMA,
        **verdict,
        "policy": policy_stamp,
        "robot": identity.get("robot"),
        "task": verdict.get("source"),
        "run": run_stamp,
    }
    write_json(out / CERTIFICATE_FILE, certificate)
    # Exactly this verdict's records — "at-fit" is a substring of
    # "at-x0.7-at-fit", so a glob would hand a sibling's trials over.
    records = verdict_file.parent / f"records-{suffix}.jsonl"
    if records.is_file():
        shutil.copy2(records, out / records.name)
    return stamp_kind(Kind.CERTIFICATE, out)


LOG_DIR_LINE = "log_dir:"


def _training_log(arm: Path, train: Path, label: str) -> str | None:
    """The run's console log: its own `train.log` (beside or inside
    `train/`), else the segment of a study log — a `*.log` beside the arm
    — whose `log_dir:` banner names this run (a study launches several
    runs into one log, and a restarted run leaves two segments: the
    longest wins)."""
    for own in (train / TRAIN_LOG, arm / TRAIN_LOG):
        if own.is_file():
            return read_text(own, errors="replace")
    best: str | None = None
    for log in sorted(arm.parent.glob("*.log")):
        lines = read_text(log, errors="replace").splitlines(keepends=True)
        starts = [i for i, line in enumerate(lines) if LOG_DIR_LINE in line]
        for n, i in enumerate(starts):
            if not lines[i].rstrip().endswith(f"/{label}/train"):
                continue
            end = starts[n + 1] if n + 1 < len(starts) else len(lines)
            segment = "".join(lines[i:end])
            if best is None or _iterations_in(segment) > _iterations_in(best):
                best = segment
    return best


def _iterations_in(text: str) -> int:
    return sum(1 for _ in ITERATION_LINE.finditer(text))


def _write_training_record(log: Path) -> None:
    """The run's training facts and reward curve, read once from the
    console log so the index never re-reads hundreds of thousands of lines."""
    record = parse_rsl_rl_log(read_text(log, errors="replace"))
    if record is not None:
        write_json(log.parent / TRAINING_FILE, record.to_json())


def import_finding(project: Project, record: str | Path) -> dict[str, Any]:
    """A record from the repo's findings ledger (by id, or a path to a
    record file) into the project's findings."""
    path = Path(record)
    if not path.is_file():
        path = ledger() / f"{record}.json"
    if not path.is_file():
        raise FileNotFoundError(f"no finding {record!r} (looked in {ledger()})")
    body = read_json(path)
    if "claim" not in body or "id" not in body:
        raise ValueError(f"{path} is not a finding record (no id/claim)")
    out = project.findings / path.name
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
    for path in sorted(ledger().glob("*.json")):
        if prefix and not path.stem.startswith(prefix):
            continue
        try:
            body = read_json(path)
        except (OSError, ValueError):
            continue
        out.append(
            {
                "id": body.get("id", path.stem),
                "date": body.get("date", ""),
                "claim": body.get("claim", ""),
            }
        )
    return out
