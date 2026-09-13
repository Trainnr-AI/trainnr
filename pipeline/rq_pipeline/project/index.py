"""Walk a project, stamp everything, read lineage, decide where it stands.

The index is what the Studio opens and what `describe_project` returns:
every artifact by kind with its stamp and the stamps it cites, and the
loop map — the eight states of docs/76 §3, each proved by an artifact
or marked missing, with the next legal move named.

The files are the truth; this is a cache. `write_index` puts it at
`<project>/.index/project.json`; deleting that directory loses nothing.

Lineage is read, never inferred: a training run's `run.json` quotes its
dataset's provenance (`bundle`, `expert`); a dataset's `provenance.json`
names its source batch and bundle; an RL run's `identity.json` names the
robot and actuator stamps it trained against; a certificate names all of
its inputs. Where an older artifact recorded no stamp, the index says
`unrecorded`, the way the datasheet does — it never invents one.
"""

from __future__ import annotations

import contextlib
import math
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rq_pipeline.bundles.hashing import is_stamp
from rq_pipeline.collect.provenance import PROVENANCE_FILE
from rq_pipeline.deploy.manifest import gate_word, read_gates
from rq_pipeline.envs.lerobot_train_log import RUN_MANIFEST_FILE
from rq_pipeline.envs.rsl_rl_log import COL_REWARD, STATUS_RUNNING, TRAINING_FILE
from rq_pipeline.evaluate.commands import describe_twist
from rq_pipeline.fleet.drift import verdict_word
from rq_pipeline.project.files import read_json, write_json, write_text
from rq_pipeline.project.kinds import (
    ACCEPTANCE_FILE,
    ACCEPTED,
    CERTIFICATE_FILE,
    DEPLOY_FILE,
    DRIFT_FILE,
    IDENTITY_FILE,
    POLICY_FILE,
    RECORDING_FILE,
    REJECTED,
    TASK_FILE,
    UNREVIEWED,
    Kind,
    UnknownKindError,
    detect,
    stamp_kind,
    stamp_run,
)
from rq_pipeline.project.locate import FOLDERS, LOOPS, Project

# The one word for a fact an artifact never recorded - never a guess.
UNRECORDED = "unrecorded"
INTERVAL_ENDS = 2  # a confidence interval is two numbers
# The readers the index keeps per kind: each takes the artifact's root.
CiteReader = Callable[[Path], dict[str, str]]
SummaryReader = Callable[[Path], dict[str, Any]]

# The loop's states, in order, and the kind whose presence proves each
# (docs/76 §3). `identified` is proved by a fit OR by a robot bundle that
# carries fit records; the checker below handles the second case.
STATES: tuple[tuple[str, Kind], ...] = (
    ("telemetry recorded", Kind.RECORDING),
    ("asset onboarded", Kind.ROBOT),
    ("system identified", Kind.FIT),
    ("environment defined", Kind.TASK),
    ("data generated", Kind.BATCH),
    ("policy trained", Kind.RUN),
    ("policy evaluated", Kind.CERTIFICATE),
    ("deployment exported", Kind.DEPLOY),
    ("drift monitored", Kind.DRIFT),
)

# What an agent does next when a state is the first missing one.
NEXT_MOVE: dict[str, str] = {
    "telemetry recorded": (
        "record the robot's telemetry (ingest_recording: a .wire file, a LeRobot "
        "dataset, a ROS 2 .mcap bag, a mocap CSV or BVH) — or start from its model"
    ),
    "asset onboarded": "onboard the robot's model as an asset (onboard_robot)",
    "system identified": (
        "run system identification on a recording, or name an identified actuator model"
    ),
    "environment defined": "define the environment (task spec) and run acceptance",
    "data generated": (
        "generate demonstrations: scripted, planned, or a walk's own rollouts"
    ),
    "policy trained": "train a policy on the dataset, or a walk by reinforcement",
    "policy evaluated": (
        "evaluate the policy with paired trials and a confidence interval"
    ),
    "deployment exported": "export a deployment manifest and pass the sim-to-sim check",
    "drift monitored": "record fresh telemetry and check for parameter drift",
}


@dataclass(frozen=True)
class Artifact:
    kind: str
    stamp: str
    path: str  # relative to the project root
    # Stamps (or `unrecorded`) of the artifacts this one cites.
    cites: dict[str, str] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    # A picture, relative to the project root, when the kind has one
    # (`project/previews.py`); None otherwise — never a placeholder.
    preview: str | None = None
    # The detail view's path, relative to the project root
    # (`project/details.py`); None when the kind has no writer yet.
    detail: str | None = None
    # When it entered and when it last changed (ISO 8601, UTC, seconds):
    # the record's own date where it keeps one (a finding), else the
    # oldest and newest file under it. Never invented: None when the
    # artifact has no files at all.
    created: str | None = None
    updated: str | None = None
    # Stamps of the artifacts in this project that cite this one — the
    # lineage read downstream (a policy's evaluations, a run's policy).
    cited_by: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class State:
    name: str
    proved_by: list[str]  # stamps
    present: bool
    # False when this loop never passes through the stage (a
    # reinforcement-learning loop has no dataset); `note` says why.
    needed: bool = True
    note: str | None = None


@dataclass(frozen=True)
class ProjectIndex:
    schema: str
    project: str
    root: str
    indexed: str  # ISO-8601 UTC
    artifacts: list[Artifact]
    states: list[State]
    next_move: str | None
    refused: list[dict[str, str]]  # paths that carried no or two markers

    def by_kind(self, kind: Kind) -> list[Artifact]:
        return [a for a in self.artifacts if a.kind == kind.value]


INDEX_SCHEMA = "trainnr-project-index/1"


def index_project(project: Project) -> ProjectIndex:
    manifest = project.manifest()
    artifacts: list[Artifact] = []
    refused: list[dict[str, str]] = []
    for folder in FOLDERS:
        for path in _candidates(project.root / folder):
            try:
                kind = detect(path)
                # An RL arm keeps its identity under <arm>/train; the ARM is
                # the artifact's name, not the folder the marker sat in.
                label = path.parent.name if path.name == "train" else None
                if kind is Kind.RUN:
                    identity = stamp_run(path, name=label)
                else:
                    identity = stamp_kind(kind, path, name=label)
                if kind is Kind.TASK:
                    identity = _task_identity(path, identity)
            except UnknownKindError as why:
                refused.append(
                    {"path": str(path.relative_to(project.root)), "reason": str(why)}
                )
                continue
            created, updated = _times(kind, path)
            artifacts.append(
                Artifact(
                    kind=kind.value,
                    stamp=identity,
                    path=str(path.relative_to(project.root)),
                    cites=_cites(kind, path),
                    summary=_summary(kind, path),
                    created=created,
                    updated=updated,
                )
            )
    _link_cited_by(artifacts)
    loop = manifest.loop or _loop_of(artifacts)
    states = _states(artifacts, loop)
    missing = [s.name for s in states if not s.present and s.needed]
    # Telemetry is the loop's first state but not a prerequisite: a robot
    # that entered as a model (a Menagerie bundle) has no recording yet
    # and does not need one before a task is declared. The next move is
    # the first missing state AFTER the first proved one.
    first_proved = next((i for i, s in enumerate(states) if s.present), None)
    if first_proved is not None:
        missing = [s.name for s in states[first_proved:] if not s.present and s.needed]
    return ProjectIndex(
        schema=INDEX_SCHEMA,
        project=manifest.name,
        root=str(project.root),
        indexed=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        artifacts=sorted(artifacts, key=lambda a: (a.kind, a.path)),
        states=states,
        next_move=NEXT_MOVE[missing[0]] if missing else None,
        refused=refused,
    )


def write_index(
    project: Project, index: ProjectIndex | None = None, *, previews: bool = True
) -> Path:
    """Write the index; with `previews`, render each artifact's picture
    first and record its path on the artifact (a cache keyed by stamp, so
    unchanged artifacts are never re-rendered)."""
    index = index if index is not None else index_project(project)
    if previews:
        from rq_pipeline.project.details import write_details  # noqa: PLC0415
        from rq_pipeline.project.previews import write_previews  # noqa: PLC0415

        found = write_previews(project, index)
        details = write_details(project, index)
        index = replace(
            index,
            artifacts=[
                replace(a, preview=found.get(a.stamp), detail=details.get(a.stamp))
                for a in index.artifacts
            ],
        )
    out = write_json(project.index_path, asdict(index))
    write_text(out.parent / ".gitignore", "*\n")
    return out


def _candidates(folder: Path) -> list[Path]:
    """The artifacts directly under a kind's folder: each subdirectory,
    plus single-file artifacts (fit records, findings, `.pt` policies).
    An RL run keeps `identity.json` beside its checkpoint under `train/`;
    that nesting is walked one level."""
    if not folder.is_dir():
        return []
    found: list[Path] = []
    for entry in sorted(folder.iterdir()):
        if entry.name.startswith("."):
            continue
        if entry.is_file():
            found.append(entry)
            continue
        names = {p.name for p in entry.iterdir()}
        if IDENTITY_FILE not in names and (entry / "train" / IDENTITY_FILE).is_file():
            found.append(entry / "train")
        else:
            found.append(entry)
    return found


def _read(path: Path) -> dict[str, Any]:
    """A record, or `{}` when the file is absent or unreadable - the
    index says `unrecorded` for what it cannot read, and never stops."""
    return read_json(path, missing_ok=True)


def interval_of(record: dict[str, Any]) -> tuple[float, float] | None:
    """The exact interval a record carries (`ci95`, or the older `ci`),
    or None when it holds none - never a made-up pair."""
    ci = record.get("ci95") or record.get("ci")
    if isinstance(ci, (list, tuple)) and len(ci) == INTERVAL_ENDS:
        try:
            return float(ci[0]), float(ci[1])
        except (TypeError, ValueError):
            return None
    return None


def ratio_of(record: dict[str, Any], k: str = "successes", n: str = "trials") -> str:
    """`k / n` as a record wrote them, or `unrecorded` when either is
    absent - never `None / None`."""
    successes, trials = record.get(k), record.get(n)
    if successes is None or trials is None:
        return UNRECORDED
    return f"{successes} / {trials}"


def _stamp_or_unrecorded(value: Any) -> str:
    return value if isinstance(value, str) and is_stamp(value) else UNRECORDED


def _cites(kind: Kind, path: Path) -> dict[str, str]:
    """The stamps an artifact names — read from its own manifest."""
    reader = _CITE_READERS.get(kind)
    return reader(path) if reader is not None else {}


def _pick(raw: dict[str, Any], keys: tuple[str, ...]) -> dict[str, str]:
    """The named keys as stamps-or-unrecorded, only those present."""
    return {key: _stamp_or_unrecorded(raw.get(key)) for key in keys if key in raw}


def _cites_dataset(path: Path) -> dict[str, str]:
    raw = _read(path / PROVENANCE_FILE)
    return {
        "bundle": _stamp_or_unrecorded(raw.get("bundle")),
        "expert": _stamp_or_unrecorded(raw.get("expert")),
        # `source` is the batch's NAME by contract; its stamp rides beside it.
        "source": _stamp_or_unrecorded(raw.get("source_stamp")),
    }


def _cites_run(path: Path) -> dict[str, str]:
    run = _read(path / RUN_MANIFEST_FILE)
    if run:
        prov = run.get("provenance") or {}
        return {
            key: _stamp_or_unrecorded(prov.get(key)) for key in ("bundle", "expert")
        }
    identity = _read(path / IDENTITY_FILE)
    return {
        key: _stamp_or_unrecorded(identity.get(key)) for key in ("robot", "actuator")
    }


def _cites_batch(path: Path) -> dict[str, str]:
    # A batch's stamps live per episode; the datasheet folds them.
    first = next(iter(sorted(path.glob("episode_*/manifest.json"))), None)
    if first is None:
        return {}
    return _pick(_read(first), ("task", "expert", "instrument"))


def _cites_policy(path: Path) -> dict[str, str]:
    manifest = _read(path / POLICY_FILE)
    if manifest:
        return _pick(manifest, ("run", "robot", "actuator"))
    return _pick(_read(path / IDENTITY_FILE), ("robot", "actuator"))


_CITE_READERS: dict[Kind, CiteReader] = {
    Kind.DATASET: _cites_dataset,
    Kind.RUN: _cites_run,
    Kind.BATCH: _cites_batch,
    Kind.POLICY: _cites_policy,
    Kind.CERTIFICATE: lambda p: _pick(
        _read(p / CERTIFICATE_FILE), ("robot", "task", "policy", "run")
    ),
    Kind.DEPLOY: lambda p: _pick(
        _read(p / DEPLOY_FILE), ("policy", "run", "robot", "task", "certificate")
    ),
    # A check cites the robot (its version carries the fit records it was
    # judged against; the drawer names them) and the recording it judged.
    Kind.DRIFT: lambda p: _pick(_read(p / DRIFT_FILE), ("robot", "recording")),
}


def _summary(kind: Kind, path: Path) -> dict[str, Any]:
    """A few glanceable facts for the Studio's list rows — never the
    whole record; the detail view reads the artifact itself."""
    reader = _SUMMARY_READERS.get(kind)
    return reader(path) if reader is not None else {}


def _take(raw: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {k: raw[k] for k in keys if k in raw}


def _summary_run(path: Path) -> dict[str, Any]:
    run = _read(path / RUN_MANIFEST_FILE)
    if run:
        return {"learning": "imitation", **_take(run, ("policy", "steps", "started"))}
    training = _read(path / TRAINING_FILE)
    ident = _read(path / IDENTITY_FILE)
    out: dict[str, Any] = {"learning": "reinforcement"}
    if training:
        out["iterations"] = training.get("iterations_logged") or UNRECORDED
        reward = (training.get("final") or {}).get(COL_REWARD)
        if reward is not None:
            out["final reward"] = round(float(reward), 1)
        status = training.get("status")
        if status:
            out["status"] = status
            planned = training.get("iterations")  # only when the console said
            logged = training.get("iterations_logged")
            if status == STATUS_RUNNING and planned and logged is not None:
                out["progress"] = f"{logged} of {planned} iterations"
    basis = ident.get("dr_basis") or ""
    if basis:
        out["randomization"] = _basis_name(basis)
    if "seed" in ident:
        out["seed"] = ident["seed"]
    return out


def _summary_recording(path: Path) -> dict[str, Any]:
    raw = _read(path / RECORDING_FILE)
    out = _take(raw, ("adapter", "source", "duration_s", "collection"))
    if "channels" in raw:
        out["channels"] = len(raw["channels"])
    return out


def _summary_certificate(path: Path) -> dict[str, Any]:
    """What a card says about an evaluation: the rate and its interval."""
    c = _read(path / CERTIFICATE_FILE)
    out: dict[str, Any] = {}
    # The envelope first: the card's subtitle is cut short, and the
    # envelope is what tells two judgments of one checkpoint apart.
    commands = describe_twist((c.get("protocol") or {}).get("commands"))
    if commands:
        out["commands"] = commands
    if "successes" in c and "trials" in c:
        out["success"] = ratio_of(c)
    interval = interval_of(c)
    if interval is not None:
        out["interval"] = f"[{interval[0]:.2f}, {interval[1]:.2f}]"
    judged = (c.get("protocol") or {}).get("judged_at")
    if judged:
        out["judged at"] = judged
    return out


def _basis_name(basis: str) -> str:
    """The domain-randomization basis in a few words: the part before its
    colon or its parenthesis ("identified-set", "caller-declared span ±0.1")."""
    return basis.split(":", 1)[0].split(" (", 1)[0].strip()


def _summary_policy(path: Path) -> dict[str, Any]:
    manifest = _read(path / POLICY_FILE)
    if manifest:
        basis = manifest.get("dr_basis") or ""
        iterations = manifest.get("iterations")
        out: dict[str, Any] = {
            "iterations": UNRECORDED if iterations is None else iterations
        }
        out["randomization"] = _basis_name(basis) if basis else UNRECORDED
        out["format"] = manifest.get("format", "")
        return out
    return _take(_read(path / IDENTITY_FILE), ("dr_basis", "seed"))


CLAIM_CHARS = 160  # a card's subtitle: the claim's first sentence, this long at most


def _task_identity(path: Path, folder_stamp: str) -> str:
    """A task's version is its spec's content hash (`Task.stamp`, kept
    in `task.json`), not a hash of its folder — so a review written
    beside it later does not rename it. The name stays the folder's."""
    recorded = str(_read(path / TASK_FILE).get("stamp", ""))
    if "@" not in recorded:
        return folder_stamp
    return f"{folder_stamp.split('@', 1)[0]}@{recorded.split('@', 1)[1]}"


def _link_cited_by(artifacts: list[Artifact]) -> None:
    """Fill every artifact's `cited_by` from the others' `cites`, so the
    lineage reads both ways without a second pass over the files. A cite
    names a version: the hash decides, the name half is a label (a
    certificate cites `kitting@7d4f…`; the project holds it as
    `tray-far@7d4f…`)."""
    by_stamp = {a.stamp: a for a in artifacts}
    by_hash: dict[str, list[Artifact]] = {}
    for a in artifacts:
        if "@" in a.stamp:
            by_hash.setdefault(a.stamp.split("@", 1)[1], []).append(a)
    for a in artifacts:
        for cited in a.cites.values():
            targets = [by_stamp[cited]] if cited in by_stamp else []
            if not targets and "@" in cited:
                targets = by_hash.get(cited.split("@", 1)[1], [])
            for target in targets:
                if a.stamp not in target.cited_by:
                    target.cited_by.append(a.stamp)
    for a in artifacts:
        a.cited_by.sort()


def _times(kind: Kind, path: Path) -> tuple[str | None, str | None]:
    """(created, updated) for an artifact: its own recorded date when it
    keeps one, else the oldest file under it; the newest file under it."""
    stamps = sorted(file_times(path))
    oldest = _iso(stamps[0]) if stamps else None
    newest = _iso(stamps[-1]) if stamps else None
    own = _own_date(kind, path)
    return (own or oldest, newest)


def file_times(path: Path) -> list[float]:
    """Modification times of every file under `path`, hidden files and
    hidden directories (an artifact's own `.index`, a `.git`) left out."""
    if path.is_file():
        return [path.stat().st_mtime]
    times: list[float] = []
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in filenames:
            if not name.startswith("."):
                with contextlib.suppress(OSError):
                    times.append((Path(dirpath) / name).stat().st_mtime)
    return times


def _own_date(kind: Kind, path: Path) -> str | None:
    """The date a record carries about itself, as an ISO instant."""
    if kind is Kind.FINDING:
        date = _read(path).get("date")
        if isinstance(date, str) and len(date) >= DATE_CHARS:
            return f"{date[:DATE_CHARS]}T00:00:00+00:00"
    if kind is Kind.DRIFT:
        created = _read(path / DRIFT_FILE).get("created_utc")
        if isinstance(created, str) and created:
            return created
    return None


UNBOUNDED = "unbounded"


def interval_text(lower: float | None, upper: float | None) -> str:
    """An interval as a drawer says it: unrecorded when an end is unknown,
    unbounded for an infinite end."""
    if lower is None or upper is None:
        return UNRECORDED

    def end(v: float) -> str:
        return UNBOUNDED if not math.isfinite(v) else f"{v:.4g}"

    return f"[{end(lower)}, {end(upper)}]"


def _iso(epoch: float) -> str:
    return (
        datetime.fromtimestamp(epoch, tz=timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )


DATE_CHARS = len("2026-09-04")


def _summary_task(path: Path) -> dict[str, Any]:
    out = _take(_read(path / TASK_FILE), ("task_id", "stamp", "kind"))
    verdict = _read(path / ACCEPTANCE_FILE)
    out["acceptance"] = (
        (ACCEPTED if verdict.get("accepted") else REJECTED) if verdict else UNREVIEWED
    )
    return out


def _hz(rate: object) -> str:
    """A control rate as the card says it, or unrecorded."""
    return f"{rate:g} Hz" if isinstance(rate, (int, float)) else UNRECORDED


def _summary_deploy(path: Path) -> dict[str, Any]:
    """A deployment at a glance: the gate's word first (what the card is
    for), then the checkpoint and the control rate."""
    m = _read(path / DEPLOY_FILE)
    gates = read_gates(path)
    out: dict[str, Any] = {
        "gate": gate_word(gates["mujoco"]) if "mujoco" in gates else "not run",
    }
    if "dds" in gates:
        out["gate (DDS)"] = gate_word(gates["dds"])
    out["checkpoint"] = m.get("checkpoint", UNRECORDED)
    out["control"] = _hz((m.get("control") or {}).get("control_hz"))
    return out


def _summary_finding(path: Path) -> dict[str, Any]:
    raw = _read(path)
    first = str(raw.get("claim", "")).split(". ", 1)[0]
    short = first[:CLAIM_CHARS] + ("…" if len(first) > CLAIM_CHARS else "")
    return {"claim": short}  # the id carries the date already


def _summary_drift(path: Path) -> dict[str, Any]:
    """What a card says about a drift check: the word, then who left."""
    d = _read(path / DRIFT_FILE)
    if not d:
        return {}
    left = list(d.get("left") or [])
    unresolved = list(d.get("unresolved") or [])
    out: dict[str, Any] = {"verdict": verdict_word(bool(d.get("drifted")))}
    if left:
        out["left"] = ", ".join(left)
    if unresolved:
        out["unresolved"] = ", ".join(unresolved)
    out["references"] = len(d.get("fit") or [])
    return out


_SUMMARY_READERS: dict[Kind, SummaryReader] = {
    Kind.DRIFT: _summary_drift,
    Kind.RECORDING: _summary_recording,
    Kind.BATCH: lambda p: {"episodes": len(list(p.glob("episode_*")))},
    Kind.RUN: _summary_run,
    Kind.CERTIFICATE: _summary_certificate,
    Kind.FINDING: _summary_finding,
    Kind.DEPLOY: _summary_deploy,
    Kind.POLICY: _summary_policy,
    Kind.ROBOT: lambda p: {
        "files": sorted(e.name for e in p.iterdir() if not e.name.startswith("."))
    },
    Kind.TASK: _summary_task,
}


# The stages a loop kind never passes through, with the reason the
# window shows in their place.
NOT_NEEDED: dict[str, dict[str, str]] = {
    "reinforcement": {
        "data generated": (
            "a reinforcement-learning loop has no dataset: the policy learns "
            "from its own rollouts in simulation"
        ),
    },
}


def _loop_of(artifacts: list[Artifact]) -> str:
    """The loop kind read off the runs when the manifest has not said:
    what the first run says it learned by; empty with no runs."""
    for a in artifacts:
        if a.kind == Kind.RUN.value and a.summary.get("learning") in LOOPS:
            return str(a.summary["learning"])
    return ""


def _states(artifacts: list[Artifact], loop: str = "") -> list[State]:
    by_kind: dict[str, list[str]] = {}
    for a in artifacts:
        by_kind.setdefault(a.kind, []).append(a.stamp)
    skipped = NOT_NEEDED.get(loop, {})
    # A robot bundle that carries fit records proves identification too.
    states: list[State] = []
    for name, kind in STATES:
        proof = list(by_kind.get(kind.value, []))
        if kind is Kind.FIT and not proof:
            proof = [
                a.stamp
                for a in artifacts
                if a.kind == Kind.ROBOT.value and "fits" in a.summary.get("files", [])
            ]
        states.append(
            State(
                name=name,
                proved_by=proof,
                present=bool(proof),
                needed=name not in skipped,
                note=skipped.get(name),
            )
        )
    return states
