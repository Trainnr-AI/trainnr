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

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rq_pipeline.bundles.hashing import is_stamp
from rq_pipeline.collect.provenance import PROVENANCE_FILE
from rq_pipeline.envs.lerobot_train_log import RUN_MANIFEST_FILE
from rq_pipeline.project.kinds import (
    CERTIFICATE_FILE,
    DEPLOY_FILE,
    DRIFT_FILE,
    IDENTITY_FILE,
    TASK_FILE,
    Kind,
    UnknownKindError,
    detect,
    stamp_kind,
)
from rq_pipeline.project.locate import FOLDERS, Project

UNRECORDED = "unrecorded"

# The loop's states, in order, and the kind whose presence proves each
# (docs/76 §3). `identified` is proved by a fit OR by a robot bundle that
# carries fit records; the checker below handles the second case.
STATES: tuple[tuple[str, Kind], ...] = (
    ("telemetry ingested", Kind.RECORDING),
    ("robot known", Kind.ROBOT),
    ("dynamics identified", Kind.FIT),
    ("task declared", Kind.TASK),
    ("data pressed", Kind.BATCH),
    ("policy trained", Kind.RUN),
    ("policy certified", Kind.CERTIFICATE),
    ("deployable", Kind.DEPLOY),
    ("loop watched", Kind.DRIFT),
)

# What an agent does next when a state is the first missing one.
NEXT_MOVE: dict[str, str] = {
    "telemetry ingested": (
        "ingest a recording of the robot (ingest_recording: a .wire file, a "
        "LeRobot dataset, a ROS 2 .mcap bag) — or skip to onboarding its model"
    ),
    "robot known": "onboard a robot bundle (onboard_robot)",
    "dynamics identified": (
        "identify dynamics from a recording, or name a certified actuator bundle"
    ),
    "task declared": "declare a task spec and run acceptance",
    "data pressed": (
        "press demonstrations (generate_demos / press_planned / press_walk)"
    ),
    "policy trained": "train a policy on the pressed data (run_chain / train_walk)",
    "policy certified": "certify the policy with paired trials",
    "deployable": "export a deploy manifest and pass the sim2sim gate",
    "loop watched": "ingest fresh telemetry and run the drift check",
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


@dataclass(frozen=True)
class State:
    name: str
    proved_by: list[str]  # stamps
    present: bool


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
                identity = stamp_kind(kind, path, name=label)
            except UnknownKindError as why:
                refused.append(
                    {"path": str(path.relative_to(project.root)), "reason": str(why)}
                )
                continue
            artifacts.append(
                Artifact(
                    kind=kind.value,
                    stamp=identity,
                    path=str(path.relative_to(project.root)),
                    cites=_cites(kind, path),
                    summary=_summary(kind, path),
                )
            )
    states = _states(artifacts)
    missing = [s.name for s in states if not s.present]
    # Telemetry is the loop's first state but not a prerequisite: a robot
    # that entered as a model (a Menagerie bundle) has no recording yet
    # and does not need one before a task is declared. The next move is
    # the first missing state AFTER the first proved one.
    first_proved = next((i for i, s in enumerate(states) if s.present), None)
    if first_proved is not None:
        missing = [s.name for s in states[first_proved:] if not s.present]
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
    from dataclasses import replace  # noqa: PLC0415

    index = index if index is not None else index_project(project)
    if previews:
        from rq_pipeline.project.previews import write_previews  # noqa: PLC0415

        found = write_previews(project, index)
        index = replace(
            index,
            artifacts=[replace(a, preview=found.get(a.stamp)) for a in index.artifacts],
        )
    out = project.index_path
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.with_name(out.name + ".tmp")
    staging.write_text(json.dumps(asdict(index), indent=1) + "\n", encoding="utf-8")
    staging.replace(out)
    (out.parent / ".gitignore").write_text("*\n")
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
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


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


_CITE_READERS: dict[Kind, Any] = {
    Kind.DATASET: _cites_dataset,
    Kind.RUN: _cites_run,
    Kind.BATCH: _cites_batch,
    Kind.CERTIFICATE: lambda p: _pick(
        _read(p / CERTIFICATE_FILE), ("robot", "task", "policy", "source")
    ),
    Kind.DEPLOY: lambda p: _pick(
        _read(p / DEPLOY_FILE), ("policy", "robot", "task", "certificate")
    ),
    Kind.DRIFT: lambda p: _pick(_read(p / DRIFT_FILE), ("robot", "recording", "fit")),
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
        return _take(run, ("policy", "steps", "started"))
    return _take(_read(path / IDENTITY_FILE), ("dr_basis", "seed"))


def _summary_recording(path: Path) -> dict[str, Any]:
    from rq_pipeline.project.kinds import RECORDING_FILE  # noqa: PLC0415

    raw = _read(path / RECORDING_FILE)
    out = _take(raw, ("adapter", "source", "duration_s"))
    if "channels" in raw:
        out["channels"] = len(raw["channels"])
    return out


_SUMMARY_READERS: dict[Kind, Any] = {
    Kind.RECORDING: _summary_recording,
    Kind.BATCH: lambda p: {"episodes": len(list(p.glob("episode_*")))},
    Kind.RUN: _summary_run,
    Kind.CERTIFICATE: lambda p: _take(
        _read(p / CERTIFICATE_FILE), ("successes", "trials", "ci95", "instrument")
    ),
    Kind.FINDING: lambda p: _take(_read(p), ("id", "date", "claim")),
    Kind.ROBOT: lambda p: {
        "files": sorted(e.name for e in p.iterdir() if not e.name.startswith("."))
    },
    Kind.TASK: lambda p: _take(_read(p / TASK_FILE), ("task_id", "stamp", "kind")),
}


def _states(artifacts: list[Artifact]) -> list[State]:
    by_kind: dict[str, list[str]] = {}
    for a in artifacts:
        by_kind.setdefault(a.kind, []).append(a.stamp)
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
        states.append(State(name=name, proved_by=proof, present=bool(proof)))
    return states
