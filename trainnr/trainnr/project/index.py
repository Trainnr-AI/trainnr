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
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trainnr.bundles.basis import BASES, BASIS_OWN, BASIS_UNKNOWN
from trainnr.bundles.bundle import BUNDLE_FILE
from trainnr.bundles.hashing import is_stamp
from trainnr.collect.provenance import PROVENANCE_FILE
from trainnr.deploy.attribution import (
    fit_line,
    marked_sensitivity,
    read_attribution,
)
from trainnr.deploy.gate import DRAW_NOW, draw_of
from trainnr.deploy.manifest import gate_word, read_gates
from trainnr.deploy.preflight import (
    PREFLIGHT_SUMMARY_KEY,
    card_line,
    read_preflight,
)
from trainnr.deploy.runtimes import DEFAULT_RUNTIME, RUNTIMES
from trainnr.deploy.viewport_source import VIEWPORT_KEY, scenes_of
from trainnr.envs.lerobot_train_log import RUN_MANIFEST_FILE
from trainnr.envs.rsl_rl_log import COL_REWARD, STATUS_RUNNING, TRAINING_FILE
from trainnr.evaluate.commands import describe_twist
from trainnr.fleet.drift import verdict_word
from trainnr.project.files import read_json, write_json, write_text
from trainnr.project.kinds import (
    ACCEPTANCE_FILE,
    ACCEPTED,
    CERTIFICATE_FILE,
    DEPLOY_FILE,
    DRIFT_FILE,
    FIT_FILE,
    FITS_DIR,
    IDENTITY_FILE,
    POLICY_FILE,
    RECORDING_FILE,
    REJECTED,
    SCENE_FILE,
    TASK_FILE,
    UNREVIEWED,
    Kind,
    UnknownKindError,
    detect,
    stamp_kind,
    stamp_run,
)
from trainnr.project.locate import (
    FOLDERS,
    INDEX_DIR,
    LOOPS,
    RECORDINGS_FOLDER,
    Project,
)
from trainnr.robot.fit_record import SPREAD_FILENAME as SPREAD_FILE
from trainnr.robots.capture import LISTENING, CaptureState
from trainnr.robots.recording import JOINT_POSITION
from trainnr.scenes.record import (
    capture_failed,
    capture_in_progress,
    capture_stage,
    proxy_from_splat,
)
from trainnr.scenes.stage import SCENE_TERRAIN_WORD, scene_name_of

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
        "dataset, a ROS 2 bag as .mcap or rosbag2 .db3, a mocap CSV or BVH), or a "
        "registered public log (ingest_public_log) — or start from its model"
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
    # The same picture drawn in the light palette, for the kinds the
    # presenter paints; the app picks by its theme (2026-10-02).
    preview_light: str | None = None
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
    # Whose robot the proof rests on (`robots.recording.BASES`, the
    # strongest among the proving records): "own robot", "public log",
    # "simulation". None for a stage that has no basis (2026-09-24: a fit
    # of a public Go2 log lights Sys ID and must say so).
    basis: str | None = None


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
    # Folders a chain is still filling (a scene being captured): the path
    # and the chain's latest stage - shown as work, not warned about.
    in_progress: list[dict[str, str]] = field(default_factory=list)

    def by_kind(self, kind: Kind) -> list[Artifact]:
        return [a for a in self.artifacts if a.kind == kind.value]


INDEX_SCHEMA = "trainnr-project-index/1"


def index_project(project: Project) -> ProjectIndex:
    manifest = project.manifest()
    artifacts: list[Artifact] = []
    refused: list[dict[str, str]] = []
    in_progress: list[dict[str, str]] = []
    for folder in FOLDERS:
        for path in _candidates(project.root / folder):
            if capture_in_progress(path):
                relative = path.relative_to(project.root).as_posix()
                failed = capture_failed(path)
                if failed is not None:  # a dead chain is a refusal, not work
                    reason = f"capture failed: {failed}"
                    refused.append({"path": relative, "reason": reason})
                else:
                    stage = capture_stage(path)
                    in_progress.append(
                        {"path": relative, "stage": stage, "kind": Kind.SCENE.value}
                    )
                continue
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
                relative = path.relative_to(project.root).as_posix()
                refused.append(
                    {
                        "path": relative,
                        "reason": str(why),
                    }
                )
                continue
            created, updated = _times(kind, path)
            artifacts.append(
                Artifact(
                    kind=kind.value,
                    stamp=identity,
                    path=path.relative_to(project.root).as_posix(),
                    cites=_cites(kind, path),
                    summary=_summary(kind, path),
                    created=created,
                    updated=updated,
                )
            )
    _link_cited_by(artifacts)
    loop = manifest.loop or _loop_of(artifacts, project.root)
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
        in_progress=in_progress + _live_capture(project),
    )


def _live_capture(project: Project) -> list[dict[str, str]]:
    """A capture listening right now, as work in progress: the recording
    it will become and how much has arrived, per topic when there are
    topics (the listener's own state file, `robots.capture`)."""
    state = CaptureState.read(project.root / INDEX_DIR)
    if state.state != LISTENING or not state.name:
        return []
    arrived = (
        ", ".join(f"{topic} {n}" for topic, n in sorted(state.topics.items()))
        if state.topics
        else f"{state.datagrams} messages"
    )
    return [
        {
            "path": f"{RECORDINGS_FOLDER}/{state.name}",
            "stage": f"capturing on {state.source}: {arrived}",
            "kind": Kind.RECORDING.value,
        }
    ]


def write_index(
    project: Project,
    index: ProjectIndex | None = None,
    *,
    previews: bool = True,
) -> Path:
    """Write the index; with `previews`, render each artifact's picture
    first, in both palettes, and record both paths on the artifact (a
    cache keyed by stamp and palette, so unchanged artifacts are never
    re-rendered); the app shows the set of its theme, instantly."""
    index = index if index is not None else index_project(project)
    if previews:
        from trainnr.project.details import write_details  # noqa: PLC0415
        from trainnr.project.previews import write_previews  # noqa: PLC0415

        found = write_previews(project, index, "dark")
        found_light = write_previews(project, index, "light")
        details = write_details(project, index)
        index = replace(
            index,
            artifacts=[
                replace(
                    a,
                    preview=found.get(a.stamp),
                    preview_light=found_light.get(a.stamp),
                    detail=details.get(a.stamp),
                )
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


def _cites_deploy(raw: dict[str, Any]) -> dict[str, str]:
    """A deployment's lineage; a staged one (docs/78 §8.1) also cites the
    scene it stands on, by the version its terrain word carries."""
    cites = _pick(raw, ("policy", "run", "robot", "task", "certificate"))
    terrain = str((raw.get("scene") or {}).get("terrain") or "")
    word, _, rest = terrain.partition(" ")
    if word == SCENE_TERRAIN_WORD and rest:
        cites["scene"] = rest
    return cites


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
    Kind.DEPLOY: lambda p: _cites_deploy(_read(p / DEPLOY_FILE)),
    # A check cites the robot (its version carries the fit records it was
    # judged against; the drawer names them) and the recording it judged.
    Kind.DRIFT: lambda p: _pick(_read(p / DRIFT_FILE), ("robot", "recording")),
}


HEADLINE_KEY = "headline"  # the card's one line; a table column never
# The words a card uses: what a person at Weights & Biases, Isaac Lab or
# LeRobot would read without a glossary (2026-09-28).
SEP = " · "


def _summary(kind: Kind, path: Path) -> dict[str, Any]:
    """A few glanceable facts for the Studio's list rows — never the
    whole record; the detail view reads the artifact itself. The first
    key is `headline`: the card's one line, in plain words; the rest are
    the table's columns."""
    reader = _SUMMARY_READERS.get(kind)
    out = reader(path) if reader is not None else {}
    headline = _headline(kind, out)
    return {HEADLINE_KEY: headline, **out} if headline else out


def _join(*parts: object) -> str:
    return SEP.join(str(x) for x in parts if x not in (None, "", UNRECORDED))


def _headline(kind: Kind, s: dict[str, Any]) -> str:  # noqa: PLR0911 - one line per kind
    """The card's line from the summary's facts."""
    if kind is Kind.RUN:
        iterations = s.get("iterations")
        return _join(
            s.get("learning") if s.get("learning") == "imitation" else "PPO",
            f"{iterations:,} iterations" if isinstance(iterations, int) else None,
            f"reward {s['final reward']}" if "final reward" in s else None,
            s.get("progress") or s.get("status"),
        )
    if kind is Kind.CERTIFICATE:
        trials = f"{s['trials']} trials" if "trials" in s else None
        return _join(s.get("condition"), trials)
    if kind is Kind.POLICY:
        return _join(
            s.get("checkpoint"),
            f"from {s['run']}" if s.get("run") else None,
            s.get("randomization"),
        )
    if kind is Kind.DEPLOY:
        preflight = s.get(PREFLIGHT_SUMMARY_KEY)
        return _join(
            f"evaluated {s[EVALUATED_KEY]}" if s.get(EVALUATED_KEY) else None,
            f"gate {s['gate']}" if s.get("gate") else None,
            f"pre-flight {preflight}" if preflight else None,
            f"on {s['scene']}" if s.get("scene") else None,
        )
    if kind is Kind.DRIFT:
        n_out = s.get("out of interval count", 0)
        n_und = s.get("undetermined count", 0)
        return _join(
            s.get("verdict"),
            f"{n_out} out of interval" if n_out else None,
            f"{n_und} undetermined" if n_und else None,
        )
    if kind is Kind.SCENE:
        splats = s.get("splats")
        return _join(
            "Gaussian splat",
            f"{splats:,} points" if isinstance(splats, int) else None,
            f"gap p95 {s['gap p95']}" if s.get("gap p95") else None,
        )
    if kind is Kind.RECORDING:
        channels = f"{s['channels']} channels" if "channels" in s else None
        return _join(s.get("duration"), s.get("rate"), channels, s.get("origin"))
    if kind is Kind.ROBOT:
        return _join(
            f"{s['joints']} joints" if "joints" in s else None,
            f"{s['dof']} DoF" if "dof" in s else None,
            s.get("format"),
        )
    if kind is Kind.TASK:
        return _join(s.get("template"), s.get("robot"), s.get("acceptance"))
    return ""


def _take(raw: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {k: raw[k] for k in keys if k in raw}


def _summary_run(path: Path) -> dict[str, Any]:
    run = _read(path / RUN_MANIFEST_FILE)
    if run:
        return {"learning": "imitation", **_take(run, ("policy", "steps", "started"))}
    training = _read(path / TRAINING_FILE)
    ident = _read(path / IDENTITY_FILE)
    # `learning` stays for the loop kind (`_loop_of_runs`); the table hides it.
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
        out["randomization"] = randomization_label(basis)
    out.update(_trained_under(ident))
    if "seed" in ident:
        out["seed"] = ident["seed"]
    return out


def randomization_label(basis: str) -> str:
    """The domain-randomization basis in a few words, as a column reads:
    `declared ±10 %` for a span around the vendor's constants, `fit
    0ad6202797c5` for a measured fit's intervals; the whole sentence
    stays in the record."""
    head = basis.split(":", 1)[0].split(" (", 1)[0].strip()
    fit = re.search(r"fit@([0-9a-f]{6,})", basis)
    if fit:
        return f"fit {fit.group(1)}"
    scale = re.search(r"±\s*([0-9.]+)\s*scale", head)
    if head.startswith("declared") and scale:
        return f"declared ±{float(scale.group(1)) * 100:g} %"
    return head


TRAINED_UNDER_KEY = "trained under"  # the card's word for a run's fit


def _trained_under(identity: dict[str, Any]) -> dict[str, str]:
    """`trained under: fit@... (public log)` when a run, a policy or a
    deployment trained under a measured fit; nothing when it trained on
    declared numbers (2026-09-25)."""
    fit = identity.get("fit")
    if not fit:
        return {}
    return {TRAINED_UNDER_KEY: f"{fit} ({identity.get('fit_basis') or UNRECORDED})"}


def _summary_recording(path: Path) -> dict[str, Any]:
    raw = _read(path / RECORDING_FILE)
    out: dict[str, Any] = {}
    duration = raw.get("duration_s")
    if isinstance(duration, (int, float)):
        out["duration"] = duration_text(float(duration))
    rate = _measured_rate(raw)
    if rate is not None:
        out["rate"] = f"{rate:g} Hz"
    if "channels" in raw:
        out["channels"] = len(raw["channels"])
    if raw.get("adapter"):
        out["format"] = raw["adapter"]
    if raw.get("collection"):
        out["collection"] = raw["collection"]
    basis = raw.get("basis") or BASIS_UNKNOWN
    if basis != BASIS_OWN:
        # Whose robot: a public log names it; the operator's own says nothing.
        out[BASIS_KEY] = basis  # the loop map's chip reads it; the table hides it
        out["origin"] = basis
        out.update(_take(raw.get("provenance") or {}, ("robot", "licence")))
    if raw.get("source"):
        out["source"] = raw["source"]
    return out


def duration_text(seconds: float) -> str:
    """`24 s`, `12 min`, `1 h 05 min`: the way a recording's length is read."""
    if seconds < 10:  # noqa: PLR2004 - a short clip keeps its tenths
        return f"{seconds:.1f} s"
    if seconds < 120:  # noqa: PLR2004 - under two minutes stays in seconds
        return f"{seconds:.0f} s"
    minutes = round(seconds / 60)
    if minutes < 60:  # noqa: PLR2004
        return f"{minutes} min"
    return f"{minutes // 60} h {minutes % 60:02d} min"


def _measured_rate(raw: dict[str, Any]) -> float | None:
    """The joint-position channel's measured rate, for the card."""
    for channel in raw.get("channels") or []:
        if channel.get("name") == JOINT_POSITION and channel.get("rate_hz"):
            return round(float(channel["rate_hz"]), 1)
    return None


def _summary_certificate(path: Path) -> dict[str, Any]:
    """What a card says about an evaluation: the rate and its interval."""
    c = _read(path / CERTIFICATE_FILE)
    out: dict[str, Any] = {}
    # The envelope first: the card's subtitle is cut short, and the
    # envelope is what tells two judgments of one checkpoint apart.
    protocol = c.get("protocol") or {}
    judged = protocol.get("judged_at")
    out["condition"] = condition_label(judged, path.name, protocol.get("delay"))
    if "successes" in c and "trials" in c:
        out["success"] = ratio_of(c)
        out["trials"] = c["trials"]
    interval = interval_of(c)
    if interval is not None:
        out["95% CI"] = f"[{interval[0]:.2f}, {interval[1]:.2f}]"
    commands = describe_twist(protocol.get("commands"))
    if commands:
        out["commands"] = commands
    return out


# What a judgment's condition says on a card, a column and the matrix's
# header: the world and the perturbation in a few words. The recorded
# sentence (`judged_at`) stays in the drawer's Protocol section.
NOMINAL_CONDITION = "nominal"


def condition_label(judged: object, name: str = "", delay: object = None) -> str:
    """`judged_at` folded to a few words: `in fit world`, `cross-eval →
    declared world`, `kp x0.8`, `delay 2 ticks`; joined with ` · `."""
    text = str(judged or "")
    parts: list[str] = []
    cross = re.match(r"CROSS-evaluation: trained in ([^,]+), judged in ([^;]+)", text)
    if cross:
        world = "fit" if cross.group(2).startswith("fit") else "declared"
        parts.append(f"cross-eval → {world} world")
    elif re.search(r"law DR: fit", text):
        parts.append("in fit world")
    elif re.search(r"law DR: declared", text):
        parts.append("in declared world")
    knob = re.search(r"(kp|kd|armature) at fit x ([0-9.]+)", text)
    if knob:
        parts.append(f"{knob.group(1)} x{knob.group(2)}")
    ticks = delay if isinstance(delay, int) else None
    if ticks is None:
        m = re.search(r"-delay-(\d+)-", name)
        ticks = int(m.group(1)) if m else None
    if ticks:
        parts.append(f"delay {ticks} tick{'s' if ticks != 1 else ''}")
    return SEP.join(parts) if parts else NOMINAL_CONDITION


def _basis_name(basis: str) -> str:
    """The domain-randomization basis in a few words: the part before its
    colon or its parenthesis ("identified-set", "caller-declared span ±0.1")."""
    return basis.split(":", 1)[0].split(" (", 1)[0].strip()


def _summary_policy(path: Path) -> dict[str, Any]:
    manifest = _read(path / POLICY_FILE)
    if manifest:
        basis = manifest.get("dr_basis") or ""
        iterations = manifest.get("iterations")
        out: dict[str, Any] = {}
        if manifest.get("checkpoint"):
            out["checkpoint"] = manifest["checkpoint"]
        if manifest.get("run"):
            out["run"] = manifest["run"]
        out["iterations"] = UNRECORDED if iterations is None else iterations
        out["randomization"] = randomization_label(basis) if basis else UNRECORDED
        out.update(_trained_under(manifest))
        out["format"] = manifest.get("format", "")
        return out
    identity = _read(path / IDENTITY_FILE)
    return {**_take(identity, ("dr_basis", "seed")), **_trained_under(identity)}


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
    if kind in (Kind.DRIFT, Kind.SCENE):
        marker = DRIFT_FILE if kind is Kind.DRIFT else SCENE_FILE
        created = _read(path / marker).get("created_utc")
        if isinstance(created, str) and created:
            return created
    if kind is Kind.ROBOT:
        # Onboarding copies the source tree with its dates; the bundle
        # record is written at onboarding, so its time is when the robot
        # joined (a robot onboarded today read "added 2026-09-10" before
        # 2026-09-28).
        record = path / BUNDLE_FILE
        if record.is_file():
            with contextlib.suppress(OSError):
                return _iso(record.stat().st_mtime)
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
    raw = _read(path / TASK_FILE)
    out: dict[str, Any] = {}
    if raw.get("task_id"):
        out["template"] = str(raw["task_id"]).split("/", 1)[-1]
    spec_raw = raw.get("spec")
    spec: dict[str, Any] = spec_raw if isinstance(spec_raw, dict) else {}
    robot = raw.get("robot") or spec.get("robot") or raw.get("rig")
    if robot:
        out["robot"] = robot
    verdict = _read(path / ACCEPTANCE_FILE)
    out["acceptance"] = (
        (ACCEPTED if verdict.get("accepted") else REJECTED) if verdict else UNREVIEWED
    )
    return out


def _hz(rate: object) -> str:
    """A control rate as the card says it, or unrecorded."""
    return f"{rate:g} Hz" if isinstance(rate, (int, float)) else UNRECORDED


# The card's word for a gate drawn before the per-trial draw
# (`deploy.gate.DRAW_NOW`): its trials are not the ones a re-run would draw.
OLD_DRAW = "{word}, drawn by count (re-run to pair)"
# What the deployment card's picture shows when it is not the robot.
PICTURE_KEY = "picture"
CLIFF_PICTURE = "the fall at the cliff, {knob} {level:g}"


# A deployment card's first fact: the success rate of the evaluation it cites.
EVALUATED_KEY = "evaluated"


def _gate_card_word(record: dict[str, Any]) -> str:
    word = gate_word(record)
    return word if draw_of(record) == DRAW_NOW else OLD_DRAW.format(word=word)


def _summary_deploy(path: Path) -> dict[str, Any]:
    """A deployment at a glance: the gate's word first (what the card is
    for), then the checkpoint and the control rate."""
    m = _read(path / DEPLOY_FILE)
    gates = read_gates(path)
    out: dict[str, Any] = {}
    # How well the policy does, first: a gate passes when the export
    # reproduces its evaluation, so "gate passed" alone read as ready for
    # a robot on a policy that succeeded 0 of 8 times (2026-10-04).
    rate = ((gates.get(DEFAULT_RUNTIME) or {}).get("verdict") or {}).get(
        "certificate_rate"
    )
    if isinstance(rate, (int, float)):
        out[EVALUATED_KEY] = f"{rate:.0%} success"
    out["gate"] = (
        _gate_card_word(gates[DEFAULT_RUNTIME])
        if DEFAULT_RUNTIME in gates
        else "not run"
    )
    for runtime in RUNTIMES:
        if runtime != DEFAULT_RUNTIME and runtime in gates:
            out[f"gate ({runtime.upper()})"] = _gate_card_word(gates[runtime])
    preflight = read_preflight(path)
    if preflight:  # before the first tick on a robot: passed, or refused by name
        out[PREFLIGHT_SUMMARY_KEY] = card_line(preflight)
    out.update(_trained_under(m))  # the fit the shipped policy trained under
    attribution = read_attribution(path)
    if attribution:  # what would break it first, right under the gate's word
        # marked when ranked under another cliff rule, or drawn by count
        sensitivity = marked_sensitivity(attribution) or UNRECORDED
        out["sensitivity"] = (
            sensitivity
            if draw_of(attribution) == DRAW_NOW
            else OLD_DRAW.format(word=sensitivity)
        )
        at_fit = fit_line(attribution)
        if at_fit is not None:  # the robot as it was measured: holds or not
            out["measured joints"] = at_fit
        still = attribution.get("still") or {}
        if still.get("file"):  # the card's picture is then the fall, not the robot
            out[PICTURE_KEY] = CLIFF_PICTURE.format(
                knob=still.get("knob", UNRECORDED), level=still.get("level", "")
            )
    scene = scene_name_of(m)
    if scene:  # a staged deployment: what it stands on, before the rest
        out["scene"] = scene
        out["terrain"] = (m.get("scene") or {}).get("terrain_kind", UNRECORDED)
    out["checkpoint"] = m.get("checkpoint", UNRECORDED)
    out["control"] = _hz((m.get("control") or {}).get("control_hz"))
    # What the Studio's MuJoCo viewport can show of it: live, each gate
    # trial, each pre-flight segment (hidden on the card; the drawer reads it).
    out[VIEWPORT_KEY] = scenes_of(path)
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
    out["out of interval count"] = len(left)
    out["undetermined count"] = len(unresolved)
    if left:
        out["out of interval"] = ", ".join(left)
    if unresolved:
        out["undetermined"] = ", ".join(unresolved)
    out["reference fits"] = len(d.get("fit") or [])
    return out


def _summary_scene(path: Path) -> dict[str, Any]:
    """What a card says about a scene: the gap first, then what it holds."""
    s = _read(path / SCENE_FILE)
    if not s:
        return {}
    gap = s.get("gap") or {}
    p95 = gap.get("p95_m")
    # a proxy built from the splat itself makes the gap self-referential;
    # the card says so beside the number (docs/78 §8.6)
    own = proxy_from_splat(s)
    out: dict[str, Any] = {
        "gap p95": (f"{p95 * 100:.1f} cm" + (" (proxy from the splat)" if own else ""))
        if isinstance(p95, (int, float))
        else UNRECORDED,
        "splats": (s.get("splat") or {}).get("count", UNRECORDED),
        "source": s.get("source", UNRECORDED),
    }
    declared = [
        p.get("name") for p in s.get("physics", []) if p.get("basis") == "declared"
    ]
    if declared:
        out["declared"] = ", ".join(declared)
    return out


def _summary_robot(p: Path) -> dict[str, Any]:
    """The bundle's files, and what its importer changed (the audit's one
    line; absent for a bundle onboarded before the audit existed)."""
    from trainnr.bundles.bundle import read_audit  # noqa: PLC0415

    out: dict[str, Any] = {
        "files": sorted(e.name for e in p.iterdir() if not e.name.startswith(".")),
        "fit_bases": fit_bases(p / FITS_DIR),
    }
    bundle = _read(p / BUNDLE_FILE)
    census = bundle.get("census") or {}
    if isinstance(census.get("joints"), int):
        out["joints"] = census["joints"]
    if isinstance(census.get("dofs"), int):
        out["dof"] = census["dofs"]
    model_file = bundle.get("model_file") or ""
    if model_file:
        out["format"] = "MJCF" if str(model_file).endswith(".xml") else "USD"
    audit = read_audit(p)
    if audit and audit.get("summary"):
        out["audit"] = audit["summary"]
    return out


# The summary facts every row carries for the index's own use and no card
# shows (`model.rs::HIDDEN_KEYS`, pinned by test_studio_mirrors).
BASIS_KEY = "basis"
HIDDEN_SUMMARY_KEYS = (
    "files",
    "fit_bases",
    VIEWPORT_KEY,
    HEADLINE_KEY,
    "learning",
    BASIS_KEY,
    TRAINED_UNDER_KEY,  # the randomization column says fit or declared already
    "out of interval count",
    "undetermined count",
)


def _summary_fit(p: Path) -> dict[str, Any]:
    """Whose robot a fit record measured (`bundles.basis`): a fit folder's
    `fit.json`, or a single-file record under `fits/`; a record written
    before the field is unknown."""
    raw = _read(p / FIT_FILE) if p.is_dir() else _read(p)
    return {BASIS_KEY: raw.get(BASIS_KEY) or BASIS_UNKNOWN}


_SUMMARY_READERS: dict[Kind, SummaryReader] = {
    Kind.SCENE: _summary_scene,
    Kind.FIT: _summary_fit,
    Kind.DRIFT: _summary_drift,
    Kind.RECORDING: _summary_recording,
    Kind.BATCH: lambda p: {"episodes": len(list(p.glob("episode_*")))},
    Kind.RUN: _summary_run,
    Kind.CERTIFICATE: _summary_certificate,
    Kind.FINDING: _summary_finding,
    Kind.DEPLOY: _summary_deploy,
    Kind.POLICY: _summary_policy,
    Kind.ROBOT: _summary_robot,
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


def _loop_of(artifacts: list[Artifact], root: Path | None = None) -> str:
    """The loop kind when the manifest has not said: what the first run
    says it learned by, else `reinforcement` as soon as a declared task is
    a walk family (a project made without `loop` kept the Dataset stage
    needed and "generate demonstrations" as the next move with a Go2 walk
    declared, stranger test 2026-10-03); empty with neither."""
    for a in artifacts:
        if a.kind == Kind.RUN.value and a.summary.get("learning") in LOOPS:
            return str(a.summary["learning"])
    if root is not None and any(
        a.kind == Kind.TASK.value and _is_walk_task(root / a.path) for a in artifacts
    ):
        return "reinforcement"
    return ""


def _is_walk_task(path: Path) -> bool:
    """Whether the declared task at `path` is a walk family (the task
    registry's `walk` mark); an unknown family is not."""
    # the registry loads plugins; imported here, not at module import
    from trainnr.tasks.registry import resolve  # noqa: PLC0415

    task_id = str(_read(path / TASK_FILE).get("task_id") or "")
    if not task_id:
        return False
    try:
        return bool(resolve(task_id).walk)
    except (KeyError, ValueError):
        return False


def fit_bases(fits: Path) -> list[str]:
    """The `basis` words of the fit records under `fits`, distinct, in
    BASES order; a record written before the field counts as unknown."""
    import json  # noqa: PLC0415

    if not fits.is_dir():
        return []
    found = set()
    for path in sorted(fits.glob("*.json")):
        if path.name == SPREAD_FILE:
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        found.add(raw.get("basis") or BASIS_UNKNOWN)
    return [basis for basis in BASES if basis in found]


# The kinds whose summaries say whose robot proved them, and what their
# silence means: a recording's summary names the basis only when it is not
# the operator's own. Every other stage has no basis (None): a policy, a
# certificate or a deployment is not a measurement of a robot.
SILENT_BASIS: dict[Kind, str] = {Kind.RECORDING: BASIS_OWN}


def strongest_basis(bases: list[str]) -> str | None:
    """The first of BASES present: an own-robot fit outranks a public
    log, which outranks a simulation."""
    return next((basis for basis in BASES if basis in bases), None)


def _states(artifacts: list[Artifact], loop: str = "") -> list[State]:
    by_kind: dict[str, list[str]] = {}
    for a in artifacts:
        by_kind.setdefault(a.kind, []).append(a.stamp)
    skipped = NOT_NEEDED.get(loop, {})
    # A robot bundle that carries fit records proves identification too,
    # and its records say whose robot they measured.
    states: list[State] = []
    for name, kind in STATES:
        proof = list(by_kind.get(kind.value, []))
        basis = None
        if kind is Kind.FIT:
            carriers = [
                a
                for a in artifacts
                if a.kind == Kind.ROBOT.value and FITS_DIR in a.summary.get("files", [])
            ]
            fits = [a for a in artifacts if a.kind == kind.value and a.stamp in proof]
            if not proof:
                proof = [a.stamp for a in carriers]
            found = [b for a in carriers for b in a.summary.get("fit_bases", [])]
            found += [str(a.summary.get(BASIS_KEY, BASIS_UNKNOWN)) for a in fits]
            basis = strongest_basis(found) if proof else None
        elif kind in SILENT_BASIS and proof:
            # A kind whose summary says whose robot it was; its silence means
            # SILENT_BASIS's word (a recording says nothing when it is ours).
            found = [
                str(a.summary.get(BASIS_KEY, SILENT_BASIS[kind]))
                for a in artifacts
                if a.kind == kind.value and a.stamp in proof
            ]
            basis = strongest_basis(found)
        states.append(
            State(
                name=name,
                proved_by=proof,
                present=bool(proof),
                needed=name not in skipped,
                note=skipped.get(name),
                basis=basis,
            )
        )
    return states
