"""What an artifact IS, decided by the file at its root — never guessed.

Every kind has its markers: the file the pipeline already writes at that
artifact's root (`datasheet.md` for a generated dataset, `run.json` for a
training run's watch directory, `identity.json` for an RL run, and so
on). The marker constants are IMPORTED from the modules that write them,
so a rename there is a rename here. An artifact with no marker, or with
two, is refused by name: a registry that guesses a kind would attach the
wrong lineage to it, silently.

The rules are one table (`RULES`, `FILE_RULES`): a new kind is a row,
not a branch in `detect`.

`stamp_kind` is the one stamping door: check the marker, then hash
through `bundles/hashing.stamp`. Downstream code never invents a name.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from rq_pipeline.bundles.hashing import (
    STAMP_LENGTH,
    STAMP_SEPARATOR,
    bundle_hash,
    content_stamp,
    stamp,
)
from rq_pipeline.collect.datasheet import DATASHEET_FILE
from rq_pipeline.collect.provenance import PROVENANCE_FILE
from rq_pipeline.deploy.manifest import MANIFEST_FILE as DEPLOY_FILE
from rq_pipeline.envs.lerobot_train_log import CHECKPOINT_WEIGHTS, RUN_MANIFEST_FILE
from rq_pipeline.fleet.drift import DRIFT_FILE as _DRIFT_FILE
from rq_pipeline.project.files import read_json
from rq_pipeline.robots.recording import MANIFEST_FILE as RECORDING_FILE
from rq_pipeline.scenes.record import SCENE_FILE

# Markers this module owns because no writer exists yet (docs/76 §2):
# a drift record. The rest are imported from their writers above or
# named here for the writers in this package.
CERTIFICATE_FILE = "certificate.json"  # `project/importer.write_certificate`
DRIFT_FILE = _DRIFT_FILE
FIT_FILE = "fit.json"
TASK_FILE = "task.json"  # `project/task_ref.write_task_reference`
# A policy's manifest beside its weights (`project/importer.write_policy`);
# the weights are the marker, the manifest carries the facts.
POLICY_FILE = "policy.json"
# The critic's verdict beside a task (`tools/accept-task.py --project`).
ACCEPTANCE_FILE = "acceptance.json"
ACCEPTANCE_SCHEMA = "trainnr-acceptance/1"
# The three words a task's review can come back as, on cards and drawers.
ACCEPTED, REJECTED, UNREVIEWED = "accepted", "rejected", "unreviewed"
IDENTITY_FILE = "identity.json"  # rq_mjlab's walk run identity (walk_train.py)
CHECKPOINT_SUFFIX = ".pt"  # rsl_rl's checkpoints (`model_<iteration>.pt`)
FINDINGS_DIR = "findings"  # single-file findings live under this folder name
FITS_DIR = "fits"  # single-file fit records under a bundle or a project


class Kind(str, Enum):
    ROBOT = "robot"
    RECORDING = "recording"
    FIT = "fit"
    TASK = "task"
    BATCH = "batch"
    DATASET = "dataset"
    RUN = "run"
    POLICY = "policy"
    CERTIFICATE = "certificate"
    DEPLOY = "deploy"
    DRIFT = "drift"
    SCENE = "scene"  # a captured scene: splat, proxy, the gap (docs/78 §3)
    FINDING = "finding"


@dataclass(frozen=True)
class Marker:
    """One way a directory proves its kind: a file of this name at the
    root, or any file with this suffix. A `fallback` marker counts only
    when no named-file marker of any kind matched — a robot bundle is an
    MJCF at the root (`profile.json` is optional: Menagerie wraps carry
    none), and a bare checkpoint directory is a policy."""

    file: str | None = None
    suffix: str | None = None
    fallback: bool = False

    def matches(self, names: Iterable[str]) -> bool:
        names = list(names)
        if self.file is not None:
            return self.file in names
        if self.suffix is not None:
            return any(n.endswith(self.suffix) for n in names)
        return False


ROBOT_MODEL_SUFFIX = ".xml"

# One row per kind; the first marker of each row is the one the kind is
# named by in a refusal.
RULES: dict[Kind, tuple[Marker, ...]] = {
    Kind.RECORDING: (Marker(file=RECORDING_FILE),),
    Kind.FIT: (Marker(file=FIT_FILE),),
    Kind.TASK: (Marker(file=TASK_FILE),),
    Kind.BATCH: (Marker(file=DATASHEET_FILE),),
    Kind.DATASET: (Marker(file=PROVENANCE_FILE),),
    # A chain run carries `run.json`; an RL run carries `identity.json`
    # beside its checkpoint.
    Kind.RUN: (Marker(file=RUN_MANIFEST_FILE), Marker(file=IDENTITY_FILE)),
    Kind.CERTIFICATE: (Marker(file=CERTIFICATE_FILE),),
    Kind.DEPLOY: (Marker(file=DEPLOY_FILE),),
    Kind.DRIFT: (Marker(file=DRIFT_FILE),),
    Kind.SCENE: (Marker(file=SCENE_FILE),),
    Kind.ROBOT: (Marker(suffix=ROBOT_MODEL_SUFFIX, fallback=True),),
    Kind.POLICY: (
        Marker(file=CHECKPOINT_WEIGHTS, fallback=True),
        Marker(suffix=CHECKPOINT_SUFFIX, fallback=True),
    ),
}
# The marker each kind is named by (the first of its row).
MARKERS: dict[Kind, str] = {
    kind: (rules[0].file or rules[0].suffix or "")
    for kind, rules in RULES.items()
    if not rules[0].fallback
}


@dataclass(frozen=True)
class FileMarker:
    """How a single file proves its kind: its name, its suffix, or its
    suffix inside a folder of this name."""

    kind: Kind
    name: str | None = None
    suffix: str | None = None
    parent: str | None = None

    def matches(self, path: Path) -> bool:
        if self.name is not None and path.name == self.name:
            return True
        if self.suffix is None or path.suffix != self.suffix:
            return False
        return self.parent is None or path.parent.name == self.parent


FILE_RULES: tuple[FileMarker, ...] = (
    FileMarker(Kind.POLICY, name=CHECKPOINT_WEIGHTS),
    FileMarker(Kind.POLICY, suffix=CHECKPOINT_SUFFIX),
    FileMarker(Kind.FINDING, suffix=".json", parent=FINDINGS_DIR),
    FileMarker(Kind.FIT, name=FIT_FILE),
    FileMarker(Kind.FIT, suffix=".json", parent=FITS_DIR),
)


class UnknownKindError(ValueError):
    """No marker, or more than one — refused rather than guessed."""


def detect(root: Path) -> Kind:
    """The kind of the artifact at `root`, by its marker."""
    root = Path(root)
    if root.is_file():
        return _detect_file(root)
    if not root.is_dir():
        raise FileNotFoundError(f"no artifact at {root}")
    names = {p.name for p in root.iterdir()}
    found = [
        kind
        for kind, rules in RULES.items()
        if any(m.matches(names) for m in rules if not m.fallback)
    ]
    if not found:
        found = [
            kind
            for kind, rules in RULES.items()
            if any(m.matches(names) for m in rules if m.fallback)
        ]
    if len(found) == 1:
        return found[0]
    if not found:
        raise UnknownKindError(
            f"{root} carries no artifact marker; one of "
            f"{sorted(MARKERS.values())}, an MJCF, or weights"
        )
    raise UnknownKindError(
        f"{root} carries markers for {sorted(k.value for k in found)}; "
        "an artifact is one kind"
    )


def _detect_file(path: Path) -> Kind:
    for rule in FILE_RULES:
        if rule.matches(path):
            return rule.kind
    raise UnknownKindError(f"{path} is not a recognised single-file artifact")


def _deploy_run_records() -> tuple[str, ...]:
    """What a deployment's runs write beside its manifest - the gates'
    records and contact sites, the attribution, the assay, the pre-flight,
    their stills, an operator's STOP, a writer's staging file: records
    ABOUT the deployment, never the deployment. A STOP file or a
    pre-flight record once moved the deployment's identity (the review of
    2026-09-24)."""
    from rq_pipeline.deploy import assay, attribution, gate, preflight  # noqa: PLC0415
    from rq_pipeline.deploy.manifest import GATE_RECORDS  # noqa: PLC0415

    return (
        *GATE_RECORDS.values(),
        gate.CONTACTS_FILE.format(runtime="*"),
        assay.ASSAY_FILE,
        attribution.ATTRIBUTION_FILE,
        attribution.STILL_FILE,
        preflight.PREFLIGHT_FILE,
        preflight.STILL_FILE,
        preflight.STOP_FILE,
        STAGING_SUFFIX_PATTERN,
    )


STAGING_SUFFIX_PATTERN = "*.tmp"
# The files an artifact's runs write inside it that are not the artifact:
# left out of its identity (patterns, relative to the artifact's root).
RUN_RECORDS: dict[Kind, Callable[[], tuple[str, ...]]] = {
    Kind.DEPLOY: _deploy_run_records,
}


def artifact_hash(kind: Kind, root: Path) -> str:
    """The artifact's content hash: `bundles.hashing.bundle_hash` with the
    run records its kind declares (`RUN_RECORDS`) left out; a kind with
    none is `bundle_hash` itself."""
    records = RUN_RECORDS.get(kind)
    return bundle_hash(Path(root), exclude=records() if records else ())


def stamp_kind(kind: Kind, root: Path, name: str | None = None) -> str:
    """The artifact's `name@hash`, after checking it IS that kind.
    `name` defaults to the directory's (or file's stem's) own name."""
    root = Path(root)
    actual = detect(root)
    if actual is not kind:
        raise UnknownKindError(f"{root} is a {actual.value}, not a {kind.value}")
    label = name if name is not None else (root.stem if root.is_file() else root.name)
    # A stamp's name half must not carry the separator; RL arms are
    # named `identified#1` and the like — safe. Version suffixes are
    # stripped so `walk@abc` re-stamps as `walk@def`, not `walk@abc@def`.
    if STAMP_SEPARATOR in label:
        label = label.split(STAMP_SEPARATOR, 1)[0]
    if kind in RUN_RECORDS:
        return f"{label}{STAMP_SEPARATOR}{artifact_hash(kind, root)[:STAMP_LENGTH]}"
    return stamp(label, root)


def stamp_run(root: Path, name: str | None = None) -> str:
    """A run's `name@hash`: the hash is of the identity it was launched
    with (`run.json`, or rq_mjlab's `identity.json`), not of its folder.
    A folder hash moves every time a checkpoint lands, so a policy that
    cited a training run pointed at a version that no longer existed
    a minute later (the Go2 run, 2026-09-10). `name` defaults to the
    folder's own."""
    root = Path(root)
    if detect(root) is not Kind.RUN:
        raise UnknownKindError(f"{root} is not a run")
    manifest = root / RUN_MANIFEST_FILE
    marker = RUN_MANIFEST_FILE if manifest.is_file() else IDENTITY_FILE
    fields = read_json(root / marker)
    label = name if name is not None else root.name
    return content_stamp(label.split("@", 1)[0], fields)
