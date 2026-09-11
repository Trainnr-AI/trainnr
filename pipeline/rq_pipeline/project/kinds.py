"""What an artifact IS, decided by the file at its root — never guessed.

Every kind has one marker: the file the pipeline already writes at that
artifact's root (`datasheet.md` for a pressed batch, `run.json` for a
training run's watch directory, `identity.json` for an RL run, and so
on). The marker constants are IMPORTED from the modules that write them,
so a rename there is a rename here. An artifact with no marker, or with
two, is refused by name: a registry that guesses a kind would attach the
wrong lineage to it, silently.

`stamp_kind` is the one stamping door: check the marker, then hash
through `bundles/hashing.stamp`. Downstream code never invents a name.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from rq_pipeline.bundles.hashing import content_stamp, stamp
from rq_pipeline.collect.datasheet import DATASHEET_FILE
from rq_pipeline.collect.provenance import PROVENANCE_FILE
from rq_pipeline.envs.lerobot_train_log import CHECKPOINT_WEIGHTS, RUN_MANIFEST_FILE

# Markers this module owns because no writer exists yet (docs/76 §2):
# a recording, a certificate directory, a deploy manifest, a drift record.
RECORDING_FILE = "recording.json"
CERTIFICATE_FILE = "certificate.json"
DEPLOY_FILE = "deploy.json"
DRIFT_FILE = "drift.json"
FIT_FILE = "fit.json"
TASK_FILE = "task.json"
# The critic's verdict beside a task (`tools/accept-task.py --project`).
ACCEPTANCE_FILE = "acceptance.json"
ACCEPTANCE_SCHEMA = "trainnr-acceptance/1"
IDENTITY_FILE = "identity.json"  # rq_mjlab's walk run identity (walk_train.py)


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
    FINDING = "finding"


# One marker per kind. A robot bundle is the exception: it is recognised
# by an MJCF at its root (`profile.json` is optional — Menagerie wraps
# carry none), so it is detected by suffix, below.
MARKERS: dict[Kind, str] = {
    Kind.RECORDING: RECORDING_FILE,
    Kind.FIT: FIT_FILE,
    Kind.TASK: TASK_FILE,
    Kind.BATCH: DATASHEET_FILE,
    Kind.DATASET: PROVENANCE_FILE,
    Kind.RUN: RUN_MANIFEST_FILE,
    Kind.CERTIFICATE: CERTIFICATE_FILE,
    Kind.DEPLOY: DEPLOY_FILE,
    Kind.DRIFT: DRIFT_FILE,
}
ROBOT_MODEL_SUFFIX = ".xml"


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
    found = [kind for kind, marker in MARKERS.items() if marker in names]
    # A dataset directory carries provenance.json; so does nothing else at
    # its root. A batch carries datasheet.md. An RL run carries
    # identity.json beside its checkpoint — that is a RUN too.
    if IDENTITY_FILE in names and Kind.RUN not in found:
        found.append(Kind.RUN)
    if not found and any(n.endswith(ROBOT_MODEL_SUFFIX) for n in names):
        found.append(Kind.ROBOT)
    if not found and (
        CHECKPOINT_WEIGHTS in names or any(n.endswith(".pt") for n in names)
    ):
        found.append(Kind.POLICY)
    if len(found) == 1:
        return found[0]
    if not found:
        raise UnknownKindError(
            f"{root} carries no artifact marker; one of "
            f"{sorted(m for m in MARKERS.values())}, an MJCF, or weights"
        )
    raise UnknownKindError(
        f"{root} carries markers for {sorted(k.value for k in found)}; "
        "an artifact is one kind"
    )


def _detect_file(path: Path) -> Kind:
    if path.suffix == ".pt" or path.name == CHECKPOINT_WEIGHTS:
        return Kind.POLICY
    if path.suffix == ".json" and path.parent.name == "findings":
        return Kind.FINDING
    if path.name == FIT_FILE or (path.suffix == ".json" and path.parent.name == "fits"):
        return Kind.FIT
    raise UnknownKindError(f"{path} is not a recognised single-file artifact")


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
    if "@" in label:
        label = label.split("@", 1)[0]
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
    fields = json.loads((root / marker).read_text())
    label = name if name is not None else root.name
    return content_stamp(label.split("@", 1)[0], fields)
