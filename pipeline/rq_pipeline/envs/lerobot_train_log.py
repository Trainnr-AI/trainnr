"""What a LeRobot training run writes, read back: the metrics lines of
`lerobot-train`, its in-loop evaluation summaries, and the run manifest
our chain writes beside them — so a dashboard (`tools/train-watch.py
--follow`) can stream a run from its files, local or mirrored from a
rented card, with every parameter visible.

LeRobot's trainer logs one line per `log_freq` steps
(`MetricsTracker.__str__`, lerobot 0.6):

    step:1.0K smpl:64K ep:45 epch:1.58 loss:0.281 grdn:12.345 lr:3.0e-05
        updt_s:0.081 data_s:0.001

with `format_big_number` on the counts — `1.0K`, `10K`, `1.2M` — which a
naive `step:(\\d+)` reads as 1 past step 999 (measured 2026-08-27 in the
first version of train-watch). Standard library only.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# LeRobot's format_big_number suffixes, in the order it applies them.
BIG_NUMBER_UNITS = {"": 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
_BIG_NUMBER = re.compile(r"^(-?\d+(?:\.\d+)?)([KMB]?)$")
# A metric token: a name that starts a word (not `ot_train.py:597`, whose
# "py" follows a dot; not a clock's "01:37", which starts with a digit;
# `smp/s:24` keeps its slash) and a numeric value — LeRobot's big-number
# suffix allowed — that ends the word.
_TOKEN = re.compile(
    r"(?<![\w./])([A-Za-z][A-Za-z0-9_/]*):"
    r"(-?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?[KMB]?)(?=\s|$)"
)
# The trainer's in-loop evaluation summary (lerobot_train.py, the
# "overall" suite): a python dict repr after this marker.
EVAL_MARKER = "Suite overall aggregated:"
_EVAL_FIELD = re.compile(r"'(pc_success|n_episodes|eval_s|avg_sum_reward)': ([\d.]+)")
COUNT_FIELDS = ("step", "smpl", "ep")
RUN_MANIFEST_FILE = "run.json"
TRAIN_LOG_FILE = "train.log"
GPU_LOG_FILE = "gpu.log"
# The trainer refuses an output directory that already exists, so what
# the chain writes BEFORE the trainer starts lives beside the run:
# `runs/<name>-act` ↔ `runs/<name>-watch` (measured 2026-08-27).
RUN_SUFFIX = "-act"
WATCH_SUFFIX = "-watch"


def watch_dir_for(run_dir: Path) -> Path:
    """The chain's sidecar directory for a trainer output directory."""
    run_dir = Path(run_dir)
    name = run_dir.name
    if name.endswith(RUN_SUFFIX):
        name = name[: -len(RUN_SUFFIX)]
    return run_dir.parent / f"{name}{WATCH_SUFFIX}"


def parse_big_number(text: str) -> int:
    """`1.0K` → 1000, `10K` → 10000, `1.2M` → 1200000, `450` → 450."""
    match = _BIG_NUMBER.match(text.strip())
    if not match:
        raise ValueError(f"not a big number: {text!r}")
    value, unit = match.groups()
    return round(float(value) * BIG_NUMBER_UNITS[unit])


@dataclass(frozen=True)
class TrainLine:
    """One metrics line: the counters and every named metric."""

    step: int
    samples: int
    episodes: int
    epochs: float
    metrics: Mapping[str, float]  # loss, grdn, lr, updt_s, data_s, l1_loss, kld_loss…


def parse_train_line(line: str) -> TrainLine | None:
    """The metrics line as data, or None for any other line."""
    tokens = dict(_TOKEN.findall(line))
    if "step" not in tokens or "loss" not in tokens:
        return None
    try:
        counts = {
            name: parse_big_number(tokens.get(name, "0")) for name in COUNT_FIELDS
        }
        epochs = float(tokens.get("epch", "0"))
    except ValueError:
        return None
    metrics: dict[str, float] = {}
    for name, value in tokens.items():
        if name in COUNT_FIELDS or name == "epch":
            continue
        try:
            metrics[name] = float(value)
        except ValueError:
            continue  # a word after a colon in prose, not a metric
    return TrainLine(
        step=counts["step"],
        samples=counts["smpl"],
        episodes=counts["ep"],
        epochs=epochs,
        metrics=metrics,
    )


@dataclass(frozen=True)
class EvalLine:
    """The trainer's in-loop evaluation summary: LeRobot's own count."""

    pc_success: float
    n_episodes: int
    eval_s: float


def parse_eval_line(line: str) -> EvalLine | None:
    if EVAL_MARKER not in line:
        return None
    fields = dict(_EVAL_FIELD.findall(line))
    if "pc_success" not in fields or "n_episodes" not in fields:
        return None
    return EvalLine(
        pc_success=float(fields["pc_success"]),
        n_episodes=int(float(fields["n_episodes"])),
        eval_s=float(fields.get("eval_s", 0.0)),
    )


@dataclass(frozen=True)
class RunManifest:
    """Every parameter of one training run, written by the chain before
    the trainer starts — the dashboard shows it from step 0, and the
    trainer's own resolved config joins it at the first checkpoint."""

    name: str
    policy: str
    steps: int
    batch_size: int
    checkpoint_every: int
    inloop_episodes: int
    eval_episodes: int
    workers: int
    learning_rate: float | None
    device: str
    dataset_root: str
    dataset_repo_id: str
    command: str
    started: str  # ISO-8601, UTC
    provenance: Mapping[str, Any] = field(default_factory=dict)  # the dataset's
    task: str = ""
    scale: str = ""

    def write(self, run_dir: Path) -> Path:
        path = Path(run_dir) / RUN_MANIFEST_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=1), encoding="utf-8")
        return path

    @classmethod
    def read(cls, run_dir: Path) -> RunManifest:
        raw = json.loads(
            (Path(run_dir) / RUN_MANIFEST_FILE).read_text(encoding="utf-8")
        )
        return cls(**raw)

    def as_markdown(self) -> str:
        """The parameters as a table, for a viewer's text panel."""
        rows = [
            ("run", self.name),
            ("scale", self.scale or "-"),
            ("policy", self.policy),
            ("task", self.task or "-"),
            ("steps", str(self.steps)),
            ("batch size", str(self.batch_size)),
            (
                "learning rate",
                "LeRobot default"
                if self.learning_rate is None
                else f"{self.learning_rate:g}",
            ),
            ("checkpoint every", str(self.checkpoint_every)),
            ("in-loop eval episodes", str(self.inloop_episodes)),
            ("final eval episodes", str(self.eval_episodes)),
            ("dataloader workers", str(self.workers)),
            ("device", self.device),
            ("dataset", f"{self.dataset_repo_id} at {self.dataset_root}"),
            ("started", self.started),
        ]
        for key in ("bundle", "expert", "episodes", "fps"):
            if key in self.provenance:
                rows.append((f"dataset {key}", str(self.provenance[key])))
        table = "\n".join(f"| {k} | {v} |" for k, v in rows)
        return (
            f"# {self.name}\n\n| parameter | value |\n|---|---|\n{table}\n\n"
            f"`{self.command}`\n"
        )


class FileFollower:
    """`tail -f` for one file: `new_lines()` returns what was appended
    since the last call (whole lines only), and copes with the file not
    existing yet or being truncated."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._offset = 0
        self._partial = ""

    def new_lines(self) -> list[str]:
        if not self.path.exists():
            return []
        size = self.path.stat().st_size
        if size < self._offset:  # truncated and rewritten
            self._offset, self._partial = 0, ""
        if size == self._offset:
            return []
        with self.path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(self._offset)
            chunk = handle.read()
            self._offset = handle.tell()
        text = self._partial + chunk
        lines = text.split("\n")
        self._partial = lines.pop()  # "" when the chunk ended on a newline
        return [line.rstrip("\r") for line in lines]
