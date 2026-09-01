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

import bisect
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from rq_pipeline.bundles.json_record import JsonRecord

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
PROVENANCE_KEYS = ("bundle", "expert", "episodes", "fps")  # of the dataset's sidecar
# Every stage of the chain, teed as it happens — the demo attempts, the
# convert, the trainer's lines, the evaluations — one file from the
# chain's first second (a dashboard that opens at the train stage shows
# an empty grid for minutes; measured 2026-08-28).
CHAIN_LOG_FILE = "chain.log"
GPU_LOG_FILE = "gpu.log"
# The chain's own narration in that log: stage headers and timings, the
# generator's verdicts, the batch's provenance line.
STAGE_LINE = re.compile(
    r"^(=== |--- \d+ s|attempt \d+: |kept \d+/|gave up|provenance: |run manifest: |"
    r"--until |e2e-smoke: |play it: )"
)


def is_stage_line(line: str) -> bool:
    return bool(STAGE_LINE.match(line))


# The trainer refuses an output directory that already exists, so what
# the chain writes BEFORE the trainer starts lives beside the run:
# `runs/<name>-act` ↔ `runs/<name>-watch` (measured 2026-08-27).
POLICY = "act"
RUN_SUFFIX = f"-{POLICY}"
WATCH_SUFFIX = "-watch"
STEP_DIR_WIDTH = 6  # LeRobot names a checkpoint step as six digits
CHECKPOINT_WEIGHTS = "model.safetensors"
TRAINER_CONFIG = "train_config.json"
# The trainer's own logger stamps every line; the dashboard places GPU
# samples by it.
LOGGER_STAMP = re.compile(r"INFO (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
# nvidia-smi's columns, as the chain samples them and the dashboard reads them.
GPU_QUERY = "timestamp,utilization.gpu,memory.used"


def watch_dir_for(run_dir: Path) -> Path:
    """The chain's sidecar directory for a trainer output directory."""
    return RunLayout.of(run_dir).watch


@dataclass(frozen=True)
class RunLayout:
    """Where one named chain run keeps its artifacts under runs/ — the
    naming scheme, spelled once for the chain, the dashboard and the
    rented-machine tool (four spellings before 2026-08-28)."""

    runs: Path
    name: str

    @classmethod
    def of(cls, run_dir: Path) -> RunLayout:
        """The layout a trainer directory (`runs/<name>-act`) belongs to."""
        run_dir = Path(run_dir)
        name = run_dir.name
        if name.endswith(RUN_SUFFIX):
            name = name[: -len(RUN_SUFFIX)]
        return cls(run_dir.parent, name)

    @property
    def demos(self) -> Path:
        return self.runs / f"{self.name}-demos"

    @property
    def dataset(self) -> Path:
        return self.runs / f"{self.name}-lerobot"

    @property
    def training(self) -> Path:
        return self.runs / f"{self.name}{RUN_SUFFIX}"

    @property
    def watch(self) -> Path:
        """The dashboard's sidecar — beside the trainer's directory, never
        inside it: the trainer refuses an output directory that exists."""
        return self.runs / f"{self.name}{WATCH_SUFFIX}"

    @property
    def evaluation(self) -> Path:
        return self.runs / f"{self.name}-eval"

    @property
    def inloop_records(self) -> Path:
        return self.training / "inloop-episodes.jsonl"

    @property
    def eval_records(self) -> Path:
        return self.evaluation / "episodes.jsonl"

    @property
    def repo_id(self) -> str:
        return f"rq-pipeline/aloha2-kitting-{self.name}"

    def checkpoint(self, step: int) -> Path:
        return (
            self.training
            / "checkpoints"
            / f"{step:0{STEP_DIR_WIDTH}d}"
            / "pretrained_model"
        )

    def checkpoint_steps(self, *, with_weights: bool = False) -> list[int]:
        """The steps with a checkpoint on disk, ascending; `with_weights`
        keeps only those whose weights have arrived (a mirror brings the
        config first)."""
        root = self.training / "checkpoints"
        if not root.is_dir():
            return []
        steps = sorted(int(e.name) for e in root.iterdir() if e.name.isdigit())
        if with_weights:
            steps = [
                s for s in steps if (self.checkpoint(s) / CHECKPOINT_WEIGHTS).exists()
            ]
        return steps


@dataclass(frozen=True)
class Scale:
    """One size of the chain: how many demonstrations, how long to
    train, how often to checkpoint, how many starts to evaluate, the
    trainer's batch and dataloader workers, the learning rate."""

    episodes: int
    steps: int
    checkpoint_every: int
    inloop_episodes: int
    eval_episodes: int
    workers: int
    batch: int
    # None keeps LeRobot's default (ACT: 1e-5, tuned for batch 8); a
    # bigger batch wants a bigger rate — the square-root rule for AdamW.
    lr: float | None = None


BATCH_SIZE = 8  # LeRobot's ACT sim recipe
SCALES = {
    # Minutes on the WSL card; the smallest run that exercises every stage.
    "smoke": Scale(
        episodes=2,
        steps=300,
        checkpoint_every=300,
        inloop_episodes=1,
        eval_episodes=2,
        workers=4,
        batch=BATCH_SIZE,
    ),
    # LeRobot's ACT sim recipe (50 episodes, 100k steps, batch 8) with a
    # checkpoint every 20k so a lost instance costs an hour, not a day,
    # and twenty paired starts for an interval worth reading (CP95 on 20
    # trials is +-0.2 wide at 50%). Measured 2026-08-27 on a B200: 11.3
    # steps/s at batch 8 with 4 workers and 11.4 with 16 — the loop is
    # CPU-bound at that batch; batch 64 gave 6.5x the throughput.
    "cloud": Scale(
        episodes=50,
        steps=100_000,
        checkpoint_every=20_000,
        inloop_episodes=4,
        eval_episodes=20,
        workers=16,
        batch=BATCH_SIZE,
    ),
}


def resolve_scale(name: str, **overrides: Any) -> Scale:
    """A preset with every given (non-None) knob overridden; the final
    checkpoint is the one evaluated, so `steps` must be a multiple of
    `checkpoint_every`."""
    scale = replace(
        SCALES[name], **{k: v for k, v in overrides.items() if v is not None}
    )
    if scale.steps % scale.checkpoint_every:
        raise ValueError(
            f"steps {scale.steps} is not a multiple of checkpoint_every "
            f"{scale.checkpoint_every}; the final checkpoint is the one evaluated"
        )
    return scale


@dataclass(frozen=True)
class GpuSample:
    taken: datetime
    util_pct: float
    mem_mib: float


def parse_gpu_line(line: str) -> GpuSample | None:
    """One `nvidia-smi --query-gpu=GPU_QUERY --format=csv,noheader` line:
    `2026/08/28 00:33:40.123, 58 %, 17052 MiB`."""
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < len(GPU_QUERY.split(",")):
        return None
    try:
        taken = datetime.fromisoformat(
            parts[0].replace("/", "-").replace(" ", "T")[:26]
        )
        return GpuSample(taken, float(parts[1].split()[0]), float(parts[2].split()[0]))
    except (ValueError, IndexError):
        return None


class StepClock:
    """(wall time → step) from the trainer's stamped metrics lines: a
    sample taken at a moment belongs to the step whose line precedes it
    — so a replay of a finished log does not pile every sample onto the
    final step."""

    def __init__(self) -> None:
        self._when: list[datetime] = []
        self._step: list[int] = []

    def add(self, when: datetime, step: int) -> None:
        self._when.append(when)
        self._step.append(step)

    def at(self, moment: datetime) -> int:
        index = bisect.bisect_right(self._when, moment)
        return self._step[index - 1] if index else 0


def parse_big_number(text: str) -> int:
    """`1.0K` → 1000, `10K` → 10000, `1.2M` → 1200000, `450` → 450."""
    match = _BIG_NUMBER.match(text.strip())
    if not match:
        raise ValueError(f"not a big number: {text!r}")
    value, unit = match.groups()
    return round(float(value) * BIG_NUMBER_UNITS[unit])


@dataclass(frozen=True)
class TrainLine:
    """One metrics line: the counters, every named metric, the logger's clock."""

    step: int
    samples: int
    episodes: int
    epochs: float
    metrics: Mapping[str, float]  # loss, grdn, lr, updt_s, data_s, l1_loss, kld_loss…
    stamp: datetime | None = None


# LeRobot's console abbreviations -> the names the dashboards plot.
# One home: train-watch and the cloud feed logged the SAME numbers
# under different entity paths, so a blueprint keyed to one could
# never match the other's recording (review 2026-09-01).
METRIC_NAMES = {
    "grdn": "grad_norm",
    "smp/s": "samples_per_s",
    "updt_s": "update_s",
    "data_s": "dataloading_s",
}


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
    stamped = LOGGER_STAMP.search(line)
    return TrainLine(
        step=counts["step"],
        samples=counts["smpl"],
        episodes=counts["ep"],
        epochs=epochs,
        metrics=metrics,
        stamp=datetime.fromisoformat(stamped.group(1)) if stamped else None,
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
class RunManifest(JsonRecord):
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

    @classmethod
    def build(  # noqa: PLR0913 - every fact of one run, named
        cls,
        layout: RunLayout,
        scale: Scale,
        *,
        device: str,
        command: Sequence[str],
        scale_name: str,
        task: str,
        started: datetime,
    ) -> RunManifest:
        """The manifest from the chain's resolved knobs and the dataset's
        provenance sidecar (its stamps, when the dataset is there yet)."""
        sidecar = layout.dataset / "provenance.json"
        provenance = (
            json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
        )
        return cls(
            name=layout.name,
            policy=POLICY,
            steps=scale.steps,
            batch_size=scale.batch,
            checkpoint_every=scale.checkpoint_every,
            inloop_episodes=scale.inloop_episodes,
            eval_episodes=scale.eval_episodes,
            workers=scale.workers,
            learning_rate=scale.lr,
            device=device,
            dataset_root=str(layout.dataset),
            dataset_repo_id=layout.repo_id,
            command=" ".join(str(part) for part in command),
            started=started.isoformat(timespec="seconds"),
            provenance={
                key: provenance[key] for key in PROVENANCE_KEYS if key in provenance
            },
            task=task,
            scale=scale_name,
        )

    def write_to(self, run_dir: Path) -> Path:
        return self.write(Path(run_dir) / RUN_MANIFEST_FILE)

    @classmethod
    def read_from(cls, run_dir: Path) -> RunManifest:
        return cls.read(Path(run_dir) / RUN_MANIFEST_FILE)

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
        for key in PROVENANCE_KEYS:
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
