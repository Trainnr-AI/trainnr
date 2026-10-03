"""The T5 chain, one command: demos -> LeRobot v3 -> lerobot-train
(in-loop eval through OUR env) -> lerobot-eval (paired starts, per-trial
records) -> the fold. Two scales, one script: `--scale smoke` (the
default) is "test everything end to end" at a size the WSL card
finishes in minutes; `--scale cloud` is the real run — the ACT sim
recipe's 50 demonstrations and 100k steps, twenty paired evaluation
starts — for a rented GPU. Any knob overrides its scale's value, and
`--from`/`--until` run any contiguous slice of the four stages: the
CPU-bound demos and convert on one box, the GPU-bound train and eval on
another.

    # WSL / Linux (train venv; the GPU renderer for the demos, the card for ACT):
    cd trainnr && ../tools/wsl-run.sh .venv-train/bin/python \\
        ../tools/e2e-smoke.py --episodes 2 --steps 300
    # macOS: none of the GL variables; the policy device is picked for you:
    cd trainnr && .venv-train/bin/python ../tools/e2e-smoke.py --episodes 2 --steps 300
    # the cloud GPU, on a dataset the box converted and pushed:
    cd trainnr && MUJOCO_GL=egl .venv-train/bin/python \\
        ../tools/e2e-smoke.py --scale cloud --name t5-cloud --from train

Every stage writes what the next one reads, and every file is one a
stranger can open: the batch's manifests (seeds, draws, verdicts), the
dataset's provenance sidecar (bundle stamp, expert stamp, every
manifest), the trainer's checkpoint, and the JSONL records the env
appends per trial. Beside the trainer's directory, from the chain's
first second, the sidecar a dashboard follows (`train-watch --follow`):
the run manifest, every stage's output teed into one log, GPU samples.
The last stage folds the records: counts, an exact interval, the
milestone funnel, the events.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from _lab import bootstrap, lerobot_eval_command, lerobot_train_command

bootstrap()

from trainnr.collect.datasheet import DATASHEET_FILE, summarize  # noqa: E402
from trainnr.collect.kitting_demos import DR_SPAN, generate_demos  # noqa: E402
from trainnr.collect.kitting_export import (  # noqa: E402
    DatasetProvenance,
    export_kitting_demos,
)
from trainnr.collect.provenance import PROVENANCE_FILE  # noqa: E402
from trainnr.envs.lerobot_plugin import TrainnrEnvConfig  # noqa: E402
from trainnr.envs.lerobot_policy import best_device  # noqa: E402
from trainnr.envs.lerobot_train_log import (  # noqa: E402
    CHAIN_LOG_FILE,
    GPU_LOG_FILE,
    GPU_QUERY,
    POLICY,
    SCALES,
    RunLayout,
    RunManifest,
    Scale,
    resolve_scale,
)
from trainnr.evaluate.records import (  # noqa: E402
    fold,
    funnel,
    passes,
    read_records,
)
from trainnr.stats.intervals import clopper_pearson  # noqa: E402
from trainnr.tasks.aloha2 import KITTING  # noqa: E402

# The seeds lerobot-eval hands the env: episode i starts at EVAL_SEED + i,
# which our env reads as trial i - the pairing the certificate needs.
EVAL_SEED = 1000
# Every control tick a frame: the rate the env observes at.
FRAME_EVERY = 1
# Runs/eval sizes small enough to stay inside the vector env's one pass.
MAX_EVAL_BATCH = 4
KNOBS = ("episodes", "steps", "checkpoint_every", "inloop_episodes", "eval_episodes")


@dataclass
class Chain:
    """One run of the chain: its layout, its scale, the log every stage
    and this tool write to, and the stage a failure names."""

    layout: RunLayout
    scale: Scale
    scale_name: str
    device: str
    seed: int
    log: Path
    current: str = "start"

    def say(self, text: str) -> None:
        print(text, flush=True)
        with self.log.open("a", encoding="utf-8") as log:
            log.write(text + "\n")

    def stage(self, title: str) -> float:
        self.current = title
        self.say(f"\n=== {title} ({time.strftime('%H:%M:%S')}) ===")
        return time.perf_counter()

    def done(self, started: float) -> None:
        self.say(f"--- {time.perf_counter() - started:.0f} s")

    def run(self, command: Sequence[object]) -> None:
        """A stage's subprocess: its output on the terminal AND in the chain
        log, line by line as it happens."""
        argv = [str(part) for part in command]
        self.say("$ " + " ".join(argv))
        process = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        assert process.stdout is not None
        with self.log.open("a", encoding="utf-8") as log:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
        if code := process.wait():
            raise StageFailed(self.current, code)

    def manifest(self, command: Sequence[object] = ()) -> None:
        path = RunManifest.build(
            self.layout,
            self.scale,
            device=self.device,
            command=command,
            scale_name=self.scale_name,
            task=KITTING,
            started=datetime.now(UTC),
        ).write_to(self.layout.watch)
        self.say(f"run manifest: {path}")


class StageFailed(SystemExit):
    """A stage's subprocess failed: say which, and what is on disk."""

    def __init__(self, title: str, code: int) -> None:
        super().__init__(
            f"e2e-smoke: stage '{title}' failed (exit {code}); its own output is "
            "above. Earlier stages' artifacts stay on disk under runs/<name>-*; "
            "rerun with --from <stage> to reuse them, or --force to start over."
        )


class GpuSampler:
    """`nvidia-smi` every few seconds into a file the dashboard reads;
    nothing where there is no nvidia-smi (a Mac, a CPU box)."""

    EVERY_S = 5

    def __init__(self, log_path: Path) -> None:
        self.process: subprocess.Popen | None = None
        if shutil.which("nvidia-smi") is None:
            return
        self._handle = log_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            [
                "nvidia-smi",
                f"--query-gpu={GPU_QUERY}",
                "--format=csv,noheader",
                "-l",
                str(self.EVERY_S),
            ],
            stdout=self._handle,
            stderr=subprocess.DEVNULL,
        )

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=10)
        self._handle.close()


# ---- the stages -----------------------------------------------------------


def demos(chain: Chain) -> None:
    # Preflight: the certified actuator-bundle store verifies before a
    # single episode is pressed — a tampered or malformed bundle fails
    # the chain here, by name, not in a training run three stages later.
    from trainnr.robot.actuator_bundle import (  # noqa: PLC0415
        BUNDLE_STORE,
        read_bundle,
    )

    store = sorted(BUNDLE_STORE.glob("*.bundle.json"))
    if not store:
        # An empty store used to pass with "0 bundles verify" — a green
        # line for a missing artifact (review 2026-09-01).
        raise StageFailed(chain.current, 1)
    for bundle_path in store:
        read_bundle(bundle_path)  # verifies; raises naming the file
    # Honest scope: this is a STORE INTEGRITY check. The kitting chain
    # does not read an actuator bundle (its dynamics come from the task
    # bundle's own model), so nothing here could fail later either —
    # the earlier wording claimed it caught what a training run would.
    chain.say(
        f"preflight: {len(store)} certified actuator bundles verify "
        "(store integrity; this chain's kitting task consumes none)"
    )

    batch = generate_demos(
        chain.layout.demos,
        episodes=chain.scale.episodes,
        seed=chain.seed,
        dr_span=DR_SPAN,
        frame_every=FRAME_EVERY,
        say=chain.say,
    )
    if not batch.complete:
        raise StageFailed(chain.current, 1)
    # The batch's own datasheet (written by the press): its warnings go
    # into the chain log verbatim — surfaced, never smoothed over.
    sheet = summarize(chain.layout.demos)
    rate = (
        f"keep rate <= {sheet.keep_rate_bound:.0%}"
        if sheet.shards == 1
        else f"{sheet.shards} shards, no single keep-rate bound"
    )
    chain.say(
        f"datasheet: {chain.layout.demos / DATASHEET_FILE} — "
        f"{sheet.episodes} episodes, {rate}, bases {list(sheet.bases)}"
    )
    for warning in sheet.warnings:
        chain.say(f"datasheet WARNING: {warning}")


def convert(chain: Chain) -> None:
    layout = chain.layout
    export_kitting_demos(layout.demos, layout.dataset, repo_id=layout.repo_id)
    provenance = DatasetProvenance.read(layout.dataset / PROVENANCE_FILE)
    chain.say(
        f"provenance: {provenance.bundle}, expert {provenance.expert}, "
        f"{provenance.episodes} episodes, {sum(provenance.frames)} frames at "
        f"{provenance.fps} fps; gain scales "
        f"{[round(m['gain_scale'], 2) for m in provenance.manifests]}"
    )


def train(chain: Chain) -> None:
    layout, scale = chain.layout, chain.scale
    episodes = min(scale.inloop_episodes, MAX_EVAL_BATCH)
    command = lerobot_train_command(
        policy=POLICY,
        device=chain.device,
        dataset=layout.repo_id,
        dataset_root=layout.dataset,
        output_dir=layout.training,
        job_name=layout.training.name,
        steps=scale.steps,
        batch_size=scale.batch,
        save_freq=scale.checkpoint_every,
        num_workers=scale.workers,
        lr=scale.lr,
        eval_freq=scale.checkpoint_every,
        eval_episodes=episodes,
        eval_batch=episodes,
        extra=TrainnrEnvConfig.cli_flags(
            KITTING,
            record_to=layout.inloop_records,
            policy_name=f"{layout.name}-inloop",
            trials=episodes,
        ),
    )
    chain.manifest(command)  # again, now with the command and the provenance
    chain.run(command)


def evaluate(chain: Chain) -> None:
    layout, scale = chain.layout, chain.scale
    checkpoint = layout.checkpoint(scale.steps)
    if not checkpoint.is_dir():
        raise SystemExit(
            f"e2e-smoke: no checkpoint at {checkpoint}; the trainer saved "
            f"{layout.checkpoint_steps()}"
        )
    chain.run(
        lerobot_eval_command(
            policy_path=checkpoint,
            device=chain.device,
            output_dir=layout.evaluation,
            seed=EVAL_SEED,
            episodes=scale.eval_episodes,
            batch_size=min(scale.eval_episodes, MAX_EVAL_BATCH),
            extra=TrainnrEnvConfig.cli_flags(
                KITTING,
                record_to=layout.eval_records,
                policy_name=f"{layout.name}-{POLICY}-{scale.steps}",
                trials=scale.eval_episodes,
            ),
        )
    )
    chain.stage("fold: counts, intervals, funnel")
    report(chain, layout.inloop_records)
    report(chain, layout.eval_records)
    chain.say(
        f"\nplay it: train-watch --play {checkpoint} --task {KITTING} "
        "--action-space bundle --look aloha2"
    )


def report(chain: Chain, path: Path) -> None:
    """Counts, intervals, funnel, events — per pass, when the file holds
    several (the trainer's in-loop eval appends one pass per checkpoint)."""
    records = read_records(path)
    chain.say(
        f"{path}: {len(records)} records; "
        f"instrument {sorted({r.instrument for r in records})}; "
        f"source {sorted({r.source for r in records})}"
    )
    chunks = passes(records)
    for number, chunk in enumerate(chunks, start=1):
        label = f"  pass {number}/{len(chunks)}: " if len(chunks) > 1 else "  "
        for score in fold(chunk):
            low, high = clopper_pearson(score.successes, score.trials)
            chain.say(
                f"{label}{score.name}: {score.successes}/{score.trials}  "
                f"CP95 [{low:.3f}, {high:.3f}]"
            )
        chain.say(f"{label}funnel {funnel(chunk)}")
        chain.say(
            f"{label}events {[(r.trial, [e['name'] for e in r.events]) for r in chunk]}"
        )


@dataclass(frozen=True)
class Stage:
    """A stage: its name, its title for the log, what it runs, what it
    writes (removed by --force) and what it needs from the stage before."""

    name: str
    title: Callable[[Chain], str]
    run: Callable[[Chain], None]
    writes: Callable[[RunLayout], list[Path]]
    needs: Callable[[Chain], Path | None] = lambda _chain: None


STAGES = (
    Stage(
        "demos",
        lambda c: (
            f"demos: {c.scale.episodes} kept at +-{DR_SPAN:.0%} DR -> {c.layout.demos}"
        ),
        demos,
        lambda layout: [layout.demos],
    ),
    Stage(
        "convert",
        lambda c: f"convert {c.layout.demos} -> {c.layout.dataset}",
        convert,
        lambda layout: [layout.dataset],
        lambda c: c.layout.demos,
    ),
    Stage(
        "train",
        lambda c: (
            f"train {POLICY} {c.scale.steps} steps on {c.device}, "
            "in-loop eval through our env"
        ),
        train,
        lambda layout: [layout.training, layout.watch],
        # A reused dataset needs no batch beside it: the dataset carries
        # every manifest in its provenance (the pod holds the dataset and
        # not the frames, by design).
        lambda c: c.layout.dataset,
    ),
    Stage(
        "eval",
        lambda c: f"lerobot-eval: {c.scale.eval_episodes} paired starts, records",
        evaluate,
        lambda layout: [layout.evaluation],
        lambda c: c.layout.checkpoint(c.scale.steps),
    ),
)
STAGE_NAMES = tuple(stage.name for stage in STAGES)


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive count, got {text}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--name", default="smoke", help="prefix of the runs/ dirs")
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument(
        "--scale",
        choices=sorted(SCALES),
        default="smoke",
        help="the preset every unset knob below takes its value from",
    )
    for knob in KNOBS:
        parser.add_argument(
            f"--{knob.replace('_', '-')}", type=positive_int, default=None
        )
    parser.add_argument(
        "--workers", type=positive_int, default=None, help="dataloader workers"
    )
    parser.add_argument(
        "--batch", type=positive_int, default=None, help="the trainer's batch"
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="learning rate (default: LeRobot's)"
    )
    parser.add_argument(
        "--seed", type=int, default=20260828, help="the generator's seed"
    )
    parser.add_argument(
        "--device", default=None, help="torch device; default: the best present"
    )
    parser.add_argument(
        "--from", dest="first", choices=STAGE_NAMES, default=STAGE_NAMES[0]
    )
    parser.add_argument(
        "--until", dest="last", choices=STAGE_NAMES, default=STAGE_NAMES[-1]
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="remove this name's earlier artifacts of the stages that run (the "
        "trainer refuses an existing output directory)",
    )
    args = parser.parse_args()
    if STAGE_NAMES.index(args.first) > STAGE_NAMES.index(args.last):
        parser.error(f"--from {args.first} is after --until {args.last}")
    return args


def prepare(chain: Chain, stages: Sequence[Stage], *, force: bool) -> None:
    """Refuse to write over a named run (the trainer refuses an existing
    output directory after minutes of work; a smaller batch would inherit
    stale episodes); `--force` removes what the stages that run wrote.
    The first stage's input must already be there."""
    needed = stages[0].needs(chain)
    if needed is not None and not needed.exists():
        raise SystemExit(f"e2e-smoke: --from {stages[0].name} needs {needed}")
    existing = [p for stage in stages for p in stage.writes(chain.layout) if p.exists()]
    if existing and not force:
        raise SystemExit(
            "e2e-smoke: this name already has artifacts: "
            + ", ".join(str(path) for path in existing)
            + " — pass --force to remove them, or --name something new"
        )
    for path in existing:
        shutil.rmtree(path)


def main() -> None:
    args = parse_args()
    try:
        scale = resolve_scale(
            args.scale,
            **{
                knob: getattr(args, knob) for knob in (*KNOBS, "workers", "batch", "lr")
            },
        )
    except ValueError as error:
        raise SystemExit(f"e2e-smoke: {error}") from None
    layout = RunLayout(args.runs, args.name)
    chain = Chain(
        layout,
        scale,
        args.scale,
        best_device(args.device),
        args.seed,
        layout.watch / CHAIN_LOG_FILE,
    )
    stages = STAGES[STAGE_NAMES.index(args.first) : STAGE_NAMES.index(args.last) + 1]
    prepare(chain, stages, force=args.force)
    # The sidecar opens before the first stage: the dashboard follows the
    # demos and the convert too, not only the trainer.
    layout.watch.mkdir(parents=True, exist_ok=True)
    chain.log.write_text("", encoding="utf-8")
    chain.manifest()
    sampler = GpuSampler(layout.watch / GPU_LOG_FILE)
    try:
        for stage in stages:
            started = chain.stage(stage.title(chain))
            stage.run(chain)
            chain.done(started)
    finally:
        sampler.stop()


if __name__ == "__main__":
    main()
