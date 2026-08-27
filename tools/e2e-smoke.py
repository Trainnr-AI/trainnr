"""The T5 chain, one command: demos -> LeRobot v3 -> lerobot-train
(in-loop eval through OUR env) -> lerobot-eval (paired starts, per-trial
records) -> the fold. Two scales, one script: `--scale smoke` (the
default) is "test everything end to end" at a size the WSL card
finishes in minutes; `--scale cloud` is the real run — the ACT sim
recipe's 50 demonstrations and 100k steps, twenty paired evaluation
starts — for a rented GPU. Any knob overrides its scale's value.

    # WSL / Linux (train venv; the GPU renderer for the demos, the card for ACT):
    cd pipeline && ../tools/wsl-run.sh .venv-train/bin/python \\
        ../tools/e2e-smoke.py --episodes 2 --steps 300
    # macOS: none of the GL variables; the policy device is picked for you:
    cd pipeline && .venv-train/bin/python ../tools/e2e-smoke.py --episodes 2 --steps 300
    # the cloud GPU (a Linux box with the train venv; EGL is the offscreen
    # renderer there too, without WSL's Mesa variables):
    cd pipeline && MUJOCO_GL=egl .venv-train/bin/python \\
        ../tools/e2e-smoke.py --scale cloud --name t5-cloud

Every stage writes what the next one reads, and every file is one a
stranger can open: the batch's manifests (seeds, draws, verdicts), the
dataset's provenance sidecar (bundle stamp, every manifest), the
trainer's checkpoint, and the JSONL records the env appends per trial
- the in-loop eval's rows are written by LeRobot's own evaluator, not
by this tool. The last stage folds those rows: counts, an exact
interval, the milestone funnel, the events.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from _lab import bootstrap, lerobot_eval_command, lerobot_train_command

bootstrap()

from rq_pipeline.collect.kitting_export import export_kitting_demos  # noqa: E402
from rq_pipeline.envs.lerobot_plugin import RobotiqEnvConfig  # noqa: E402
from rq_pipeline.envs.lerobot_policy import best_device  # noqa: E402
from rq_pipeline.evaluate.records import fold, funnel, read_records  # noqa: E402
from rq_pipeline.stats.intervals import clopper_pearson  # noqa: E402
from rq_pipeline.tasks.aloha2 import KITTING  # noqa: E402

TOOLS = Path(__file__).resolve().parent
DEMO_GENERATOR = TOOLS / "kitting-demos.py"
# The generator's knobs for training data: every control tick a frame
# (the rate the env observes at), the corrected +-30% dynamics draw.
FRAME_EVERY = 1
DR_SPAN = 0.30
# The seeds lerobot-eval hands the env: episode i starts at EVAL_SEED + i,
# which our env reads as trial i - the pairing the certificate needs.
EVAL_SEED = 1000
POLICY = "act"
BATCH_SIZE = 8
# Runs/eval sizes small enough to stay inside the vector env's one pass.
MAX_EVAL_BATCH = 4
STEP_DIR_WIDTH = 6  # LeRobot names a checkpoint step as six digits
STAGES = ("demos", "convert", "train", "eval")


@dataclass(frozen=True)
class Scale:
    """One size of the chain: how many demonstrations, how long to
    train, how often to checkpoint, how many starts to evaluate."""

    episodes: int
    steps: int
    checkpoint_every: int
    inloop_episodes: int
    eval_episodes: int
    # Dataloader workers and the batch. Measured 2026-08-27 on a B200:
    # 11.3 steps/s at batch 8 with FOUR workers and 11.4 with SIXTEEN, the
    # card at 17% — the training loop itself is CPU-bound at batch 8
    # (the desktop 3090 Ti does 14.8 on faster cores). Workers are not
    # the lever; the batch is: a bigger batch moves the same samples
    # through the card in fewer steps. The cloud preset keeps the ACT sim
    # recipe's batch 8 for comparability; `--batch` overrides it.
    workers: int
    batch: int


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
    # and twenty paired starts for an interval worth reading (CP95 on
    # 20 trials is +-0.2 wide at 50%). Measured rates on the 3090 Ti
    # (docs/31): ACT 14.8 steps/s at batch 8, the converter 38 s/episode,
    # lerobot-eval 78 s/episode.
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


@dataclass(frozen=True)
class RunLayout:
    """Where one named chain run keeps its four artifacts under runs/."""

    runs: Path
    name: str

    @property
    def demos(self) -> Path:
        return self.runs / f"{self.name}-demos"

    @property
    def dataset(self) -> Path:
        return self.runs / f"{self.name}-lerobot"

    @property
    def training(self) -> Path:
        return self.runs / f"{self.name}-{POLICY}"

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
        return f"rq-pipeline/aloha2-{KITTING}-{self.name}"

    def checkpoint(self, step: int) -> Path:
        return (
            self.training
            / "checkpoints"
            / f"{step:0{STEP_DIR_WIDTH}d}"
            / "pretrained_model"
        )


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive count, got {text}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--name", default="smoke", help="prefix of the four runs/ dirs")
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument(
        "--scale",
        choices=sorted(SCALES),
        default="smoke",
        help="the preset every unset knob below takes its value from",
    )
    parser.add_argument(
        "--episodes", type=positive_int, default=None, help="demonstrations to keep"
    )
    parser.add_argument(
        "--seed", type=int, default=20260828, help="the generator's seed"
    )
    parser.add_argument(
        "--steps", type=positive_int, default=None, help="training steps"
    )
    parser.add_argument(
        "--checkpoint-every",
        type=positive_int,
        default=None,
        help="the trainer saves every this many steps (and at the last)",
    )
    parser.add_argument(
        "--inloop-episodes",
        type=positive_int,
        default=None,
        help="episodes of lerobot-train's own in-loop eval at every checkpoint",
    )
    parser.add_argument(
        "--eval-episodes",
        type=positive_int,
        default=None,
        help="lerobot-eval paired starts",
    )
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=None,
        help="the trainer's dataloader workers (not the bottleneck; measured)",
    )
    parser.add_argument(
        "--batch",
        type=positive_int,
        default=None,
        help="the trainer's batch size: the lever when the card idles at batch 8",
    )
    parser.add_argument(
        "--device", default=None, help="torch device; default: the best present"
    )
    parser.add_argument(
        "--until",
        choices=STAGES,
        default=STAGES[-1],
        help="stop after this stage: the CPU-bound stages (demos, convert) on "
        "one box, the GPU-bound ones (train, eval) on another",
    )
    parser.add_argument(
        "--skip-demos",
        action="store_true",
        help="reuse the batch already under <runs>/<name>-demos",
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="reuse the dataset already under <runs>/<name>-lerobot (implies "
        "--skip-demos): the CPU-bound stages done elsewhere, the GPU-bound "
        "ones here",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="remove this name's earlier artifacts first (the trainer refuses "
        "an existing output directory, and a smaller batch would otherwise "
        "inherit stale episodes)",
    )
    args = parser.parse_args()
    if args.skip_convert:
        args.skip_demos = True
    scale = SCALES[args.scale]
    for knob in (
        "episodes",
        "steps",
        "checkpoint_every",
        "inloop_episodes",
        "eval_episodes",
        "workers",
        "batch",
    ):
        if getattr(args, knob) is None:
            setattr(args, knob, getattr(scale, knob))
    if args.steps % args.checkpoint_every:
        raise SystemExit(
            f"e2e-smoke: --steps {args.steps} is not a multiple of "
            f"--checkpoint-every {args.checkpoint_every}; the final checkpoint "
            "is the one evaluated"
        )
    return args


_CURRENT_STAGE = "start"


def stage(title: str) -> float:
    global _CURRENT_STAGE  # noqa: PLW0603 - the one line a failure names
    _CURRENT_STAGE = title
    print(f"\n=== {title} ({time.strftime('%H:%M:%S')}) ===", flush=True)
    return time.perf_counter()


def done(started: float) -> None:
    print(f"--- {time.perf_counter() - started:.0f} s", flush=True)


def run(command: list[str]) -> None:
    print("$", " ".join(str(part) for part in command), flush=True)
    try:
        subprocess.run([str(part) for part in command], check=True)
    except subprocess.CalledProcessError as error:
        raise StageFailed(_CURRENT_STAGE, error.returncode) from None


class StageFailed(SystemExit):
    """A stage's subprocess failed: say which, and what is on disk."""

    def __init__(self, title: str, code: int) -> None:
        super().__init__(
            f"e2e-smoke: stage '{title}' failed (exit {code}); its own output is "
            "above. Earlier stages' artifacts stay on disk under runs/<name>-*; "
            "rerun with --skip-demos to keep the batch, or --force to start over."
        )


def prepare(
    layout: RunLayout, *, skip_demos: bool, skip_convert: bool, force: bool
) -> None:
    """Refuse to write over a named run: the trainer refuses an existing
    output directory after minutes of work, and a smaller batch would
    inherit stale episodes. `--force` removes the earlier artifacts —
    never the ones a --skip flag says to reuse."""
    # A reused dataset needs no batch beside it: the dataset carries every
    # manifest in its provenance, and the machine that trains never
    # needs the frames (measured 2026-08-27: a refusal on the pod, which
    # held the dataset and not the demos, by design).
    if skip_demos and not skip_convert and not layout.demos.is_dir():
        raise SystemExit(f"e2e-smoke: --skip-demos but no batch at {layout.demos}")
    if skip_convert and not layout.dataset.is_dir():
        raise SystemExit(
            f"e2e-smoke: --skip-convert but no dataset at {layout.dataset}"
        )
    targets = [layout.training, layout.evaluation]
    if not skip_convert:
        targets.insert(0, layout.dataset)
    if not skip_demos:
        targets.insert(0, layout.demos)
    existing = [path for path in targets if path.exists()]
    if existing and not force:
        raise SystemExit(
            "e2e-smoke: this name already has artifacts: "
            + ", ".join(str(path) for path in existing)
            + " — pass --force to remove them, or --name something new"
        )
    for path in existing:
        shutil.rmtree(path)


def generate_demos(layout: RunLayout, episodes: int, seed: int) -> None:
    run(
        [
            sys.executable,
            DEMO_GENERATOR,
            episodes,
            layout.demos,
            "--frame-every",
            FRAME_EVERY,
            "--dr-span",
            DR_SPAN,
            "--seed",
            seed,
        ]
    )


def convert(layout: RunLayout) -> None:
    export_kitting_demos(layout.demos, layout.dataset, repo_id=layout.repo_id)
    provenance = json.loads(
        (layout.dataset / "provenance.json").read_text(encoding="utf-8")
    )
    print(
        f"provenance: {provenance['bundle']}, expert {provenance['expert']}, "
        f"{provenance['episodes']} episodes, "
        f"{sum(provenance['frames'])} frames at {provenance['fps']} fps; gain scales "
        f"{[round(m['gain_scale'], 2) for m in provenance['manifests']]}",
        flush=True,
    )


def train(layout: RunLayout, args: argparse.Namespace, device: str) -> None:
    inloop = RobotiqEnvConfig.cli_flags(
        KITTING, record_to=layout.inloop_records, policy_name=f"{layout.name}-inloop"
    )
    episodes = min(args.inloop_episodes, MAX_EVAL_BATCH)
    run(
        lerobot_train_command(
            policy=POLICY,
            device=device,
            dataset=layout.repo_id,
            dataset_root=layout.dataset,
            output_dir=layout.training,
            job_name=layout.training.name,
            steps=args.steps,
            batch_size=args.batch,
            save_freq=args.checkpoint_every,
            num_workers=args.workers,
            eval_freq=args.checkpoint_every,
            eval_episodes=episodes,
            eval_batch=episodes,
            extra=inloop,
        )
    )


def evaluate(layout: RunLayout, args: argparse.Namespace, device: str) -> None:
    checkpoint = layout.checkpoint(args.steps)
    if not checkpoint.is_dir():
        saved = sorted(p.name for p in (layout.training / "checkpoints").glob("*"))
        raise SystemExit(
            f"e2e-smoke: no checkpoint at {checkpoint}; the trainer saved {saved}"
        )
    run(
        lerobot_eval_command(
            policy_path=checkpoint,
            device=device,
            output_dir=layout.evaluation,
            seed=EVAL_SEED,
            episodes=args.eval_episodes,
            batch_size=min(args.eval_episodes, MAX_EVAL_BATCH),
            extra=RobotiqEnvConfig.cli_flags(
                KITTING,
                record_to=layout.eval_records,
                policy_name=f"{layout.name}-{POLICY}-{args.steps}",
            ),
        )
    )


def report(path: Path) -> None:
    records = read_records(path)
    print(
        f"{path}: {len(records)} records; "
        f"instrument {sorted({r.instrument for r in records})}; "
        f"source {sorted({r.source for r in records})}"
    )
    for score in fold(records):
        low, high = clopper_pearson(score.successes, score.trials)
        print(
            f"  {score.name}: {score.successes}/{score.trials}  CP95 [{low:.3f}, "
            f"{high:.3f}]"
        )
    print(f"  funnel {funnel(records)}")
    print(f"  events {[(r.trial, [e['name'] for e in r.events]) for r in records]}")


def stops_after(args: argparse.Namespace, stage_name: str) -> bool:
    if args.until != stage_name:
        return False
    print(f"\n--until {stage_name}: stopping here, as asked", flush=True)
    return True


def main() -> None:
    args = parse_args()
    layout = RunLayout(args.runs, args.name)
    device = best_device(args.device)
    prepare(
        layout,
        skip_demos=args.skip_demos,
        skip_convert=args.skip_convert,
        force=args.force,
    )
    if not args.skip_demos:
        started = stage(
            f"demos: {args.episodes} kept at +-{DR_SPAN:.0%} DR -> {layout.demos}"
        )
        generate_demos(layout, args.episodes, args.seed)
        done(started)
    if stops_after(args, "demos"):
        return
    if not args.skip_convert:
        started = stage(f"convert {layout.demos} -> {layout.dataset}")
        convert(layout)
        done(started)
    if stops_after(args, "convert"):
        return
    started = stage(
        f"train {POLICY} {args.steps} steps on {device}, in-loop eval through our env"
    )
    train(layout, args, device)
    done(started)
    if stops_after(args, "train"):
        return
    started = stage(f"lerobot-eval: {args.eval_episodes} paired starts, records")
    evaluate(layout, args, device)
    done(started)
    stage("fold: counts, intervals, funnel")
    report(layout.inloop_records)
    report(layout.eval_records)
    print(
        f"\nplay it: train-watch --play {layout.checkpoint(args.steps)} --task "
        f"{KITTING} "
        "--action-space bundle --look aloha2",
        flush=True,
    )


if __name__ == "__main__":
    main()
