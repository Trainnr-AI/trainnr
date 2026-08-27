"""The T5 chain at smoke scale, one command: demos -> LeRobot v3 ->
lerobot-train (in-loop eval through OUR env) -> lerobot-eval (paired
starts, per-trial records) -> the fold. "Test everything end to end"
on this repo, at a size the WSL card finishes in minutes.

    # WSL / Linux (train venv; the GPU renderer for the demos, the card for ACT):
    cd pipeline && ../tools/wsl-run.sh .venv-train/bin/python \\
        ../tools/e2e-smoke.py --episodes 2 --steps 300
    # macOS: none of the GL variables; the policy device is picked for you:
    cd pipeline && .venv-train/bin/python ../tools/e2e-smoke.py --episodes 2 --steps 300

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
        "--episodes", type=positive_int, default=2, help="demonstrations to keep"
    )
    parser.add_argument(
        "--seed", type=int, default=20260828, help="the generator's seed"
    )
    parser.add_argument(
        "--steps", type=positive_int, default=300, help="training steps"
    )
    parser.add_argument(
        "--inloop-episodes",
        type=positive_int,
        default=1,
        help="episodes of lerobot-train's own in-loop eval at the last step",
    )
    parser.add_argument(
        "--eval-episodes",
        type=positive_int,
        default=2,
        help="lerobot-eval paired starts",
    )
    parser.add_argument(
        "--device", default=None, help="torch device; default: the best present"
    )
    parser.add_argument(
        "--skip-demos",
        action="store_true",
        help="reuse the batch already under <runs>/<name>-demos",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="remove this name's earlier artifacts first (the trainer refuses "
        "an existing output directory, and a smaller batch would otherwise "
        "inherit stale episodes)",
    )
    return parser.parse_args()


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


def prepare(layout: RunLayout, *, skip_demos: bool, force: bool) -> None:
    """Refuse to write over a named run: the trainer refuses an existing
    output directory after minutes of work, and a smaller batch would
    inherit stale episodes. `--force` removes the earlier artifacts."""
    if skip_demos and not layout.demos.is_dir():
        raise SystemExit(f"e2e-smoke: --skip-demos but no batch at {layout.demos}")
    targets = [layout.dataset, layout.training, layout.evaluation]
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
        f"provenance: {provenance['bundle']}, {provenance['episodes']} episodes, "
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
            batch_size=BATCH_SIZE,
            save_freq=args.steps,
            eval_freq=args.steps,
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


def main() -> None:
    args = parse_args()
    layout = RunLayout(args.runs, args.name)
    device = best_device(args.device)
    prepare(layout, skip_demos=args.skip_demos, force=args.force)
    if not args.skip_demos:
        started = stage(
            f"demos: {args.episodes} kept at +-{DR_SPAN:.0%} DR -> {layout.demos}"
        )
        generate_demos(layout, args.episodes, args.seed)
        done(started)
    started = stage(f"convert {layout.demos} -> {layout.dataset}")
    convert(layout)
    done(started)
    started = stage(
        f"train {POLICY} {args.steps} steps on {device}, in-loop eval through our env"
    )
    train(layout, args, device)
    done(started)
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
