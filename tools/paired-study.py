"""C1, the paired study — generate the two-condition datasets.

    cd pipeline && uv run --env-file wsl.env --extra sim python \\
        ../tools/paired-study.py generate <out> \\
        [--episodes 8] [--seed 17] [--truth-damping 1.18] [--truth-gain 0.85] \\
        [--interval 0.05] [--guessed-span 0.30] [--frame-every 5]

docs/e2e-research/62 is the protocol. `generate` presses the SAME
open-loop expert on the SAME task under two dynamics-draw conditions
that differ ONLY in their range and basis: GUESSED (the folklore span
around nominal) and IDENTIFIED (a tight interval around the study's
declared truth — the stand-in a real fit record replaces). It writes
`<out>/guessed/`, `<out>/identified/` (each a DemoLayout batch with
its datasheet) and `<out>/study.json`, the record of truth, conditions
and seeds the later train/eval phases must cite.
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from _lab import bootstrap, lerobot_eval_command, lerobot_train_command

bootstrap()

from rq_pipeline.collect.scripted_demos import generate_scripted_demos  # noqa: E402
from rq_pipeline.tasks.so101 import (  # noqa: E402
    LIFT_STUDY,
    build_lift_study,
    scripted_pick,
)

EXPERT = "scripted-pick@lift-study"


def conditions(args: argparse.Namespace) -> dict[str, dict]:
    """The two arms of the study. The ONLY allowed difference is the
    range and the basis string."""
    span = args.guessed_span
    rel = args.interval
    return {
        "guessed": {
            "dr": {
                "damping": (1.0 - span, 1.0 + span),
                "gain": (1.0 - span, 1.0 + span),
            },
            "basis": f"guessed span ±{span:g} around nominal (folklore DR)",
        },
        "identified": {
            "dr": {
                "damping": (
                    args.truth_damping * (1 - rel),
                    args.truth_damping * (1 + rel),
                ),
                "gain": (args.truth_gain * (1 - rel), args.truth_gain * (1 + rel)),
            },
            "basis": (
                f"declared interval ±{rel:g} around the study truth "
                "(stand-in for an identified fit record - docs/e2e-research/62)"
            ),
        },
    }


def _checkpoint(training_dir: Path) -> Path:
    """The newest checkpoint's pretrained_model directory."""
    checkpoints = sorted((training_dir / "checkpoints").glob("[0-9]*"))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints under {training_dir}")
    return checkpoints[-1] / "pretrained_model"


def evaluate(out: Path, *, trials: int, alpha: float, delta: float) -> int:
    """Phase 4: both checkpoints judged AT the truth on matched trials —
    the dynamics pinned by degenerate variation ranges, so every trial
    of every policy runs the same world (62 §2)."""
    import subprocess  # noqa: PLC0415

    from rq_pipeline.envs.lerobot_plugin import RobotiqEnvConfig  # noqa: PLC0415
    from rq_pipeline.envs.lerobot_policy import best_device  # noqa: PLC0415
    from rq_pipeline.evaluate.records import fold, funnel, read_records  # noqa: PLC0415
    from rq_pipeline.stats.effects import main_effect  # noqa: PLC0415

    study = json.loads((out / "study.json").read_text())
    truth = study["truth"]
    pins = [
        f"joints.damping_scale={truth['damping']}:{truth['damping']}",
        f"actuators.gain_scale={truth['gain']}:{truth['gain']}",
    ]
    device = best_device()
    outcomes = {}
    for name in study["conditions"]:
        records_path = out / f"{name}-records.jsonl"
        records_path.unlink(missing_ok=True)
        command = lerobot_eval_command(
            policy_path=_checkpoint(out / f"{name}-training"),
            device=device,
            output_dir=out / f"{name}-eval",
            seed=1000,
            episodes=trials,
            batch_size=trials,
            extra=RobotiqEnvConfig.cli_flags(
                LIFT_STUDY,
                record_to=records_path,
                policy_name=f"paired-{name}",
                variations=pins,
                trials=trials,
            ),
        )
        print(f"== evaluating {name} at truth {truth} ({trials} paired trials)")
        subprocess.run(command, check=True)
        records = read_records(records_path)
        (score,) = fold(records)
        print(f"{name}: {score.successes}/{score.trials}; funnel {funnel(records)}")
        outcomes[name] = [r.success for r in sorted(records, key=lambda r: r.trial)]

    effect = main_effect(
        "dr_basis",
        "guessed",
        "identified",
        outcomes["guessed"],
        outcomes["identified"],
        alpha=alpha,
        delta=delta,
    )
    verdict = {
        "guessed": f"{effect.successes_a}/{effect.trials_a} CI {effect.interval_a}",
        "identified": f"{effect.successes_b}/{effect.trials_b} CI {effect.interval_b}",
        "difference": effect.difference,
        "p": effect.p_value,
        "verdict": effect.verdict,
        "alpha": alpha,
        "delta": delta,
        "truth": truth,
        "trials": trials,
    }
    (out / "verdict.json").write_text(json.dumps(verdict, indent=1))
    print(json.dumps(verdict, indent=1))
    print(f"verdict -> {out / 'verdict.json'}")
    return 0


def train(out: Path, *, steps: int, batch: int) -> int:
    """Phase 3: the SAME trainer config on each arm's dataset — nothing
    about the condition may leak into the config (62 §1). No in-loop
    eval: the judging happens once, paired, at truth (phase 4)."""
    import subprocess  # noqa: PLC0415

    from rq_pipeline.envs.lerobot_policy import best_device  # noqa: PLC0415

    study = json.loads((out / "study.json").read_text())
    device = best_device()
    for name in study["conditions"]:
        output = out / f"{name}-training"
        command = lerobot_train_command(
            policy="act",
            device=device,
            dataset=f"rq-pipeline/paired-{name}",
            dataset_root=out / f"{name}-lerobot",
            output_dir=output,
            job_name=f"paired-{name}",
            steps=steps,
            batch_size=batch,
            save_freq=steps,
        )
        print(f"== training {name}: {steps} steps at batch {batch} on {device}")
        subprocess.run(command, check=True)
        print(f"{name}: checkpoint under {output}")
    return 0


def convert(out: Path) -> int:
    """Phase 2: both arms to LeRobot datasets (train venv - the
    converter needs the `train` extra)."""
    from rq_pipeline.collect.demo_export import export_demos  # noqa: PLC0415

    study = json.loads((out / "study.json").read_text())
    task = build_lift_study()
    for name in study["conditions"]:
        root = out / f"{name}-lerobot"
        export_demos(out / name, root, task=task, repo_id=f"rq-pipeline/paired-{name}")
        print(f"{name}: dataset -> {root}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("phase", choices=["generate", "convert", "train", "evaluate"])
    parser.add_argument("out", type=Path)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--truth-damping", type=float, default=1.18)
    parser.add_argument("--truth-gain", type=float, default=0.85)
    parser.add_argument("--interval", type=float, default=0.05)
    parser.add_argument("--guessed-span", type=float, default=0.30)
    parser.add_argument("--frame-every", type=int, default=5)
    parser.add_argument("--train-steps", type=int, default=800)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=0.15)
    args = parser.parse_args()

    if args.phase == "convert":
        return convert(args.out)
    if args.phase == "train":
        return train(args.out, steps=args.train_steps, batch=args.batch)
    if args.phase == "evaluate":
        return evaluate(
            args.out, trials=args.trials, alpha=args.alpha, delta=args.delta
        )

    the_conditions = conditions(args)
    batches = {}
    for name, condition in the_conditions.items():
        print(f"== condition {name}: {condition['basis']}")
        batches[name] = generate_scripted_demos(
            args.out / name,
            task_factory=build_lift_study,
            policy=scripted_pick,
            expert=EXPERT,
            dr=condition["dr"],
            basis=condition["basis"],
            episodes=args.episodes,
            # One seed per condition, disjoint on purpose: the pairing
            # happens at EVALUATION (matched trials), not at generation.
            seed=args.seed + list(the_conditions).index(name),
            frame_every=args.frame_every,
        )

    study = {
        "task": LIFT_STUDY,
        "expert": EXPERT,
        "truth": {"damping": args.truth_damping, "gain": args.truth_gain},
        "conditions": {
            name: {"dr": c["dr"], "basis": c["basis"], "kept": batches[name].kept}
            for name, c in the_conditions.items()
        },
        "episodes_per_condition": args.episodes,
        "seed": args.seed,
        "generated": date.today().isoformat(),
        "protocol": "docs/e2e-research/62-paired-study.md",
    }
    (args.out / "study.json").write_text(json.dumps(study, indent=1))
    print(f"study record -> {args.out / 'study.json'}")
    complete = all(b.complete for b in batches.values())
    for name, batch in batches.items():
        print(f"{name}: kept {batch.kept}/{batch.wanted} in {batch.attempts} attempts")
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
