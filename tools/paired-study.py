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

from _lab import bootstrap

bootstrap()

from rq_pipeline.collect.scripted_demos import generate_scripted_demos  # noqa: E402
from rq_pipeline.tasks.so101 import LIFT, build_lift, scripted_pick  # noqa: E402

EXPERT = "scripted-pick@lift"


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("phase", choices=["generate"])
    parser.add_argument("out", type=Path)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--truth-damping", type=float, default=1.18)
    parser.add_argument("--truth-gain", type=float, default=0.85)
    parser.add_argument("--interval", type=float, default=0.05)
    parser.add_argument("--guessed-span", type=float, default=0.30)
    parser.add_argument("--frame-every", type=int, default=5)
    args = parser.parse_args()

    the_conditions = conditions(args)
    batches = {}
    for name, condition in the_conditions.items():
        print(f"== condition {name}: {condition['basis']}")
        batches[name] = generate_scripted_demos(
            args.out / name,
            task_factory=build_lift,
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
        "task": LIFT,
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
