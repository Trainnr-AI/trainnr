"""Multiply kept kitting seeds into new episodes at device scale — B3.2.

    cd pipeline && ../tools/wsl-run.sh .venv-train/bin/python \\
        ../tools/press-multiply.py <seeds-dir> <out> \\
        [--episodes 4] [--worlds 64] [--seed 11] [--dr-span 0.30] [--impl warp]

Seeds are `DemoLayout` episode directories (the kitting press's output).
Per round: one damping/gain draw rescales the scene, `--worlds`
candidates (spawns jittered within the measured ±5 mm replay basin,
actions noised) ride one batched device rollout, and only the device's
keepers pay for the CPU reference — which renders the frames and issues
the shipping verdict. The core is
`rq_pipeline.collect.multiply.multiply` (tested there); this is its
kitting command line. Ends with the datasheet over the multiplied set.
"""

import argparse
import sys
from pathlib import Path

from _lab import bootstrap

bootstrap()

from rq_pipeline.collect.datasheet import write_datasheet  # noqa: E402
from rq_pipeline.collect.kitting_demos import (  # noqa: E402
    DR_SPAN,
    kitting_dynamics_fn,
    kitting_variant_fn,
)
from rq_pipeline.collect.multiply import MultiplyPlan, multiply  # noqa: E402
from rq_pipeline.collect.press_batch import Seed  # noqa: E402

# Contact/constraint capacity per world: the kitting bundle's measured
# pair (docs/07 2026-08-27) was 4096/8192 for a HANDFUL of worlds;
# 512/1024 per world held at W=64 (docs/07 B3.1).
NACONMAX_PER_WORLD = 512
NJMAX_PER_WORLD = 1024


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("seeds", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--worlds", type=int, default=64)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--dr-span", type=float, default=DR_SPAN)
    parser.add_argument("--impl", default=None, help="device impl (default: backend's)")
    parser.add_argument("--first-episode", type=int, default=0)
    args = parser.parse_args()

    from rq_pipeline.physics.mujoco_backend import (  # noqa: PLC0415
        MuJoCoBackend,
        keyframe_state,
    )
    from rq_pipeline.tasks.aloha2 import KittingSpec, build_kitting  # noqa: PLC0415

    seed_dirs = sorted(args.seeds.glob("episode_*"))
    if not seed_dirs:
        print(f"no episode_* directories under {args.seeds}", file=sys.stderr)
        return 1
    seeds = [Seed.read(d) for d in seed_dirs]

    spec = KittingSpec()
    task = build_kitting(spec=spec)
    backend = MuJoCoBackend()
    backend.load_spec(build_kitting(spec=spec).spec)
    home = keyframe_state(backend.model, task.protocol.home)

    engine: dict = {
        "naconmax": NACONMAX_PER_WORLD * args.worlds,
        "njmax": NJMAX_PER_WORLD * args.worlds,
    }
    if args.impl:
        engine["impl"] = args.impl
    else:
        # The backend's default impl is Warp — the GPU instrument. On a
        # machine without CUDA that dies deep inside Warp; refuse here,
        # naming the flag (review 2026-09-01: this tool now runs on the
        # Mac too).
        try:
            import warp as wp  # noqa: PLC0415

            cuda = wp.is_cuda_available()
        except ImportError:
            # No warp at all (plain sim extra): same refusal, not a
            # traceback (second review, 2026-09-01).
            cuda = False
        if not cuda:
            print(
                "no CUDA device: the default MJX impl is 'warp' (the GPU "
                "instrument). Pass --impl jax to multiply on this machine, "
                "or run on the box.",
                file=sys.stderr,
            )
            return 1

    plan = MultiplyPlan(
        variant_fn=kitting_variant_fn(home),
        dynamics_fn=kitting_dynamics_fn(spec, args.dr_span),
        episodes=args.episodes,
        worlds=args.worlds,
        engine=engine,
    )
    result = multiply(
        args.out,
        task=task,
        seeds=seeds,
        plan=plan,
        seed=args.seed,
        instrument=backend.instrument,
        control_hz=task.control_hz,
        first_episode=args.first_episode,
    )
    print(
        f"kept {result.kept}/{result.wanted} from {result.candidates} candidates "
        f"in {result.rounds} round(s); device kept {result.device_kept}, "
        f"CPU refused {result.false_positives} (filter false-positives, not shipped)"
    )
    if result.kept:
        print(f"datasheet -> {write_datasheet(args.out)}")
    return 0 if result.complete else 1


if __name__ == "__main__":
    sys.exit(main())
