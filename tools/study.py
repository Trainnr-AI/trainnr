"""A study with arms as DATA: press, convert, train, judge — paired.

    cd pipeline && uv run --extra sim python ../tools/study.py generate SPEC OUT
    cd pipeline && .venv-train/bin/python ../tools/study.py convert  <spec.json> <out>
    cd pipeline && .venv-train/bin/python ../tools/study.py train    <spec.json> <out>
    cd pipeline && .venv-train/bin/python ../tools/study.py evaluate <spec.json> <out>
    cd pipeline && .venv-train/bin/python ../tools/study.py finding  <spec.json> <out>

The generalisation of `paired-study.py` (C1, docs/e2e-research/62):
that tool hard-codes two arms that differ in their dynamics range; a
spec declares ANY arms — episode counts (the demo-count curve), visual
DR on or off (the visual-DR ablation), dynamics ranges (C1 at a
competent recipe) — with everything else held equal by construction:
one task, one expert, one trainer config, one evaluation protocol at a
pinned truth on matched seeds, so every between-arm comparison is
paired. The spec is the study's record; `finding` writes the result as
a tracked `docs/findings/<id>.json` with every stamp it rests on.

Spec shape (JSON):
  {"id": "demo-count-lift", "claim": "...",
   "truth": {"damping": 1.18, "gain": 0.85},
   "train": {"steps": 20000, "batch": 32},
   "eval": {"trials": 80, "variations": ["headlight.diffuse_scale=0.6:1.4"]},
   "compare": [["n8", "n128"]],
   "arms": {"n8": {"episodes": 8, "dr": {"damping": [0.7, 1.3], "gain": [0.7, 1.3]},
                   "basis": "...", "visuals": [], "visual_basis": ""}, ...}}
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from _lab import REPO, bootstrap, lerobot_eval_command, lerobot_train_command

bootstrap()

from rq_pipeline.bundles.hashing import fields_hash  # noqa: E402
from rq_pipeline.stats.effects import main_effect  # noqa: E402
from rq_pipeline.stats.intervals import clopper_pearson  # noqa: E402

EVAL_SEED = 1000  # shared by every arm: trial k starts identically everywhere
EXPERT = "scripted-pick@lift-study"
PROTOCOL = "docs/e2e-research/62-paired-study.md (arms generalised: tools/study.py)"
STUDY_FILE = "study.json"


def load_spec(path: Path) -> dict[str, Any]:
    spec = json.loads(Path(path).read_text())
    for key in ("id", "claim", "truth", "train", "eval", "arms"):
        if key not in spec:
            raise ValueError(f"{path}: spec needs {key!r}")
    for name, arm in spec["arms"].items():
        for key in ("episodes", "dr", "basis"):
            if key not in arm:
                raise ValueError(f"{path}: arm {name!r} needs {key!r}")
        if arm.get("visuals") and not arm.get("visual_basis"):
            raise ValueError(f"{path}: arm {name!r} draws visuals without a basis")
    return spec


def generate(spec: dict[str, Any], out: Path, *, seed: int, frame_every: int) -> int:
    from rq_pipeline.collect.scripted_demos import (  # noqa: PLC0415
        generate_scripted_demos,
    )
    from rq_pipeline.evaluate.variations import parse_variation  # noqa: PLC0415
    from rq_pipeline.tasks.so101 import (  # noqa: PLC0415
        LIFT_STUDY,
        build_lift_study,
        scripted_pick,
    )

    batches = {}
    for index, (name, arm) in enumerate(spec["arms"].items()):
        print(f"== arm {name}: {arm['episodes']} episodes, {arm['basis']}")
        batches[name] = generate_scripted_demos(
            out / name,
            task_factory=build_lift_study,
            policy=scripted_pick,
            expert=EXPERT,
            dr={k: tuple(v) for k, v in arm["dr"].items()},
            basis=arm["basis"],
            episodes=int(arm["episodes"]),
            seed=seed + index,  # disjoint per arm: pairing happens at evaluation
            frame_every=frame_every,
            visuals=[parse_variation(v) for v in arm.get("visuals", [])],
            visual_basis=arm.get("visual_basis", ""),
            cameras=spec.get("cameras"),
        )
    record = {
        "spec": spec,
        "spec_hash": fields_hash(spec),
        "task": LIFT_STUDY,
        "expert": EXPERT,
        "seed": seed,
        "frame_every": frame_every,
        "kept": {name: b.kept for name, b in batches.items()},
        "attempts": {name: b.attempts for name, b in batches.items()},
        "protocol": PROTOCOL,
    }
    (out / STUDY_FILE).write_text(json.dumps(record, indent=1))
    for name, b in batches.items():
        print(f"{name}: kept {b.kept}/{b.wanted} in {b.attempts} attempts")
    return 0 if all(b.complete for b in batches.values()) else 1


def convert(spec: dict[str, Any], out: Path) -> int:
    from rq_pipeline.collect.demo_export import export_demos  # noqa: PLC0415
    from rq_pipeline.tasks.so101 import build_lift_study  # noqa: PLC0415

    task = build_lift_study()
    for name in spec["arms"]:
        root = out / f"{name}-lerobot"
        export_demos(
            out / name, root, task=task, repo_id=f"rq-pipeline/{spec['id']}-{name}"
        )
        print(f"{name}: dataset -> {root}")
    return 0


def train(spec: dict[str, Any], out: Path) -> int:
    from rq_pipeline.envs.lerobot_policy import best_device  # noqa: PLC0415

    steps, batch = int(spec["train"]["steps"]), int(spec["train"]["batch"])
    device = best_device()
    for name in spec["arms"]:
        output = out / f"{name}-training"
        command = lerobot_train_command(
            policy="act",
            device=device,
            dataset=f"rq-pipeline/{spec['id']}-{name}",
            dataset_root=out / f"{name}-lerobot",
            output_dir=output,
            job_name=f"{spec['id']}-{name}",
            steps=steps,
            batch_size=batch,
            save_freq=steps,
        )
        print(f"== training {name}: {steps} steps at batch {batch} on {device}")
        subprocess.run(command, check=True)
    return 0


def _checkpoint(training_dir: Path) -> Path:
    checkpoints = sorted(
        (training_dir / "checkpoints").glob("[0-9]*"), key=lambda p: int(p.name)
    )
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints under {training_dir}")
    return checkpoints[-1] / "pretrained_model"


def evaluate(spec: dict[str, Any], out: Path) -> int:
    """Every arm judged AT the truth on matched trials; the spec's extra
    eval variations (visual sweeps) are drawn by trial index, so every
    arm sees the identical factor vector on trial k."""
    from rq_pipeline.envs.lerobot_plugin import RobotiqEnvConfig  # noqa: PLC0415
    from rq_pipeline.envs.lerobot_policy import best_device  # noqa: PLC0415
    from rq_pipeline.evaluate.records import fold, funnel, read_records  # noqa: PLC0415
    from rq_pipeline.tasks.so101 import LIFT_STUDY  # noqa: PLC0415

    truth = spec["truth"]
    trials = int(spec["eval"]["trials"])
    variations = [
        f"joints.damping_scale={truth['damping']}:{truth['damping']}",
        f"actuators.gain_scale={truth['gain']}:{truth['gain']}",
        *spec["eval"].get("variations", []),
    ]
    device = best_device()
    results: dict[str, Any] = {}
    for name in spec["arms"]:
        records_path = out / f"{name}-records.jsonl"
        records_path.unlink(missing_ok=True)
        command = lerobot_eval_command(
            policy_path=_checkpoint(out / f"{name}-training"),
            device=device,
            output_dir=out / f"{name}-eval",
            seed=EVAL_SEED,
            episodes=trials,
            batch_size=min(trials, 8),
            extra=RobotiqEnvConfig.cli_flags(
                LIFT_STUDY,
                record_to=records_path,
                policy_name=f"{spec['id']}-{name}",
                variations=variations,
                trials=trials,
            ),
        )
        print(f"== evaluating {name} at truth {truth} ({trials} paired trials)")
        subprocess.run(command, check=True)
        records = read_records(records_path)
        (score,) = fold(records)
        low, high = clopper_pearson(score.successes, score.trials)
        results[name] = {
            "successes": score.successes,
            "trials": score.trials,
            "ci95": [round(low, 4), round(high, 4)],
            "funnel": funnel(records),
            "instrument": sorted({r.instrument for r in records}),
            "outcomes": [r.success for r in sorted(records, key=lambda r: r.trial)],
        }
        print(f"{name}: {score.successes}/{score.trials} CI95 [{low:.3f}, {high:.3f}]")
    alpha, delta = (
        float(spec["eval"].get("alpha", 0.05)),
        float(spec["eval"].get("delta", 0.15)),
    )
    effects = []
    for a, b in spec.get("compare", []):
        effect = main_effect(
            f"{a} vs {b}",
            a,
            b,
            results[a]["outcomes"],
            results[b]["outcomes"],
            alpha=alpha,
            delta=delta,
        )
        effects.append(
            {
                "a": a,
                "b": b,
                "difference": effect.difference,
                "p": effect.p_value,
                "verdict": effect.verdict,
            }
        )
        print(
            f"{a} vs {b}: diff {effect.difference:+.3f}, "
            f"p {effect.p_value:.3g}, {effect.verdict}"
        )
    verdict = {
        "arms": {
            k: {kk: vv for kk, vv in v.items() if kk != "outcomes"}
            for k, v in results.items()
        },
        "effects": effects,
        "alpha": alpha,
        "delta": delta,
        "truth": truth,
        "variations": variations,
        "trials": trials,
    }
    (out / "verdict.json").write_text(json.dumps(verdict, indent=1))
    print(f"verdict -> {out / 'verdict.json'}")
    return 0


def finding(spec: dict[str, Any], out: Path, argv: list[str]) -> int:
    """The tracked record: the claim with every stamp it rests on."""
    from rq_pipeline.collect.datasheet import summarize  # noqa: PLC0415
    from rq_pipeline.evaluate.findings import (  # noqa: PLC0415
        Finding,
        repo_commit,
        today,
    )

    study = json.loads((out / STUDY_FILE).read_text())
    verdict = json.loads((out / "verdict.json").read_text())
    # Each arm's declared episode count rides with its result: the
    # figure draws a curve over it when the arms differ in count.
    for name, arm in spec["arms"].items():
        if name in verdict.get("arms", {}):
            verdict["arms"][name]["episodes"] = int(arm["episodes"])
    instruments = sorted(
        {i for arm in verdict["arms"].values() for i in arm["instrument"]}
    )
    inputs: dict[str, Any] = {
        "spec_hash": study["spec_hash"],
        "task": study["task"],
        "expert": study["expert"],
    }
    artifacts: dict[str, str] = {
        "verdict": str(out / "verdict.json"),
        "study": str(out / STUDY_FILE),
    }
    for name in spec["arms"]:
        sheet = summarize(out / name)
        inputs[f"{name}.datasheet"] = (
            f"{sheet.episodes} episodes, bases {list(sheet.bases)}, "
            f"visual bases {list(sheet.visual_bases)}"
        )
        artifacts[f"{name}.records"] = str(out / f"{name}-records.jsonl")
        artifacts[f"{name}.datasheet"] = str(out / name / "datasheet.md")
    record = Finding(
        id=f"{spec['id']}-{today()}",
        claim=spec["claim"],
        date=today(),
        repo_commit=repo_commit(REPO),
        argv=argv,
        instrument=", ".join(instruments),
        outcome=verdict,
        inputs=inputs,
        artifacts=artifacts,
        protocol=PROTOCOL,
        caveats=tuple(spec.get("caveats", [])),
    )
    from rq_pipeline.evaluate.figures import render  # noqa: PLC0415

    figures = render(record, REPO)
    record = Finding(
        **{
            **record.__dict__,
            "artifacts": {
                **record.artifacts,
                **{f"figure.{k}": v for k, v in figures.items()},
            },
        }
    )
    path = record.path(REPO)
    record.write(path)
    print(f"finding -> {path}")
    print(f"figure -> {figures['svg']} (+pdf, png, csv)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "phase", choices=["generate", "convert", "train", "evaluate", "finding"]
    )
    parser.add_argument("spec", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--frame-every", type=int, default=5)
    args = parser.parse_args()
    spec = load_spec(args.spec)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.phase == "generate":
        return generate(spec, args.out, seed=args.seed, frame_every=args.frame_every)
    if args.phase == "convert":
        return convert(spec, args.out)
    if args.phase == "train":
        return train(spec, args.out)
    if args.phase == "evaluate":
        return evaluate(spec, args.out)
    return finding(spec, args.out, sys.argv)


if __name__ == "__main__":
    sys.exit(main())
