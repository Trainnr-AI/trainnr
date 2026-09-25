"""Run the sim-to-sim gate on a project's deployment (docs/76 A6).

    cd pipeline && uv run --extra sim --extra deploy \\
        python ../tools/gate-deployment.py --project ../projects/go2-walk \\
        --name go2-c1-final --trials 20 [--runtime mujoco|dds]

Drives the exported ONNX policy through its manifest under the named
runtime (`rq_pipeline.deploy.runtimes`: plain MuJoCo, or Unitree's own
simulator and controller over DDS, stood up around the gate), judges
every trial the evaluation's way, writes the runtime's record beside
the manifest, reindexes the project. Exits 1 when the gate fails.
"""

import argparse
import sys
from pathlib import Path

from _lab import bootstrap, running, trial_reporter

bootstrap()

from rq_pipeline.deploy.gate import (  # noqa: E402
    DEFAULT_SEED,
    DEFAULT_TOLERANCE,
    DEFAULT_TRIALS,
    gate,
    gate_stream_name,
)
from rq_pipeline.deploy.manifest import Key, load_manifest  # noqa: E402
from rq_pipeline.deploy.runtime import assets_dir_of  # noqa: E402
from rq_pipeline.deploy.runtimes import (  # noqa: E402
    DEFAULT_RUNTIME,
    runtime_names,
    runtime_spec,
)
from rq_pipeline.deploy.viewport_source import scene_text  # noqa: E402
from rq_pipeline.project import index_project, write_index  # noqa: E402
from rq_pipeline.project.cited import cited_certificate  # noqa: E402
from rq_pipeline.project.locate import DEPLOY_FOLDER, Project  # noqa: E402
from rq_pipeline.scenes.stage import scene_name_of  # noqa: E402
from rq_pipeline.viz import viewer_file  # noqa: E402

# The verdict as the Running now panel's last line says it.
VERDICT_WORDS = {
    True: "passed",
    False: "failed",
    None: "judged nothing (no evaluation cited, or another protocol)",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True, help="the deployment's folder name")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument(
        "--runtime",
        default=DEFAULT_RUNTIME,
        choices=runtime_names(),
        help="what drives the policy: "
        + "; ".join(
            f"{name}: {runtime_spec(name).description}" for name in runtime_names()
        ),
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="the reference checkout with their simulator and controller built "
        "(dds only; default $TRAINNR_UNITREE_REFERENCE, else the cache)",
    )
    args = parser.parse_args()
    project = Project(args.project.resolve()).use()
    folder = project.folder(DEPLOY_FOLDER) / args.name
    manifest = load_manifest(folder)
    spec = runtime_spec(args.runtime)
    try:
        assets_dir = assets_dir_of(manifest) if spec.needs_assets else None
        stack = spec.stack_for(manifest, reference=args.reference)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    try:
        certificate = cited_certificate(project, manifest.raw.get(Key.CERTIFICATE))
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    scene_name = scene_name_of(manifest.raw)
    scene_dir = project.scenes / scene_name if scene_name else None
    if scene_dir is not None and not scene_dir.is_dir():
        print(
            f"[gate] scene {scene_name!r} not in this project: no splat in the picture"
        )
        scene_dir = None
    with (
        running(
            project.root,
            name=f"{args.name} ({spec.name})",
            viewport=scene_text(args.name),
            viewer=viewer_file(folder, gate_stream_name(spec.name)),
        ) as run,
        stack,
    ):
        run.stage(f"standing up the {spec.name} runtime")
        # What the stack started that the runtime must share (the DDS
        # stack's one virtual pad); a library runtime's NoStack shares nothing.
        shared = stack.runtime_options()
        opener = spec.open()
        run.progress(0, args.trials, "trials", f"trial 1 of {args.trials}")
        record = gate(
            folder,
            assets_dir=assets_dir,
            runtime=spec.name,
            trials=args.trials,
            seed=args.seed,
            tolerance=args.tolerance,
            certificate=certificate,
            open=lambda manifest, assets_dir=None: opener(
                manifest, assets_dir=assets_dir, **shared
            ),
            narrate=True,
            scene_dir=scene_dir,
            on_trial=trial_reporter(run),
        )
        run.progress(
            record["trials"],
            record["trials"],
            "trials",
            f"{record['successes']}/{record['trials']} tracked: "
            f"{VERDICT_WORDS[record['verdict'].get('passed')]}",
        )
    verdict = record["verdict"]
    print(
        f"[gate] {spec.name}: {record['successes']}/{record['trials']} "
        f"ci95 {record['ci95']} passed={verdict.get('passed')}",
        flush=True,
    )
    write_index(project, index_project(project))
    sys.exit(0 if verdict.get("passed") in (True, None) else 1)


if __name__ == "__main__":
    main()
