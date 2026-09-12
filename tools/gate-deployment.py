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
import json
import sys
from pathlib import Path
from typing import Any

from _lab import bootstrap

bootstrap()

from rq_pipeline.deploy.gate import (  # noqa: E402
    DEFAULT_SEED,
    DEFAULT_TOLERANCE,
    DEFAULT_TRIALS,
    gate,
)
from rq_pipeline.deploy.manifest import Key, load_manifest  # noqa: E402
from rq_pipeline.deploy.runtime import assets_dir_of  # noqa: E402
from rq_pipeline.deploy.runtimes import (  # noqa: E402
    DEFAULT_RUNTIME,
    runtime_names,
    runtime_spec,
)
from rq_pipeline.project import index_project, write_index  # noqa: E402
from rq_pipeline.project.kinds import CERTIFICATE_FILE, Kind, stamp_kind  # noqa: E402
from rq_pipeline.project.locate import Project  # noqa: E402

DEPLOY_FOLDER = "deploy"
CERTIFICATES_FOLDER = "certificates"


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
    certificate = _certificate(project, manifest.raw.get(Key.CERTIFICATE))
    with stack:
        # What the stack started that the runtime must share (the DDS
        # stack's one virtual pad); a library runtime shares nothing.
        shared = getattr(stack, "runtime_options", dict)()
        opener = spec.open()
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
        )
    verdict = record["verdict"]
    print(
        f"[gate] {spec.name}: {record['successes']}/{record['trials']} "
        f"ci95 {record['ci95']} passed={verdict.get('passed')}",
        flush=True,
    )
    write_index(project, index_project(project))
    sys.exit(0 if verdict.get("passed") in (True, None) else 1)


def _certificate(project: Project, stamp: str | None) -> dict[str, Any] | None:
    """The cited evaluation's record, found by its version's hash among
    the project's evaluations; None when the manifest cites none."""
    if not stamp or "@" not in stamp:
        return None
    wanted = stamp.split("@", 1)[1]
    for folder in sorted(project.folder(CERTIFICATES_FOLDER).iterdir()):
        path = folder / CERTIFICATE_FILE
        if not path.is_file():
            continue
        if stamp_kind(Kind.CERTIFICATE, folder).split("@", 1)[1] == wanted:
            return json.loads(path.read_text(encoding="utf-8"))
    raise SystemExit(
        f"the manifest cites evaluation {stamp!r}, which is not in "
        f"{project.folder(CERTIFICATES_FOLDER)}"
    )


if __name__ == "__main__":
    main()
