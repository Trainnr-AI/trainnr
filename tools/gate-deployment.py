"""Run the sim-to-sim gate on a project's deployment (docs/76 A6).

    cd pipeline && uv run --extra sim --extra deploy \\
        python ../tools/gate-deployment.py --project ../projects/go2-walk \\
        --name go2-c1-final --trials 20

Drives the exported ONNX policy through its manifest in plain MuJoCo,
judges every trial the certificate's way, writes `gate.json` beside the
manifest, reindexes the project. Exits 1 when the gate fails.
"""

import argparse
import json
import sys
from pathlib import Path

from _lab import bootstrap

bootstrap()

from rq_pipeline.deploy.gate import (  # noqa: E402
    DEFAULT_TOLERANCE,
    DEFAULT_TRIALS,
    gate,
)
from rq_pipeline.deploy.manifest import load_manifest  # noqa: E402
from rq_pipeline.deploy.runtime import assets_dir_of  # noqa: E402
from rq_pipeline.project import index_project, write_index  # noqa: E402
from rq_pipeline.project.kinds import CERTIFICATE_FILE  # noqa: E402
from rq_pipeline.project.locate import Project  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True, help="the deployment's folder name")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = parser.parse_args()
    project = Project(args.project.resolve()).use()
    folder = project.folder("deploy") / args.name
    manifest = load_manifest(folder)
    try:
        assets_dir = assets_dir_of(manifest)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    certificate = _certificate(project, manifest.raw.get("certificate"))
    record = gate(
        folder,
        assets_dir=assets_dir,
        trials=args.trials,
        seed=args.seed,
        tolerance=args.tolerance,
        certificate=certificate,
    )
    verdict = record["verdict"]
    print(
        f"[gate] {record['successes']}/{record['trials']} ci95 {record['ci95']} "
        f"passed={verdict.get('passed')}",
        flush=True,
    )
    write_index(project, index_project(project))
    sys.exit(0 if verdict.get("passed") in (True, None) else 1)


def _certificate(project: Project, stamp: str | None) -> dict | None:
    """The cited evaluation's record, found by its version's hash."""
    if not stamp or "@" not in stamp:
        return None
    wanted = stamp.split("@", 1)[1]
    for folder in project.folder("certificates").iterdir():
        path = folder / CERTIFICATE_FILE
        if not path.is_file():
            continue
        from rq_pipeline.project.kinds import Kind, stamp_kind  # noqa: PLC0415

        try:
            if stamp_kind(Kind.CERTIFICATE, folder).split("@", 1)[1] == wanted:
                return json.loads(path.read_text())
        except Exception:  # a folder that is not a certificate any more
            continue
    return None


if __name__ == "__main__":
    main()
