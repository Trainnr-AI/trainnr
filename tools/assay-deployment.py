#!/usr/bin/env python3
"""The perturbation assay: a deployment on a captured scene, nominal and
moved (docs/78 §4.1), each stage gated with one seed; the success cliff
recorded on the nominal stage. Spawned by the MCP door `assay_deployment`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _lab import bootstrap

bootstrap()

from trainnr.deploy.assay import assay  # noqa: E402
from trainnr.deploy.gate import DEFAULT_SEED, DEFAULT_TRIALS  # noqa: E402
from trainnr.deploy.manifest import Key, load_manifest  # noqa: E402
from trainnr.deploy.runtime import assets_dir_of  # noqa: E402
from trainnr.deploy.runtimes import DEFAULT_RUNTIME  # noqa: E402
from trainnr.project import index_project, write_index  # noqa: E402
from trainnr.project.cited import cited_certificate  # noqa: E402
from trainnr.project.locate import DEPLOY_FOLDER, Project  # noqa: E402
from trainnr.scenes.terrain import DEFAULT_TERRAIN, terrain_names  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True, help="the deployment's folder name")
    parser.add_argument("--scene", required=True, help="the scene's folder name")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--terrain", default=DEFAULT_TERRAIN, choices=terrain_names())
    parser.add_argument(
        "--narrate", action="store_true", help="mirror every stage into the Studio"
    )
    args = parser.parse_args()
    project = Project(args.project.resolve()).use()
    deployment = project.folder(DEPLOY_FOLDER) / args.name
    scene = project.scenes / args.scene
    try:
        manifest = load_manifest(deployment)
        certificate = cited_certificate(project, manifest.raw.get(Key.CERTIFICATE))
        record = assay(
            deployment,
            scene,
            project.folder(DEPLOY_FOLDER),
            f"{args.name}-on-{args.scene}",
            assets_dir=assets_dir_of(manifest),
            trials=args.trials,
            seed=args.seed,
            runtime=DEFAULT_RUNTIME,
            narrate=args.narrate,
            certificate=certificate,
            terrain=args.terrain,
        )
    except (FileNotFoundError, FileExistsError, ValueError, ImportError) as exc:
        raise SystemExit(str(exc)) from exc
    cliff = record["cliff"]
    for row in record["perturbations"]:
        print(
            f"[assay] {row['perturbation']['label']:>9}: "
            f"{row['successes']}/{row['trials']} ci95 {row['ci95']}",
            flush=True,
        )
    print(
        f"[assay] cliff: nominal {cliff['nominal_rate']} -> worst "
        f"{cliff['worst_rate']} ({cliff['worst']}), drop {cliff['drop']}"
        + (f"; {cliff['note']}" if cliff["note"] else ""),
        flush=True,
    )
    write_index(project, index_project(project))
    sys.exit(0)


if __name__ == "__main__":
    main()
