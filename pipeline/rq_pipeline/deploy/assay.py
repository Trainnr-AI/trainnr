"""The perturbation assay (docs/78 §4.1, §9 step 3): the same
deployment gated on the same scene with the terrain moved by the
field's ±20 mm on each axis and ±5° of yaw about the start
(`scenes.stage.ASSAY`), the policy left believing nothing moved. The
success cliff - how far the rate falls from nominal under the worst
move - is the number that sets the collision tolerance a task declares
and the geometric span it randomizes over; a visual metric never does
(2608.21416: a 20 mm mesh shift took a task from 30 % to 0 % with the
image similarity unchanged).

One stage per perturbation, each a deployment folder beside the nominal
one; one record, `assay.json`, in the nominal one. Honest when nothing
walked at nominal: the cliff is then unmeasurable and the record says so.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rq_pipeline.bundles.hashing import stamp
from rq_pipeline.deploy.gate import DEFAULT_SEED, DEFAULT_TRIALS, gate
from rq_pipeline.deploy.manifest import Key, load_manifest
from rq_pipeline.deploy.runtimes import DEFAULT_RUNTIME, Opener
from rq_pipeline.scenes.stage import ASSAY, NOMINAL, Perturbation, stage_deployment
from rq_pipeline.scenes.terrain import DEFAULT_TERRAIN

ASSAY_FILE = "assay.json"
ASSAY_SCHEMA = "trainnr-assay/1"
STAGE_NAME = "{name}-{label}"
UNMEASURABLE = "no walk succeeded at nominal: the cliff is unmeasurable here"


def stage_name(name: str, perturbation: Perturbation) -> str:
    """Where a perturbation's stage lives: the nominal name, or the
    nominal name with the perturbation's label."""
    if perturbation == NOMINAL:
        return name
    return STAGE_NAME.format(name=name, label=perturbation.label)


OTHER_STAGE = (
    "{folder} is staged {have}, this assay stages {want}: nine rows are one "
    "cliff only on one terrain from one start; assay under another name, "
    "or remove that stage"
)


def require_same_stage(folder: Path, perturbation: Perturbation, terrain: str) -> None:
    """A stage the assay reuses must be the one it would have built: the
    same terrain kind and the same perturbation (the nominal stage's
    name is also `stage_deployment`'s default, so a stage made by hand
    on other terms could otherwise sit as the assay's nominal row)."""
    block = load_manifest(folder).raw.get(Key.SCENE) or {}
    have = (
        str(block.get("terrain_kind")),
        str((block.get("perturbation") or {}).get("label")),
    )
    want = (terrain, perturbation.label)
    if have != want:
        raise ValueError(
            OTHER_STAGE.format(
                folder=folder.name,
                have=f"on {have[0]} as {have[1]}",
                want=f"on {want[0]} as {want[1]}",
            )
        )


def assay(  # noqa: PLR0913 - the assay's own knobs, each named
    deployment_dir: Path,
    scene_dir: Path,
    deploy_root: Path,
    name: str,
    *,
    assets_dir: Path,
    trials: int = DEFAULT_TRIALS,
    seed: int = DEFAULT_SEED,
    runtime: str = DEFAULT_RUNTIME,
    perturbations: tuple[Perturbation, ...] = ASSAY,
    open: Opener | None = None,
    narrate: bool = False,
    terrain: str | None = None,
    certificate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stage and gate every perturbation, the nominal first; returns
    the record written to the nominal stage. A stage that exists is
    reused (its gate re-run), so an assay resumes."""
    terrain = terrain or DEFAULT_TERRAIN
    rows: list[dict[str, Any]] = []
    for p in perturbations:
        folder = deploy_root / stage_name(name, p)
        if folder.exists():
            require_same_stage(folder, p, terrain)
        else:
            stage_deployment(
                deployment_dir,
                scene_dir,
                folder,
                assets_dir=assets_dir,
                perturbation=p,
                terrain=terrain,
            )
        record = gate(
            folder,
            assets_dir=assets_dir,
            runtime=runtime,
            trials=trials,
            seed=seed,
            open=open,
            narrate=narrate,
            scene_dir=scene_dir,
            certificate=certificate,
        )
        rows.append(
            {
                "perturbation": asdict(p),
                "deployment": folder.name,
                "successes": record["successes"],
                "trials": record["trials"],
                "ci95": record["ci95"],
            }
        )
    nominal = rows[0]
    nominal_rate = nominal["successes"] / max(nominal["trials"], 1)
    worst = min(
        rows[1:], key=lambda r: r["successes"] / max(r["trials"], 1), default=nominal
    )
    worst_rate = worst["successes"] / max(worst["trials"], 1)
    cliff = {
        "nominal_rate": round(nominal_rate, 4),
        "worst_rate": round(worst_rate, 4),
        "drop": round(nominal_rate - worst_rate, 4),
        "worst": worst["perturbation"]["label"],
        "measurable": nominal["successes"] > 0,
        "note": "" if nominal["successes"] > 0 else UNMEASURABLE,
    }
    out = {
        "schema": ASSAY_SCHEMA,
        "deployment": name,
        "scene": stamp(scene_dir.name, scene_dir),
        "runtime": runtime,
        "protocol": {"trials": trials, "seed": seed, "terrain": terrain},
        "perturbations": rows,
        "cliff": cliff,
        "judged": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    (deploy_root / name / ASSAY_FILE).write_text(
        json.dumps(out, indent=1) + "\n", encoding="utf-8"
    )
    return out


def read_assay(folder: Path) -> dict[str, Any] | None:
    path = Path(folder) / ASSAY_FILE
    if not path.is_file():
        return None
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return raw
