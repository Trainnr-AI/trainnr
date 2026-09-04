"""Dataset-level operations — the first: a union that keeps its story.

The inventory of 2026-09-02 named "no dataset operations" as gap 11:
export took a directory whole, and two batches could not become one
dataset. The DAgger loop (docs/66 §6) needs exactly that — the base
teacher dataset plus each round's relabeled episodes, as one dataset
the next student trains on. LeRobot ships the aggregator; this module
wraps it so the union carries provenance: every source's sidecar
verbatim, the union's own stamp, and a refusal when the sources
disagree on the dynamics basis, the visual basis, the expert or the
frame rate — a union of two stories is two datasets, not one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rq_pipeline.bundles.json_record import JsonRecord
from rq_pipeline.collect.provenance import PROVENANCE_FILE

# What every source must agree on for a union to be ONE dataset.
MUST_AGREE = ("fps", "expert", "bundle")
BASIS_KEYS = ("dynamics_basis", "visual_basis")
MIN_SOURCES = 2  # a union of one is the dataset itself


@dataclass(frozen=True)
class UnionProvenance(JsonRecord):
    """The union's sidecar: which datasets, in order, with their own
    sidecars verbatim, and the counts the union adds up to."""

    sources: list[str]  # the source datasets' directory NAMES, in order
    source_provenance: list[dict[str, Any]]  # each source's sidecar, verbatim
    episodes: int
    fps: int
    expert: str
    bundle: str
    note: str = (
        "a union keeps every source's episodes and manifests; nothing is resampled"
    )


def _read_provenance(root: Path) -> dict[str, Any]:
    path = Path(root) / PROVENANCE_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"{root} carries no {PROVENANCE_FILE}: a dataset without a sidecar "
            "cannot join a union (export it through demo_export first)"
        )
    return json.loads(path.read_text())


def _manifest_bases(provenance: dict[str, Any]) -> dict[str, set[str]]:
    bases: dict[str, set[str]] = {key: set() for key in BASIS_KEYS}
    for manifest in provenance.get("manifests", []):
        for key in BASIS_KEYS:
            bases[key].add(str(manifest.get(key) or ""))
    return bases


def check_union(sources: list[Path]) -> list[dict[str, Any]]:
    """Every source's sidecar, after refusing any disagreement on what
    a single dataset must share. Returns the sidecars in source order."""
    if len(sources) < MIN_SOURCES:
        raise ValueError(f"a union needs at least two datasets, got {len(sources)}")
    sidecars = [_read_provenance(root) for root in sources]
    for key in MUST_AGREE:
        values = {str(s.get(key)) for s in sidecars}
        if len(values) != 1:
            raise ValueError(f"sources disagree on {key}: {sorted(values)}")
    for key in BASIS_KEYS:
        values: set[str] = set()
        for s in sidecars:
            values |= _manifest_bases(s)[key]
        if len(values) > 1:
            raise ValueError(
                f"sources disagree on {key}: {sorted(values)} — a union of two "
                "bases is two datasets (docs/e2e-research/58 §8)"
            )
    return sidecars


def concat_datasets(sources: list[Path], out: Path, *, repo_id: str) -> Path:
    """The sources, in order, as one LeRobot dataset at `out`, with a
    `UnionProvenance` sidecar beside it."""
    from lerobot.datasets.aggregate import aggregate_datasets  # noqa: PLC0415

    sources = [Path(s) for s in sources]
    sidecars = check_union(sources)
    out = Path(out)
    aggregate_datasets(
        repo_ids=[f"{repo_id}-src{i}" for i in range(len(sources))],
        aggr_repo_id=repo_id,
        roots=sources,
        aggr_root=out,
    )
    UnionProvenance(
        sources=[s.name for s in sources],
        source_provenance=sidecars,
        episodes=sum(int(s.get("episodes", 0)) for s in sidecars),
        fps=int(sidecars[0]["fps"]),
        expert=str(sidecars[0]["expert"]),
        bundle=str(sidecars[0]["bundle"]),
    ).write(out / PROVENANCE_FILE)
    return out
