"""The instrument's MCP surface: what this repo measures, as queryable tools.

Any MCP client — the Studio's agent panel first, Claude Code in a
terminal equally — gets the same read-only window: the robot bundles
with their hash identity and fit verdicts, the provenance-gated actuator
library, the task and engine registries, and the run manifests. Reading
through the repo's public seams (`bundles`, `actuator_library`, the two
registries), never around them, so the answers an agent gets are the
same ones the pipeline itself acts on.

Read-only by design in this first pass: the server is a window into the
instrument, not a lever on it. Launching runs and mutating bundles stay
with the operator until the cheap-and-reversible phase has earned the
next one (the Studio doc's own sequencing).

The query functions are plain functions returning JSON-able dicts, with
no MCP import anywhere near them — the suite tests them directly and the
`mcp` extra is only needed to actually serve:

    cd pipeline && uv run --extra sim --extra mcp \\
        python ../tools/mcp-server.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rq_pipeline.bundles.hashing import stamp
from rq_pipeline.bundles.locate import robots_dir
from rq_pipeline.physics.registry import engines
from rq_pipeline.robot.actuator_library import (
    list_actuators,
    list_models,
    load_actuator,
)
from rq_pipeline.tasks.registry import resolve, tasks

# robots/actuators is the actuator LIBRARY (per-servo friction models,
# grown by tools/sync-bam-actuators.py), not a robot bundle — it has its
# own two tools below and stays out of the bundle census.
NOT_A_BUNDLE = ("actuators",)


def bundle_names() -> list[str]:
    return sorted(
        entry.name
        for entry in robots_dir().iterdir()
        if entry.is_dir() and entry.name not in NOT_A_BUNDLE
    )


def describe_bundles() -> list[dict[str, Any]]:
    """Every robot bundle: name@hash identity and a file census."""
    described = []
    for name in bundle_names():
        root = robots_dir() / name
        files = sorted(p.name for p in root.iterdir())
        described.append(
            {
                "name": name,
                "stamp": stamp(name, root),
                "files": files,
                "has_profile": "profile.json" in files,
                "has_fits": "fits" in files,
            }
        )
    return described


def describe_bundle(name: str) -> dict[str, Any]:
    """One bundle in full: identity, profile, every fit record, SPREAD."""
    root = robots_dir() / name
    if not root.is_dir() or name in NOT_A_BUNDLE:
        raise KeyError(f"no bundle {name!r}; one of {bundle_names()}")
    detail: dict[str, Any] = {
        "stamp": stamp(name, root),
        "files": sorted(p.name for p in root.iterdir()),
    }
    profile = root / "profile.json"
    if profile.exists():
        # Raw JSON, deliberately not through RobotProfile's validating
        # loader: that schema describes the rig drivetrain's constants,
        # and this window reports what a bundle SAYS, schema or not.
        detail["profile"] = json.loads(profile.read_text())
    fits = root / "fits"
    if fits.is_dir():
        detail["fits"] = {
            record.name: json.loads(record.read_text())
            for record in sorted(fits.glob("*.json"))
        }
    return detail


def describe_actuators() -> list[dict[str, Any]]:
    """The vendored actuator library: every servo, its tiers, provenance."""
    described = []
    for slug in list_actuators():
        model = load_actuator(slug, list_models(slug)[0])
        described.append(
            {
                "slug": slug,
                "tiers": list(list_models(slug)),
                "source": model.provenance.source,
                "citation": model.provenance.citation,
                "license": model.provenance.license,
            }
        )
    return described


def describe_actuator(slug: str, tier: str = "m6") -> dict[str, Any]:
    """One servo at one friction tier: every parameter, with provenance."""
    from dataclasses import asdict  # noqa: PLC0415 - tiny, keeps the top clean

    model = load_actuator(slug, tier)
    return {
        "slug": model.slug,
        "tier": model.tier,
        "servo": asdict(model.servo),
        "friction": asdict(model.friction),
        "provenance": asdict(model.provenance),
    }


def describe_tasks() -> list[dict[str, Any]]:
    """The task registry: what `lerobot-eval --env.task=<id>` can run."""
    return [
        {"task_id": entry.task_id, "name": entry.name, "rig": entry.rig}
        for entry in tasks().values()
    ]


def describe_task(task_id: str) -> dict[str, Any]:
    """One task built for real: its spec's numbers and its content stamp.

    Compiling the scene is what makes the stamp honest — this needs the
    sim extra, same as everything else that touches the model.
    """
    from dataclasses import asdict  # noqa: PLC0415 - tiny, keeps the top clean

    entry = resolve(task_id)
    task = entry.build()
    detail: dict[str, Any] = {
        "task_id": entry.task_id,
        "rig": entry.rig,
        "stamp": task.stamp,
    }
    spec = getattr(task, "task_spec", None)
    if spec is not None:
        detail["spec"] = asdict(spec)
    return detail


def describe_engines() -> list[dict[str, Any]]:
    """The engine registry: every physics backend an evaluation can name."""
    return [{"name": entry.name, "doc": entry.doc} for entry in engines().values()]


def describe_runs(runs_root: Path | None = None) -> list[dict[str, Any]]:
    """Every run with a manifest under `pipeline/runs/` — the same
    `run.json` the dashboard follows, verbatim."""
    root = runs_root if runs_root is not None else Path(__file__).parents[1] / "runs"
    if not root.is_dir():
        return []
    described = []
    for manifest in sorted(root.glob("*/run.json")):
        described.append(
            {"run": manifest.parent.name, "manifest": json.loads(manifest.read_text())}
        )
    return described


def build_server() -> Any:
    """The MCP server over the query functions. Needs the `mcp` extra."""
    from mcp.server import MCPServer  # noqa: PLC0415 - mcp extra

    server = MCPServer(
        name="robotiq",
        instructions=(
            "Read-only window into the robotiq instrument: robot bundles "
            "(hash-stamped, with fit records and their honesty verdicts), "
            "the provenance-gated actuator library, the task and physics-"
            "engine registries, and training-run manifests. Every answer "
            "comes through the same code paths the pipeline itself uses."
        ),
    )
    server.tool(description="Every robot bundle: name@hash, file census")(
        describe_bundles
    )
    server.tool(description="One bundle in full: profile, fit records, SPREAD verdict")(
        describe_bundle
    )
    server.tool(description="The actuator library: servos, tiers, provenance")(
        describe_actuators
    )
    server.tool(description="One servo at one friction tier (m1..m6): all parameters")(
        describe_actuator
    )
    server.tool(description="The task registry: ids, names, rigs")(describe_tasks)
    server.tool(description="One task built for real: its spec and content stamp")(
        describe_task
    )
    server.tool(description="The physics-engine registry")(describe_engines)

    # A typed no-arg wrapper: describe_runs' `runs_root` parameter exists
    # for the tests, not for clients — a Path in the tool schema would
    # only invite an argument nobody should pass.
    def runs() -> list[dict[str, Any]]:
        """Training-run manifests under pipeline/runs/."""
        return describe_runs()

    server.tool(
        name="describe_runs",
        description="Training-run manifests under pipeline/runs/",
    )(runs)
    return server


def main() -> None:
    build_server().run("stdio")
