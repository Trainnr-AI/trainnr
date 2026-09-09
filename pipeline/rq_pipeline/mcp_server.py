"""The instrument's MCP surface: what this repo measures, as queryable tools.

Any MCP client — the Studio's agent panel first, Claude Code in a
terminal equally — gets the same read-only window: the robot bundles
with their hash identity and fit verdicts, the provenance-gated actuator
library, the task and engine registries, and the run manifests. Reading
through the repo's public seams (`bundles`, `actuator_library`, the two
registries), never around them, so the answers an agent gets are the
same ones the pipeline itself acts on.

Two kinds of tools (docs/64 §3): the DESCRIBE family — read-only
windows through the repo's public seams — and, since S2, the ACT
family (`rq_pipeline.mcp_actions`): thin doors that spawn the CLI
owning the work as a background job and hand back a handle
(`job_status` polls, artifacts land under `runs/` as always). The
agent lives in the developer's own tool; these tools are how it
presses, trains, certifies and opens the Studio.

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

# BUNDLE_STORE has ONE home (the bundle module itself); it was spelled
# three ways once — review 2026-09-01.
from rq_pipeline.robot.actuator_bundle import BUNDLE_STORE
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


def describe_actuator_bundles() -> list[dict[str, Any]]:
    """Every certified actuator bundle in the committed store — stamp,
    check flags (optimizer rails/floors) and honesty advisories."""
    from rq_pipeline.robot.actuator_bundle import read_bundle, verify  # noqa: PLC0415

    if not BUNDLE_STORE.is_dir():
        raise FileNotFoundError(
            f"no bundle store at {BUNDLE_STORE} — an empty list here would "
            "read as 'no bundles' when the path is simply wrong; wrap the "
            "vendored fits with tools/actuator-bundle.py wrap --all"
        )
    described = []
    for path in sorted(BUNDLE_STORE.glob("*.bundle.json")):
        bundle = read_bundle(path)  # verifies on read; a bad file raises by name
        described.append(
            {
                "file": path.name,
                "stamp": bundle["stamp"],
                "checks": bundle["checks"],
                "advisories": verify(bundle),
            }
        )
    return described


def describe_actuator_bundle(slug: str, tier: str = "m6") -> dict[str, Any]:
    """One certified bundle in full — BAM's params verbatim plus the
    envelope — with its advisories riding along."""
    from rq_pipeline.robot.actuator_bundle import read_bundle, verify  # noqa: PLC0415

    path = BUNDLE_STORE / f"{slug}.{tier}.bundle.json"
    if not path.exists():
        available = sorted(p.name for p in BUNDLE_STORE.glob("*.bundle.json"))
        raise KeyError(f"no bundle {path.name!r} in the store; have {available}")
    bundle = read_bundle(path)
    return {"bundle": bundle, "advisories": verify(bundle)}


def describe_datasheet(demos_dir: str) -> dict[str, Any]:
    """A demo batch's datasheet, as data: kept episodes, keep-rate
    bound, stamps, dynamics spreads with their bases, warnings."""
    from dataclasses import asdict  # noqa: PLC0415

    from rq_pipeline.collect.datasheet import summarize  # noqa: PLC0415

    summary = summarize(Path(demos_dir))
    return {
        **asdict(summary),
        # None across shards: each shard restarts its attempt counter,
        # so no single bound exists (the render refuses too).
        "keep_rate_bound": summary.keep_rate_bound if summary.shards == 1 else None,
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
    root = _runs_root(runs_root)
    if not root.is_dir():
        return []
    described = []
    for manifest in sorted(root.glob("*/run.json")):
        described.append(
            {"run": manifest.parent.name, "manifest": json.loads(manifest.read_text())}
        )
    return described


def _jobs_root() -> Path:
    """Where the MCP job table lives: the current project's root, else
    the legacy `pipeline/runs`. The job manager appends `mcp-jobs/`."""
    from rq_pipeline.project import current_project  # noqa: PLC0415

    try:
        return current_project().root
    except FileNotFoundError:
        return _runs_root(None)


def _runs_root(runs_root: Path | None) -> Path:
    return runs_root if runs_root is not None else Path(__file__).parents[1] / "runs"


def list_eval_records(runs_root: Path | None = None) -> list[dict[str, Any]]:
    """Every episode-record file under `pipeline/runs/` — the JSONL the
    evaluation layer writes, wherever a run keeps one."""
    root = _runs_root(runs_root)
    if not root.is_dir():
        return []
    found = []
    for path in sorted(root.glob("*/*episodes.jsonl")):
        found.append(
            {
                "run": path.parent.name,
                "file": path.name,
                "records": sum(1 for line in path.read_text().splitlines() if line),
            }
        )
    return found


def describe_eval(run: str, runs_root: Path | None = None) -> dict[str, Any]:
    """One run's episode records, folded the way the certificate is:
    trials, successes, the milestone funnel, and every trial's verdict —
    through `rq_pipeline.evaluate.records`, never a private re-parse."""
    from rq_pipeline.evaluate.records import (  # noqa: PLC0415 - keeps import cheap
        funnel,
        milestones,
        read_records,
    )

    root = _runs_root(runs_root)
    paths = sorted((root / run).glob("*episodes.jsonl"))
    if not paths:
        known = [entry["run"] for entry in list_eval_records(runs_root)]
        raise KeyError(f"no episode records under {run!r}; runs with records: {known}")
    detail: dict[str, Any] = {"run": run, "files": {}}
    for path in paths:
        records = read_records(path)
        detail["files"][path.name] = {
            "records": len(records),
            "successes": sum(1 for r in records if r.success),
            "milestones": milestones(records),
            "funnel": funnel(records),
            "trials": [
                {
                    "trial": r.trial,
                    "policy": r.policy,
                    "success": r.success,
                    "steps": r.steps,
                    "instrument": r.instrument,
                    "events": [dict(event) for event in r.events],
                }
                for r in records
            ],
        }
    return detail


def friction_curve(
    slug: str, tier: str = "m6", points: int = 101, tau_external: float = 0.3
) -> dict[str, Any]:
    """The actuator's friction-torque budget over its velocity range, as
    plottable curves — computed by `friction_torque_budget` itself, so a
    chart of this data is a chart of the model, not of a re-derivation.

    Two curves: unloaded, and under `tau_external` N·m of external
    torque (the load-dependent M3-M6 terms are invisible without load).
    """
    import numpy as np  # noqa: PLC0415 - sim/numpy extra

    from rq_pipeline.robot.friction_budget import (  # noqa: PLC0415
        friction_torque_budget,
    )

    model = load_actuator(slug, tier)
    top = model.servo.max_velocity if model.servo.max_velocity is not None else 8.0
    velocity = np.linspace(0.0, top, points)
    zero = np.zeros_like(velocity)
    unloaded = friction_torque_budget(model.friction, velocity, zero, zero)
    loaded = friction_torque_budget(
        model.friction, velocity, zero, np.full_like(velocity, tau_external)
    )
    return {
        "slug": slug,
        "tier": tier,
        "tau_external": tau_external,
        "velocity": velocity.tolist(),
        "unloaded": np.asarray(unloaded).tolist(),
        "loaded": np.asarray(loaded).tolist(),
    }


def describe_project(refresh: bool = True) -> dict[str, Any]:
    """The current project (`$TRAINNR_PROJECT`, else `projects/default`):
    every artifact by kind with its stamp and the stamps it cites, the
    loop map with each state proved or missing, and the next legal move.
    Re-indexes from the files by default; the index is a cache."""
    from rq_pipeline.project import (  # noqa: PLC0415
        current_project,
        index_project,
        write_index,
    )

    project = current_project()
    if refresh or not project.index_path.is_file():
        # write_index renders previews and records their paths; return
        # what was WRITTEN, so the agent sees the same index the Studio does.
        write_index(project, index_project(project))
    return dict(json.loads(project.index_path.read_text(encoding="utf-8")))


def list_project_dirs() -> list[dict[str, Any]]:
    """Every project under `projects/`: name, root, and its loop progress
    (stages proved of eight) from its index when one exists."""
    from rq_pipeline.project import list_projects  # noqa: PLC0415

    listed = []
    for project in list_projects():
        entry: dict[str, Any] = {"name": project.name, "root": str(project.root)}
        if project.index_path.is_file():
            raw = json.loads(project.index_path.read_text())
            entry["stages_proved"] = sum(
                1 for s in raw.get("states", []) if s.get("present")
            )
            entry["stages"] = len(raw.get("states", []))
            entry["artifacts"] = len(raw.get("artifacts", []))
            entry["indexed"] = raw.get("indexed")
        listed.append(entry)
    return listed


def list_robot_adapters() -> list[dict[str, str]]:
    """Every source format a recording can enter through: this repo's
    wire telemetry, a LeRobot dataset, a ROS 2 MCAP bag, plus plugins."""
    from rq_pipeline.robots import list_adapters  # noqa: PLC0415

    return [{"name": e.name, "doc": e.doc} for e in list_adapters().values()]


def ingest_recording(
    source: str, name: str | None = None, adapter: str | None = None
) -> dict[str, Any]:
    """Read a robot's telemetry (a .wire file, a LeRobot dataset directory,
    a ROS 2 .mcap bag) into the current project as a stamped recording:
    named channels with units and rates, a census of what the robot
    reported, and the adapter's honest notes. The first move of the loop."""
    from rq_pipeline.project import current_project  # noqa: PLC0415
    from rq_pipeline.robots.ingest import ingest  # noqa: PLC0415

    return ingest(current_project(), Path(source), name=name, adapter=adapter)


# -- the Studio's control surface (docs/76 §10.1) ----------------------------------


def describe_studio() -> dict[str, Any]:
    """What the Studio shows right now — project, page, selected artifact,
    the viewer's recording and time cursor, whether the presenter runs —
    with `alive` (a fresh heartbeat from a live pid) and the presenter's
    last status. Read from `<project>/.index/studio-state.json`."""
    import json  # noqa: PLC0415

    from rq_pipeline.project import current_project  # noqa: PLC0415
    from rq_pipeline.project.control import state  # noqa: PLC0415
    from rq_pipeline.project.locate import INDEX_DIR  # noqa: PLC0415

    project = current_project()
    out = state(project)
    status = project.root / INDEX_DIR / "present-status.json"
    if status.is_file():
        try:
            out["presenter"] = json.loads(status.read_text(encoding="utf-8"))
        except ValueError:
            out["presenter"] = {"error": "unreadable present-status.json"}
    return out


def launch_studio() -> dict[str, Any]:
    """Start the Studio window on the current project (the built binary:
    `$TRAINNR_STUDIO`, else crates/studio-shell/target/release) and wait
    for its first heartbeat. Refuses when one already runs."""
    from rq_pipeline.project import current_project  # noqa: PLC0415
    from rq_pipeline.project.control import launch  # noqa: PLC0415

    return launch(current_project())


def quit_studio() -> dict[str, Any]:
    """Close the Studio: a `quit` command first; past the timeout, the
    process is terminated by pid."""
    from rq_pipeline.project import current_project  # noqa: PLC0415
    from rq_pipeline.project.control import quit as quit_  # noqa: PLC0415

    return quit_(current_project())


def open_in_studio(
    section: str | None = None,
    artifact: str | None = None,
    project: str | None = None,
    table: str | None = None,
) -> dict[str, Any]:
    """Navigate the Studio: a page by name (projects, overview, robots,
    environments, recordings, datasets, experiments, policies, evaluations,
    findings, deployments, monitoring, live), an artifact by version (its
    page opens with the detail drawer), another project by root path, or
    one of the selected artifact's tables by title (Joints, Actuators,
    Episodes…) in the exploration modal — an empty string closes it."""
    from rq_pipeline.project import current_project  # noqa: PLC0415
    from rq_pipeline.project.control import SECTIONS, command  # noqa: PLC0415

    if section is not None and section.strip().lower() not in SECTIONS:
        return {
            "status": "refused",
            "reason": f"no page {section!r}; one of {', '.join(SECTIONS)}",
        }
    return command(
        current_project(),
        "open",
        section=section,
        artifact=artifact,
        project=project,
        table=table,
    )


def show_in_studio(artifact: str) -> dict[str, Any]:
    """Stream one artifact into the Studio's viewer as itself (a robot as
    its meshes in 3D, a recording as time series, an experiment as its
    curves, an evaluation as its funnel) and switch to the Live view."""
    from rq_pipeline.project import current_project  # noqa: PLC0415
    from rq_pipeline.project.control import command  # noqa: PLC0415

    return command(current_project(), "show", artifact=artifact)


def compare_in_studio(a: str, b: str) -> dict[str, Any]:
    """Two artifacts side by side in the viewer, `a` left and `b` right —
    two robots, two recordings, an experiment beside its evaluation."""
    from rq_pipeline.project import current_project  # noqa: PLC0415
    from rq_pipeline.project.control import command  # noqa: PLC0415

    return command(current_project(), "compare", a=a, b=b)


# One door, one timeline: every knob the viewer's own time panel has.
def set_studio_time(  # noqa: PLR0913, PLR0917
    timeline: str | None = None,
    seconds: float | None = None,
    sequence: int | None = None,
    play: bool | None = None,
    speed: float | None = None,
    start: float | None = None,
    end: float | None = None,
    follow: bool | None = None,
    step: int | None = None,
) -> dict[str, Any]:
    """Drive the viewer's timeline: pick a timeline by name, put the cursor
    at `seconds` (duration or timestamp timelines) or `sequence` (step,
    frame, episode), play or pause, set the playback speed, select a
    time range (`start`..`end`, seconds), follow the newest data, or step
    by frames. Refused when nothing is streaming."""
    from rq_pipeline.project import current_project  # noqa: PLC0415
    from rq_pipeline.project.control import command  # noqa: PLC0415

    return command(
        current_project(),
        "time",
        timeline=timeline,
        seconds=seconds,
        sequence=sequence,
        play=play,
        speed=speed,
        start=start,
        end=end,
        follow=follow,
        step=step,
    )


def set_studio_panels(
    blueprint: str | None = None, selection: str | None = None
) -> dict[str, Any]:
    """The viewer's side panels: `expand` or `toggle` the blueprint panel
    (left: what each view shows) and the selection panel (right: the
    selected entity's properties)."""
    from rq_pipeline.project import current_project  # noqa: PLC0415
    from rq_pipeline.project.control import PANEL_ACTIONS, command  # noqa: PLC0415

    for name, value in (("blueprint", blueprint), ("selection", selection)):
        if value is not None and value not in PANEL_ACTIONS:
            return {
                "status": "refused",
                "reason": f"{name}: {value!r} is not one of {', '.join(PANEL_ACTIONS)}",
            }
    return command(
        current_project(), "panels", blueprint=blueprint, selection=selection
    )


def simulate_in_studio(task: str | None = None) -> dict[str, Any]:
    """Run a scene in the Studio's MuJoCo viewport — a preview task by name
    (kitting, lift, duck) or `walk` for the newest trained walk policy —
    and switch to the Live view; with no task, stop the viewport. The
    state file then reports `viewport_task` and `viewport_fps`, the
    frames drawn to the screen in the last second."""
    from rq_pipeline.project import current_project  # noqa: PLC0415
    from rq_pipeline.project.control import command  # noqa: PLC0415

    return command(current_project(), "simulate", task=task)


# One door, one simulate section: every knob of simulate's Simulation panel.
def control_simulator(  # noqa: PLR0913, PLR0917
    run: bool | None = None,
    step: int | None = None,
    reset: bool | None = None,
    keyframe: str | None = None,
    speed: float | None = None,
    manual: bool | None = None,
) -> dict[str, Any]:
    """MuJoCo simulate's Simulation section on the running scene: `run`
    (True runs, False pauses), `step` n physics steps (pauses and takes
    manual control), `reset` to the initial state or to a `keyframe` by
    name, `speed` as a real-time factor (0.01..100), `manual` (True: the
    sliders drive the scene; False: its own motion, from its start).
    Refused when no scene runs — simulate_in_studio first."""
    from rq_pipeline.project import current_project  # noqa: PLC0415
    from rq_pipeline.project.control import command  # noqa: PLC0415

    return command(
        current_project(),
        "simulator",
        run=run,
        step=step,
        reset=reset,
        keyframe=keyframe,
        speed=speed,
        manual=manual,
    )


def set_simulator_input(
    value: float, actuator: str | None = None, joint: str | None = None
) -> dict[str, Any]:
    """One slider of simulate's Control or Joint panel: an actuator's
    control value by name, or a hinge/slide joint's position by name
    (free and ball joints have no scalar). Takes manual control of the
    scene. Names come from the robot's Actuators and Joints tables."""
    from rq_pipeline.project import current_project  # noqa: PLC0415
    from rq_pipeline.project.control import command  # noqa: PLC0415

    if (actuator is None) == (joint is None):
        return {"status": "refused", "reason": "name exactly one of actuator, joint"}
    return command(
        current_project(), "simulator", actuator=actuator, joint=joint, value=value
    )


def set_simulator_view(
    flag: str | None = None,
    on: bool | None = None,
    camera: str | None = None,
    inspect: str | None = None,
) -> dict[str, Any]:
    """What the simulator shows: a MuJoCo visualization or rendering
    `flag` by its own name with `on` (contactpoint, contactforce, joint,
    actuator, constraint, inertia, com, transparent, perturbforce, camera,
    light, tendon… or shadow, reflection, skybox, fog, wireframe…); the
    `camera` to a named view (front, side, top, reset); the Inspect drawer
    by tab (control, joints, physics) or close."""
    from rq_pipeline.project import current_project  # noqa: PLC0415
    from rq_pipeline.project.control import command  # noqa: PLC0415

    if flag is not None and on is None:
        return {"status": "refused", "reason": "a flag needs `on`"}
    if flag is None and camera is None and inspect is None:
        return {"status": "refused", "reason": "name a flag, a camera view or inspect"}
    return command(
        current_project(),
        "simulator",
        flag=flag,
        on=on,
        view=camera,
        inspect=inspect,
    )


def screenshot_studio(
    section: str | None = None, artifact: str | None = None, width: int = 1600
) -> dict[str, Any]:
    """See the window: capture the whole Studio as a PNG (scaled to `width`,
    never upscaled) and return its path — read that file to look at it.
    Pass a page name or an artifact version to navigate there first."""
    from rq_pipeline.project import current_project  # noqa: PLC0415
    from rq_pipeline.project.control import screenshot  # noqa: PLC0415

    return screenshot(
        current_project(), section=section, artifact=artifact, width=width
    )


def read_studio_events(since_ns: int = 0, limit: int = 200) -> list[dict[str, Any]]:
    """What the human did in the Studio after `since_ns` (epoch nanoseconds;
    0 for everything): page opened, artifact selected, artifact shown,
    project switched, time scrubbed. The agent's context for 'look at this'."""
    from rq_pipeline.project import current_project  # noqa: PLC0415
    from rq_pipeline.project.control import events  # noqa: PLC0415

    return events(current_project(), since_ns=since_ns, limit=limit)


def create_project_dir(path: str, name: str, description: str = "") -> dict[str, Any]:
    """Make a project directory: the manifest and one folder per artifact
    kind. Never overwrites an existing project."""
    from rq_pipeline.project import create_project  # noqa: PLC0415

    project = create_project(Path(path), name, description)
    return {"root": str(project.root), "name": project.name}


# A registration list: one statement per door, read top to bottom.
def build_server() -> Any:  # noqa: PLR0915
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
    server.tool(
        description="Identified actuator models: versions, parameter-bound checks, "
        "advisories"
    )(describe_actuator_bundles)
    server.tool(
        description="One identified actuator model in full (fit parameters + "
        "provenance)"
    )(describe_actuator_bundle)
    server.tool(
        description="A generated dataset's datasheet: success-rate bound, versions, "
        "randomization ranges"
    )(describe_datasheet)
    server.tool(description="The task registry: ids, names, rigs")(describe_tasks)
    server.tool(
        description="One environment built for real: its task spec and version"
    )(describe_task)
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

    def evals() -> list[dict[str, Any]]:
        """Every episode-record file under pipeline/runs/."""
        return list_eval_records()

    def eval_detail(run: str) -> dict[str, Any]:
        """One run's records folded: successes, funnel, per-trial verdicts."""
        return describe_eval(run)

    server.tool(
        name="list_eval_records",
        description="Every episode-record file under pipeline/runs/",
    )(evals)
    server.tool(
        name="describe_eval",
        description="One run's episode records: successes, milestone funnel, trials",
    )(eval_detail)
    server.tool(
        description="An actuator's friction-torque curves over velocity, from its model"
    )(friction_curve)

    # The PROJECT family (docs/76): where one effort lives and where it
    # stands in the loop.
    server.tool(
        description="The current project: artifacts by kind with versions and "
        "lineage, the loop map (each state proved or missing), the next move."
    )(describe_project)
    server.tool(
        name="list_projects",
        description="Every project under projects/: name, root, stages proved, count.",
    )(list_project_dirs)
    server.tool(description="Every source format a recording can enter through.")(
        list_robot_adapters
    )
    server.tool(
        description="Ingest robot telemetry (.wire, LeRobot dataset, ROS 2 .mcap) "
        "into the project as a stamped recording with channels, units, census."
    )(ingest_recording)
    server.tool(
        name="create_project",
        description="Make a project directory with its manifest and one folder "
        "per artifact kind; never overwrites.",
    )(create_project_dir)

    # The Studio's control surface: every door a file under <project>/.index
    # (commands in, state and events out), so the agent drives the window
    # in real time and reads back what it shows (docs/76 §10.1).
    server.tool(
        description="What the Studio shows now: project, page, selected artifact, "
        "viewer recording and time cursor, presenter; with alive/heartbeat."
    )(describe_studio)
    server.tool(
        description="Start the Studio window on the current project; waits for "
        "its heartbeat. Refuses when one already runs."
    )(launch_studio)
    server.tool(description="Close the Studio (a quit command, then by pid).")(
        quit_studio
    )
    server.tool(
        description="Navigate the Studio: a page by name, an artifact by version "
        "(its drawer opens), a table of it by title (the modal), or another project."
    )(open_in_studio)
    server.tool(
        description="Stream one artifact into the viewer as itself (3D robot, "
        "time-series recording, experiment curves, evaluation funnel); Live view."
    )(show_in_studio)
    server.tool(
        description="Two artifacts side by side in the viewer, a left and b right."
    )(compare_in_studio)
    server.tool(
        description="Drive the viewer timeline: cursor (seconds or sequence), "
        "play/pause, speed, time selection, follow, step. Refused if nothing streams."
    )(set_studio_time)
    server.tool(
        description="Expand or toggle the viewer's blueprint (left) and selection "
        "(right) panels."
    )(set_studio_panels)
    server.tool(
        description="Run a scene in the MuJoCo viewport (kitting, lift, duck, walk) "
        "and open the Live view; no task stops it. State reports viewport_fps."
    )(simulate_in_studio)
    server.tool(
        description="simulate's Simulation section on the running scene: run/pause, "
        "step n, reset (to a keyframe), speed, manual control."
    )(control_simulator)
    server.tool(
        description="One Control or Joint slider by name: an actuator's control value "
        "or a joint's position; takes manual control."
    )(set_simulator_input)
    server.tool(
        description="What the simulator shows: a MuJoCo flag by name with on, a "
        "camera view (front/side/top/reset), the Inspect drawer "
        "(control/joints/physics/close)."
    )(set_simulator_view)
    server.tool(
        description="See the window: a PNG of the whole Studio (optionally after "
        "opening a page or an artifact); read the returned path to look at it."
    )(screenshot_studio)
    server.tool(
        description="What the human did in the Studio since a time: pages opened, "
        "artifacts selected or shown, project switches, time scrubs."
    )(read_studio_events)

    # The ACT family — S2's doors (docs/64 §3 stage 1), each spawning
    # the CLI that owns the work as a job.
    from rq_pipeline.mcp_actions import Actions  # noqa: PLC0415
    from rq_pipeline.mcp_jobs import JobManager  # noqa: PLC0415

    # Jobs live in the PROJECT when one is open (its Overview shows them);
    # otherwise the legacy runs root, so a checkout with no project still
    # works exactly as before.
    actions = Actions(JobManager(_jobs_root()))
    server.tool(
        description="Generate demonstrations with a scripted policy; only successful "
        "episodes are kept (DR draws recorded). Returns a job handle."
    )(actions.generate_demos)
    server.tool(
        description="Augment seed demonstrations (device filters, CPU verifies, "
        "success criterion gates). Needs the GPU box. Returns a job handle."
    )(actions.multiply_demos)
    server.tool(
        description="The whole chain: generate -> dataset -> train -> paired "
        "evaluation -> fold with intervals. smoke scale runs on a laptop. Job handle."
    )(actions.run_chain)
    server.tool(
        description="Train the microduck walk policy with identified actuator models "
        "(rq_mjlab). agent=smoke is minutes; agent=g3 is the flagship recipe. Job "
        "handle."
    )(actions.train_walk)
    server.tool(
        description="Evaluate the walk policy: paired episodes, exact confidence "
        "intervals, stamps on every row. Job handle."
    )(actions.certify_walk)
    server.tool(
        description="The RL teacher generates demonstrations (docs/66 D2): the walk "
        "checkpoint rolls out, keepers become a stamped batch with chase-camera "
        "frames, discards a failures.jsonl. Job handle."
    )(actions.press_walk)
    server.tool(
        description="The planner policy generates demonstrations (docs/66 D3) on an "
        "SO-101 task: beats written from the seated scene, executed by chained IK, "
        "kept by the task's referee, streamed to the Studio. Job handle."
    )(actions.press_planned)
    server.tool(
        description="Launch the Studio (release build); anything speaking the "
        "Rerun SDK streams into it on :9876."
    )(actions.open_studio)
    server.tool(
        description="Onboard a robot: its MJCF directory becomes a hash-stamped "
        "bundle under robots/, compiled once as the honesty check."
    )(actions.onboard_robot)
    server.tool(description="A job's state and log tail")(actions.job_status)
    server.tool(description="SIGTERM a job's process group")(actions.cancel_job)
    server.tool(description="Every job on record, newest first")(actions.list_jobs)
    return server


def main() -> None:
    build_server().run("stdio")
