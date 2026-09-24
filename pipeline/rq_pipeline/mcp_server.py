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
generates data, trains, evaluates and opens the Studio.

The query functions are plain functions returning JSON-able dicts, with
no MCP import anywhere near them — the suite tests them directly and the
`mcp` extra is only needed to actually serve:

    cd pipeline && uv run --extra sim --extra mcp \\
        python ../tools/mcp-server.py
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rq_pipeline.bundles.hashing import stamp
from rq_pipeline.bundles.locate import bundle_dirs, find_bundle
from rq_pipeline.deploy.manifest import TWIST_RELEASE, TWIST_SHORT
from rq_pipeline.deploy.runtimes import DEFAULT_RUNTIME
from rq_pipeline.mcp_jobs import DONE, JobHandle, Refusal, refusal
from rq_pipeline.physics.registry import engines
from rq_pipeline.project.locate import (
    DEPLOY_FOLDER,
    ROBOTS_FOLDER,
    RUNS_FOLDER,
    SCENES_FOLDER,
)

# BUNDLE_STORE has ONE home (the bundle module itself); it was spelled
# three ways once — review 2026-09-01.
from rq_pipeline.robot.actuator_bundle import BUNDLE_STORE
from rq_pipeline.robot.actuator_library import (
    list_actuators,
    list_models,
    load_actuator,
)
from rq_pipeline.scenes.splatters import DEFAULT_SPLATTER
from rq_pipeline.scenes.terrain import DEFAULT_TERRAIN
from rq_pipeline.tasks.registry import resolve, tasks

# robots/actuators is the actuator LIBRARY (per-servo friction models,
# grown by tools/sync-bam-actuators.py), not a robot bundle — it has its
# own two tools below and stays out of the bundle census.
NOT_A_BUNDLE = ("actuators", "actuator-bundles")


def bundle_names() -> list[str]:
    _use_project_if_any()
    return sorted(name for name in bundle_dirs() if name not in NOT_A_BUNDLE)


def _use_project_if_any() -> None:
    """Search the current project's robots first, when there is one."""
    from rq_pipeline.project import current_project  # noqa: PLC0415

    with contextlib.suppress(FileNotFoundError):
        current_project()


def describe_bundles() -> list[dict[str, Any]]:
    """Every robot bundle the project or the library holds: name@hash
    identity and a file census."""
    described = []
    for name in bundle_names():
        root = bundle_dirs()[name]
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
    _use_project_if_any()
    root = find_bundle(name)
    if root is None or name in NOT_A_BUNDLE:
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


def _reason(why: BaseException) -> str:
    """An exception as a refusal's reason: a KeyError's own message,
    not its repr with quotes."""
    return str(why.args[0]) if isinstance(why, KeyError) and why.args else str(why)


def describe_task_families() -> dict[str, Any]:
    """The families a task can be declared over — every registered task
    whose builder takes a spec — with each spec's fields, types and
    defaults. Read this, then write only what you change in `create_task`."""
    from rq_pipeline.tasks.overlay import families, spec_fields  # noqa: PLC0415

    return {
        task_id: {"rig": entry.rig, "fields": spec_fields(task_id)}
        for task_id, entry in families().items()
    }


def create_task(
    task_id: str, name: str, overlay: dict[str, Any] | None = None
) -> dict[str, Any] | Refusal:
    """Declare an environment into the project: a family (see
    `describe_task_families`) with `overlay` — only the spec fields you
    change — built for real and stamped by its content. Refuses, by name,
    an unknown field, a task with no spec, or a name already taken. The
    next move is `accept_task(name)`."""
    from rq_pipeline.project import (  # noqa: PLC0415
        current_project,
        index_project,
        write_index,
    )
    from rq_pipeline.project.task_ref import declare_task  # noqa: PLC0415

    project = current_project()
    try:
        out = declare_task(project, task_id, name, overlay)
    except (KeyError, ValueError, FileExistsError, TypeError) as why:
        return refusal(_reason(why))
    write_index(project, index_project(project))
    return {"status": DONE, **out, "next": f"accept_task({name!r})"}


def onboard_robot(
    model_path: str,
    name: str,
    variants: dict[str, str] | None = None,
    root: str | None = None,
    accept_changes: bool = False,
) -> dict[str, Any] | Refusal:
    """A robot enters as a hash-stamped bundle, by its file's format: an
    MJCF's directory copied whole (meshes and includes ride along) and
    compiled once as the honesty check; a URDF through MuJoCo's own
    loader; a USD asset (.usd/.usda/.usdc/.usdz) read by Newton's
    importer and written as a bundle with leaf names, mesh files, sensors
    and a home key — `variants` selects its variant sets (e.g.
    {"Physics": "Newton_compliant"}) and `root` is "fixed" (default) or
    "free". Every door then AUDITS what the importer changed against the
    source as authored (masses, inertias, joint limits and parameters,
    dropped elements); an unexplained change is refused by name and the
    bundle removed unless `accept_changes` is true, and the record then
    says so. The reply carries the audit's one line. Into the current
    project's `robots/` when a project is open, else the library. Never
    overwrites; refuses by name, including an option the format does
    not take."""
    from rq_pipeline.mcp_actions import Actions  # noqa: PLC0415
    from rq_pipeline.mcp_jobs import JobManager  # noqa: PLC0415
    from rq_pipeline.project import current_project  # noqa: PLC0415
    from rq_pipeline.robot.onboarding import ACCEPT_CHANGES  # noqa: PLC0415

    into: Path | None = None
    with contextlib.suppress(FileNotFoundError):
        into = current_project().folder(ROBOTS_FOLDER)
    actions = Actions(JobManager(_jobs_root()))
    options: dict[str, Any] = {}
    if variants is not None:
        options["variants"] = variants
    if root is not None:
        options["root"] = root
    if accept_changes:
        options[ACCEPT_CHANGES] = True
    try:
        out = actions.onboard_robot(
            model_path, name, into=str(into) if into else None, options=options
        )
    except (FileNotFoundError, FileExistsError, ValueError, ImportError) as why:
        return refusal(str(why))
    if into is not None:
        from rq_pipeline.project import index_project, write_index  # noqa: PLC0415

        project = current_project()
        write_index(project, index_project(project))
    return {"status": DONE, **out}


def _project_root_if_any() -> Path | None:
    from rq_pipeline.project import current_project  # noqa: PLC0415

    with contextlib.suppress(FileNotFoundError):
        return current_project().root
    return None


def train_walk(  # noqa: PLR0913, PLR0917 - the trainer's own knobs, each named
    agent: str = "smoke",
    envs: int | None = None,
    iterations: int | None = None,
    robot: str | None = None,
    name: str | None = None,
    seed: int | None = None,
    task: str | None = None,
    scene: str | None = None,
    cameras: bool = True,
) -> JobHandle | Refusal:
    """Train a walk policy through rq_mjlab. `task` names a declared walk
    in the project: its robot and randomization span are used and its
    version cited by the run; `robot` alone names a registered walk
    family's robot (`describe_task_families`). With neither, the project's
    one declared walk is taken; several or none is a refusal by name.
    With a project open the trainer searches its robots first, and `name`
    — the experiment's folder under the project's `runs/` — makes the run
    an artifact the index sees (`agent="g3"` only; a smoke archives
    nothing). `scene` names a captured scene in the project: the walk
    trains on its heightfield from the course's start with the head
    camera seeing its splat (docs/78 E2; the Go2); `cameras=False` trains
    on the scene without the camera (the rate without pictures). Minutes
    to hours; returns a job handle."""
    from rq_pipeline.mcp_actions import Actions  # noqa: PLC0415
    from rq_pipeline.mcp_jobs import JobManager  # noqa: PLC0415

    root = _project_root_if_any()
    dr_span: float | None = None
    task_stamp: str | None = None
    if task is None and robot is None:
        task = _the_project_walk(root)
        if task is None:
            return refusal(_no_single_walk(root))
    if task is not None:
        declared = _declared_walk(root, task)
        if isinstance(declared, str):
            return refusal(declared)
        robot, dr_span, task_stamp = declared.robot, declared.dr_span, declared.stamp
    try:
        log_dir = _run_dir(root, name)
        scene_dir = _scene_dir(root, scene)
    except ValueError as why:
        return refusal(str(why))
    try:
        return Actions(JobManager(_jobs_root())).train_walk(
            agent,
            envs,
            iterations,
            robot=robot,
            project=str(root) if root else None,
            log_dir=log_dir,
            seed=seed,
            dr_span=dr_span,
            task_stamp=task_stamp,
            scene=scene_dir,
            cameras=cameras,
        )
    except ValueError as why:
        return refusal(str(why))


def _run_dir(root: Path | None, name: str | None) -> str | None:
    """The experiment's folder under the open project's runs; None for
    an unnamed run; ValueError by name otherwise."""
    from rq_pipeline.project.locate import plain_name  # noqa: PLC0415

    if name is None:
        return None
    plain_name(name, "experiment name")
    if root is None:
        raise ValueError("an experiment name needs an open project")
    return str(root / "runs" / name)


def _scene_dir(root: Path | None, scene: str | None) -> str | None:
    """A named scene's folder in the open project, for a walk that trains
    on it; None for no scene; ValueError by name otherwise."""
    from rq_pipeline.project.locate import SCENES_FOLDER, plain_name  # noqa: PLC0415
    from rq_pipeline.scenes.record import SCENE_FILE  # noqa: PLC0415

    if scene is None:
        return None
    plain_name(scene, "scene name")
    if root is None:
        raise ValueError("a scene needs an open project")
    if not (root / SCENES_FOLDER / scene / SCENE_FILE).is_file():
        raise ValueError(f"no scene {scene!r} in this project")
    return str(root / SCENES_FOLDER / scene)


@dataclass(frozen=True)
class DeclaredWalk:
    """A declared walk as the doors use it: its robot (the walk family's
    rig), its randomization span, its version."""

    name: str
    robot: str
    dr_span: float | None
    stamp: str


def _declared_walk(root: Path | None, task: str) -> DeclaredWalk | str:
    """A declared walk's robot, span and version — or, as a string, the
    reason it is not one."""
    from rq_pipeline.project.locate import Project  # noqa: PLC0415
    from rq_pipeline.project.task_ref import read_task_reference  # noqa: PLC0415
    from rq_pipeline.tasks.walks import walk_robot  # noqa: PLC0415

    if root is None:
        return "a declared task needs an open project"
    try:
        ref = read_task_reference(Project(root), task)
        walk = walk_robot(ref.task_id)
    except (FileNotFoundError, KeyError, ValueError) as why:
        return _reason(why)
    if walk is None:
        return f"{task!r} is not a walk"
    return DeclaredWalk(name=task, robot=walk, dr_span=ref.dr_span, stamp=ref.stamp)


def _declared_walks(root: Path | None) -> list[DeclaredWalk]:
    """Every declared walk in the project, by folder order."""
    from rq_pipeline.project.locate import Project  # noqa: PLC0415
    from rq_pipeline.project.task_ref import task_references  # noqa: PLC0415

    if root is None:
        return []
    found = []
    for ref in task_references(Project(root)):
        walk = _declared_walk(root, ref.name)
        if isinstance(walk, DeclaredWalk):
            found.append(walk)
    return found


def _the_project_walk(root: Path | None) -> str | None:
    """The project's one declared walk, by name, when there is exactly
    one — the walk a door takes when the caller names neither task nor
    robot. Else None (refused by `_no_single_walk`)."""
    walks = _declared_walks(root)
    return walks[0].name if len(walks) == 1 else None


def _no_single_walk(root: Path | None) -> str:
    walks = _declared_walks(root)
    if not walks:
        return (
            "name the walk: `task` (a declared walk in the project) or `robot` "
            "(a registered walk family's robot); this project declares no walk"
        )
    return "name the walk with `task`; this project declares several: " + ", ".join(
        w.name for w in walks
    )


def evaluate_walk(  # noqa: PLR0913, PLR0917 - the evaluation's knobs, each named
    checkpoint: str,
    trials: int = 40,
    seed: int = 1000,
    device: str | None = None,
    student: str | None = None,
    horizon: int = 20,
    robot: str | None = None,
    scene: str | None = None,
) -> JobHandle | Refusal:
    """Evaluate a walk policy: seeded paired episodes, exact intervals,
    the run's versions on every row; `robot` names the walk the checkpoint
    belongs to, else the project's one declared walk. With a project open
    its robots are searched first. `scene` names the captured scene the
    checkpoint trained on (its identity says): the certificate is judged
    on it, and its protocol names it. Job handle."""
    from rq_pipeline.mcp_actions import Actions  # noqa: PLC0415
    from rq_pipeline.mcp_jobs import JobManager  # noqa: PLC0415

    root = _project_root_if_any()
    if robot is None:
        walk = _the_project_walk(root)
        if walk is None:
            return refusal(_no_single_walk(root))
        declared = _declared_walk(root, walk)
        assert isinstance(declared, DeclaredWalk)
        robot = declared.robot
    try:
        return Actions(JobManager(_jobs_root())).evaluate_walk(
            checkpoint,
            trials=trials,
            seed=seed,
            device=device,
            student=student,
            horizon=horizon,
            robot=robot,
            project=str(root) if root else None,
            scene=_scene_dir(root, scene),
        )
    except ValueError as why:
        return refusal(str(why))


def export_deployment(  # noqa: PLR0911 - each return is one named refusal
    run: str,
    checkpoint: str,
    name: str,
    certificate: str | None = None,
    unevaluated: bool = False,
) -> JobHandle | Refusal:
    """Export a trained policy for deployment: `run` is an experiment in
    the project (its folder under runs/), `checkpoint` a file in it
    (model_7999.pt), `name` the deployment's folder. The policy artifact
    and, unless named, the newest evaluation of that checkpoint are cited
    from the index. A checkpoint with no evaluation is refused by name —
    a deployment the gates cannot judge is one nobody can trust — unless
    `unevaluated` says the export is deliberate (a smoke, a mechanics
    check); the order that closes the loop is evaluate, export, gate.
    The manifest carries everything a runtime needs —
    joint and actuator orders, gains, home pose, action scale, the
    ordered observations, the control rate — read from the built
    environment; the ONNX has normalization folded in; the trained scene
    rides along as MJCF. Job handle; next: `gate_deployment(name)`."""
    from rq_pipeline.mcp_actions import Actions  # noqa: PLC0415
    from rq_pipeline.mcp_jobs import JobManager  # noqa: PLC0415
    from rq_pipeline.project import current_project, index_project  # noqa: PLC0415
    from rq_pipeline.project.locate import plain_name  # noqa: PLC0415

    project = current_project()
    try:
        plain_name(run, "run name")
        plain_name(name, "deployment name")
    except ValueError as why:
        return refusal(str(why))
    run_dir = project.folder(RUNS_FOLDER) / run
    path = run_dir / checkpoint
    if not path.is_file():
        return refusal(f"no checkpoint {checkpoint!r} in run {run!r}")
    if (project.folder(DEPLOY_FOLDER) / name).exists():
        return refusal(f"deployment {name!r} already exists")
    index = index_project(project)
    run_art = next(
        (
            a
            for a in index.artifacts
            if a.kind == "run" and a.path == project.relative(RUNS_FOLDER, run)
        ),
        None,
    )
    if run_art is None:
        return refusal(f"run {run!r} is not in the index")
    robot = _walk_of_run(index.artifacts, run_dir)
    if robot is None:
        return refusal(
            f"run {run!r} names no walk (no task in its identity); pass one by name"
        )
    policy = policy_of_checkpoint(index.artifacts, run, run_art.stamp, checkpoint)
    if certificate is None and policy is not None:
        newest = newest_evaluation_of(index.artifacts, policy.stamp)
        certificate = newest.stamp if newest else None
    if certificate is None and not unevaluated:
        return refusal(
            f"{run}/{checkpoint} has no evaluation to cite: evaluate it first "
            "(evaluate_walk), or pass unevaluated=True for a deliberate export "
            "whose gates will report and judge nothing"
        )
    return Actions(JobManager(_jobs_root())).export_deployment(
        str(path),
        name=name,
        robot=robot,
        project=str(project.root),
        certificate=certificate,
        policy_stamp=policy.stamp if policy else None,
    )


def policy_of_checkpoint(
    artifacts: Iterable[Any], run: str, run_stamp: str, checkpoint: str
) -> Any | None:
    """The policy artifact of a run's checkpoint. The live loop names a
    judged checkpoint's policy `<run>-<checkpoint stem>` and it cites the
    run; either mark finds it. None when the checkpoint was never judged."""
    stem = Path(checkpoint).stem
    return next(
        (
            a
            for a in artifacts
            if a.kind == "policy"
            and (
                a.stamp.split("@", 1)[0] == f"{run}-{stem}"
                or (
                    a.cites.get("run") == run_stamp
                    and a.stamp.split("@", 1)[0].endswith(stem)
                )
            )
        ),
        None,
    )


def newest_evaluation_of(artifacts: Iterable[Any], policy_stamp: str) -> Any | None:
    """The newest evaluation citing a policy by version, or None."""
    judged = [
        a
        for a in artifacts
        if a.kind == "certificate" and a.cites.get("policy") == policy_stamp
    ]
    judged.sort(key=lambda a: a.updated or "", reverse=True)
    return judged[0] if judged else None


def _walk_of_run(artifacts: Iterable[Any], run_dir: Path) -> str | None:
    """The walk robot a run belongs to: the family of the declared task
    its identity cites, else the robot its identity names when that is
    a registered walk's."""
    from rq_pipeline.tasks.walks import walk_robot  # noqa: PLC0415

    task_id = _task_id_in_project(artifacts, _run_task(run_dir))
    robot = walk_robot(task_id) if task_id else None
    return robot if robot is not None else _robot_of_run(run_dir)


def _run_task(run_dir: Path) -> str | None:
    """The declared task a run cites, from its identity record."""
    from rq_pipeline.project.kinds import IDENTITY_FILE  # noqa: PLC0415

    identity = run_dir / IDENTITY_FILE
    if not identity.is_file():
        return None
    return json.loads(identity.read_text()).get("task")


def _task_id_in_project(artifacts: Iterable[Any], ref: str | None) -> str | None:
    """The task family a run's declared task belongs to. A run trained
    by the door cites its task by the project's stamp (`go2-walk@0e7e…`,
    the environment card), and the family id (`robotiq/go2-walk`) is
    what that card's spec records; a run that names the family id
    directly is taken as is. Unknown: None (the box, 2026-09-11 - the
    export door handed the stamp to the registry and was refused)."""
    if not ref:
        return None
    if "@" not in ref:
        return ref
    version = ref.split("@", 1)[1]
    for a in artifacts:
        if a.kind != "task":
            continue
        recorded = str(a.summary.get("stamp", a.stamp))
        if recorded.split("@", 1)[-1] == version:
            return a.summary.get("task_id")
    return None


def _robot_of_run(run_dir: Path) -> str | None:
    """The walk a run belongs to, from the robot its identity names
    (`go2@…` → go2) when that is a walk the trainer knows."""
    from rq_pipeline.mcp_actions import walk_robots  # noqa: PLC0415
    from rq_pipeline.project.kinds import IDENTITY_FILE  # noqa: PLC0415

    identity = run_dir / IDENTITY_FILE
    if not identity.is_file():
        return None
    robot = str(json.loads(identity.read_text()).get("robot", "")).split("@", 1)[0]
    return robot if robot in walk_robots() else None


def gate_deployment(
    name: str,
    trials: int = 20,
    seed: int = 1000,
    tolerance: float | None = None,
    runtime: str = DEFAULT_RUNTIME,
) -> JobHandle | Refusal:
    """Run the sim-to-sim gate on a deployment: the exported policy is
    driven through its manifest alone by a registered runtime — no
    training stack — over seeded held commands and judged the
    evaluation's way; passes when its rate is within `tolerance` (the
    gate's own default when unset) of the evaluation it cites. `runtime`
    is one of `list_gate_runtimes()`: plain MuJoCo through our manifest,
    or Unitree's own simulator and controller over DDS (the second gate,
    Linux only). A deployment staged on a scene (`stage_deployment`) is
    judged along the scene's course instead - forward, steered to the
    next waypoint, success = arrival (docs/78 §8.4). Job handle; the
    runtime's record lands beside the manifest and shows in the Studio."""
    from rq_pipeline.deploy.manifest import MANIFEST_FILE  # noqa: PLC0415
    from rq_pipeline.deploy.runtimes import (  # noqa: PLC0415
        require_platform,
        runtime_spec,
    )
    from rq_pipeline.mcp_actions import Actions  # noqa: PLC0415
    from rq_pipeline.mcp_jobs import JobManager  # noqa: PLC0415
    from rq_pipeline.project import current_project  # noqa: PLC0415
    from rq_pipeline.project.locate import plain_name  # noqa: PLC0415

    project = current_project()
    try:
        plain_name(name, "deployment name")
        require_platform(runtime_spec(runtime))  # unknown, or not for this OS
    except (ValueError, RuntimeError) as why:
        return refusal(str(why))
    if not (project.folder(DEPLOY_FOLDER) / name / MANIFEST_FILE).is_file():
        return refusal(f"no deployment {name!r} in this project")
    return Actions(JobManager(_jobs_root())).gate_deployment(
        name,
        project=str(project.root),
        trials=trials,
        seed=seed,
        tolerance=tolerance,
        runtime=runtime,
    )


def stage_deployment(  # noqa: PLR0913, PLR0917 - the stage's own knobs, each named
    deployment: str,
    scene: str,
    name: str | None = None,
    start_x: float | None = None,
    start_y: float | None = None,
    heading_deg: float | None = None,
    terrain: str = DEFAULT_TERRAIN,
) -> dict[str, Any] | Refusal:
    """Put a deployment on a captured scene, as a new deployment (docs/78
    §4 E2): the trained scene's plane floor is replaced by the scene's
    collision proxy as `terrain` - `heightfield` (the proxy's top
    surface on a 2 cm grid, the field's own representation for legged
    terrain; undersides absent), `hulls` (CoACD convex parts; MuJoCo
    collides a mesh as its hull, so a course as one mesh would be a box)
    or `overhangs` (the ground below the clearance as a heightfield, what
    stands above it as convex parts of its occupancy volume: a table the
    robot walks under) - each with its gap measured and recorded, with the
    scene's declared friction; the robot starts one metre before the
    scene's first waypoint along its course unless a start is given; a
    head camera and a course camera are added for the splat renderer.
    Synchronous, seconds. The result is a deployment: `gate_deployment`
    judges the policy on it and the Studio shows the run over the splat.
    Refused by name: an unknown deployment, scene or terrain, a scene
    laying out no course when no start is given, a name already taken."""
    from rq_pipeline.deploy.manifest import (  # noqa: PLC0415
        MANIFEST_FILE,
        load_manifest,
    )
    from rq_pipeline.deploy.runtime import assets_dir_of  # noqa: PLC0415
    from rq_pipeline.project import index_project, write_index  # noqa: PLC0415
    from rq_pipeline.project.locate import current_project, plain_name  # noqa: PLC0415
    from rq_pipeline.scenes import stage as staging  # noqa: PLC0415
    from rq_pipeline.scenes.record import SCENE_FILE  # noqa: PLC0415

    project = current_project()
    name = name or f"{deployment}-on-{scene}"
    try:
        plain_name(deployment, "deployment name")
        plain_name(scene, "scene name")
        plain_name(name, "deployment name")
        source = project.folder(DEPLOY_FOLDER) / deployment
        if not (source / MANIFEST_FILE).is_file():
            raise FileNotFoundError(f"no deployment {deployment!r} in this project")
        scene_dir = project.scenes / scene
        if not (scene_dir / SCENE_FILE).is_file():
            raise FileNotFoundError(f"no scene {scene!r} in this project")
        start_xy = None
        if start_x is not None and start_y is not None:
            start_xy = (float(start_x), float(start_y))
        elif start_x is not None or start_y is not None:
            raise ValueError("a start needs both start_x and start_y")
        staged = staging.stage_deployment(
            source,
            scene_dir,
            project.folder(DEPLOY_FOLDER) / name,
            assets_dir=assets_dir_of(load_manifest(source)),
            start_xy=start_xy,
            heading_deg=heading_deg,
            terrain=terrain,
        )
    except (FileNotFoundError, FileExistsError, ValueError, ImportError) as why:
        return refusal(_reason(why))
    write_index(project, index_project(project))
    facts = staged.facts()
    return {
        "status": DONE,
        "deployment": name,
        "scene": staged.scene,
        "terrain": facts["terrain"],
        "terrain_kind": facts["terrain_kind"],
        "terrain_geoms": facts["terrain_geoms"],
        "terrain_gap": facts["terrain_gap"],
        "start": facts["start"],
        "heading_deg": facts["heading_deg"],
        "surface_z": facts["surface_z"],
        "floor": facts["floor"],
        "cameras": facts["cameras"],
        "trained_floor_removed": staged.floor_removed,
        "next": f"gate_deployment({name!r}) judges the policy on the scene",
    }


def assay_deployment(
    deployment: str, scene: str, trials: int = 20, seed: int = 1000
) -> JobHandle | Refusal:
    """The perturbation assay (docs/78 §4.1): the deployment on the scene
    nine times - nominal, the terrain shifted ±20 mm on each axis, turned
    ±5° about the start - gated on each with the same seed. The success
    cliff (nominal rate minus the worst) is what sets the collision
    tolerance a task declares and the span it randomizes over; a visual
    metric never does. A job; `assay.json` lands on the nominal stage
    beside its gate. Honest when nothing walked at nominal: the cliff is
    then unmeasurable and the record says so."""
    from rq_pipeline.deploy.manifest import MANIFEST_FILE  # noqa: PLC0415
    from rq_pipeline.mcp_actions import Actions  # noqa: PLC0415
    from rq_pipeline.mcp_jobs import JobManager  # noqa: PLC0415
    from rq_pipeline.project import current_project  # noqa: PLC0415
    from rq_pipeline.project.locate import plain_name  # noqa: PLC0415
    from rq_pipeline.scenes.record import SCENE_FILE  # noqa: PLC0415

    project = current_project()
    try:
        plain_name(deployment, "deployment name")
        plain_name(scene, "scene name")
    except ValueError as why:
        return refusal(str(why))
    if not (project.folder(DEPLOY_FOLDER) / deployment / MANIFEST_FILE).is_file():
        return refusal(f"no deployment {deployment!r} in this project")
    if not (project.scenes / scene / SCENE_FILE).is_file():
        return refusal(f"no scene {scene!r} in this project")
    return Actions(JobManager(_jobs_root())).assay_deployment(
        deployment, scene, project=str(project.root), trials=trials, seed=seed
    )


def attribute_deployment(
    deployment: str, runtime: str = "mujoco", trials: int = 20, seed: int = 1000
) -> JobHandle | Refusal:
    """Which parameter would break this policy first (docs/77 §9): a
    deployment whose plane gate PASSED is re-run in plain MuJoCo with one
    dynamics knob turned at a time up a ladder of the field's deployment
    deviations - actions applied late, friction, payload, servo stiffness
    and damping, encoder noise, a slope believed flat, pushes. A knob's
    cliff is the first rung where the tracked rate's exact lower bound
    falls under the certificate's; the knobs ranked by that rung are the
    answer to "it walked in simulation and fell on the robot". A job;
    `attribution.json` and a picture of the fall land beside the manifest,
    the Deployments card reads "most sensitive to ...". Refused by name
    when the gate did not pass, cites no evaluation, or is staged on a
    scene (another protocol)."""
    from rq_pipeline.deploy.attribution import read_attribution  # noqa: PLC0415
    from rq_pipeline.deploy.manifest import MANIFEST_FILE  # noqa: PLC0415
    from rq_pipeline.deploy.runtimes import runtime_names  # noqa: PLC0415
    from rq_pipeline.mcp_actions import Actions  # noqa: PLC0415
    from rq_pipeline.mcp_jobs import JobManager  # noqa: PLC0415
    from rq_pipeline.project import current_project  # noqa: PLC0415
    from rq_pipeline.project.locate import plain_name  # noqa: PLC0415

    project = current_project()
    try:
        plain_name(deployment, "deployment name")
    except ValueError as why:
        return refusal(str(why))
    if runtime not in runtime_names():
        return refusal(
            f"unknown runtime {runtime!r}; one of {', '.join(runtime_names())}"
        )
    folder = project.folder(DEPLOY_FOLDER) / deployment
    if not (folder / MANIFEST_FILE).is_file():
        return refusal(f"no deployment {deployment!r} in this project")
    try:
        read_attribution(folder)  # a record of another schema is refused now
    except ValueError as why:
        return refusal(str(why))
    return Actions(JobManager(_jobs_root())).attribute_deployment(
        deployment, project=str(project.root), runtime=runtime, trials=trials, seed=seed
    )


def list_gate_runtimes() -> list[dict[str, Any]]:
    """Every runtime the sim-to-sim gate can drive an exported policy
    through, with the platforms it runs on (empty: every platform)."""
    from rq_pipeline.deploy.runtimes import RUNTIMES  # noqa: PLC0415

    return [
        {
            "name": spec.name,
            "description": spec.description,
            "platforms": list(spec.platforms),
        }
        for spec in RUNTIMES.values()
    ]


def play_walk(
    run: str, checkpoint: str, envs: int = 9, viewer: str = "viser"
) -> JobHandle | Refusal:
    """Open a checkpoint of an experiment in mjlab's own viewer - `viser`,
    its browser viewer (the URL is on the job's log), or `native`, its
    MuJoCo window - with the same rollout streamed into the Studio's Live
    view by the recorder. `run` is the experiment's folder under runs/,
    `checkpoint` a file in it. Job handle; the viewer lives until closed."""
    from rq_pipeline.mcp_actions import Actions  # noqa: PLC0415
    from rq_pipeline.mcp_jobs import JobManager  # noqa: PLC0415
    from rq_pipeline.project import current_project, index_project  # noqa: PLC0415
    from rq_pipeline.project.locate import plain_name  # noqa: PLC0415

    project = current_project()
    try:
        plain_name(run, "run name")
        run_dir = project.folder(RUNS_FOLDER) / run
        path = run_dir / checkpoint
        if not path.is_file():
            return refusal(f"no checkpoint {checkpoint!r} in run {run!r}")
        robot = _walk_of_run(index_project(project).artifacts, run_dir)
        if robot is None:
            return refusal(f"run {run!r} names no walk")
        return Actions(JobManager(_jobs_root())).play_walk(
            str(path), robot=robot, envs=envs, project=str(project.root), viewer=viewer
        )
    except ValueError as why:
        return refusal(str(why))


def preview_rewards(
    task: str, controller: str = "untrained", seconds: float = 5.0, seed: int = 1000
) -> JobHandle | Refusal:
    """See the reward before training: roll the declared walk for a few
    seconds under `controller` - `untrained` (the recipe's actor at its
    random start) or `stand` (the held posture) - with every reward term
    streamed per step into the Studio's viewer beside the 3D world, and
    write a per-term summary beside the task (`preview-<controller>.json`).
    A silent term, a dominating term or a surprising scale shows here,
    before a run is paid for. Job handle."""
    from rq_pipeline.mcp_actions import Actions  # noqa: PLC0415
    from rq_pipeline.mcp_jobs import JobManager  # noqa: PLC0415
    from rq_pipeline.project.locate import plain_name  # noqa: PLC0415

    root = _project_root_if_any()
    try:
        plain_name(task, "task name")
        declared = _declared_walk(root, task)
        if isinstance(declared, str):
            return refusal(declared)
        assert root is not None
        out = root / "tasks" / task / f"preview-{controller}.json"
        return Actions(JobManager(_jobs_root())).preview_rewards(
            robot=declared.robot,
            controller=controller,
            seconds=seconds,
            seed=seed,
            project=str(root),
            out=str(out),
        )
    except ValueError as why:
        return refusal(str(why))


def accept_task(name: str) -> JobHandle | Refusal:
    """Review a declared environment with the acceptance critic (the
    scripted policy must succeed on every paired trial, the floor policy
    on none). Minutes of simulation: returns a job handle; the verdict,
    counts, funnel and reasons land beside the task as acceptance.json
    and in its Studio drawer. Refuses a name the project does not hold,
    or a family with no scripted expert."""
    from rq_pipeline.mcp_actions import Actions  # noqa: PLC0415
    from rq_pipeline.mcp_jobs import JobManager  # noqa: PLC0415
    from rq_pipeline.project import current_project  # noqa: PLC0415
    from rq_pipeline.project.task_ref import read_task_reference  # noqa: PLC0415
    from rq_pipeline.tasks.experts import expert_for  # noqa: PLC0415
    from rq_pipeline.tasks.walks import walk_robot  # noqa: PLC0415

    project = current_project()
    try:
        ref = read_task_reference(project, name)
        if walk_robot(ref.task_id) is None:
            expert_for(ref.task_id)
    except (FileNotFoundError, KeyError, ValueError) as why:
        return refusal(_reason(why))
    return Actions(JobManager(_jobs_root())).accept_task(name, str(project.root))


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


MIN_FITS_FOR_SPREAD = 2  # a spread needs two fits to disagree (tools/fit-report.py)


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
    """One run's episode records, folded the way the evaluation is:
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
    a ROS 2 bag as .mcap or rosbag2 sqlite3) into the current project as
    a stamped recording: named channels with units and measured rates, a
    census of what the robot reported, and the adapter's honest notes.
    The first move of the loop."""
    from rq_pipeline.project import current_project  # noqa: PLC0415
    from rq_pipeline.project.ingest import ingest  # noqa: PLC0415

    return ingest(current_project(), Path(source), name=name, adapter=adapter)


# -- live capture (docs/76 §5.1): the rig's UDP stream into a recording -----------

# One listener per project, held by the server process that started it
# (the state on disk is what every other process reads). Keyed by the
# project's root so two projects can listen on two ports.
_CAPTURES: dict[str, Any] = {}


def start_capture(  # noqa: PLR0913 - the listener's knobs, each named
    name: str,
    *,
    source: str | None = None,
    port: int | None = None,
    network: str | None = None,
    basis: str | None = None,
    window_s: float | None = None,
) -> dict[str, Any] | Refusal:
    """Listen for a robot's telemetry and land it in the current project;
    `stop_capture` ingests it as a stamped recording. `source` names the
    protocol (`list_capture_sources`): `udp` (default) is the Pico rig's
    port; `dds` is Unitree's bus, `rt/lowstate` and `rt/lowcmd` as one
    recording, on `network` (`lo` for their simulator, the robot's
    interface for the robot), its `basis` "own robot" unless declared
    "simulation" for the stand-in. The state on disk
    (`<project>/.index/capture.json`) is what `capture_status` and the
    Studio read. Refused by name: no project open, a capture already
    listening for this project, a recording of that name already present,
    an unknown source, an option the source does not take, a platform it
    does not run on, a port another process holds, a missing SDK.
    `window_s` caps the listen (default ten minutes)."""
    from rq_pipeline.project import current_project  # noqa: PLC0415
    from rq_pipeline.project.ingest import capture as live_capture  # noqa: PLC0415
    from rq_pipeline.robots.capture import (  # noqa: PLC0415
        DEFAULT_SOURCE,
        DEFAULT_WINDOW_S,
        LISTENING,
    )

    try:
        project = current_project()
    except FileNotFoundError as why:
        return refusal(str(why))
    key = str(project.root)
    held = _CAPTURES.get(key)
    if held is not None and held.state.state == LISTENING:
        return refusal(
            f"a capture named {held.name!r} is already listening on "
            f"{held.state.source} for this project; stop_capture first"
        )
    given = {"port": port, "network": network, "basis": basis}
    options = {k: v for k, v in given.items() if v is not None}
    try:
        listener = live_capture(
            project, name, source=source or DEFAULT_SOURCE, **options
        )
        state = listener.start(
            DEFAULT_WINDOW_S if window_s is None else float(window_s)
        )
    except (FileExistsError, OSError, RuntimeError, ValueError) as why:
        return refusal(str(why))
    _CAPTURES[key] = listener
    return {"status": DONE, **_capture_state(state)}


def list_capture_sources() -> list[dict[str, Any]]:
    """Every protocol a live capture listens on: its name, what it is,
    the options it takes, where it runs, and whether it runs here."""
    from rq_pipeline.robots.capture import sources  # noqa: PLC0415

    return sources()


def stop_capture() -> dict[str, Any] | Refusal:
    """Stop the project's listener and ingest what it captured: the raw
    file becomes a stamped recording through the same ingest as
    `ingest_recording` (adapter `wire`), the index is rewritten so the
    loop's telemetry stage is proved by it. A session with no datagrams
    fails by name and leaves nothing. Refused when nothing is listening
    for the current project in this server."""
    from rq_pipeline.project import current_project  # noqa: PLC0415

    try:
        project = current_project()
    except FileNotFoundError as why:
        return refusal(str(why))
    listener = _CAPTURES.pop(str(project.root), None)
    if listener is None:
        return refusal(
            "no capture is listening for this project in this server; "
            "capture_status reads the state any process wrote"
        )
    state = listener.stop()
    from rq_pipeline.project import index_project, write_index  # noqa: PLC0415

    write_index(project, index_project(project))
    return {"status": DONE, **_capture_state(state)}


def capture_status() -> dict[str, Any] | Refusal:
    """The current project's capture state as written on disk: idle,
    listening (datagrams so far, last one when), ingested (the
    recording's stamp), or failed (why). Readable from any process."""
    from rq_pipeline.project import current_project  # noqa: PLC0415
    from rq_pipeline.project import ingest as ingest_doors  # noqa: PLC0415

    try:
        project = current_project()
    except FileNotFoundError as why:
        return refusal(str(why))
    return {"status": DONE, **ingest_doors.capture_status(project)}


def _capture_state(state: Any) -> dict[str, Any]:
    from dataclasses import asdict  # noqa: PLC0415

    return {"capture": asdict(state)}


def list_public_logs() -> list[dict[str, Any]]:
    """The public recordings of real robots the registry can fetch: robot,
    source, recorded date, the data's licence state, whether fetched."""
    from rq_pipeline.robots.public_logs import listing  # noqa: PLC0415

    return listing()


def ingest_public_log(
    name: str, recording_name: str | None = None
) -> dict[str, Any] | Refusal:
    """Fetch a registered public log (checked against its byte count and
    digest, cached under runs/public-logs) and ingest it into the current
    project with a provenance block: origin "public log", source, url,
    licence state, robot. The telemetry stage then reads "public log" -
    a real robot, not ours. Refuses an unknown name, a download that
    differs from the registry, and a network that is not there."""
    import urllib.error  # noqa: PLC0415

    from rq_pipeline.project import current_project  # noqa: PLC0415
    from rq_pipeline.project.ingest import ingest  # noqa: PLC0415
    from rq_pipeline.robots import public_logs  # noqa: PLC0415

    try:
        entry = public_logs.resolve(name)
        source = public_logs.fetch(name)
    except KeyError as why:
        return refusal(str(why))
    except (urllib.error.URLError, OSError, ValueError) as why:
        return refusal(f"fetch {name!r}: {why}")
    return ingest(
        current_project(),
        source,
        name=recording_name or entry.name,
        adapter=entry.adapter,
        provenance=entry.provenance(),
        basis=entry.basis,
    )


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


def open_in_studio(  # noqa: PLR0913, PLR0917 - one door, one argument per thing it can open
    section: str | None = None,
    artifact: str | None = None,
    project: str | None = None,
    table: str | None = None,
    view: str | None = None,
    search: str | None = None,
) -> dict[str, Any] | Refusal:
    """Navigate the Studio: a page by name (projects, overview, robots,
    environments, recordings, datasets, experiments, policies, evaluations,
    findings, deployments, monitoring, live), an artifact by version (its
    page opens with the detail drawer), another project by root path, one
    of the selected artifact's tables by title (Joints, Actuators,
    Episodes…) in the exploration modal — an empty string closes it — or
    the page's view: cards, table, or matrix (evaluations judged under
    two or more conditions: policies by condition). `search` opens the
    command palette (the user's ⌘K) with that query typed, to point the
    user at something by name."""
    from rq_pipeline.project import current_project  # noqa: PLC0415
    from rq_pipeline.project.control import SECTIONS, command  # noqa: PLC0415

    if section is not None and section.strip().lower() not in SECTIONS:
        return refusal(f"no page {section!r}; one of {', '.join(SECTIONS)}")
    return command(
        current_project(),
        "open",
        section=section,
        artifact=artifact,
        project=project,
        table=table,
        view=view,
        search=search,
    )


def show_in_studio(artifact: str) -> dict[str, Any]:
    """Stream one artifact into the Studio's viewer as itself (a robot as
    its meshes in 3D, a recording as time series, an experiment as its
    curves, an evaluation as its funnel) and switch to the Live view."""
    from rq_pipeline.project import current_project  # noqa: PLC0415
    from rq_pipeline.project.control import command, wait_presented  # noqa: PLC0415

    project = current_project()
    since = time.time()
    answer = command(project, "show", artifact=artifact)
    if answer.get("status") != "done":
        return answer
    # The Studio only accepted the request; the presenter answers later,
    # and a kind it cannot show is a failure the agent must hear about.
    return {**answer, **wait_presented(project, artifact, since=since)}


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
) -> dict[str, Any] | Refusal:
    """The viewer's side panels: `expand` or `toggle` the blueprint panel
    (left: what each view shows) and the selection panel (right: the
    selected entity's properties)."""
    from rq_pipeline.project import current_project  # noqa: PLC0415
    from rq_pipeline.project.control import PANEL_ACTIONS, command  # noqa: PLC0415

    for name, value in (("blueprint", blueprint), ("selection", selection)):
        if value is not None and value not in PANEL_ACTIONS:
            return refusal(
                f"{name}: {value!r} is not one of {', '.join(PANEL_ACTIONS)}"
            )
    return command(
        current_project(), "panels", blueprint=blueprint, selection=selection
    )


def simulate_in_studio(task: str | None = None) -> dict[str, Any]:
    """Run a scene in the Studio's MuJoCo viewport — a preview scene by a
    task's name (`describe_tasks`; the viewport's own list is what the
    Simulator page offers) or `walk:<robot>` for the newest trained walk
    policy of a registered walk family — and switch to the Live view;
    with no task, stop the viewport. The
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
    follow: str | None = None,
) -> dict[str, Any]:
    """MuJoCo simulate's Simulation section on the running scene: `run`
    (True runs, False pauses), `step` n physics steps (pauses and takes
    manual control), `reset` to the initial state or to a `keyframe` by
    name, `speed` as a real-time factor (0.01..100), `manual` (True: the
    sliders drive the scene; False: its own motion, from its start). In a
    many-worlds scene (walk), `follow` keeps the camera on a world: an
    index, worst (lowest reward), failing (an ended episode), cycle, none.
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
        follow=follow,
    )


def set_simulator_input(
    value: float | None = None,
    actuator: str | None = None,
    joint: str | None = None,
    command: str | None = None,
) -> dict[str, Any] | Refusal:
    """One slider of simulate's Control or Joint panel: an actuator's
    control value by name, or a hinge/slide joint's position by name
    (free and ball joints have no scalar); either takes manual control of
    the scene. Or, in a walk scene, one axis of the commanded twist -
    `command` vx (forward, m/s), vy (left) or wz (turn, rad/s) with
    `value`, sent to the followed world (else w0) through mjlab's own
    joystick override; `command="own"` hands the commands back to the
    task. Names come from the robot's Actuators and Joints tables."""
    from rq_pipeline.project import current_project  # noqa: PLC0415
    from rq_pipeline.project.control import command as send  # noqa: PLC0415

    named = [n for n in (actuator, joint, command) if n is not None]
    if len(named) != 1:
        return refusal("name exactly one of actuator, joint, command")
    if command is not None and command not in (*TWIST_SHORT, TWIST_RELEASE):
        return refusal(
            f"command is one of {TWIST_SHORT} or {TWIST_RELEASE}, not {command!r}"
        )
    if value is None and command != TWIST_RELEASE:
        return refusal("a value is needed")
    return send(
        current_project(),
        "simulator",
        actuator=actuator,
        joint=joint,
        command=command,
        value=value,
    )


def set_simulator_view(  # noqa: PLR0913, PLR0917 - one door, one switch per kind of thing shown
    flag: str | None = None,
    on: bool | None = None,
    group: int | None = None,
    kind: str | None = None,
    camera: str | None = None,
    inspect: str | None = None,
    fullscreen: bool | None = None,
) -> dict[str, Any] | Refusal:
    """What the simulator shows: a MuJoCo visualization or rendering
    `flag` by its own name with `on` (contactpoint, contactforce, joint,
    actuator, constraint, inertia, com, transparent, perturbforce, camera,
    light, tendon… or shadow, reflection, skybox, fog, wireframe…); a
    `group` number 0-5 with `on`, of `kind` geom (the default), site,
    joint, tendon, actuator, flex or skin - MuJoCo's group enable; the
    `camera`
    to a named view (front, side, top, reset); the Inspect drawer by tab
    (control, joints, physics, visuals) or close; `fullscreen` puts the
    viewport alone on the page (what the `f` key does) or restores it."""
    from rq_pipeline.project import current_project  # noqa: PLC0415
    from rq_pipeline.project.control import command  # noqa: PLC0415

    if (flag is not None or group is not None) and on is None:
        return refusal("a flag or group needs `on`")
    if (flag, group, camera, inspect, fullscreen) == (None,) * 5:
        return refusal("name a flag, a group, a camera view, inspect or fullscreen")
    return command(
        current_project(),
        "simulator",
        flag=flag,
        on=on,
        group=group,
        kind=kind,
        view=camera,
        inspect=inspect,
        fullscreen=fullscreen,
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


# -- import what exists: experiments, policies, evaluations, findings ------------


def import_experiment(path: str, name: str | None = None) -> dict[str, Any] | Refusal:
    """Bring a trained rq_mjlab experiment into the project: its run
    (identity, log), its checkpoint as a policy citing the run, robot and
    actuator model, and one evaluation per verdict file with its trial
    records. `path` is the arm's directory (holding `train/`) or the
    `train/` directory itself. Refuses, by name, a name already in the
    project or a directory that is not an rq_mjlab run."""
    from rq_pipeline.project import (  # noqa: PLC0415
        current_project,
        index_project,
        write_index,
    )
    from rq_pipeline.project.importer import (  # noqa: PLC0415
        import_experiment as run_import,
    )

    project = current_project()
    try:
        out = run_import(project, Path(path), name=name)
    except (FileExistsError, FileNotFoundError, ValueError) as why:
        return refusal(str(why))
    write_index(project, index_project(project))
    return {"status": DONE, **out}


def import_finding(record: str) -> dict[str, Any] | Refusal:
    """Bring a record from the repository's findings ledger into the
    project, by id (e.g. walk-c1-2026-09-04) or by path."""
    from rq_pipeline.project import (  # noqa: PLC0415
        current_project,
        index_project,
        write_index,
    )
    from rq_pipeline.project.importer import (  # noqa: PLC0415
        import_finding as run_import,
    )

    project = current_project()
    try:
        out = run_import(project, record)
    except (FileExistsError, FileNotFoundError, ValueError) as why:
        return refusal(str(why))
    write_index(project, index_project(project))
    return {"status": DONE, **out}


def list_ledger_findings(prefix: str = "") -> list[dict[str, str]]:
    """The repository's findings ledger: id, date and claim of every
    record, optionally those whose id starts with `prefix`."""
    from rq_pipeline.project.importer import ledger_findings  # noqa: PLC0415

    return ledger_findings(prefix)


# -- system identification (stage ②, the door: docs/76 §6) ----------------------


def _project_artifact(stamp: str, kind: str) -> Any:
    from rq_pipeline.project import current_project, index_project  # noqa: PLC0415

    project = current_project()
    artifact = next(
        (a for a in index_project(project).artifacts if a.stamp == stamp), None
    )
    if artifact is None:
        raise KeyError(f"no artifact {stamp!r} in {project.root}")
    if artifact.kind != kind:
        raise ValueError(f"{stamp} is a {artifact.kind}, not a {kind}")
    return project, artifact


def list_identification_methods() -> list[dict[str, str]]:
    """Every way a robot's dynamics can be identified from a recording:
    the built-in drivetrain ratio fit, and any method a package registers
    under the rq_pipeline.identification_methods entry-point group."""
    from rq_pipeline.robot.methods import methods  # noqa: PLC0415

    return [
        {"name": e.name, "doc": " ".join(e.doc.split())} for e in methods().values()
    ]


def identify_system(
    robot: str, recording: str, method: str | None = None
) -> dict[str, Any] | Refusal:
    """System identification as a door: fit the robot's dynamics from a
    recording in this project (both by version), with a method by name
    or the one that accepts the pair. Writes a fit record into the
    robot's bundle — every parameter's estimate and confidence interval,
    identified or not (interval within 10 % of its allowed range), the
    anchor statement verbatim — and, past two records, the cross-run
    spread verdict; re-indexes so the loop's 'system identified' state is
    proved by the record. Refused by name: an unknown version, a robot
    no method can fit from this recording."""
    from rq_pipeline.project import index_project, write_index  # noqa: PLC0415
    from rq_pipeline.robot.fit_record import (  # noqa: PLC0415
        load_fit_records,
        spread_verdicts,
        write_spread_record,
    )
    from rq_pipeline.robot.methods import detect, resolve  # noqa: PLC0415

    try:
        project, bundle = _project_artifact(robot, "robot")
        _, rec = _project_artifact(recording, "recording")
        bundle_dir = project.root / bundle.path
        recording_dir = project.root / rec.path
        entry = resolve(method) if method else detect(bundle_dir, recording_dir)
        fitter = entry.build()
        why = fitter.accepts(bundle_dir, recording_dir)
        if why is not None:
            raise ValueError(f"{entry.name} cannot fit {robot} from {recording}: {why}")
        result, path = fitter.fit(bundle_dir, recording_dir, write=True)
    except (FileNotFoundError, KeyError, ValueError) as refused:
        return refusal(_reason(refused))
    records = load_fit_records(bundle_dir)
    spread = None
    if len(records) >= MIN_FITS_FOR_SPREAD:
        write_spread_record(bundle_dir)
        spread = {
            name: {
                "lowest": v.lowest,
                "highest": v.highest,
                "mean_half_width": v.mean_half_width,
                "exceeds": v.exceeds,
                "verdict": v.verdict,
            }
            for name, v in spread_verdicts(records).items()
        }
    index = index_project(project)
    write_index(project, index)
    state = next(s for s in index.states if s.name == "system identified")
    # The record lives inside the bundle, so the robot's version moved:
    # an identified robot is a different artifact from an unidentified
    # one, and every later citation names the identified version.
    robot_now = next(
        (
            a.stamp
            for a in index.artifacts
            if a.kind == "robot" and a.path == bundle.path
        ),
        robot,
    )
    return {
        "status": DONE,
        "method": entry.name,
        "robot": robot_now,
        "robot_before": robot,
        "recording": recording,
        "record": str(path.relative_to(project.root)) if path else None,
        "summary": result.summary(),
        "parameters": [
            {
                "name": p.name,
                "estimate": p.estimate,
                "half_width": p.half_width,
                "allowed_range": p.allowed_range,
                "identified": p.pinned,
            }
            for p in result.parameters
        ],
        "confidence": result.confidence,
        "anchor": records[-1].anchor if records else None,
        "records": len(records),
        "spread": spread,
        "state": {
            "system identified": state.present,
            "proved_by": state.proved_by,
            "basis": state.basis,
        },
    }


def check_drift(
    robot: str, recording: str, method: str | None = None, name: str | None = None
) -> dict[str, Any] | Refusal:
    """Drift monitoring as a door: identify fresh telemetry (a recording in
    this project, by version) with the robot's identification method
    WITHOUT writing a fit record, and judge every parameter against the
    union of the robot's identified intervals. Writes a drift record under
    the project's monitoring folder — the verdict per parameter (within,
    left, unresolved, anchored), the parameters that left, the
    recommendation (re-identify, then re-evaluate) — and re-indexes so the
    loop's 'drift monitored' state is proved by it. Refused by name: an
    unknown version, a robot with no fit record yet, a robot no method can
    fit from this recording, a check name already taken."""
    from rq_pipeline.fleet.drift import DRIFT_FILE, judge  # noqa: PLC0415
    from rq_pipeline.project import index_project, write_index  # noqa: PLC0415
    from rq_pipeline.project.index import UNRECORDED  # noqa: PLC0415
    from rq_pipeline.project.locate import plain_name  # noqa: PLC0415
    from rq_pipeline.robot.methods import detect, resolve  # noqa: PLC0415

    try:
        project, bundle = _project_artifact(robot, "robot")
        _, rec = _project_artifact(recording, "recording")
        bundle_dir = project.root / bundle.path
        recording_dir = project.root / rec.path
        check_name = plain_name(
            name if name else f"{recording.split('@', 1)[0]}-check", "check name"
        )
        out_dir = project.monitoring / check_name
        if out_dir.exists():
            raise ValueError(
                f"a drift check named {check_name!r} exists; a check is an "
                "artifact and is never overwritten - name this one"
            )
        entry = resolve(method) if method else detect(bundle_dir, recording_dir)
        fitter = entry.build()
        why = fitter.accepts(bundle_dir, recording_dir)
        if why is not None:
            raise ValueError(f"{entry.name} cannot fit {robot} from {recording}: {why}")
        record = judge(
            bundle_dir, recording_dir, fitter, robot=robot, recording=recording
        )
    except (FileNotFoundError, KeyError, ValueError) as refused:
        return refusal(_reason(refused))
    path = record.write(out_dir / DRIFT_FILE)
    index = index_project(project)
    write_index(project, index)
    state = next(s for s in index.states if s.name == "drift monitored")
    stamp = next(
        (
            a.stamp
            for a in index.artifacts
            if a.kind == "drift" and a.path == str(out_dir.relative_to(project.root))
        ),
        UNRECORDED,
    )
    return {
        "status": DONE,
        "check": stamp,
        "robot": robot,
        "recording": recording,
        "method": entry.name,
        "record": str(path.relative_to(project.root)),
        "drifted": record.drifted,
        "left": list(record.left),
        "unresolved": list(record.unresolved),
        "recommendation": record.recommendation,
        "references": record.references,
        "parameters": [
            {
                "name": p.name,
                "verdict": p.verdict,
                "reference": [p.reference_lower, p.reference_upper],
                "fresh": [p.fresh_lower, p.fresh_upper],
                "estimate": p.fresh_estimate,
                "shift": p.shift,
                "note": p.note,
            }
            for p in record.parameters
        ],
        "state": {"drift monitored": state.present, "proved_by": state.proved_by},
    }


def import_scene(source: str, name: str) -> dict[str, Any] | Refusal:
    """Bring a captured scene into the project as an artifact (docs/78
    §3): today a Neverwhere benchmark scene folder (MIT; a web splat, a
    collision mesh already in the world frame, the alignment that takes
    the splat there, a MuJoCo XML declaring the floor's friction). The
    splat is moved into the world frame and written as a 3DGS PLY, the
    mesh becomes the collision proxy with an MJCF wrapper, the
    visible-surface-to-proxy GAP is measured (chamfer, 95th percentile,
    the fraction beyond 2 cm, the proxy fraction unseen) and recorded
    whatever it is, the declared friction is recorded as declared with
    the span this project randomizes over. Refused by name: a folder
    that is not a scene, a name already taken, a bad name."""
    from rq_pipeline.project import index_project, write_index  # noqa: PLC0415
    from rq_pipeline.project.index import UNRECORDED  # noqa: PLC0415
    from rq_pipeline.project.locate import current_project, plain_name  # noqa: PLC0415
    from rq_pipeline.scenes import neverwhere  # noqa: PLC0415
    from rq_pipeline.scenes.record import load_scene_record  # noqa: PLC0415

    project = current_project()
    try:
        plain_name(name, "scene name")
        folder = Path(source).expanduser().resolve()
        if not folder.is_dir():
            raise FileNotFoundError(f"no folder {folder}")
        record_path = neverwhere.import_scene(folder, project.scenes / name, name=name)
    except (FileNotFoundError, ValueError, OSError) as refused:
        return refusal(_reason(refused))
    record = load_scene_record(record_path)
    index = index_project(project)
    write_index(project, index)
    stamp = next(
        (
            a.stamp
            for a in index.artifacts
            if a.kind == "scene" and a.path == project.relative(SCENES_FOLDER, name)
        ),
        UNRECORDED,
    )
    g = record.gap
    return {
        "status": DONE,
        "scene": stamp,
        "source": record.source,
        "record": str(record_path.relative_to(project.root)),
        "splats": record.splat.get("count"),
        "proxy_faces": record.proxy.get("faces"),
        "proxy_watertight": record.proxy.get("watertight"),
        "gap": {
            "chamfer_m": g.chamfer_m,
            "p95_m": g.p95_m,
            "beyond_tolerance_fraction": g.beyond_tolerance_fraction,
            "hidden_fraction": g.hidden_fraction,
            "tolerance_m": g.tolerance_m,
            "note": g.note,
        },
        "declared": list(record.declared),
        "notes": list(record.notes),
    }


def capture_scene(  # noqa: PLR0913, PLR0917 - the capture's knobs, each named
    source: str,
    name: str,
    fps: float = 2.0,
    steps: int = 30_000,
    scale: float | None = None,
    floor_friction: list[float] | None = None,
    brush: str | None = None,
    splatter: str = DEFAULT_SPLATTER,
) -> JobHandle | Refusal:
    """Capture a scene from a phone video (or a folder of still frames):
    frames by ffmpeg, poses by COLMAP (one camera, sequential matching
    with loop detection), the splat by Brush (Apache; Metal on a Mac,
    CUDA elsewhere), aligned so the fitted floor is z=0 with +z up,
    scaled by `scale` metres per COLMAP unit when you measured a length
    in the capture (unrecorded otherwise: the scene's metres are then
    COLMAP's units), the proxy as the visible surface itself (Poisson
    over the gaussian centres; no dense reconstruction on a laptop), the
    gap measured, `floor_friction` recorded as declared. `splatter` names
    the trainer: gsplat (CUDA, the train environment) or brush (a binary,
    any OS); auto takes gsplat where it answers. A job of minutes to an
    hour; streams into the Studio when one is open. The
    scene appears under the project's scenes when the chain completes.
    Refused by name: a missing tool (colmap, ffmpeg, Brush's binary by
    path or on PATH as brush_app), a bad name, a source that is neither
    a file nor a folder."""
    from rq_pipeline.mcp_actions import Actions  # noqa: PLC0415
    from rq_pipeline.mcp_jobs import JobManager  # noqa: PLC0415
    from rq_pipeline.project import current_project  # noqa: PLC0415
    from rq_pipeline.project.locate import plain_name  # noqa: PLC0415
    from rq_pipeline.scenes.capture import MissingToolError, Tools  # noqa: PLC0415
    from rq_pipeline.scenes.record import SCENE_FILE  # noqa: PLC0415

    project = current_project()
    try:
        plain_name(name, "scene name")
        where = Path(source).expanduser().resolve()
        if not where.exists():
            raise FileNotFoundError(f"no video file or frames folder at {where}")
        Tools.find(
            brush=Path(brush).expanduser() if brush else None,
            video=where.is_file(),
            splatter=splatter,
        )
    except (MissingToolError, FileNotFoundError, ValueError) as why:
        return refusal(_reason(why))
    if (project.scenes / name / SCENE_FILE).is_file():
        return refusal(f"scene {name!r} already exists in this project")
    return Actions(JobManager(_jobs_root())).capture_scene(
        str(where),
        name,
        project=str(project.root),
        fps=fps,
        steps=steps,
        scale=scale,
        floor_friction=floor_friction,
        brush=brush,
        splatter=splatter,
    )


def describe_scene(scene: str) -> dict[str, Any] | Refusal:
    """A captured scene's record (by version): its source and capture,
    the chain that made it with every licence, the splat's and proxy's
    facts, the alignment, the gap's four numbers, and every physics
    parameter with its basis - measured with an interval, or declared
    with a span."""
    from dataclasses import asdict  # noqa: PLC0415

    from rq_pipeline.scenes.record import SCENE_FILE, load_scene_record  # noqa: PLC0415

    try:
        project, found = _any_project_artifact(scene)
        if found.kind != "scene":
            raise ValueError(f"{scene} is a {found.kind}, not a scene")
        record = load_scene_record(project.root / found.path / SCENE_FILE)
    except (FileNotFoundError, KeyError, ValueError) as refused:
        return refusal(_reason(refused))
    return {"status": DONE, "scene": found.stamp, **asdict(record)}


def describe_viewer_recording(
    artifact: str, values: bool = False
) -> dict[str, Any] | Refusal:
    """The saved viewer streams an artifact carries (by version): every
    feed that narrated it into the Studio also wrote its stream into the
    artifact's folder, so a run with no window leaves the same picture.
    Each file described by Rerun's own reader: entity paths, timelines,
    components, chunk counts, size; and, with `values` and the viz-query
    extra, rows per timeline and every scalar series' count, minimum,
    maximum and last value (tens of seconds on a gate's file, so asked
    for, not assumed). Refused by name: an unknown version."""
    from rq_pipeline.project.viewer import describe_folder  # noqa: PLC0415
    from rq_pipeline.viz import VIEWER_FILE_ENV  # noqa: PLC0415

    try:
        project, found = _any_project_artifact(artifact)
    except (FileNotFoundError, KeyError, ValueError) as refused:
        return refusal(_reason(refused))
    folder = project.root / found.path
    files = describe_folder(folder, values=values)
    return {
        "status": DONE,
        "artifact": found.stamp,
        "kind": found.kind,
        "recordings": files,
        "note": (
            "no saved viewer stream: the artifact was made by a feed that "
            f"streams only, or with {VIEWER_FILE_ENV} off"
            if not files
            else f"{len(files)} saved stream(s); show_in_studio replays them"
        ),
    }


def _any_project_artifact(stamp: str) -> Any:
    from rq_pipeline.project import current_project, index_project  # noqa: PLC0415

    project = current_project()
    artifact = next(
        (a for a in index_project(project).artifacts if a.stamp == stamp), None
    )
    if artifact is None:
        raise KeyError(f"no artifact {stamp!r} in {project.root}")
    return project, artifact


def describe_identification(robot: str) -> dict[str, Any] | Refusal:
    """A robot's system identification as recorded: every fit record with
    its parameters, intervals and verdicts, the anchor statements, and the
    cross-run spread verdict when two or more records exist."""
    from rq_pipeline.robot.fit_record import (  # noqa: PLC0415
        load_fit_records,
        spread_verdicts,
    )

    try:
        project, bundle = _project_artifact(robot, "robot")
    except (FileNotFoundError, KeyError, ValueError) as refused:
        return refusal(_reason(refused))
    records = load_fit_records(project.root / bundle.path)
    return {
        "status": DONE,
        "robot": robot,
        "records": [
            {
                "recording": r.recording,
                "created_utc": r.created_utc,
                "confidence": r.confidence,
                "anchor": r.anchor,
                "parameters": [
                    {
                        "name": p.name,
                        "estimate": p.estimate,
                        "half_width": p.half_width,
                        "identified": p.pinned,
                        "unit": (r.units or {}).get(p.name),
                    }
                    for p in r.parameters
                ],
            }
            for r in records
        ],
        "spread": {
            name: {
                "lowest": v.lowest,
                "highest": v.highest,
                "mean_half_width": v.mean_half_width,
                "exceeds": v.exceeds,
                "verdict": v.verdict,
            }
            for name, v in spread_verdicts(records).items()
        }
        if len(records) >= MIN_FITS_FOR_SPREAD
        else None,
    }


def create_project_dir(
    path: str, name: str, description: str = "", loop: str = ""
) -> dict[str, Any] | Refusal:
    """Make a project directory: the manifest and one folder per artifact
    kind. `loop` says how its policy learns — `imitation` (from a
    dataset) or `reinforcement` (from its own rollouts; the dataset stage
    is then marked not needed) — or is left unset and read off the runs.
    Never overwrites an existing project."""
    from rq_pipeline.project import create_project  # noqa: PLC0415

    try:
        project = create_project(Path(path), name, description, loop=loop)
    except (FileExistsError, ValueError) as why:
        return refusal(str(why))
    return {"status": DONE, "root": str(project.root), "name": project.name}


# A registration list: one statement per door, read top to bottom.
def build_server() -> Any:  # noqa: PLR0915
    """The MCP server over the query functions. Needs the `mcp` extra."""
    from mcp.server import MCPServer  # noqa: PLC0415 - mcp extra

    server = MCPServer(
        name="robotiq",
        # The name is bound by clients' .mcp.json entries; it changes with
        # the S3 rename (docs/70 §1), not here.
        instructions=(
            "The instrument's doors: describe (robot bundles hash-stamped "
            "with fit records and their honesty verdicts, the provenance-"
            "gated actuator library, the task and physics-engine registries, "
            "runs, evaluations, projects), act (onboard a robot, ingest or "
            "capture telemetry, identify, create and accept a task, press "
            "demonstrations, train, evaluate, export, gate, check drift, "
            "capture a scene — long work returns a job handle), and drive "
            "the Studio (open, show, compare, time, simulate, screenshot). "
            "Every answer comes through the same code paths the pipeline "
            "itself uses; a refusal names its reason."
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
        description="Ingest robot telemetry (.wire, LeRobot dataset, ROS 2 .mcap or "
        "rosbag2 .db3) into the project as a stamped recording with channels, "
        "units, measured rates, census."
    )(ingest_recording)
    server.tool(
        description="Every protocol a live capture listens on (udp, dds), its "
        "options, and whether it runs on this machine."
    )(list_capture_sources)
    server.tool(
        description="Record a robot's telemetry live into the project: udp (the "
        "rig) or dds (Unitree's rt/lowstate + rt/lowcmd as one recording, the "
        "robot or their simulator); stop_capture ingests it."
    )(start_capture)
    server.tool(
        description="Stop the project's listener and ingest the capture as a stamped "
        "recording; a session with no datagrams fails by name."
    )(stop_capture)
    server.tool(
        description="The project's capture state on disk: idle, listening, ingested, "
        "failed — readable from any process."
    )(capture_status)
    server.tool(
        description="The public real-robot recordings the registry can fetch: "
        "robot, source, recorded date, licence state, fetched or not."
    )(list_public_logs)
    server.tool(
        description="Fetch a registered public log (size and digest checked) and "
        "ingest it with its provenance; the telemetry stage reads 'public log'."
    )(ingest_public_log)
    server.tool(
        description="Bring a trained rq_mjlab experiment into the project: run, policy "
        "and one evaluation per verdict, each citing the others by version."
    )(import_experiment)
    server.tool(
        description="Bring a findings-ledger record into the project by id or path."
    )(import_finding)
    server.tool(
        description="The findings ledger: id, date, claim; optional id prefix."
    )(list_ledger_findings)
    server.tool(
        description="Every way a robot's dynamics can be identified from a recording."
    )(list_identification_methods)
    server.tool(
        description="System identification: fit a robot's dynamics from a recording "
        "(both by version); writes the fit record with intervals and verdicts."
    )(identify_system)
    server.tool(
        description="The saved viewer streams an artifact carries (a headless run "
        "leaves the same picture a window shows): each file's entity paths, "
        "timelines, components and size; with the viz-query extra, the scalar "
        "series' count, min, max and last value."
    )(describe_viewer_recording)
    server.tool(
        description="Bring a captured scene into the project (a Neverwhere benchmark "
        "scene folder today): the splat in the world frame, the collision proxy, the "
        "visible-surface-to-proxy gap measured, the floor's declared friction with its "
        "span. Refuses a folder that is not a scene."
    )(import_scene)
    server.tool(
        description="Capture a scene from a phone video or a folder of frames: ffmpeg, "
        "COLMAP (poses), Brush (the splat), the floor fitted to z=0, the proxy as "
        "the visible surface, the gap measured, friction recorded as declared. A "
        "job of minutes to an hour; refuses a missing tool by name."
    )(capture_scene)
    server.tool(
        description="A captured scene's record: source, capture, the tool chain with "
        "licences, the splat and proxy facts, the alignment, the gap's four numbers, "
        "and every physics parameter with its basis (measured with an interval, or "
        "declared with a span)."
    )(describe_scene)
    server.tool(
        description="Put a deployment on a captured scene as a new deployment: the "
        "plane floor replaced by the scene's collision proxy in convex parts with "
        "its declared friction, the robot started on the scene's course, cameras "
        "added for the splat renderer. gate_deployment then judges the policy there."
    )(stage_deployment)
    server.tool(
        description="The perturbation assay on a captured scene: the deployment "
        "staged nine times (nominal, ±20 mm per axis, ±5° yaw about the start) "
        "and gated on each with one seed; the success cliff that sets a task's "
        "collision tolerance and span. A job; assay.json on the nominal stage."
    )(assay_deployment)
    server.tool(
        description="Which parameter would break this policy first: a passing "
        "plane gate re-run with one dynamics knob turned at a time up its ladder "
        "(latency, friction, payload, kp, kd, encoder noise, tilt, pushes); the "
        "cliff per knob against the certificate's lower bound, the knobs ranked, "
        "the fall pictured. A job; attribution.json beside the manifest."
    )(attribute_deployment)
    server.tool(
        description="Drift monitoring: identify fresh telemetry (a recording, by "
        "version) without writing a fit record and judge every parameter against "
        "the robot's identified intervals; writes a drift record naming what left "
        "and recommending re-identification."
    )(check_drift)
    server.tool(
        description="A robot's fit records: parameters, intervals, identified or not, "
        "anchors, the cross-run spread."
    )(describe_identification)
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
        description="Run a scene in the MuJoCo viewport (a task's preview scene, or "
        "walk:<robot> for the newest trained walk) and open the Live view; no task "
        "stops it. State reports viewport_fps."
    )(simulate_in_studio)
    server.tool(
        description="simulate's Simulation section on the running scene: run/pause, "
        "step n, reset (to a keyframe), speed, manual control; follow a world."
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
        description="Export a trained walk policy for deployment: ONNX with "
        "normalization folded in, a manifest read from the built environment "
        "(joint and actuator orders, gains, home pose, action scale, ordered "
        "observations, control rate), the trained scene as MJCF; cites the "
        "checkpoint's newest evaluation and refuses one that has none unless "
        "unevaluated=True. Job handle."
    )(export_deployment)
    server.tool(
        description="The sim-to-sim gate: drive the exported policy through its "
        "manifest alone by a registered runtime (plain MuJoCo, or Unitree's "
        "simulator over DDS), judge it the evaluation's way, pass within a stated "
        "tolerance of the cited evaluation. Job handle."
    )(gate_deployment)
    server.tool(
        description="Every runtime the sim-to-sim gate can drive a policy through, "
        "with the platforms each runs on."
    )(list_gate_runtimes)
    server.tool(
        description="The families an environment can be declared over, with every "
        "spec field, type and default."
    )(describe_task_families)
    server.tool(
        description="Declare an environment: a family plus the spec fields you "
        "change, built for real and stamped by content. Next: accept_task."
    )(create_task)
    server.tool(
        description="Review a declared environment with the acceptance critic "
        "(scripted policy every trial, floor policy none). Job handle; the "
        "verdict lands beside the task."
    )(accept_task)
    server.tool(
        description="Generate kitting demonstrations with the scripted policy; only "
        "successful episodes are kept (DR draws recorded). Returns a job handle."
    )(actions.generate_kitting_demos)
    server.tool(
        description="Augment seed demonstrations (device filters, CPU verifies, "
        "success criterion gates). Needs the GPU box. Returns a job handle."
    )(actions.multiply_demos)
    server.tool(
        description="The whole chain: generate -> dataset -> train -> paired "
        "evaluation -> fold with intervals. smoke scale runs on a laptop. Job handle."
    )(actions.run_chain)
    server.tool(
        description="Train a walk policy through rq_mjlab on a declared walk (task) "
        "or a registered walk family's robot (describe_task_families); name the "
        "experiment to archive it in the project. agent=smoke is minutes; agent=g3 "
        "is the flagship recipe. Job handle."
    )(train_walk)
    server.tool(
        description="Evaluate a walk policy: paired episodes, exact confidence "
        "intervals, versions on every row; robot names the walk, else the "
        "project's one declared walk. Job handle."
    )(evaluate_walk)
    server.tool(
        description="Open a checkpoint of an experiment in mjlab's own viewer "
        "(viser in the browser, or native), the same rollout streamed into the "
        "Studio's Live view. Job handle; the viewer lives until closed."
    )(play_walk)
    server.tool(
        description="See the reward before training: roll the declared walk for "
        "a few seconds untrained or standing, every reward term streamed into the "
        "Studio, a per-term summary written beside the task. Job handle."
    )(preview_rewards)
    server.tool(
        description="The RL teacher generates demonstrations (docs/66 D2): the walk "
        "checkpoint rolls out, keepers become a stamped batch with chase-camera "
        "frames, discards a failures.jsonl. Job handle."
    )(actions.generate_walk_demos)
    server.tool(
        description="The planner policy generates demonstrations (docs/66 D3) on a "
        "task from the registry: beats written from the seated scene, executed by "
        "chained IK, kept by the task's success criterion, streamed to the Studio. "
        "Job handle."
    )(actions.generate_planned_demos)
    server.tool(
        description="Launch the Studio (release build); anything speaking the "
        "Rerun SDK streams into it on :9876."
    )(actions.open_studio)
    server.tool(
        description="Onboard a robot: an MJCF directory, or a USD asset read by "
        "Newton, becomes a hash-stamped bundle under robots/, compiled once as "
        "the honesty check."
    )(onboard_robot)
    server.tool(description="A job's state and log tail")(actions.job_status)
    server.tool(description="SIGTERM a job's process group")(actions.cancel_job)
    server.tool(description="Every job on record, newest first")(actions.list_jobs)
    return server


def main() -> None:
    build_server().run("stdio")
