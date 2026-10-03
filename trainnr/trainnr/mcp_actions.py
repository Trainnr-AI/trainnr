"""The MCP surface's act-locally tools — docs/64 §3 stage 1.

Every door here is THIN: it spawns the CLI that already owns the work,
through the venv that CLI documents, with output teed to a job log —
`mcp_jobs.JobManager` hands back the id, `job_status` polls it, and the
artifacts land under `runs/` exactly where the tool always put them.
Nothing is re-implemented; a door's whole contract is its command line,
which is why the tests pin those lines verbatim against a fake spawner.

The one synchronous tool is `onboard_robot`: copying an MJCF's
directory into `robots/<name>/`, compiling it once as the honesty
check, and stamping it — seconds, not minutes, and the caller wants
the stamp in the reply.

Environment shapes (each the wrapped tool's own documented launch):
- pipeline tools:  uv run --project trainnr [--extra …] python tools/…
- the T5 chain:    trainnr/.venv-train/bin/python tools/e2e-smoke.py
- trainnr_mjlab (walk): uv run --project trainnr-mjlab python -m trainnr_mjlab.…
- the Studio:      cargo run --release, in crates/trainnr-studio
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from trainnr.bundles.locate import robots_dir
from trainnr.deploy.runtimes import DEFAULT_RUNTIME
from trainnr.deploy.unitree_stage import REFERENCE_CACHE, REFERENCE_ENV
from trainnr.mcp_jobs import UV_NO_SYNC, Cancelled, JobHandle, JobManager, JobStatus
from trainnr.paths import train_python

# The checkout this package runs from (trainnr/ -> trainnr/ -> the
# repo). Not the robot library's parent: `TRAINNR_ROBOTS_DIR` moves the
# library without moving the tools.
REPO_ROOT = Path(__file__).resolve().parents[2]  # <repo>/trainnr/trainnr/<file>
PIPELINE_DIR = REPO_ROOT / "trainnr"
TRAINNR_MJLAB_DIR = REPO_ROOT / "trainnr-mjlab"
TOOLS_DIR = REPO_ROOT / "tools"


# The T5 chain's documented interpreter (its own docstring: the train
# venv, python 3.12 + lerobot) — NOT the pipeline's default venv.
TRAIN_PYTHON = train_python()

# The reference checkout Unitree's simulator and controller are built in,
# for the DDS gate runtime (docs/77 §7): named here, handed to the gate
# tool, never assumed by it.
UNITREE_REFERENCE_ENV = REFERENCE_ENV
UNITREE_REFERENCE_DEFAULT = REFERENCE_CACHE


def unitree_reference() -> Path:
    """`$TRAINNR_UNITREE_REFERENCE`, else the documented cache location."""
    override = os.environ.get(UNITREE_REFERENCE_ENV)
    return (Path(override) if override else UNITREE_REFERENCE_DEFAULT).expanduser()


# The box's launch environment (trainnr/wsl.env: GL to the card, CUDA's
# library path, one BLAS thread). Every door carries it on Linux, because
# the developer's agent launches the MCP server with NO environment of
# its own (.mcp.json) — and a warp child without LD_LIBRARY_PATH falls to
# the CPU SILENTLY: a stranger's evaluate_walk (certify_walk then) would
# have judged on the wrong instrument (found 2026-09-02, the first run of
# the doors on the GPU box). Harmless on native Linux (the file's own
# header); absent on
# macOS and Windows. Tests pass None to keep the command lines verbatim.
ENV_FILE: Path | None = (
    PIPELINE_DIR / "wsl.env" if sys.platform.startswith("linux") else None
)
WSL_RUN = TOOLS_DIR / "wsl-run.sh"  # the same env for interpreters uv does not launch


def _uv(project: Path, *extras: str, env_file: Path | None = None) -> list[str]:
    argv = ["uv", "run", UV_NO_SYNC, "--project", str(project)]
    if env_file is not None:
        argv += ["--env-file", str(env_file)]
    for extra in extras:
        argv += ["--extra", extra]
    return [*argv, "python"]


def walk_robots() -> tuple[str, ...]:
    """The robots a walk family is registered for — the task registry's
    word, so the doors refuse an unknown walk by name."""
    from trainnr.tasks.walks import walk_robots as registered  # noqa: PLC0415

    return registered()


def require_walk_robot(robot: str | None) -> str:
    """The robot a walk door acts on, or a refusal naming the legal set.
    None is refused too: no door assumes a robot."""
    known = walk_robots()
    if robot is None:
        raise ValueError(f"name the robot of the walk: one of {', '.join(known)}")
    if robot not in known:
        raise ValueError(f"robot is one of {', '.join(known)}, got {robot!r}")
    return robot


def walk_train_argv(  # noqa: PLR0913 - the trainer's own knobs, each named
    *,
    agent: str,
    robot: str,
    project: str | None = None,
    envs: int | None = None,
    iterations: int | None = None,
    seed: int | None = None,
    log_dir: str | None = None,
    dr_span: float | None = None,
    task_stamp: str | None = None,
    recorder: bool = True,
    scene: str | None = None,
    cameras: bool = True,
    fit: str | None = None,
    lag_dr_ms: float = 0.0,
    env_file: Path | None = ENV_FILE,
) -> list[str]:
    """The one command line that trains a walk through trainnr_mjlab, for the
    door and for the acceptance smoke alike — with the launch environment
    (`env_file`) the trainer needs on the box, so no caller can drop it."""
    argv = [
        *_uv(TRAINNR_MJLAB_DIR, env_file=env_file),
        "-m",
        "trainnr_mjlab.walk_train",
    ]
    argv += ["--agent", agent, "--robot", require_walk_robot(robot)]
    argv += _given("--project", project)
    argv += _given("--envs", envs)
    argv += _given("--iterations", iterations)
    argv += _given("--seed", seed)
    argv += _given("--log-dir", log_dir)
    argv += _given("--dr-span", dr_span)
    argv += _given("--task-stamp", task_stamp)
    if not recorder:
        argv.append("--no-recorder")
    if scene is not None:  # the walk on a captured scene (docs/78 E2)
        argv += ["--scene", scene]
        if not cameras:
            argv.append("--no-cameras")
    if fit is not None:  # the joints at a measured fit (trainnr_mjlab.fit_walk)
        argv += ["--fit", fit]
    if lag_dr_ms < 0:
        raise ValueError(f"a command lag is not negative: {lag_dr_ms} ms")
    if lag_dr_ms:  # a random command lag in training (trainnr_mjlab.lag_dr)
        argv += ["--lag-dr-ms", f"{lag_dr_ms:g}"]
    return argv


# walk_verdict's word for moving every law axis at a cliff rung (trainnr_mjlab
# cannot be imported here; the CLI refuses any other mismatch the same way)
JUDGE_ALL_AXES = "all"


def _given(flag: str, value: object) -> list[str]:
    """A command-line option only when a value was given (the tool's own
    default, never a second copy of it here, when not)."""
    return [] if value is None else [flag, str(value)]


class Actions:
    """The doors, bound to one JobManager (tests inject a fake spawner)
    and to the platform's launch environment (tests pass None)."""

    def __init__(self, jobs: JobManager, env_file: Path | None = ENV_FILE) -> None:
        self.jobs = jobs
        self.env_file = env_file

    def _uv(self, project: Path, *extras: str) -> list[str]:
        return _uv(project, *extras, env_file=self.env_file)

    def _under_env(self, argv: list[str]) -> list[str]:
        """A non-uv command line under the same environment (wsl-run.sh
        sources the file, then execs)."""
        return [str(WSL_RUN), *argv] if self.env_file is not None else argv

    # -- data ---------------------------------------------------------

    def generate_kitting_demos(
        self, episodes: int = 10, seed: int = 20260826, out: str = "runs/kitting-demos"
    ) -> JobHandle:
        """Generate kitting demonstrations with the scripted expert; the
        success criterion keeps or discards each, DR draws recorded per
        episode (`tools/kitting-demos.py`, kitting only). Long: returns
        a job handle."""
        if episodes < 1:
            raise ValueError(f"episodes must be >= 1, got {episodes}")
        argv = [
            *self._uv(PIPELINE_DIR, "sim"),
            str(TOOLS_DIR / "kitting-demos.py"),
            str(episodes),
            out,
            "--seed",
            str(seed),
        ]
        return self.jobs.start("generate-kitting-demos", argv, PIPELINE_DIR)

    def generate_planned_demos(  # noqa: PLR0913, PLR0917 - the generator's knobs, named
        self,
        task: str,
        episodes: int = 8,
        seed: int = 17,
        dr_span: float = 0.0,
        out: str | None = None,
        shards: int = 1,
    ) -> JobHandle:
        """The planner expert generates demonstrations (docs/66 D3) on a
        task from the registry (`describe_tasks`): the planner reads each
        seated scene, writes the beats and executes them by chained IK;
        the success criterion keeps or discards. Streams to the Studio."""
        argv = [
            *self._uv(PIPELINE_DIR, "sim", "viz"),
            str(TOOLS_DIR / "planner-demos.py"),
            task,
        ]
        if out is not None:
            argv.append(out)
        argv += ["--episodes", str(episodes), "--seed", str(seed)]
        argv += ["--dr-span", str(dr_span)]
        if shards > 1:
            argv += ["--shards", str(shards)]  # docs/66 D4: N runs, one merged batch
        return self.jobs.start("generate-planned-demos", argv, PIPELINE_DIR)

    def multiply_demos(
        self,
        seeds_dir: str,
        out: str,
        episodes: int = 4,
        seed: int = 11,
        worlds: int = 64,
    ) -> JobHandle:
        """Multiply seed demonstrations (Mimic contract: device filters,
        CPU verifies, referee gates). Needs the GPU box — the tool
        itself refuses loudly on a CUDA-less machine."""
        argv = [
            *self._uv(PIPELINE_DIR, "sim", "mjx"),
            str(TOOLS_DIR / "press-multiply.py"),
            seeds_dir,
            out,
            "--episodes",
            str(episodes),
            "--seed",
            str(seed),
            "--worlds",
            str(worlds),
        ]
        return self.jobs.start("multiply-demos", argv, PIPELINE_DIR)

    # -- the chain (generate → dataset → train → eval → fold) ----------

    def run_chain(  # noqa: PLR0913, PLR0917 - the chain's own knobs, each named
        self,
        name: str = "mcp",
        scale: str = "smoke",
        episodes: int | None = None,
        steps: int | None = None,
        from_stage: str | None = None,
        until_stage: str | None = None,
    ) -> JobHandle:
        """The whole T5 chain, one job: demos → LeRobot dataset →
        lerobot-train (in-loop eval) → paired evaluation → the fold
        with intervals and funnels. `scale="smoke"` finishes in minutes
        on a laptop; `scale="cloud"` is the real recipe for a GPU."""
        argv = self._under_env([str(TRAIN_PYTHON), str(TOOLS_DIR / "e2e-smoke.py")])
        argv += ["--name", name]
        argv += ["--scale", scale]
        if episodes is not None:
            argv += ["--episodes", str(episodes)]
        if steps is not None:
            argv += ["--steps", str(steps)]
        if from_stage is not None:
            argv += ["--from", from_stage]
        if until_stage is not None:
            argv += ["--until", until_stage]
        return self.jobs.start("chain", argv, PIPELINE_DIR)

    # -- the walk (flagship RL) ----------------------------------------

    def train_walk(  # noqa: PLR0913 - the trainer's own knobs, each named
        self,
        agent: str = "smoke",
        envs: int | None = None,
        iterations: int | None = None,
        *,
        robot: str | None,
        project: str | None = None,
        log_dir: str | None = None,
        seed: int | None = None,
        dr_span: float | None = None,
        task_stamp: str | None = None,
        scene: str | None = None,
        cameras: bool = True,
        fit: str | None = None,
        lag_dr_ms: float = 0.0,
    ) -> JobHandle:
        """Train a walk through trainnr_mjlab (the certified stack: stamped
        bundles, declared DR bases, the linter green by construction).
        `robot` names the walk (a registered walk family's robot: see
        `describe_task_families`); `project` is where the robot's bundle
        is searched first and where `log_dir` — the run's own folder —
        should live so the project's index sees it. `agent="smoke"` is
        the box's 2-minute check; `agent="g3"` is the flagship recipe."""
        argv = walk_train_argv(
            agent=agent,
            robot=require_walk_robot(robot),
            project=project,
            envs=envs,
            iterations=iterations,
            seed=seed,
            log_dir=log_dir,
            dr_span=dr_span,
            task_stamp=task_stamp,
            scene=scene,
            cameras=cameras,
            fit=fit,
            lag_dr_ms=lag_dr_ms,
            env_file=self.env_file,
        )
        return self.jobs.start("train-walk", argv, TRAINNR_MJLAB_DIR)

    def evaluate_walk(  # noqa: PLR0913 - the evaluation's knobs, each named
        self,
        checkpoint: str,
        *,
        trials: int = 40,
        seed: int = 1000,
        device: str | None = None,
        student: str | None = None,
        horizon: int = 20,
        robot: str | None,
        project: str | None = None,
        scene: str | None = None,
        judge_in_fit: str | None = None,
        judge_at_scale: float | None = None,
        judge_param: str = JUDGE_ALL_AXES,
        delay: int = 0,
    ) -> JobHandle:
        """The locomotion evaluation (C1's shape): seeded paired
        episodes, tracking error and fall counts with exact intervals,
        the run's versions on every row. With `student` (a LeRobot
        checkpoint distilled from this teacher's data, docs/66 D2) the
        vision student is judged instead, through the policy bridge,
        seeing the same chase camera the data generation wrote."""
        argv = [
            *self._uv(TRAINNR_MJLAB_DIR),
            "-m",
            "trainnr_mjlab.walk_verdict",
            checkpoint,
            "--trials",
            str(trials),
            "--seed",
            str(seed),
            "--robot",
            require_walk_robot(robot),
        ]
        if project is not None:
            argv += ["--project", project]
        if device is not None:
            argv += ["--device", device]
        if student is not None:
            argv += ["--student", student, "--horizon", str(horizon)]
        if scene is not None:  # judged on the captured scene it trained on (docs/78 E2)
            argv += ["--scene", scene]
        if judge_in_fit is not None:  # a cross-evaluation in another robot world
            argv += ["--judge-in-fit", judge_in_fit]
        if judge_param != JUDGE_ALL_AXES and judge_at_scale is None:
            raise ValueError(f"judge_param {judge_param!r} needs judge_at_scale")
        if delay < 0:
            raise ValueError(f"delay is control ticks late, not {delay}")
        if judge_at_scale is not None:  # a cliff rung: one law axis pinned
            argv += [
                "--judge-at-scale",
                f"{judge_at_scale:g}",
                "--judge-param",
                judge_param,
            ]
        if delay:  # the teacher's action late by `delay` control ticks
            argv += ["--delay", str(delay)]
        return self.jobs.start("evaluate-walk", argv, TRAINNR_MJLAB_DIR)

    def play_walk(  # noqa: PLR0913 - the play's knobs, each named
        self,
        checkpoint: str,
        *,
        robot: str | None,
        envs: int = 9,
        project: str | None = None,
        viewer: str = "native",
        scene: str | None = None,
    ) -> JobHandle:
        """A checkpoint in mjlab's own viewer - its MuJoCo window or its
        browser viewer - the walk in play mode, streamed to the Studio at
        the same time by the recorder."""
        argv = [
            *self._uv(TRAINNR_MJLAB_DIR),
            "-m",
            "trainnr_mjlab.walk_play",
            checkpoint,
            "--robot",
            require_walk_robot(robot),
            "--envs",
            str(envs),
            "--viewer",
            viewer,
        ]
        if project is not None:
            argv += ["--project", project]
        if scene is not None:  # played on the captured scene it trained on
            argv += ["--scene", scene]
        return self.jobs.start("play-walk", argv, TRAINNR_MJLAB_DIR)

    def preview_rewards(  # noqa: PLR0913 - the preview's knobs, each named
        self,
        *,
        robot: str | None,
        controller: str = "untrained",
        seconds: float = 5.0,
        seed: int = 1000,
        project: str | None = None,
        out: str | None = None,
    ) -> JobHandle:
        """The reward before a run: a short rollout under an untrained
        actor or the held posture, every term streamed to the Studio,
        a summary written beside the task."""
        argv = [
            *self._uv(TRAINNR_MJLAB_DIR),
            "-m",
            "trainnr_mjlab.reward_preview",
            "--robot",
            require_walk_robot(robot),
            "--controller",
            controller,
            "--seconds",
            str(seconds),
            "--seed",
            str(seed),
        ]
        if project is not None:
            argv += ["--project", project]
        if out is not None:
            argv += ["--out", out]
        return self.jobs.start("preview-rewards", argv, TRAINNR_MJLAB_DIR)

    def generate_walk_demos(  # noqa: PLR0913 - the press's knobs, each named
        self,
        checkpoint: str | None = None,
        episodes: int = 12,
        worlds: int = 9,
        seed: int = 1000,
        out: str = "../runs/walk-demos",
        *,
        robot: str | None = None,
        scene: str | None = None,
        project: str | None = None,
        frame_size: tuple[int, int] | None = None,
    ) -> JobHandle:
        """The RL teacher generates demonstrations (docs/66 D2): the walk
        checkpoint (newest by default) rolls out in the batched env,
        each episode judged by the evaluation's criterion; keepers
        become a stamped DemoLayout batch with chase-camera frames,
        discards a failures.jsonl. On a captured `scene` (docs/78 E3)
        the rollouts stand on it and the frames are the head camera's
        picture of its splat. Export with export_batch."""
        argv = [*self._uv(TRAINNR_MJLAB_DIR), "-m", "trainnr_mjlab.walk_press"]
        argv += [checkpoint] if checkpoint else ["--latest"]
        argv += ["--out", out, "--episodes", str(episodes)]
        argv += ["--worlds", str(worlds), "--seed", str(seed)]
        if robot is not None:
            argv += ["--robot", require_walk_robot(robot)]
        if project is not None:
            argv += ["--project", project]
        if scene is not None:
            argv += ["--scene", scene]
        if frame_size is not None:
            argv += ["--width", str(frame_size[0]), "--height", str(frame_size[1])]
        return self.jobs.start("generate-walk-demos", argv, TRAINNR_MJLAB_DIR)

    # -- the Studio ----------------------------------------------------

    def accept_task(self, name: str, project_root: str) -> JobHandle:
        """Review a declared task with the acceptance critic: the scripted
        policy must succeed on every paired trial and the floor policy on
        none. Minutes of simulation: returns a job handle; the verdict
        lands beside the task as acceptance.json."""
        from trainnr.project.locate import plain_name  # noqa: PLC0415

        plain_name(name, "task name")
        argv = [
            *self._uv(PIPELINE_DIR, "sim"),
            str(TOOLS_DIR / "accept-task.py"),
            "--project",
            project_root,
            "--name",
            name,
        ]
        return self.jobs.start("accept-task", argv, PIPELINE_DIR)

    def export_deployment(  # noqa: PLR0913 - the export's own knobs, each named
        self,
        checkpoint: str,
        *,
        name: str,
        robot: str,
        project: str,
        certificate: str | None = None,
        policy_stamp: str | None = None,
    ) -> JobHandle:
        """Export a walk policy for deployment (docs/76 A6): the actor as
        ONNX, the manifest read from the built environment, the scene as
        MJCF — into the project's `deploy/<name>/`. Seconds; a job so the
        Studio shows it."""
        from trainnr.project.locate import plain_name  # noqa: PLC0415

        plain_name(name, "deployment name")
        argv = [
            *self._uv(TRAINNR_MJLAB_DIR),
            "-m",
            "trainnr_mjlab.walk_export",
            checkpoint,
            "--project",
            project,
            "--robot",
            require_walk_robot(robot),
            "--name",
            name,
        ]
        if certificate is not None:
            argv += ["--certificate", certificate]
        if policy_stamp is not None:
            argv += ["--policy-stamp", policy_stamp]
        return self.jobs.start("export-deployment", argv, TRAINNR_MJLAB_DIR)

    def gate_deployment(  # noqa: PLR0913 - the gate's knobs, each named
        self,
        name: str,
        *,
        project: str,
        trials: int = 20,
        seed: int = 1000,
        tolerance: float | None = None,
        runtime: str = DEFAULT_RUNTIME,
    ) -> JobHandle:
        """The sim-to-sim gate: the exported policy driven through its
        manifest alone by a registered runtime (`deploy.runtimes`: plain
        MuJoCo, or Unitree's simulator and controller over DDS), judged
        the evaluation's way. A job; the runtime's record lands beside
        the manifest. The DDS runtime's reference checkout is handed to
        the tool from `$TRAINNR_UNITREE_REFERENCE` (or its documented
        default); the tool never assumes one."""
        from trainnr.deploy.runtimes import runtime_spec  # noqa: PLC0415
        from trainnr.project.locate import plain_name  # noqa: PLC0415

        plain_name(name, "deployment name")
        runtime = runtime_spec(runtime).name  # refuses an unknown one by name
        argv = [
            *self._uv(PIPELINE_DIR, "sim", "deploy", "viz"),
            str(TOOLS_DIR / "gate-deployment.py"),
            "--project",
            project,
            "--name",
            name,
            "--trials",
            str(trials),
            "--seed",
            str(seed),
        ]
        if tolerance is not None:
            argv += ["--tolerance", str(tolerance)]
        argv += ["--runtime", runtime, "--reference", str(unitree_reference())]
        return self.jobs.start("gate-deployment", argv, PIPELINE_DIR)

    def capture_scene(  # noqa: PLR0913 - the capture's knobs, each named
        self,
        source: str,
        name: str,
        *,
        project: str,
        fps: float = 2.0,
        steps: int = 30_000,
        scale: float | None = None,
        floor_friction: list[float] | None = None,
        brush: str | None = None,
        splatter: str | None = None,
    ) -> JobHandle:
        """A phone video (or a folder of frames) into a scene artifact
        through ffmpeg, COLMAP and Brush (docs/78 §3): a job of minutes
        to an hour on a laptop; the record lands under the project's
        scenes when the chain completes, the log beside it meanwhile."""
        from trainnr.project.locate import plain_name  # noqa: PLC0415

        plain_name(name, "scene name")
        argv = [
            *self._uv(PIPELINE_DIR, "sim", "viz", "scene"),
            str(TOOLS_DIR / "capture-scene.py"),
            "--project",
            project,
            "--source",
            source,
            "--name",
            name,
            "--fps",
            str(fps),
            "--steps",
            str(steps),
        ]
        if scale is not None:
            argv += ["--scale", str(scale)]
        if floor_friction is not None:
            argv += ["--floor-friction", *(str(v) for v in floor_friction)]
        if brush is not None:
            argv += ["--brush", brush]
        if splatter is not None:
            argv += ["--splatter", splatter]
        return self.jobs.start("capture-scene", argv, PIPELINE_DIR)

    def assay_deployment(
        self,
        name: str,
        scene: str,
        *,
        project: str,
        trials: int = 20,
        seed: int = 1000,
    ) -> JobHandle:
        """The perturbation assay (docs/78 §4.1): the deployment staged on
        the scene nine times - nominal, ±20 mm on each axis, ±5° of yaw
        about the start - and gated on each with the same seed; the
        success cliff lands in `assay.json` on the nominal stage. A job."""
        from trainnr.project.locate import plain_name  # noqa: PLC0415

        plain_name(name, "deployment name")
        plain_name(scene, "scene name")
        argv = [
            *self._uv(PIPELINE_DIR, "sim", "deploy", "viz", "scene"),
            str(TOOLS_DIR / "assay-deployment.py"),
            "--project",
            project,
            "--name",
            name,
            "--scene",
            scene,
            "--trials",
            str(trials),
            "--seed",
            str(seed),
            "--narrate",  # law 0: every stage's gate in the Studio
        ]
        return self.jobs.start("assay-deployment", argv, PIPELINE_DIR)

    def attribute_deployment(
        self,
        name: str,
        *,
        project: str,
        runtime: str = DEFAULT_RUNTIME,
        trials: int | None = None,
        seed: int | None = None,
    ) -> JobHandle:
        """Attribution (docs/77 §9): a passing plane gate re-run with one
        dynamics knob turned at a time up its ladder; the cliff per knob,
        the knobs ranked by the rung they fall at, the fall pictured; the
        record `attribution.json` beside the manifest. Trials and seed are
        the gate's own unless given. A job."""
        from trainnr.project.locate import plain_name  # noqa: PLC0415

        plain_name(name, "deployment name")
        argv = [
            *self._uv(PIPELINE_DIR, "sim", "deploy", "viz"),
            str(TOOLS_DIR / "attribute-deployment.py"),
            "--project",
            project,
            "--name",
            name,
            "--runtime",
            runtime,
            *_given("--trials", trials),
            *_given("--seed", seed),
        ]
        return self.jobs.start("attribute-deployment", argv, PIPELINE_DIR)

    def preflight_deployment(
        self,
        name: str,
        *,
        project: str,
        runtime: str = DEFAULT_RUNTIME,
        seed: int | None = None,
    ) -> JobHandle:
        """Pre-flight (docs/77 §10): every check before the first tick, the
        ramp-in and the stops measured, the record `preflight.json` beside
        the manifest; `runtime="dds"` reads the robot's state from
        Unitree's simulator over DDS. The gate's seed unless given. A job."""
        from trainnr.project.locate import plain_name  # noqa: PLC0415

        plain_name(name, "deployment name")
        argv = [
            *self._uv(PIPELINE_DIR, "sim", "deploy", "viz"),
            str(TOOLS_DIR / "preflight-deployment.py"),
            "--project",
            project,
            "--name",
            name,
            "--runtime",
            runtime,
            *_given("--seed", seed),
        ]
        return self.jobs.start("preflight-deployment", argv, PIPELINE_DIR)

    def ingest_public_log(
        self, name: str, *, project: str, recording_name: str | None = None
    ) -> JobHandle:
        """A registered public log fetched (size and digest checked) and
        ingested into the project with its provenance. A job: the fetch is
        tens to hundreds of megabytes."""
        argv = [
            *self._uv(PIPELINE_DIR, "sim"),
            str(TOOLS_DIR / "public-log.py"),
            "ingest",
            name,
            "--project",
            project,
            *_given("--as", recording_name),
        ]
        return self.jobs.start("ingest-public-log", argv, PIPELINE_DIR)

    # -- onboarding ----------------------------------------------------

    def onboard_robot(
        self,
        model_path: str,
        name: str,
        into: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """A new robot enters as a hash-stamped bundle under `<into or the
        library>/<name>/`, by its file's format (`robot.onboarding`: an
        MJCF copied whole and compiled once, a USD read by Newton and
        written as a bundle); `options` are the format's own (a USD's
        variant selections and root), refused by name otherwise.
        Synchronous — seconds, and the caller wants the stamp in the reply."""
        from trainnr.project.locate import plain_name  # noqa: PLC0415
        from trainnr.robot.onboarding import onboard  # noqa: PLC0415

        plain_name(name, "robot name")
        destination = (Path(into) if into else robots_dir()) / name
        return onboard(Path(model_path), name, destination, options)

    # -- jobs ----------------------------------------------------------

    def job_status(self, job_id: str) -> JobStatus:
        """A job's state and its log tail."""
        return self.jobs.status(job_id)

    def cancel_job(self, job_id: str) -> Cancelled:
        """SIGTERM a job's process group."""
        return self.jobs.cancel(job_id)

    def list_jobs(self) -> list[JobStatus]:
        """Every job on record, newest first."""
        return self.jobs.list()
