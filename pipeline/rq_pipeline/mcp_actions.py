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
- pipeline tools:  uv run --project pipeline [--extra …] python tools/…
- the T5 chain:    pipeline/.venv-train/bin/python tools/e2e-smoke.py
- rq_mjlab (walk): uv run --project rq_mjlab python -m rq_mjlab.…
- the Studio:      cargo run --release, in crates/studio-shell
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

from rq_pipeline.bundles.hashing import stamp
from rq_pipeline.bundles.locate import robots_dir
from rq_pipeline.mcp_jobs import JobManager

REPO_ROOT = robots_dir().parent
PIPELINE_DIR = REPO_ROOT / "pipeline"
RQ_MJLAB_DIR = REPO_ROOT / "rq_mjlab"
STUDIO_DIR = REPO_ROOT / "crates" / "studio-shell"
TOOLS_DIR = REPO_ROOT / "tools"

# The T5 chain's documented interpreter (its own docstring: the train
# venv, python 3.12 + lerobot) — NOT the pipeline's default venv.
TRAIN_PYTHON = PIPELINE_DIR / ".venv-train" / "bin" / "python"


# The box's launch environment (pipeline/wsl.env: GL to the card, CUDA's
# library path, one BLAS thread). Every door carries it on Linux, because
# the developer's agent launches the MCP server with NO environment of
# its own (.mcp.json) — and a warp child without LD_LIBRARY_PATH falls to
# the CPU SILENTLY: a stranger's certify_walk would have certified on the
# wrong instrument (found 2026-09-02, the first run of the doors on the
# GPU box). Harmless on native Linux (the file's own header); absent on
# macOS and Windows. Tests pass None to keep the command lines verbatim.
ENV_FILE: Path | None = (
    PIPELINE_DIR / "wsl.env" if sys.platform.startswith("linux") else None
)
WSL_RUN = TOOLS_DIR / "wsl-run.sh"  # the same env for interpreters uv does not launch


def _uv(project: Path, *extras: str, env_file: Path | None = None) -> list[str]:
    argv = ["uv", "run", "--project", str(project)]
    if env_file is not None:
        argv += ["--env-file", str(env_file)]
    for extra in extras:
        argv += ["--extra", extra]
    return [*argv, "python"]


# The walks rq_mjlab knows (`rq_mjlab.walks.ROBOTS`), pinned here so the
# door refuses by name without importing the other venv's package.
WALK_ROBOTS = ("microduck", "go1", "go2")


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

    def generate_demos(
        self, episodes: int = 10, seed: int = 20260826, out: str = "runs/kitting-demos"
    ) -> dict[str, Any]:
        """Press referee-gated kitting demonstrations (scripted expert,
        DR draws recorded per episode). Long: returns a job handle."""
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
        return self.jobs.start("generate-demos", argv, PIPELINE_DIR)

    def press_planned(  # noqa: PLR0913, PLR0917 - the press's own knobs, each named
        self,
        task: str = "lift",
        episodes: int = 8,
        seed: int = 17,
        dr_span: float = 0.0,
        out: str | None = None,
        shards: int = 1,
    ) -> dict[str, Any]:
        """The planner expert presses demonstrations (docs/66 D3): on
        an SO-101 task (lift, block_stack, tool_insert) the planner reads
        each seated scene, writes the beats and executes them by chained
        IK; the referee keeps or discards. Streams to the Studio."""
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
        return self.jobs.start("press-planned", argv, PIPELINE_DIR)

    def multiply_demos(
        self,
        seeds_dir: str,
        out: str,
        episodes: int = 4,
        seed: int = 11,
        worlds: int = 64,
    ) -> dict[str, Any]:
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

    # -- the chain (press → dataset → train → eval → fold) -------------

    def run_chain(  # noqa: PLR0913, PLR0917 - the chain's own knobs, each named
        self,
        name: str = "mcp",
        scale: str = "smoke",
        episodes: int | None = None,
        steps: int | None = None,
        from_stage: str | None = None,
        until_stage: str | None = None,
    ) -> dict[str, Any]:
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
        robot: str = "microduck",
        project: str | None = None,
        log_dir: str | None = None,
        seed: int | None = None,
        dr_span: float | None = None,
        task_stamp: str | None = None,
    ) -> dict[str, Any]:
        """Train a walk through rq_mjlab (the certified stack: stamped
        bundles, declared DR bases, the linter green by construction).
        `robot` names the walk (microduck, go1, go2); `project` is where
        the robot's bundle is searched first and where `log_dir` — the
        run's own folder — should live so the project's index sees it.
        `agent="smoke"` is the box's 2-minute check; `agent="g3"` is the
        flagship recipe."""
        if robot not in WALK_ROBOTS:
            raise ValueError(f"robot is one of {', '.join(WALK_ROBOTS)}, got {robot!r}")
        argv = [*self._uv(RQ_MJLAB_DIR), "-m", "rq_mjlab.walk_train", "--agent", agent]
        argv += ["--robot", robot]
        if project is not None:
            argv += ["--project", project]
        if envs is not None:
            argv += ["--envs", str(envs)]
        if iterations is not None:
            argv += ["--iterations", str(iterations)]
        if seed is not None:
            argv += ["--seed", str(seed)]
        if log_dir is not None:
            argv += ["--log-dir", log_dir]
        if dr_span is not None:
            argv += ["--dr-span", str(dr_span)]
        if task_stamp is not None:
            argv += ["--task-stamp", task_stamp]
        return self.jobs.start("train-walk", argv, RQ_MJLAB_DIR)

    def certify_walk(  # noqa: PLR0913 - the certificate's knobs, each named
        self,
        checkpoint: str,
        *,
        trials: int = 40,
        seed: int = 1000,
        device: str | None = None,
        student: str | None = None,
        horizon: int = 20,
        robot: str = "microduck",
        project: str | None = None,
    ) -> dict[str, Any]:
        """The locomotion certificate (C1's shape): seeded paired
        episodes, tracking error and fall counts with exact intervals,
        the run's stamps on every row. With `student` (a LeRobot
        checkpoint distilled from this teacher's data, docs/66 D2) the
        vision student is judged instead, through the policy bridge,
        seeing the same chase camera the press wrote."""
        argv = [
            *self._uv(RQ_MJLAB_DIR),
            "-m",
            "rq_mjlab.walk_verdict",
            checkpoint,
            "--trials",
            str(trials),
            "--seed",
            str(seed),
            "--robot",
            robot,
        ]
        if robot not in WALK_ROBOTS:
            raise ValueError(f"robot is one of {', '.join(WALK_ROBOTS)}, got {robot!r}")
        if project is not None:
            argv += ["--project", project]
        if device is not None:
            argv += ["--device", device]
        if student is not None:
            argv += ["--student", student, "--horizon", str(horizon)]
        return self.jobs.start("certify-walk", argv, RQ_MJLAB_DIR)

    def play_walk(
        self,
        checkpoint: str,
        *,
        robot: str = "microduck",
        envs: int = 9,
        project: str | None = None,
        viewer: str = "native",
    ) -> dict[str, Any]:
        """A checkpoint in mjlab's own viewer - its MuJoCo window or its
        browser viewer - the walk in play mode, streamed to the Studio at
        the same time by the recorder."""
        if robot not in WALK_ROBOTS:
            raise ValueError(f"robot is one of {', '.join(WALK_ROBOTS)}, got {robot!r}")
        argv = [
            *self._uv(RQ_MJLAB_DIR),
            "-m",
            "rq_mjlab.walk_play",
            checkpoint,
            "--robot",
            robot,
            "--envs",
            str(envs),
            "--viewer",
            viewer,
        ]
        if project is not None:
            argv += ["--project", project]
        return self.jobs.start("play-walk", argv, RQ_MJLAB_DIR)

    def preview_rewards(  # noqa: PLR0913 - the preview's knobs, each named
        self,
        *,
        robot: str = "microduck",
        controller: str = "untrained",
        seconds: float = 5.0,
        seed: int = 1000,
        project: str | None = None,
        out: str | None = None,
    ) -> dict[str, Any]:
        """The reward before a run: a short rollout under an untrained
        actor or the held posture, every term streamed to the Studio,
        a summary written beside the task."""
        if robot not in WALK_ROBOTS:
            raise ValueError(f"robot is one of {', '.join(WALK_ROBOTS)}, got {robot!r}")
        argv = [
            *self._uv(RQ_MJLAB_DIR),
            "-m",
            "rq_mjlab.reward_preview",
            "--robot",
            robot,
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
        return self.jobs.start("preview-rewards", argv, RQ_MJLAB_DIR)

    def press_walk(
        self,
        checkpoint: str | None = None,
        episodes: int = 12,
        worlds: int = 9,
        seed: int = 1000,
        out: str = "../runs/walk-demos",
    ) -> dict[str, Any]:
        """The RL teacher presses demonstrations (docs/66 D2): the walk
        checkpoint (newest by default) rolls out in the batched env,
        each episode judged by the certificate's criterion; keepers
        become a stamped DemoLayout batch with chase-camera frames,
        discards a failures.jsonl. Export with export_batch."""
        argv = [*self._uv(RQ_MJLAB_DIR), "-m", "rq_mjlab.walk_press"]
        argv += [checkpoint] if checkpoint else ["--latest"]
        argv += ["--out", out, "--episodes", str(episodes)]
        argv += ["--worlds", str(worlds), "--seed", str(seed)]
        return self.jobs.start("press-walk", argv, RQ_MJLAB_DIR)

    # -- the Studio ----------------------------------------------------

    def accept_task(self, name: str, project_root: str) -> dict[str, Any]:
        """Review a declared task with the acceptance critic: the scripted
        policy must succeed on every paired trial and the floor policy on
        none. Minutes of simulation: returns a job handle; the verdict
        lands beside the task as acceptance.json."""
        from rq_pipeline.project.locate import plain_name  # noqa: PLC0415

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
    ) -> dict[str, Any]:
        """Export a walk policy for deployment (docs/76 A6): the actor as
        ONNX, the manifest read from the built environment, the scene as
        MJCF — into the project's `deploy/<name>/`. Seconds; a job so the
        Studio shows it."""
        from rq_pipeline.project.locate import plain_name  # noqa: PLC0415

        plain_name(name, "deployment name")
        if robot not in WALK_ROBOTS:
            raise ValueError(f"robot is one of {', '.join(WALK_ROBOTS)}, got {robot!r}")
        argv = [
            *self._uv(RQ_MJLAB_DIR),
            "-m",
            "rq_mjlab.walk_export",
            checkpoint,
            "--project",
            project,
            "--robot",
            robot,
            "--name",
            name,
        ]
        if certificate is not None:
            argv += ["--certificate", certificate]
        if policy_stamp is not None:
            argv += ["--policy-stamp", policy_stamp]
        return self.jobs.start("export-deployment", argv, RQ_MJLAB_DIR)

    def gate_deployment(
        self,
        name: str,
        *,
        project: str,
        trials: int = 20,
        seed: int = 1000,
        tolerance: float | None = None,
    ) -> dict[str, Any]:
        """The sim-to-sim gate: the exported policy driven through its
        manifest alone in plain MuJoCo, judged the certificate's way.
        A job; `gate.json` lands beside the manifest."""
        from rq_pipeline.project.locate import plain_name  # noqa: PLC0415

        plain_name(name, "deployment name")
        argv = [
            *self._uv(PIPELINE_DIR, "sim", "deploy"),
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
        return self.jobs.start("gate-deployment", argv, PIPELINE_DIR)

    def open_studio(self) -> dict[str, Any]:
        """Launch the Studio (release build — the debug viewer's slow
        ingest is a measured hazard). Everything that speaks the Rerun
        SDK streams into its window on :9876."""
        argv = ["cargo", "run", "--release"]
        if self.env_file is not None:
            # WSLg: the embedded viewer re-asserts client-drawn chrome
            # under Wayland; unsetting the display var restores the
            # window frame (docs/07 2026-09-01).
            argv = ["env", "-u", "WAYLAND_DISPLAY", *argv]
        return self.jobs.start("studio", argv, STUDIO_DIR)

    # -- onboarding ----------------------------------------------------

    def onboard_robot(
        self, mjcf_path: str, name: str, into: str | None = None
    ) -> dict[str, Any]:
        """A new robot enters as a hash-stamped bundle: the MJCF's whole
        directory copied under `<into or the library>/<name>/`, compiled
        once as the honesty check, its model file and census recorded in
        `bundle.json`, stamped. Synchronous — seconds, and the caller
        wants the stamp in the reply."""
        from rq_pipeline.bundles.bundle import write_bundle_record  # noqa: PLC0415
        from rq_pipeline.project.locate import plain_name  # noqa: PLC0415

        plain_name(name, "robot name")
        source = Path(mjcf_path).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"no MJCF at {source}")
        destination = (Path(into) if into else robots_dir()) / name
        if destination.exists():
            raise FileExistsError(
                f"robots/{name} already exists (stamp: {stamp(name, destination)}) "
                "— onboarding never overwrites a bundle; pick another name or "
                "remove it deliberately"
            )
        # Compile FIRST — a model that does not compile is refused
        # before a single byte lands in robots/.
        import mujoco  # noqa: PLC0415 - sim extra

        model = mujoco.MjModel.from_xml_path(str(source))
        # The whole directory rides along: meshes and includes resolve
        # relative to the MJCF, and a bundle must be self-contained.
        shutil.copytree(source.parent, destination)
        write_bundle_record(destination, name, source.name, model, source=source)
        bundle_stamp = stamp(name, destination)
        return {
            "stamp": bundle_stamp,
            "path": str(destination),
            "model_file": source.name,
            "bodies": int(model.nbody),
            "joints": int(model.njnt),
            "actuators": int(model.nu),
        }

    # -- jobs ----------------------------------------------------------

    def job_status(self, job_id: str) -> dict[str, Any]:
        """A job's state and its log tail."""
        return self.jobs.status(job_id)

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        """SIGTERM a job's process group."""
        return self.jobs.cancel(job_id)

    def list_jobs(self) -> list[dict[str, Any]]:
        """Every job on record, newest first."""
        return self.jobs.list()
