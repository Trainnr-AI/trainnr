"""Job handles for the MCP surface's long-running tools.

The pattern docs/64 §5.2 picked from the MCP spec's own blessing:
a `*_start` tool returns a job id immediately, `job_status` polls it,
and the artifacts land under `runs/` exactly as the wrapped CLI always
put them. The manager is deliberately dumb: spawn the command with its
output teed to a log file, remember the pid, record the exit code when
the child ends. A job survives this server's restart (it is its own
session), and so does its exit code: the child runs under a small
runner (this module as `python -m rq_pipeline.mcp_jobs --exit-file`)
that writes the `.exit` file itself, so a door called from a script
that returned, or a server that restarted, still leaves the code the
Studio reads (two certificates showed "running" for an hour after
they ended, 2026-09-10). The in-process watcher stays as the second
recorder. A job whose pid is gone but whose exit file never appeared
(the runner itself killed) reports "ended (exit unrecorded)" rather
than guessing.

The job table is JSON files under `runs/mcp-jobs/`, one per job,
readable by a human when the tooling is not around; liveness and the
stop are the Studio's (`project/control`), one probe for every process.

2026-09-25: the table is every running piece of work, not only the
doors'. A tool started from a terminal, or by an agent's fork, enters it
through `track()` (the operator: "I want to see in real time some
information about what is currently running in the studio" - a gate run
from a terminal showed "idle"). Each job also keeps a `<id>.status` beside
its record: the stage it is at and its progress (`Tracker.stage`,
`Tracker.progress`), rewritten atomically, which the Studio's Running
now panel reads. A door's child inherits `JOB_ID_ENV` and adopts the
door's own record instead of opening a second one.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path

# pydantic (the MCP surface) reads these signatures; on Python < 3.12 it
# accepts only typing_extensions' TypedDict (the box's 3.11 venv, 2026-09-12).
if sys.version_info >= (3, 12):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

from rq_pipeline.bundles.json_record import JsonRecord
from rq_pipeline.project.control import pid_alive, terminate_group

JOBS_DIR_NAME = "mcp-jobs"

# Who started a job: a door (the MCP surface), a tool from a terminal, or
# an agent's fork (which says so through SOURCE_ENV). One vocabulary, read
# by the Studio (`model.rs::JOB_SOURCES`, pinned by test_studio_mirrors).
SOURCE_DOOR = "door"
SOURCE_TOOL = "tool"
SOURCE_AGENT = "agent"
JOB_SOURCES = (SOURCE_DOOR, SOURCE_TOOL, SOURCE_AGENT)
SOURCE_ENV = "TRAINNR_RUN_SOURCE"
# A door's child learns its own job here, so `track()` adopts the door's
# record rather than opening a second one for the same work.
JOB_ID_ENV = "TRAINNR_JOB_ID"
JOBS_DIR_ENV = "TRAINNR_JOBS_DIR"
# The Studio's viewport scene that plays the project's walk (its newest
# checkpoint): what a training run's "watch in viewport" opens
# (`viewport.rs::WALK_TASK`, pinned by test_studio_mirrors).
VIEWPORT_WALK = "walk"

# The three shapes a door answers with, so every caller reads one word
# (`status`) before anything else. A refusal names the reason; a handle
# names the job to poll; `done` carries the door's own fields beside it.
REFUSED = "refused"
DONE = "done"


class JobHandle(TypedDict):
    """A started job: what `job_status` polls."""

    job_id: str
    log: str
    pid: int


class Refusal(TypedDict):
    """A door that did not act, and why."""

    status: str  # REFUSED
    reason: str


class JobStatus(TypedDict):
    """A job's state and its log tail, with who started it and where it is."""

    job_id: str
    tool: str
    state: str
    argv: list[str]
    log: str
    log_tail: list[str]
    source: str
    name: str
    stage: str
    done: int
    total: int
    unit: str


STATE_RUNNING = "running"
# The process is gone and no exit code was recorded: killed, crashed, or
# its watcher gone with it. Said as that, never guessed as done.
STATE_DIED = "died (no exit recorded)"


class Cancelled(TypedDict):
    job_id: str
    cancelled: str


def refusal(reason: str) -> Refusal:
    return {"status": REFUSED, "reason": reason}


# How much of a job's log `job_status` returns by default — enough to
# see the current stage line and the last error, not the whole run.
DEFAULT_TAIL_LINES = 20


@dataclass(frozen=True)
class JobRecord(JsonRecord):
    """One started job, as written to `runs/mcp-jobs/<id>.json`."""

    id: str
    tool: str
    argv: list[str]
    cwd: str
    log: str
    pid: int
    started: float
    source: str = SOURCE_DOOR
    # What the work is about, for a person: the deployment, the run, the
    # scene. Empty for a door job that never said.
    name: str = ""
    # The Studio's MuJoCo viewport scene that shows this work (a
    # `deploy:<name>` scene, the walk), and the Rerun stream file it
    # writes; empty when it has none.
    viewport: str = ""
    viewer: str = ""


# The spawner is injectable so tests assert the exact command lines
# without ever running the heavy tools (docs/64 §6 S2: "a test that
# fakes the heavy call").
Spawner = Callable[[Sequence[str], Path, Path], subprocess.Popen]


EXIT_SUFFIX = ".exit"


def exit_path_for(log_path: Path) -> Path:
    """The job's exit file, beside its log (`<id>.log` -> `<id>.exit`)."""
    return log_path.with_suffix(EXIT_SUFFIX)


STATUS_SUFFIX = ".status"
STATUS_SCHEMA = "trainnr-job-status/1"
# A per-trial or per-iteration loop may call progress() thousands of
# times; the file is rewritten at most this often (the last step always).
STATUS_EVERY_S = 0.25


def status_path_for(log_path: Path) -> Path:
    """The job's live status, beside its log (`<id>.log` -> `<id>.status`)."""
    return log_path.with_suffix(STATUS_SUFFIX)


@dataclass(frozen=True)
class RunStatus:
    """Where a running job is: what it is doing, and how far along.
    `total` 0 means no count is known (a stage line alone)."""

    stage: str = ""
    done: int = 0
    total: int = 0
    unit: str = ""
    updated: float = 0.0
    schema: str = STATUS_SCHEMA


def write_status(path: Path, status: RunStatus) -> None:
    """Atomic, like the exit file: a reader sees the old status or the new."""
    staged = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    staged.write_text(json.dumps(asdict(status)), encoding="utf-8")
    os.replace(staged, path)


def read_status(path: Path) -> RunStatus | None:
    """The status a job last wrote, or None when it wrote none."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    known = set(RunStatus.__dataclass_fields__)
    return RunStatus(**{k: v for k, v in raw.items() if k in known})


def record_exit(exit_path: Path, code: int) -> None:
    """Atomic: a reader that sees the file sees the code. The
    create-then-write of write_text let status() read an EMPTY file
    mid-write (int('') - the lifecycle test, on the GPU box's faster
    fake exit, 2026-09-02)."""
    staged = exit_path.with_suffix(".exit.tmp")
    staged.write_text(str(code))
    os.replace(staged, exit_path)


def runner_argv(argv: Sequence[str], exit_path: Path) -> list[str]:
    """The command as the runner launches it: this module wrapping the
    tool, recording the tool's exit code to `exit_path` when it ends."""
    return [sys.executable, "-m", __name__, "--exit-file", str(exit_path), "--", *argv]


def _spawn(argv: Sequence[str], cwd: Path, log_path: Path) -> subprocess.Popen:
    log = open(log_path, "ab")  # noqa: SIM115 - the child owns it past this frame
    # The child adopts this job in `track()`: its stages land in the
    # door's own record (the job id is the log's stem).
    env = {**os.environ, JOB_ID_ENV: log_path.stem, JOBS_DIR_ENV: str(log_path.parent)}
    try:
        return subprocess.Popen(
            runner_argv(argv, exit_path_for(log_path)),
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            # Its own session: the job outlives this MCP server, and
            # cancel() can signal the whole process group.
            start_new_session=True,
        )
    finally:
        log.close()


# A door launches `uv run --no-sync`: the environment is made ready once,
# under a lock, BEFORE the job starts (`prepare_uv`). Four evaluations
# launched together after a branch switch each re-installed the project
# and collided on its dist-info (2026-09-25, three of four died at 1 s).
UV_NO_SYNC = "--no-sync"
UV_PREPARE_TIMEOUT_S = 900.0  # a cold install of the train extras


@contextmanager
def exclusive(path: Path) -> Iterator[None]:
    """A lock across processes on `path` (created if missing): `flock` on
    POSIX, `msvcrt.locking` on Windows; released on exit either way."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt  # noqa: PLC0415

            handle.seek(0)
            while True:
                try:  # LK_LOCK gives up after ~10 s; a sync may take longer
                    msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    continue
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl  # noqa: PLC0415

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def uv_prepare_argv(argv: Sequence[str]) -> list[str] | None:
    """The sync a `uv run --no-sync ... python <rest>` launch needs: the same
    `uv run` flags (project, env file, extras) with the sync left on, running
    nothing. None for any other command line."""
    if list(argv[:3]) != ["uv", "run", UV_NO_SYNC] or "python" not in argv:
        return None
    head = list(argv[: list(argv).index("python") + 1])
    return [head[0], head[1], *head[3:], "-c", "pass"]


def uv_lock_path(argv: Sequence[str]) -> Path:
    """One lock per uv project (the `--project` it names), in the temp dir:
    two doors on different venvs never wait on each other."""
    import hashlib  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    args = list(argv)
    project = args[args.index("--project") + 1] if "--project" in args else os.getcwd()
    digest = hashlib.sha256(str(Path(project).resolve()).encode()).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / f"trainnr-uv-{digest}.lock"


def prepare_uv(argv: Sequence[str], cwd: Path) -> None:
    """Make a `uv run --no-sync` launch's environment ready, one door at a
    time per project; refused by name, with uv's own words, if it cannot."""
    prepare = uv_prepare_argv(argv)
    if prepare is None:
        return
    with exclusive(uv_lock_path(argv)):
        done = subprocess.run(
            prepare,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=UV_PREPARE_TIMEOUT_S,
            check=False,
        )
    if done.returncode != 0:
        raise RuntimeError(
            f"the environment for {' '.join(prepare)} is not ready: "
            f"{(done.stderr or done.stdout).strip()[-800:]}"
        )


Preparer = Callable[[Sequence[str], Path], None]


class JobManager:
    """Start, poll, tail and cancel the doors' subprocesses."""

    def __init__(
        self,
        runs_root: Path,
        *,
        spawner: Spawner = _spawn,
        preparer: Preparer = prepare_uv,
    ) -> None:
        self.jobs_dir = Path(runs_root) / JOBS_DIR_NAME
        self._spawner = spawner
        self._preparer = preparer
        self._watchers: list[threading.Thread] = []

    def join(self, timeout: float | None = None) -> None:
        """Wait for every watcher to record its exit — the tests' teardown
        (a tempdir removed under a still-writing watcher raced,
        2026-09-02) and any orderly server shutdown."""
        for watcher in self._watchers:
            watcher.join(timeout)

    def start(self, tool: str, argv: Sequence[str], cwd: Path) -> JobHandle:
        """Spawn `argv` in `cwd`; returns the job's id, log path and pid.
        Its environment is made ready first (`prepare_uv`), so the job
        itself never installs anything."""
        self._preparer(argv, Path(cwd))
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        job_id = f"{tool}-{uuid.uuid4().hex[:8]}"
        log_path = self.jobs_dir / f"{job_id}.log"
        process = self._spawner(argv, Path(cwd), log_path)
        record = JobRecord(
            id=job_id,
            tool=tool,
            argv=list(argv),
            cwd=str(cwd),
            log=str(log_path),
            pid=process.pid,
            started=time.time(),
        )
        record.write(self._record_path(job_id))

        # The watcher records the exit code while this server lives; a
        # job that outlives the server ends "unrecorded", said honestly.
        def watch() -> None:
            record_exit(exit_path_for(log_path), process.wait())

        watcher = threading.Thread(target=watch, daemon=True)
        self._watchers.append(watcher)
        watcher.start()
        return {"job_id": job_id, "log": str(log_path), "pid": process.pid}

    def status(self, job_id: str, *, tail: int = DEFAULT_TAIL_LINES) -> JobStatus:
        """The job's state — running / done(exit) / ended (unrecorded) —
        with the log's tail riding along."""
        record = self._record(job_id)
        exit_path = self.jobs_dir / f"{job_id}.exit"
        if exit_path.exists():
            code = int(exit_path.read_text())
            state = "done" if code == 0 else f"failed (exit {code})"
        elif pid_alive(record.pid):
            state = STATE_RUNNING
        else:
            state = STATE_DIED
        log_path = Path(record.log)
        lines = (
            log_path.read_text(errors="replace").splitlines()
            if log_path.exists()
            else []
        )
        live = read_status(self.jobs_dir / f"{job_id}{STATUS_SUFFIX}") or RunStatus()
        return {
            "job_id": job_id,
            "tool": record.tool,
            "state": state,
            "argv": record.argv,
            "log": record.log,
            "log_tail": lines[-tail:],
            "source": record.source,
            "name": record.name,
            "stage": live.stage,
            "done": live.done,
            "total": live.total,
            "unit": live.unit,
        }

    def cancel(self, job_id: str) -> Cancelled:
        """SIGTERM the job's whole process group (its own session — the
        same gesture as the Studio's stop button)."""
        record = self._record(job_id)
        return {"job_id": job_id, "cancelled": terminate_group(record.pid)}

    def list(self) -> list[JobStatus]:
        """Every job on record, newest first, with its current state."""
        if not self.jobs_dir.is_dir():
            return []
        ids = sorted(
            (p.stem for p in self.jobs_dir.glob("*.json")),
            key=lambda job_id: self._record(job_id).started,
            reverse=True,
        )
        return [self.status(job_id, tail=1) for job_id in ids]

    def _record_path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.json"

    def _record(self, job_id: str) -> JobRecord:
        path = self._record_path(job_id)
        if not path.exists():
            known = sorted(p.stem for p in self.jobs_dir.glob("*.json"))
            raise KeyError(f"no job {job_id!r}; known: {known}")
        return JobRecord.read(path)


class Tracker:
    """A running job's voice: `stage(text)` says what it is doing,
    `progress(done, total, unit)` how far along. Both land in the job's
    `.status` (atomic) and the stage line also in its log, so the Studio
    shows the one and "open log" the history."""

    def __init__(self, jobs_dir: Path, job_id: str) -> None:
        self.job_id = job_id
        self._status_path = jobs_dir / f"{job_id}{STATUS_SUFFIX}"
        self._log_path = jobs_dir / f"{job_id}.log"
        self._status = RunStatus()
        self._written = 0.0

    def stage(self, text: str) -> None:
        """A new stage: the line replaces the last and joins the log."""
        self._status = replace(self._status, stage=text, done=0, total=0, unit="")
        with self._log_path.open("a", encoding="utf-8") as log:
            log.write(f"[{time.strftime('%H:%M:%S')}] {text}\n")
        self._write(force=True)

    def progress(self, done: int, total: int, unit: str = "", detail: str = "") -> None:
        """How far along: `done` of `total` `unit`s, with `detail` as the
        stage line when given (a rung's verdict, a reward)."""
        stage = detail or self._status.stage
        # A new line or a new count is news; only repeats are throttled.
        news = stage != self._status.stage or int(total) != self._status.total
        self._status = replace(
            self._status, stage=stage, done=int(done), total=int(total), unit=unit
        )
        self._write(force=news or done >= total)

    def _write(self, *, force: bool) -> None:
        now = time.time()
        if not force and now - self._written < STATUS_EVERY_S:
            return
        self._written = now
        write_status(self._status_path, replace(self._status, updated=now))


def jobs_dir_of(project_root: Path) -> Path:
    """The job table of a project (the one `JobManager` writes)."""
    return Path(project_root) / JOBS_DIR_NAME


# Where the job table lives when no project is open: the pipeline's own
# runs folder (the legacy home of `mcp-jobs/`).
PIPELINE_RUNS = Path(__file__).parents[1] / "runs"


def default_jobs_root() -> Path:
    """The root whose `mcp-jobs/` the doors and the tools share: the
    current project's, else `PIPELINE_RUNS`."""
    from rq_pipeline.project import current_project  # noqa: PLC0415

    try:
        return current_project().root
    except FileNotFoundError:
        return PIPELINE_RUNS


# The shell's convention for a run ended by Ctrl-C.
INTERRUPTED_EXIT = 130


def exit_code_of(error: BaseException | None) -> int:
    """The code a tracked block ends with: 0, a SystemExit's own code,
    `INTERRUPTED_EXIT` for an interrupt, else 1."""
    if error is None:
        return 0
    if isinstance(error, SystemExit):
        code = error.code
        return code if isinstance(code, int) else (0 if code is None else 1)
    if isinstance(error, KeyboardInterrupt):
        return INTERRUPTED_EXIT
    return 1


def _adopted(jobs_dir: Path) -> Path | None:
    """The door's own record this process was started under, if any."""
    job_id = os.environ.get(JOB_ID_ENV, "")
    table = os.environ.get(JOBS_DIR_ENV, "")
    if not job_id or not table or Path(table).resolve() != jobs_dir.resolve():
        return None
    path = jobs_dir / f"{job_id}.json"
    return path if path.is_file() else None


@contextmanager
def track(  # noqa: PLR0913 - a job's identity, each field named
    kind: str,
    *,
    jobs_dir: Path,
    name: str = "",
    argv: Sequence[str] | None = None,
    viewport: str = "",
    viewer: str = "",
) -> Iterator[Tracker]:
    """This process's work, in the project's job table while it runs.

    Under a door (the environment names the job, `JOB_ID_ENV`) the door's
    record is adopted: its name, viewport and viewer are filled in and the
    runner records the exit. Otherwise a record is opened with this
    process's pid and `source` from `SOURCE_ENV` (an agent's fork sets
    `agent`; a terminal leaves `tool`), and the exit is recorded here:
    0, the SystemExit's code, 130 for Ctrl-C, 1 for an error. Refused by
    name: a source word the Studio does not know."""
    jobs_dir = Path(jobs_dir)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    adopted = _adopted(jobs_dir)
    if adopted is not None:
        record = JobRecord.read(adopted)
        replace(
            record, name=name or record.name, viewport=viewport, viewer=viewer
        ).write(adopted)
        yield Tracker(jobs_dir, record.id)
        return
    source = os.environ.get(SOURCE_ENV, SOURCE_TOOL)
    if source not in JOB_SOURCES:
        raise ValueError(f"{SOURCE_ENV}={source!r}; known: {JOB_SOURCES}")
    job_id = f"{kind}-{uuid.uuid4().hex[:8]}"
    log_path = jobs_dir / f"{job_id}.log"
    log_path.touch()
    JobRecord(
        id=job_id,
        tool=kind,
        argv=list(argv if argv is not None else sys.argv),
        cwd=str(Path.cwd()),
        log=str(log_path),
        pid=os.getpid(),
        started=time.time(),
        source=source,
        name=name,
        viewport=viewport,
        viewer=viewer,
    ).write(jobs_dir / f"{job_id}.json")
    error: BaseException | None = None
    try:
        yield Tracker(jobs_dir, job_id)
    except BaseException as caught:
        error = caught
        raise
    finally:
        record_exit(exit_path_for(log_path), exit_code_of(error))


def main(args: Sequence[str] | None = None) -> int:
    """The runner: run the tool with this process's stdout and stderr
    (the job's log), record its exit code, and exit with it."""
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="run a job and record its exit code")
    parser.add_argument("--exit-file", type=Path, required=True)
    parser.add_argument("argv", nargs=argparse.REMAINDER)
    parsed = parser.parse_args(args)
    argv = parsed.argv[1:] if parsed.argv[:1] == ["--"] else parsed.argv
    if not argv:
        parser.error("no command after --")
    code = subprocess.call(argv, stdin=subprocess.DEVNULL)
    record_exit(parsed.exit_file, code)
    return code


if __name__ == "__main__":
    sys.exit(main())
