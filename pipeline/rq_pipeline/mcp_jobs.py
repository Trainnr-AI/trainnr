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
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
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
    """A job's state and its log tail."""

    job_id: str
    tool: str
    state: str
    argv: list[str]
    log: str
    log_tail: list[str]


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


# The spawner is injectable so tests assert the exact command lines
# without ever running the heavy tools (docs/64 §6 S2: "a test that
# fakes the heavy call").
Spawner = Callable[[Sequence[str], Path, Path], subprocess.Popen]


EXIT_SUFFIX = ".exit"


def exit_path_for(log_path: Path) -> Path:
    """The job's exit file, beside its log (`<id>.log` -> `<id>.exit`)."""
    return log_path.with_suffix(EXIT_SUFFIX)


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
    try:
        return subprocess.Popen(
            runner_argv(argv, exit_path_for(log_path)),
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            # Its own session: the job outlives this MCP server, and
            # cancel() can signal the whole process group.
            start_new_session=True,
        )
    finally:
        log.close()


class JobManager:
    """Start, poll, tail and cancel the doors' subprocesses."""

    def __init__(self, runs_root: Path, *, spawner: Spawner = _spawn) -> None:
        self.jobs_dir = Path(runs_root) / JOBS_DIR_NAME
        self._spawner = spawner
        self._watchers: list[threading.Thread] = []

    def join(self, timeout: float | None = None) -> None:
        """Wait for every watcher to record its exit — the tests' teardown
        (a tempdir removed under a still-writing watcher raced,
        2026-09-02) and any orderly server shutdown."""
        for watcher in self._watchers:
            watcher.join(timeout)

    def start(self, tool: str, argv: Sequence[str], cwd: Path) -> JobHandle:
        """Spawn `argv` in `cwd`; returns the job's id, log path and pid."""
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
            state = "running"
        else:
            state = "ended (exit unrecorded — the watching server restarted)"
        log_path = Path(record.log)
        lines = (
            log_path.read_text(errors="replace").splitlines()
            if log_path.exists()
            else []
        )
        return {
            "job_id": job_id,
            "tool": record.tool,
            "state": state,
            "argv": record.argv,
            "log": record.log,
            "log_tail": lines[-tail:],
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
