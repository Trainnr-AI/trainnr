"""Job handles for the MCP surface's long-running tools.

The pattern docs/64 §5.2 picked from the MCP spec's own blessing:
a `*_start` tool returns a job id immediately, `job_status` polls it,
and the artifacts land under `runs/` exactly as the wrapped CLI always
put them. The manager is deliberately dumb: spawn the command with its
output teed to a log file, remember the pid, record the exit code when
the child ends. A job survives this server's restart (it is its own
session); what does not survive is the exit-code watcher — a job whose
pid is gone but whose exit file never appeared reports
"ended (exit unrecorded)" rather than guessing.

Stdlib only: the job table is JSON files under `runs/mcp-jobs/`, one
per job, readable by a human when the tooling is not around.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from rq_pipeline.bundles.json_record import JsonRecord

JOBS_DIR_NAME = "mcp-jobs"

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


def _spawn(argv: Sequence[str], cwd: Path, log_path: Path) -> subprocess.Popen:
    log = open(log_path, "ab")  # noqa: SIM115 - the child owns it past this frame
    try:
        return subprocess.Popen(
            list(argv),
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

    def start(self, tool: str, argv: Sequence[str], cwd: Path) -> dict[str, object]:
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
            code = process.wait()
            (self.jobs_dir / f"{job_id}.exit").write_text(str(code))

        threading.Thread(target=watch, daemon=True).start()
        return {"job_id": job_id, "log": str(log_path), "pid": process.pid}

    def status(
        self, job_id: str, *, tail: int = DEFAULT_TAIL_LINES
    ) -> dict[str, object]:
        """The job's state — running / done(exit) / ended (unrecorded) —
        with the log's tail riding along."""
        record = self._record(job_id)
        exit_path = self.jobs_dir / f"{job_id}.exit"
        if exit_path.exists():
            code = int(exit_path.read_text())
            state = "done" if code == 0 else f"failed (exit {code})"
        elif _pid_alive(record.pid):
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

    def cancel(self, job_id: str) -> dict[str, object]:
        """SIGTERM the job's whole process group (its own session — the
        same gesture as the Studio's stop button)."""
        record = self._record(job_id)
        try:
            os.killpg(record.pid, signal.SIGTERM)
            note = "SIGTERM sent to the process group"
        except ProcessLookupError:
            note = "already gone"
        return {"job_id": job_id, "cancelled": note}

    def list(self) -> list[dict[str, object]]:
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


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # alive, someone else's — cannot signal, can report
    return True
