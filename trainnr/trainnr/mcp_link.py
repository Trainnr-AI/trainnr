"""Which agents are connected, as files the Studio can read.

An MCP server over stdio lives exactly as long as the client session that
spawned it, so a server's own liveness *is* the answer to "is my agent
connected?". Nothing recorded it until now: a fresh install's Studio looks
the same whether Claude Code is wired up or not, and the only way to find
out was to call a tool and see whether anything happened (first-run report,
2026-10-10).

`trainnr mcp` writes `<user home>/agents/<pid>.json` while it serves and
removes its own file on the way out, refreshing the heartbeat on a daemon
thread so an idle session still reads as connected. The Studio reads the
folder. This keeps the halves as they were: the Studio reads files the
Python side wrote and still never speaks MCP
(`crates/trainnr-studio/src/model.rs`).

**One file per server, not one per machine.** Two sessions are ordinary -
two Claude Code windows, or Claude Code and Cursor - and a single file
made them overwrite each other every heartbeat while the first to exit
deleted the file out from under the other (measured on review,
2026-10-11: two servers half a second apart flipped the recorded pid
twice a second, and the file went missing for 0.7 s of 3 s when one
left). A folder lets each session own exactly its own name, and lets the
reader say how many are there.

They live under the user's home and not in a project because an agent is
connected to the machine, not to a project: the Welcome page, which is
shown before any project exists, is exactly where the answer is needed.

The reader is the Studio's, in Rust (`model.rs::read_agent_link`), and it
judges these files the way the Python side judges the Studio's own state
file: a heartbeat that is fresh and not in the future, *and* a live pid,
so a session killed hard reads as gone rather than as an agent that will
never answer. The rule lives on the reader's side alone - one truth - and
the folder, the schema and the two thresholds are pinned across the two
languages by `tests/test_studio_mirrors.py`.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from trainnr.paths import user_home
from trainnr.safe_write import remove, write_text

SCHEMA = "trainnr-agent-link/1"
#: The folder under `user_home()` holding one file per serving process.
LINK_DIR = "agents"
#: How often the heartbeat is refreshed while the server idles. Matches the
#: Studio's own `HEARTBEAT_EVERY` (crates/trainnr-studio/src/control.rs).
HEARTBEAT_EVERY_S = 1.0
#: A heartbeat older than this is nobody's: the reader treats that agent as
#: gone. Mirrored in `crates/trainnr-studio/src/model.rs` (`LINK_STALE_S`).
STALE_S = 10.0
#: How far ahead of the reader's clock a heartbeat may sit and still count.
#: Two processes on one machine can disagree by a little, and a clock that
#: steps back (a VM resume) would otherwise leave a file reading as
#: connected for ever. The same slack as `project/control.py`'s judge.
FUTURE_SLACK_S = 5.0
#: A file whose heartbeat is older than this is debris from a session
#: killed before it could tidy up; the next server to start removes it.
#: Far beyond `STALE_S`, so a reader never races the sweep.
SWEEP_AFTER_S = 3600.0


def link_dir() -> Path:
    """Where the connection files live: `<user home>/agents`."""
    return user_home() / LINK_DIR


def link_path(pid: int | None = None) -> Path:
    """One serving process's file, named by its pid."""
    return link_dir() / f"{pid if pid is not None else os.getpid()}.json"


class AgentLink:
    """This process's connection, as a file. Start it when the server
    starts, `touch` it as tools are called, close it on the way out."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or link_path()
        self._started = time.time()
        self._calls = 0
        self._last_tool: str | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._beat: threading.Thread | None = None

    def _state(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "pid": os.getpid(),
            "started": self._started,
            "heartbeat": time.time(),
            "calls": self._calls,
            "last_tool": self._last_tool,
        }

    def _write(self) -> None:
        """Best effort: a server that cannot write this still serves. A
        read-only or missing home must never take the tools down."""
        with contextlib.suppress(OSError):
            write_text(
                self.path,
                json.dumps(self._state(), indent=1) + "\n",
                root=self.path.parent.parent,
            )

    def touch(self, tool: str | None = None) -> None:
        """Note a tool call. The counters move in memory and the heartbeat
        thread writes them within the second: a call is already the most
        expensive thing happening and should not also pay for a file write
        (review, 2026-10-11)."""
        with self._lock:
            if tool is not None:
                self._calls += 1
                self._last_tool = tool

    def open(self) -> AgentLink:
        """Sweep any debris, write this process's file, keep it fresh."""
        sweep(self.path.parent)
        with self._lock:
            self._write()
        self._beat = threading.Thread(
            target=self._keep_beating, name="trainnr-agent-link", daemon=True
        )
        self._beat.start()
        return self

    def _keep_beating(self) -> None:
        while not self._stop.wait(HEARTBEAT_EVERY_S):
            with self._lock:
                self._write()

    def close(self) -> None:
        """Stop the heartbeat and remove *this process's* file, never
        another session's. A hard kill skips this; the stale heartbeat is
        what the reader falls back on, and the next server sweeps it."""
        self._stop.set()
        if self._beat is not None:
            self._beat.join(timeout=HEARTBEAT_EVERY_S * 2)
            self._beat = None
        with contextlib.suppress(OSError):
            remove(self.path, root=self.path.parent.parent)

    def __enter__(self) -> AgentLink:
        return self.open()

    def __exit__(self, *_: object) -> None:
        self.close()


def sweep(folder: Path | None = None) -> int:
    """Remove files left by sessions killed before they could tidy up, by
    age alone: a pid check here would drag the project layer into the
    serving path for no gain, since `SWEEP_AFTER_S` is an hour and the
    reader already ignores anything older than ten seconds. Returns how
    many went."""
    folder = folder or link_dir()
    gone = 0
    try:
        names = sorted(folder.iterdir())
    except OSError:
        return 0
    for path in names:
        if path.suffix != ".json":
            continue
        try:
            stamp = float(json.loads(path.read_text(encoding="utf-8"))["heartbeat"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if time.time() - stamp > SWEEP_AFTER_S:
            # Never through a link (`safe_write`, checked up to the user
            # home): a linked `agents` folder would sweep files elsewhere.
            with contextlib.suppress(OSError):
                remove(path, root=folder.parent)
                gone += 1
    return gone
