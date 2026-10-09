"""Whether an agent is connected, as a file the Studio can read.

An MCP server over stdio lives exactly as long as the client session that
spawned it, so the server's own liveness *is* the answer to "is my agent
connected?". Nothing recorded it until now: a fresh install's Studio looks
the same whether Claude Code is wired up or not, and the only way to find
out was to call a tool and see whether anything happened (first-run report,
2026-10-10).

`trainnr mcp` writes `<user home>/agent.json` while it serves and removes
it on the way out, refreshing the heartbeat on a daemon thread so an idle
session still reads as connected. The Studio polls that file. This keeps
the halves as they were: the Studio reads a file the Python side wrote and
still never speaks MCP (`crates/trainnr-studio/src/model.rs`).

It lives under the user's home and not in a project because an agent is
connected to the machine, not to a project: the Welcome page, which is
shown before any project exists, is exactly where the answer is needed.

The reader is the Studio's, in Rust (`model.rs::read_agent_link`), and it
judges this file the way the Python side judges the Studio's own state
file: a fresh heartbeat *and* a live pid, so a session killed hard leaves
a stale file that reads as disconnected rather than as a connected agent
that will never answer. The rule lives on the reader's side alone — one
truth — and the names, the schema and the staleness are pinned across the
two languages by `tests/test_studio_mirrors.py`.
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
from trainnr.safe_write import write_text

SCHEMA = "trainnr-agent-link/1"
#: The file, under `user_home()`.
LINK_FILE = "agent.json"
#: How often the heartbeat is refreshed while the server idles. Matches the
#: Studio's own `HEARTBEAT_EVERY` (crates/trainnr-studio/src/control.rs).
HEARTBEAT_EVERY_S = 1.0
#: A heartbeat older than this is nobody's: the reader treats the link as
#: gone. Mirrored in `crates/trainnr-studio/src/model.rs` (`LINK_STALE_S`).
STALE_S = 10.0


def link_path() -> Path:
    """Where the connection file lives: `<user home>/agent.json`."""
    return user_home() / LINK_FILE


class AgentLink:
    """The open connection, as a file. Start it when the server starts,
    `touch` it as tools are called, and close it on the way out."""

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
            write_text(self.path, json.dumps(self._state(), indent=1) + "\n")

    def touch(self, tool: str | None = None) -> None:
        """Record a tool call (or just refresh the heartbeat)."""
        with self._lock:
            if tool is not None:
                self._calls += 1
                self._last_tool = tool
            self._write()

    def open(self) -> AgentLink:
        """Write the file and start refreshing it."""
        self.touch()
        self._beat = threading.Thread(
            target=self._keep_beating, name="trainnr-agent-link", daemon=True
        )
        self._beat.start()
        return self

    def _keep_beating(self) -> None:
        while not self._stop.wait(HEARTBEAT_EVERY_S):
            self.touch()

    def close(self) -> None:
        """Stop the heartbeat and remove the file. A hard kill skips this;
        the stale heartbeat is what the reader falls back on."""
        self._stop.set()
        if self._beat is not None:
            self._beat.join(timeout=HEARTBEAT_EVERY_S * 2)
            self._beat = None
        with contextlib.suppress(OSError):
            self.path.unlink()

    def __enter__(self) -> AgentLink:
        return self.open()

    def __exit__(self, *_: object) -> None:
        self.close()
