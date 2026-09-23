"""The control surface: how an agent drives the Studio in real time.

Decided 2026-09-09 (docs/76 §10.1): the Studio reads files and never talks to
the MCP server, so control is three records under `<project>/.index/`:

- `commands/<id>.json` — one file per command the agent sends (`open`,
  `show`, `compare`, `time`, `panels`, `quit`). The Studio watches the
  directory at 20 Hz, applies each, and answers with `<id>.ack.json`:
  `done`, `refused` (with the reason), or `failed`. Every command is a
  record; a session replays from the directory.
- `studio-state.json` — what the Studio shows right now: project, page,
  selected artifact, the viewer's recording, timeline and cursor, whether
  the presenter runs, a heartbeat. Written on change and once a second.
- `events.jsonl` — what the human did: a click, a page change, a time
  scrub. Appended by the Studio; read by the agent as context.

Every door here refuses by name when no Studio is alive (a stale
heartbeat or a dead pid) instead of waiting on a file nobody will answer.
`launch` starts the built binary on the project; `quit` asks politely
through a command and, past the timeout, terminates by pid.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from rq_pipeline.paths import checkout
from rq_pipeline.project.locate import INDEX_DIR, PROJECT_ENV, Project
from rq_pipeline.viz import STUDIO_ADDRESS

COMMANDS_DIR = "commands"
STATE_FILE = "studio-state.json"
# The state file's schema (control.rs `STATE_SCHEMA`); the family before
# the `/` is what a reader checks.
STATE_SCHEMA = "trainnr-studio-state/1"
EVENTS_FILE = "events.jsonl"
STUDIO_LOG = "studio.log"
STUDIO_ENV = "TRAINNR_STUDIO"  # a built Studio binary, when not in the repo
SCHEMA = "trainnr-command/1"

STALE_S = 3.0  # a heartbeat older than this is a Studio that died
ACK_TIMEOUT_S = 3.0
SCREENSHOT_TIMEOUT_S = 6.0  # a capture waits for a frame, then encodes
SCREENSHOT_WIDTH = 1600
LAUNCH_TIMEOUT_S = 15.0
QUIT_TIMEOUT_S = 5.0
POLL_S = 0.02
KEEP_COMMANDS = 200  # older command + ack files are pruned on send
SETTLE_S = 0.15  # after navigating, before a capture: the page must draw

VERBS = (
    "open",
    "show",
    "compare",
    "time",
    "panels",
    "simulate",
    "simulator",
    "screenshot",
    "quit",
)
# The rail's page names as the Studio parses them (pages.rs `Section::parse`).
SECTIONS = (
    "projects",
    "overview",
    "robots",
    "environments",
    "recordings",
    "datasets",
    "experiments",
    "policies",
    "evaluations",
    "findings",
    "deployments",
    "monitoring",
    "simulator",
    "live",  # the Simulator page's name until 2026-09-09; the Studio still parses it
)
PANEL_ACTIONS = ("expand", "toggle")
# The built Studio, relative to the checkout.
STUDIO_RELEASE = Path("crates") / "studio-shell" / "target" / "release"


# -- paths ---------------------------------------------------------------------


def commands_dir(project: Project) -> Path:
    return project.root / INDEX_DIR / COMMANDS_DIR


def state_path(project: Project) -> Path:
    return project.root / INDEX_DIR / STATE_FILE


def events_path(project: Project) -> Path:
    return project.root / INDEX_DIR / EVENTS_FILE


# -- the state and the events (Studio -> agent) --------------------------------


def state(project: Project) -> dict[str, Any]:
    """What the Studio shows, plus `alive`: a fresh heartbeat from a live pid."""
    path = state_path(project)
    if not path.is_file():
        return {
            "alive": False,
            "reason": f"no Studio has run on {project.root}",
            "project": str(project.root),
        }
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as why:
        return {"alive": False, "reason": f"unreadable state file: {why}"}
    family = str(raw.get("schema", "")).split("/", 1)[0]
    if family != STATE_SCHEMA.split("/", 1)[0]:
        schema = raw.get("schema")
        return {
            "alive": False,
            "reason": f"state file schema {schema!r} is not {STATE_SCHEMA!r}",
        }
    pid = int(raw.get("pid") or 0)
    age = time.time() - float(raw.get("heartbeat") or 0.0)
    alive = pid > 0 and pid_alive(pid) and age < STALE_S
    raw["alive"] = alive
    raw["heartbeat_age_s"] = round(age, 3)
    if not alive:
        raw["reason"] = (
            f"pid {pid} is gone"
            if not pid_alive(pid)
            else f"heartbeat is {age:.1f} s old (stale past {STALE_S:g} s)"
        )
    return raw


def events(
    project: Project, since_ns: int = 0, limit: int = 200
) -> list[dict[str, Any]]:
    """The human's actions after `since_ns` (an epoch in nanoseconds),
    newest last, at most `limit`."""
    path = events_path(project)
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if int(event.get("t", 0)) > since_ns:
            out.append(event)
    return out[-limit:]


# -- commands (agent -> Studio) ------------------------------------------------


def send(project: Project, verb: str, /, **args: Any) -> str:
    """Write one command file atomically; return its id (time-ordered)."""
    if verb not in VERBS:
        raise ValueError(f"unknown verb {verb!r}; one of {', '.join(VERBS)}")
    folder = commands_dir(project)
    folder.mkdir(parents=True, exist_ok=True)
    cid = f"{time.time_ns()}-{verb}"
    body = {"schema": SCHEMA, "id": cid, "verb": verb}
    body.update({k: v for k, v in args.items() if v is not None})
    tmp = folder / f"{cid}.json.tmp"
    tmp.write_text(json.dumps(body), encoding="utf-8")
    tmp.replace(folder / f"{cid}.json")
    _prune(folder)
    return cid


def ack(project: Project, cid: str) -> dict[str, Any] | None:
    path = commands_dir(project) / f"{cid}.ack.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def wait(
    project: Project, cid: str, timeout_s: float = ACK_TIMEOUT_S
) -> dict[str, Any]:
    """The Studio's answer to a command, or `no answer` past the timeout."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        answer = ack(project, cid)
        if answer is not None:
            return answer
        time.sleep(POLL_S)
    return {
        "status": "no answer",
        "reason": f"the Studio did not acknowledge within {timeout_s:g} s",
    }


def command(
    project: Project, verb: str, /, timeout_s: float = ACK_TIMEOUT_S, **args: Any
) -> dict[str, Any]:
    """Send a command to a live Studio and wait for its answer. Refuses by
    name when no Studio is alive on the project."""
    current = state(project)
    if not current.get("alive"):
        return {
            "status": "refused",
            "reason": f"no Studio is running on {project.root}: "
            f"{current.get('reason')}; launch_studio first",
        }
    cid = send(project, verb, **args)
    answer = wait(project, cid, timeout_s=timeout_s)
    answer["command"] = cid
    return answer


def screenshot(
    project: Project,
    section: str | None = None,
    artifact: str | None = None,
    width: int = SCREENSHOT_WIDTH,
) -> dict[str, Any]:
    """Navigate first when asked (a page, or an artifact whose drawer
    opens), let the page render, then capture the whole window as a PNG
    under `.index/screenshots/`. The answer names the file and its size."""
    if section is not None or artifact is not None:
        opened = command(project, "open", section=section, artifact=artifact)
        if opened.get("status") != "done":
            return opened
        time.sleep(SETTLE_S)  # two frames: the page lays out, the pictures load
    answer = command(project, "screenshot", timeout_s=SCREENSHOT_TIMEOUT_S, width=width)
    if answer.get("status") == "done" and "path" in answer:
        answer["read"] = "open the PNG at `path` to see the window"
    return answer


# -- lifecycle -----------------------------------------------------------------


PRESENT_STATUS_FILE = "present-status.json"
PRESENT_TIMEOUT_S = 20.0
PRESENT_POLL_S = 0.2


def present_status_path(project: Project) -> Path:
    return project.root / INDEX_DIR / PRESENT_STATUS_FILE


def wait_presented(
    project: Project, stamp: str, *, since: float, timeout_s: float = PRESENT_TIMEOUT_S
) -> dict[str, Any]:
    """The presenter's answer for `stamp` written after `since` (a wall
    clock reading taken before the request): `shown`, or `failed` with
    the reason; `pending` when none came within the timeout (a big scene
    still streaming, or no presenter running)."""
    deadline = time.monotonic() + timeout_s
    path = present_status_path(project)
    while time.monotonic() < deadline:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            written = float(raw.get("t") or path.stat().st_mtime)
            if written >= since:
                if raw.get("stamp") == stamp and raw.get("error"):
                    return {
                        "status": "failed",
                        "reason": raw["error"],
                        "artifact": stamp,
                    }
                if raw.get("shown") == stamp:
                    return {"status": "shown", "artifact": stamp}
        except (OSError, ValueError):
            pass
        time.sleep(PRESENT_POLL_S)
    return {
        "status": "pending",
        "reason": f"no answer from the presenter within {timeout_s:g} s",
        "artifact": stamp,
    }


def studio_binary() -> Path | None:
    """The built Studio: `$TRAINNR_STUDIO`, else the repo's release build."""
    named = os.environ.get(STUDIO_ENV)
    if named:
        path = Path(named)
        return path if path.is_file() else None
    exe = "studio-shell.exe" if sys.platform.startswith("win") else "studio-shell"
    path = checkout() / STUDIO_RELEASE / exe
    return path if path.is_file() else None


# The Studio's embedded Rerun server: the port of `viz.STUDIO_ADDRESS`,
# the one address every feed connects to.
VIEWER_PORT = urlsplit(STUDIO_ADDRESS).port or 0
PORT_FREE_TIMEOUT_S = 8.0


def viewer_port_free(port: int | None = None) -> bool:
    """Whether the viewer's port can be bound right now (the module's
    VIEWER_PORT, read at call time so a test can point it elsewhere)."""
    import socket  # noqa: PLC0415

    port = VIEWER_PORT if port is None else port

    # No SO_REUSEADDR: on macOS it lets the probe bind beside a live
    # listener, which is the one case the check exists for.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("0.0.0.0", port))
        except OSError:
            return False
    return True


# Studios this process launched, by pid: a child that has exited stays a
# zombie - "existing" to every probe - until its parent reaps it. The
# MCP server launches and later quits in one process, and `quit` waited
# its whole timeout on a Studio that had left in 0.2 s (2026-09-12).
_LAUNCHED: dict[int, subprocess.Popen] = {}


def launch(project: Project, binary: Path | None = None) -> dict[str, Any]:
    """Start the Studio on the project; wait for its first heartbeat.
    Waits for the viewer's port first: a window quit a moment ago can
    still hold it, and a Studio started then runs without a viewer
    server ("Address already in use" in its log; nothing streams in —
    seen 2026-09-09)."""
    current = state(project)
    if current.get("alive"):
        return {
            "status": "refused",
            "reason": f"a Studio (pid {current.get('pid')}) already runs "
            f"on {project.root}",
            "pid": current.get("pid"),
        }
    binary = binary or studio_binary()
    if binary is None or not binary.is_file():
        return {
            "status": "refused",
            "reason": (
                f"no Studio binary at {binary}"
                if binary is not None
                else "no Studio binary: build it with "
                "`cargo build --release -p studio-shell` or set $TRAINNR_STUDIO"
            ),
        }
    deadline = time.monotonic() + PORT_FREE_TIMEOUT_S
    while not viewer_port_free() and time.monotonic() < deadline:
        time.sleep(0.1)
    if not viewer_port_free():
        return {
            "status": "refused",
            "reason": f"port {VIEWER_PORT} (the viewer server) is held by another "
            "process; quit the other Studio or Rerun viewer first",
        }
    log = project.root / INDEX_DIR / STUDIO_LOG
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env[PROJECT_ENV] = str(project.root)
    with log.open("ab") as sink:
        child = subprocess.Popen(
            [str(binary)],
            cwd=str(checkout()),
            env=env,
            stdout=sink,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    _LAUNCHED[child.pid] = child
    deadline = time.monotonic() + LAUNCH_TIMEOUT_S
    while time.monotonic() < deadline:
        if child.poll() is not None:
            return {
                "status": "failed",
                "reason": f"the Studio exited with code {child.returncode}; see {log}",
                "log": str(log),
            }
        current = state(project)
        if current.get("alive") and int(current.get("pid") or 0) == child.pid:
            return {
                "status": "done",
                "pid": child.pid,
                "binary": str(binary),
                "log": str(log),
            }
        time.sleep(0.1)
    return {
        "status": "failed",
        "reason": f"no heartbeat from pid {child.pid} within {LAUNCH_TIMEOUT_S:g} s; "
        f"see {log}",
        "pid": child.pid,
        "log": str(log),
    }


def quit(project: Project, timeout_s: float = QUIT_TIMEOUT_S) -> dict[str, Any]:
    """Ask the Studio to close; past the timeout, terminate it by pid."""
    current = state(project)
    pid = int(current.get("pid") or 0)
    if not current.get("alive"):
        return {"status": "done", "reason": "no Studio was running", "pid": pid}
    answer = command(project, "quit")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not pid_alive(pid):
            return {"status": "done", "pid": pid, "answer": answer.get("status")}
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as why:
        return {"status": "failed", "reason": f"terminate {pid}: {why}", "pid": pid}
    return {"status": "done", "pid": pid, "reason": "terminated after the timeout"}


# -- helpers -------------------------------------------------------------------


def pid_alive(pid: int) -> bool:
    """Whether a process with this id is running. psutil's probe, because
    `os.kill(pid, 0)` is a liveness check on POSIX and a TERMINATE on
    Windows - it would have killed the Studio it asked after. A child of
    this process is reaped first; a zombie is not alive."""
    if pid <= 0:
        return False
    child = _LAUNCHED.get(pid)
    if child is not None and child.poll() is not None:
        del _LAUNCHED[pid]
        return False
    try:
        import psutil  # noqa: PLC0415 - the `mcp` extra
    except ImportError as why:
        raise ImportError(
            "psutil is needed to check a Studio's process: install the `mcp` extra"
        ) from why
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        return True  # it exists; another user's process, or a locked-down OS


def _prune(folder: Path, keep: int = KEEP_COMMANDS) -> None:
    files = sorted(p for p in folder.glob("*.json") if not p.name.endswith(".ack.json"))
    for old in files[:-keep] if len(files) > keep else []:
        old.unlink(missing_ok=True)
        old.with_name(old.name[: -len(".json")] + ".ack.json").unlink(missing_ok=True)


def terminate_group(pid: int) -> str:
    """Ask a process started in its own session (`start_new_session=True`)
    and everything under it to stop: SIGTERM to the process group where
    the OS has one, psutil's walk of the tree where it has not (Windows).
    Returns the note a caller reports."""
    if pid <= 0:
        return "no process"
    if hasattr(os, "killpg"):
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return "already gone"
        return "SIGTERM sent to the process group"
    import psutil  # noqa: PLC0415 - the `mcp` extra

    try:
        root = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return "already gone"
    for proc in [*root.children(recursive=True), root]:
        with contextlib.suppress(psutil.NoSuchProcess):
            proc.terminate()
    return "terminate sent to the process tree"
