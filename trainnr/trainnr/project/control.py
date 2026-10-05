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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from trainnr import safe_write
from trainnr.paths import CHECKOUT_ENV, checkout
from trainnr.project.files import read_json
from trainnr.project.locate import (
    INDEX_DIR,
    PROJECT_ENV,
    PROJECTS_ENV,
    Project,
    projects_home,
)
from trainnr.viz import STUDIO_ADDRESS

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
# The pointer a project switch leaves in the old project's state file
# (control.rs `StudioState::moved_to`), and how many it is followed through.
MOVED_TO = "moved_to"
MOVED_HOPS = 8
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
    "theme",
    "simulate",
    "simulator",
    "screenshot",
    "focus",
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
STUDIO_RELEASE = Path("crates") / "trainnr-studio" / "target" / "release"


# -- paths ---------------------------------------------------------------------


def commands_dir(project: Project) -> Path:
    return project.root / INDEX_DIR / COMMANDS_DIR


def state_path(project: Project) -> Path:
    return project.root / INDEX_DIR / STATE_FILE


def events_path(project: Project) -> Path:
    return project.root / INDEX_DIR / EVENTS_FILE


# -- the state and the events (Studio -> agent) --------------------------------


def _read_state(root: Path) -> dict[str, Any]:
    """One project's state file, judged on its own: `alive` from a fresh
    heartbeat and a live pid; a pointer (`moved_to`) is never alive."""
    path = root / INDEX_DIR / STATE_FILE
    if not path.is_file():
        return {
            "alive": False,
            "reason": f"no Studio has run on {root}",
            "project": str(root),
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
    # a heartbeat from the future is a planted file, not a live window
    # (a shared project's state file once made quit_studio signal any pid:
    # security review, 2026-10-04)
    fresh = -HEARTBEAT_FUTURE_SLACK_S <= age < STALE_S
    alive = pid > 0 and pid_alive(pid) and fresh and not raw.get(MOVED_TO)
    raw["alive"] = alive
    raw["heartbeat_age_s"] = round(age, 3)
    if alive:
        rebuilt = binary_rebuilt_since(pid)
        if rebuilt is not None:
            raw["stale_binary"] = rebuilt
    if not alive:
        raw["reason"] = (
            f"pid {pid} is gone"
            if not pid_alive(pid)
            else f"heartbeat is {age:.1f} s old (stale past {STALE_S:g} s)"
        )
    return raw


def state(project: Project) -> dict[str, Any]:
    """What the Studio shows, plus `alive`: a fresh heartbeat from a live
    pid. A Studio that switched to another project leaves a pointer
    behind (`moved_to`); it is followed, and the live state comes back
    with `elsewhere: True` and `asked` naming the project this was called
    under, so a door called under the old project finds the window
    instead of reading it as dead (2026-09-27)."""
    root = project.root
    seen: set[str] = set()
    current = _read_state(root)
    while (
        current.get(MOVED_TO)
        and pid_alive(int(current.get("pid") or 0))
        and len(seen) < MOVED_HOPS
    ):
        target = str(current[MOVED_TO])
        if target in seen:
            break
        seen.add(target)
        current = _read_state(Path(target))
    if seen:
        if current.get("alive"):
            current["elsewhere"] = True
            current["asked"] = str(root)
        else:
            current["reason"] = (
                f"the Studio moved from {root} to {current.get('project')}: "
                f"{current.get('reason')}"
            )
    return current


def home(project: Project) -> Project:
    """The project the live Studio is on: `project` itself, or the one its
    pointer leads to."""
    found = state(project)
    if found.get("alive") and found.get("elsewhere"):
        return Project(Path(str(found["project"])))
    return project


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
    cid = f"{time.time_ns()}-{verb}"
    body = {"schema": SCHEMA, "id": cid, "verb": verb}
    body.update({k: v for k, v in args.items() if v is not None})
    # The running Studio's session token: it applies no command without
    # it, so a command file that came with a project never runs.
    token = read_json(state_path(project), missing_ok=True).get("session")
    if isinstance(token, str):
        body["session"] = token
    safe_write.write_text(folder / f"{cid}.json", json.dumps(body), project.root)
    _prune(folder, root=project.root)
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


# Commands about the open project's own artifacts: sent to a Studio that
# sits on another project they would name stamps its index has not got.
PROJECT_BOUND = frozenset({"show", "compare", "simulate"})


def bound_elsewhere(verb: str, args: dict[str, Any]) -> bool:
    """Whether `verb` with `args` only makes sense on the project it was
    called under: a show, a compare, a simulate, or an open of an artifact
    that does not also name a project to switch to."""
    if verb in PROJECT_BOUND:
        return True
    opens_artifact = verb == "open" and args.get("artifact") is not None
    return opens_artifact and not args.get("project")


def command(
    project: Project, verb: str, /, timeout_s: float = ACK_TIMEOUT_S, **args: Any
) -> dict[str, Any]:
    """Send a command to a live Studio and wait for its answer. Refuses by
    name when no Studio is alive on the project. A Studio that moved to
    another project answers page, time, screenshot and quit commands from
    there; one about this project's artifacts is refused with the door
    that brings the window back."""
    current = state(project)
    if not current.get("alive"):
        return {
            "status": "refused",
            "reason": f"no Studio is running on {project.root}: "
            f"{current.get('reason')}; launch_studio first",
        }
    target = project
    if current.get("elsewhere"):
        where = str(current.get("project"))
        if bound_elsewhere(verb, args):
            return {
                "status": "refused",
                "reason": f"the Studio (pid {current.get('pid')}) is open on {where}, "
                f"not {project.root}; open_in_studio(project={str(project.root)!r}) "
                "first",
                "project": where,
            }
        target = Project(Path(where))
    cid = send(target, verb, **args)
    answer = wait(target, cid, timeout_s=timeout_s)
    answer["command"] = cid
    if target is not project:
        answer["project"] = str(target.root)
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
    the reason; `presenting` when the presenter has taken it up but a big
    scene is still streaming at the timeout (it flushes for up to
    `present.FLUSH_S`; `describe_studio` reports the outcome); `pending`
    when no presenter answered at all."""
    deadline = time.monotonic() + timeout_s
    path = present_status_path(project)
    taken_up = False
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
                taken_up = taken_up or raw.get("presenting") == stamp
        except (OSError, ValueError):
            pass
        time.sleep(PRESENT_POLL_S)
    if taken_up:
        return {
            "status": "presenting",
            "reason": f"the presenter is still streaming {stamp} after "
            f"{timeout_s:g} s (a big scene flushes for minutes); describe_studio "
            "reports `presenter.shown` when it lands",
            "artifact": stamp,
        }
    return {
        "status": "pending",
        "reason": f"no answer from the presenter within {timeout_s:g} s",
        "artifact": stamp,
    }


def binary_rebuilt_since(pid: int) -> str | None:
    """The word when the file the running Studio was started from is newer
    than the process (a rebuild after launch: the window runs the old code
    and refuses the new kinds by name, 2026-09-23); None when it is
    current, when the process cannot be read, or when the file is gone
    (cargo replacing it between two looks). The process's own executable,
    not the path a launch would take now, so a Studio started from another
    build is judged against its own file. Same host, so no clock skew;
    the start time's second is the only ambiguity, and harmless."""
    psutil = _psutil()
    try:
        process = psutil.Process(pid)
        started = process.create_time()
        exe = Path(process.exe())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None
    try:
        built = exe.stat().st_mtime
    except OSError:
        return None
    if built <= started:
        return None
    when = datetime.fromtimestamp(built, tz=timezone.utc).replace(microsecond=0)
    return (
        f"the Studio binary was rebuilt at {when.isoformat()} (UTC) after this "
        "Studio launched; quit_studio then launch_studio to run it"
    )


def _psutil() -> Any:
    """psutil, or the one refusal naming the extra that carries it."""
    try:
        import psutil  # noqa: PLC0415 - the `mcp` extra
    except ImportError as why:
        raise ImportError(
            "psutil is needed to check a Studio's process: install the `mcp` extra"
        ) from why
    return psutil


def studio_binary() -> Path | None:
    """The Studio to run: `$TRAINNR_STUDIO`, else the checkout's own
    release build, else the prebuilt one downloaded for this version
    (`trainnr.studio_install`), else None."""
    from trainnr.studio_install import installed_binary  # noqa: PLC0415

    named = os.environ.get(STUDIO_ENV)
    if named:
        path = Path(named)
        return path if path.is_file() else None
    exe = "trainnr-studio.exe" if sys.platform.startswith("win") else "trainnr-studio"
    path = checkout() / STUDIO_RELEASE / exe
    if path.is_file():
        return path
    return installed_binary()


def ensure_studio_binary() -> Path | dict[str, Any]:
    """The Studio's binary, downloading the prebuilt one when nothing is
    built or installed (the plugin's first launch: its session hook
    normally fetched it already), or a refusal saying why there is none."""
    from trainnr.studio_install import StudioInstallError, install  # noqa: PLC0415

    found = studio_binary()
    if found is not None:
        return found
    if os.environ.get(STUDIO_ENV):
        return {"status": "refused", "reason": f"${STUDIO_ENV} names no file"}
    try:
        return install(say=lambda _line: None)
    except StudioInstallError as why:
        return {"status": "refused", "reason": str(why)}


# The Studio's embedded Rerun server: the port of `viz.STUDIO_ADDRESS`,
# the one address every feed connects to.
VIEWER_PORT = urlsplit(STUDIO_ADDRESS).port or 0
PORT_FREE_TIMEOUT_S = 8.0


def viewer_port_free(port: int | None = None) -> bool:
    """Whether the viewer's port can be bound right now (the module's
    VIEWER_PORT, read at call time so a test can point it elsewhere)."""
    import socket  # noqa: PLC0415

    host, bound = viewer_bind()
    if port is None:
        port = bound if os.environ.get(VIEWER_BIND_ENV) else VIEWER_PORT
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        # On Linux, SO_REUSEADDR still refuses a live listener but ignores
        # connections closing in TIME_WAIT, as the Studio's own server
        # does; without it a job retrying its stream kept the probe
        # refusing an empty port (2026-10-04). Not on macOS, where it
        # binds beside a live listener, nor Windows, where it can steal one.
        if sys.platform.startswith("linux"):
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def port_holder(port: int) -> str | None:
    """Who listens on `port`, as `pid N (name)`, when the system says
    (macOS lists other processes' sockets only to root): a refusal named
    "another process" left an agent unable to tell which (2026-10-05)."""
    try:
        import psutil  # noqa: PLC0415

        for conn in psutil.net_connections(kind="tcp"):
            if (
                conn.status == psutil.CONN_LISTEN
                and conn.laddr
                and conn.laddr.port == port
            ):
                if conn.pid is None:
                    return None
                return f"pid {conn.pid} ({psutil.Process(conn.pid).name()})"
    except Exception:  # a name is a courtesy; the refusal stands
        return None
    return None


# The Studio's viewer server binds this host unless TRAINNR_VIEWER_BIND
# says otherwise (crates/trainnr-studio/src/main.rs, GRPC_BIND); the probe
# checks the same address the server will take.
VIEWER_BIND_ENV = "TRAINNR_VIEWER_BIND"
VIEWER_BIND_HOST = "127.0.0.1"


def viewer_bind() -> tuple[str, int]:
    """The host and port the Studio's viewer server binds:
    `$TRAINNR_VIEWER_BIND` (`0.0.0.0:9876` opens it; `[::1]:9876` is IPv6,
    brackets dropped), else this machine only on the viewer's port."""
    value = os.environ.get(VIEWER_BIND_ENV, "")
    host, _, port = value.rpartition(":") if ":" in value else (value, "", "")
    host = host.strip("[]") or VIEWER_BIND_HOST
    try:
        return host, int(port)
    except ValueError:
        return host, VIEWER_PORT


def viewer_bind_host() -> str:
    """The host part of `viewer_bind`."""
    return viewer_bind()[0]


# Studios this process launched, by pid: a child that has exited stays a
# zombie - "existing" to every probe - until its parent reaps it. The
# MCP server launches and later quits in one process, and `quit` waited
# its whole timeout on a Studio that had left in 0.2 s (2026-09-12).
_LAUNCHED: dict[int, subprocess.Popen] = {}


def _already_running(project: Project, current: dict[str, Any]) -> dict[str, Any]:
    """launch() on a live Studio: one on another project is moved here
    (one window at a time: the viewer's port); one on this project is
    refused, with a rebuilt binary named."""
    if current.get("elsewhere"):
        moved = command(project, "open", project=str(project.root))
        moved["pid"] = current.get("pid")
        moved["switched_from"] = current.get("project")
        if moved.get("status") != "done":
            moved["status"] = "failed"
        return moved
    stale = current.get("stale_binary")
    return {
        "status": "refused",
        "reason": f"a Studio (pid {current.get('pid')}) already runs "
        f"on {project.root}" + (f"; {stale}" if stale else ""),
        "pid": current.get("pid"),
    }


# Under WSL the window's own graphics go through Mesa's Vulkan-over-Direct3D
# layer, which finds the GPU only with the WSL library directory on the
# loader path (and MuJoCo's EGL only with the d3d12 Gallium driver). The
# documented launch line (`trainnr/wsl.env`) sets both; a launch through
# this door did not, and the window drew a frame in half a second on a
# software fallback (2026-10-03, "I know we fixed the mujoco smoothness").
# The loader reads the path at process start, so the launcher sets it.
WSL_LIB_DIR = "/usr/lib/wsl/lib"
WSL_GALLIUM_DRIVER = "d3d12"


def on_wsl() -> bool:
    """WSL, by the one rule the package and the Studio share."""
    from trainnr.paths import on_wsl as detected  # noqa: PLC0415

    return detected()


def wsl_gpu_environment(env: dict[str, str]) -> dict[str, str]:
    """The environment with the WSL GPU libraries first on the loader path
    and the Direct3D Gallium driver named, when this is WSL and the
    directory exists; unchanged elsewhere, and never overriding a driver
    the caller chose."""
    if not on_wsl() or not Path(WSL_LIB_DIR).is_dir():
        return env
    env.setdefault("GALLIUM_DRIVER", WSL_GALLIUM_DRIVER)
    existing = env.get("LD_LIBRARY_PATH", "")
    if WSL_LIB_DIR not in existing.split(":"):
        env["LD_LIBRARY_PATH"] = (
            f"{WSL_LIB_DIR}:{existing}" if existing else WSL_LIB_DIR
        )
    return env


# WSLg serves Wayland from its own runtime directory and links it into
# the user's; on a boot where that link is missing (no /run/user/<uid>),
# the window found no compositor and the Studio exited at once
# (2026-10-05: "WaylandError(Connection(NoCompositor))"). The launcher
# points the child at WSLg's directory when the user's lacks the socket.
WSLG_RUNTIME_DIR = Path("/mnt/wslg/runtime-dir")


def wsl_display_environment(
    env: dict[str, str], wslg_runtime: Path = WSLG_RUNTIME_DIR
) -> dict[str, str]:
    """The environment with `XDG_RUNTIME_DIR` at WSLg's own directory when
    this is WSL, the Wayland socket is missing where the environment says
    it is, and WSLg's directory holds it; unchanged otherwise."""
    display = env.get("WAYLAND_DISPLAY", "")
    if not on_wsl() or not display or Path(display).is_absolute():
        return env
    runtime = env.get("XDG_RUNTIME_DIR", "")
    if runtime and (Path(runtime) / display).exists():
        return env
    if (wslg_runtime / display).exists():
        env["XDG_RUNTIME_DIR"] = str(wslg_runtime)
    return env


def launch(  # noqa: PLR0911 - each refusal names its own reason
    project: Project, binary: Path | None = None
) -> dict[str, Any]:
    """Start the Studio on the project; wait for its first heartbeat.
    Waits for the viewer's port first: a window quit a moment ago can
    still hold it, and a Studio started then runs without a viewer
    server ("Address already in use" in its log; nothing streams in —
    seen 2026-09-09)."""
    current = state(project)
    if current.get("alive"):
        return _already_running(project, current)
    if binary is None:
        found = ensure_studio_binary()
        if isinstance(found, dict):
            return found
        binary = found
    if not binary.is_file():
        return {"status": "refused", "reason": f"no Studio binary at {binary}"}
    deadline = time.monotonic() + PORT_FREE_TIMEOUT_S
    while not viewer_port_free() and time.monotonic() < deadline:
        time.sleep(0.1)
    if not viewer_port_free():
        holder = port_holder(viewer_bind()[1]) or "another process"
        return {
            "status": "refused",
            "reason": f"port {viewer_bind()[1]} (the viewer server) is held by "
            f"{holder}; quit that Studio or Rerun viewer first (a Studio on "
            "another project: quit_studio from that project)",
        }
    log = project.root / INDEX_DIR / STUDIO_LOG
    log.parent.mkdir(parents=True, exist_ok=True)
    env = wsl_display_environment(wsl_gpu_environment(dict(os.environ)))
    env[PROJECT_ENV] = str(project.root)
    # the projects home the switcher lists, the one the tools create in
    env.setdefault(PROJECTS_ENV, str(projects_home()))
    # A downloaded Studio lives in the user's cache, not in the checkout:
    # it finds the simulator's scripts through the checkout's path.
    env.setdefault(CHECKOUT_ENV, str(checkout()))
    with safe_write.open_append(log, project.root) as sink:
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


# A heartbeat this far ahead of the clock is accepted (clock skew between
# the Studio and this process); further ahead, the state file is not trusted.
HEARTBEAT_FUTURE_SLACK_S = 5.0
# The Studio's process name, as the OS reports it.
STUDIO_PROCESS = "trainnr-studio"


def is_studio_process(pid: int) -> bool:
    """Whether `pid` is a Studio: only then may quit signal it."""
    psutil = _psutil()
    try:
        name = psutil.Process(pid).name()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False
    return name.removesuffix(".exe").startswith(STUDIO_PROCESS)


def quit(project: Project, timeout_s: float = QUIT_TIMEOUT_S) -> dict[str, Any]:
    """Ask the Studio to close; past the timeout, terminate it by pid, and
    only a process that is a Studio."""
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
    if not is_studio_process(pid):
        return {
            "status": "failed",
            "reason": f"pid {pid} is not a Studio process; it was not signalled",
            "pid": pid,
        }
    try:
        note = terminate_group(pid)
    except OSError as why:
        return {"status": "failed", "reason": f"terminate {pid}: {why}", "pid": pid}
    return {"status": "done", "pid": pid, "reason": f"after the timeout: {note}"}


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
    psutil = _psutil()
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        return True  # it exists; another user's process, or a locked-down OS


def _prune(folder: Path, keep: int = KEEP_COMMANDS, root: Path | None = None) -> None:
    """Keep the newest commands; delete only in a real folder of the
    project (a linked `.index/commands` emptied a folder outside it)."""
    files = sorted(p for p in folder.glob("*.json") if not p.name.endswith(".ack.json"))
    for old in files[:-keep] if len(files) > keep else []:
        safe_write.remove(old, root)
        safe_write.remove(old.with_name(old.name[: -len(".json")] + ".ack.json"), root)


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
    psutil = _psutil()
    try:
        root = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return "already gone"
    for proc in [*root.children(recursive=True), root]:
        with contextlib.suppress(psutil.NoSuchProcess):
            proc.terminate()
    return "terminate sent to the process tree"
