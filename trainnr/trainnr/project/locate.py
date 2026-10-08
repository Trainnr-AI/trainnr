"""Which project, and where its parts are — one answer, overridable, loud.

Projects live in the user's projects home: `$TRAINNR_PROJECTS`, else
`~/trainnr/projects` — never inside a checkout or a plugin's install
folder, which an update replaces. The current project is
`$TRAINNR_PROJECT`, else the one last created or chosen (`use_project`),
remembered in the projects home's `.current` file; a checkout that still
has a `projects/default` keeps opening it. With none of those, a tool is
refused with the two ways out (`create_project`, `use_project`) instead
of a stack trace from whatever tried to write into a missing directory.

A project is portable: it holds its own recordings, fits, tasks, batches,
datasets, runs, policies, certificates, deploy manifests and findings.
The checkout's `robots/` and `robots/actuator-bundles/` stay a SHARED,
READ-ONLY library that projects reference by stamp — a project may carry
its own robot bundles too, and those are the ones a tool writes.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from trainnr.bundles.hashing import FITS_DIR
from trainnr.paths import checkout, user_home
from trainnr.project.files import read_json, write_json

PROJECT_ENV = "TRAINNR_PROJECT"
PROJECTS_ENV = "TRAINNR_PROJECTS"  # the projects home
CURRENT_FILE = ".current"  # in the projects home: the chosen project's path
USER_HOME_DIR = "trainnr"  # ~/trainnr/projects
MANIFEST_FILE = "project.json"
INDEX_DIR = ".index"
INDEX_FILE = "project.json"
# The MCP job table (`trainnr.mcp_jobs`) lives beside the kinds.
JOBS_DIR = "mcp-jobs"
PROJECTS_DIR_NAME = "projects"
DEFAULT_PROJECT = "default"
SCHEMA = "trainnr-project/1"

# The one folder per artifact kind, each named once here and reached
# through `Project.folder`. The names are the kinds' plural forms so a
# directory listing reads as an inventory.
ROBOTS_FOLDER = "robots"
RECORDINGS_FOLDER = "recordings"
FITS_FOLDER = FITS_DIR  # the one spelling lives in bundles.hashing
TASKS_FOLDER = "tasks"
BATCHES_FOLDER = "batches"
DATASETS_FOLDER = "datasets"
RUNS_FOLDER = "runs"
POLICIES_FOLDER = "policies"
CERTIFICATES_FOLDER = "certificates"
DEPLOY_FOLDER = "deploy"
MONITORING_FOLDER = "monitoring"  # drift checks: fresh telemetry judged
SCENES_FOLDER = "scenes"  # captured scenes: splat, proxy, the gap (docs/78)
FINDINGS_FOLDER = "findings"
FOLDERS = (
    ROBOTS_FOLDER,
    RECORDINGS_FOLDER,
    FITS_FOLDER,
    TASKS_FOLDER,
    BATCHES_FOLDER,
    DATASETS_FOLDER,
    RUNS_FOLDER,
    POLICIES_FOLDER,
    CERTIFICATES_FOLDER,
    DEPLOY_FOLDER,
    MONITORING_FOLDER,
    SCENES_FOLDER,
    FINDINGS_FOLDER,
)


@dataclass(frozen=True)
class Manifest:
    """`project.json`: who this project is. Deliberately small — what the
    project CONTAINS is discovered by walking it, never declared here,
    so the manifest cannot drift from the directory."""

    schema: str
    name: str
    created: str  # ISO-8601, UTC
    description: str = ""
    # Stamps of shared-library bundles this project relies on, recorded
    # so a project moved to another checkout can say what it expects.
    library: dict[str, list[str]] = field(default_factory=dict)
    # How the policy learns: `imitation` (from a dataset of
    # demonstrations) or `reinforcement` (from its own rollouts — no
    # dataset stage). Empty when the project has not said; the index
    # then reads it off the runs it holds.
    loop: str = ""


LOOPS = ("imitation", "reinforcement")


@dataclass(frozen=True)
class Project:
    """A project directory and its parts."""

    root: Path

    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_FILE

    def manifest(self) -> Manifest:
        raw = read_json(self.manifest_path)
        if raw.get("schema") != SCHEMA:
            raise ValueError(
                f"{self.manifest_path} has schema {raw.get('schema')!r}; "
                f"this code speaks {SCHEMA!r}"
            )
        known = {f for f in Manifest.__dataclass_fields__}
        return Manifest(**{k: v for k, v in raw.items() if k in known})

    @property
    def name(self) -> str:
        return self.manifest().name

    def use(self) -> Project:
        """Make this the project the bundle locator searches first, so a
        robot onboarded here builds tasks like a library rig."""
        from trainnr.bundles.locate import add_search_root  # noqa: PLC0415

        add_search_root(self.robots)
        return self

    def folder(self, kind_folder: str) -> Path:
        if kind_folder not in FOLDERS:
            raise KeyError(f"no folder {kind_folder!r} in a project; one of {FOLDERS}")
        return self.root / kind_folder

    def relative(self, kind_folder: str, name: str) -> str:
        """The path the index records for `<kind_folder>/<name>`: POSIX
        on every OS (`Artifact.path`, the Studio reads it as such)."""
        return (self.folder(kind_folder) / name).relative_to(self.root).as_posix()

    @property
    def robots(self) -> Path:
        return self.folder(ROBOTS_FOLDER)

    @property
    def recordings(self) -> Path:
        return self.folder(RECORDINGS_FOLDER)

    @property
    def tasks(self) -> Path:
        return self.folder(TASKS_FOLDER)

    @property
    def batches(self) -> Path:
        return self.folder(BATCHES_FOLDER)

    @property
    def runs(self) -> Path:
        return self.folder(RUNS_FOLDER)

    @property
    def policies(self) -> Path:
        return self.folder(POLICIES_FOLDER)

    @property
    def certificates(self) -> Path:
        return self.folder(CERTIFICATES_FOLDER)

    @property
    def deploy(self) -> Path:
        return self.folder(DEPLOY_FOLDER)

    @property
    def monitoring(self) -> Path:
        return self.folder(MONITORING_FOLDER)

    @property
    def scenes(self) -> Path:
        return self.folder(SCENES_FOLDER)

    @property
    def findings(self) -> Path:
        return self.folder(FINDINGS_FOLDER)

    @property
    def index_path(self) -> Path:
        return self.root / INDEX_DIR / INDEX_FILE


def projects_home() -> Path:
    """Where projects are created: `$TRAINNR_PROJECTS`, else
    `<user home>/projects` (`$TRAINNR_HOME`, else ~/trainnr)."""
    named = os.environ.get(PROJECTS_ENV, "").strip()
    if named:
        return Path(named).expanduser().absolute()
    return user_home() / PROJECTS_DIR_NAME


def checkout_projects() -> Path | None:
    """A checkout's own `projects/`, when it has one: the maintainers'
    working projects and the committed sample are listed from it too.
    None for a plugin install: its copy of the repository is replaced on
    every update, and its sample read as a project of the user's that
    was not under the projects home (fresh-install audit, 2026-10-08)."""
    if _is_plugin_install(checkout()):
        return None
    folder = checkout() / PROJECTS_DIR_NAME
    return folder if folder.is_dir() else None


# Claude Code installs a plugin under its configuration folder
# (`~/.claude`, or `$CLAUDE_CONFIG_DIR`), in `plugins/`.
CLAUDE_CONFIG_ENV = "CLAUDE_CONFIG_DIR"
CLAUDE_PLUGINS_DIR = "plugins"


def _is_plugin_install(root: Path) -> bool:
    config = os.environ.get(CLAUDE_CONFIG_ENV, "").strip()
    base = Path(config).expanduser() if config else Path.home() / ".claude"
    return (base / CLAUDE_PLUGINS_DIR).resolve() in root.resolve().parents


def projects_dir() -> Path:
    """The projects home (kept under this name for callers that predate it)."""
    return projects_home()


def _is_project(root: Path) -> bool:
    return (root / MANIFEST_FILE).is_file()


def remembered_project() -> Path | None:
    """The project last created or chosen, when its manifest is still there."""
    marker = projects_home() / CURRENT_FILE
    try:
        text = marker.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    root = Path(text) if text else None
    return root if root is not None and _is_project(root) else None


# The project this process chose (create_project, use_project). It wins
# over the projects home's `.current`, which only seeds a new process: two
# agent sessions shared that file, so one session's use_project moved the
# other's training and exports into its own project (review, 2026-10-04).
_session_root: Path | None = None


def session_project() -> Path | None:
    """The project this process chose, while it is still a project."""
    if _session_root is not None and _is_project(_session_root):
        return _session_root
    return None


def remember_project(root: Path) -> None:
    """Make `root` the current project for every later call in this
    process that names none, and the starting project of later sessions."""
    global _session_root  # noqa: PLW0603 - the one per-process choice
    _session_root = Path(root).absolute()
    home = projects_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / CURRENT_FILE).write_text(
        str(Path(root).absolute()) + "\n", encoding="utf-8"
    )


def resolve_project_path(target: str | Path) -> Path:
    """A project named by the user: an absolute path as given; a bare name
    or relative path under the projects home, else under a checkout's
    `projects/` when the project is there."""
    path = Path(target).expanduser()
    if path.is_absolute():
        return path
    under_home = projects_home() / path
    if _is_project(under_home):
        return under_home
    in_checkout = checkout_projects()
    if in_checkout is not None and _is_project(in_checkout / path):
        return in_checkout / path
    return under_home


NO_PROJECT = (
    "no project is selected: create one with `create_project` (it becomes the "
    "current project) or choose one with `use_project`; `list_projects` lists "
    f"them. ${PROJECT_ENV} names one for this process."
)


def current_project() -> Project:
    """`$TRAINNR_PROJECT`, else the project this process chose, else the
    remembered one (the projects home's `.current`), else a checkout's
    `projects/default`; refused by name when none of them is a project."""
    override = os.environ.get(PROJECT_ENV, "").strip()
    if override:
        # absolute: the Studio is spawned with the checkout as its working
        # directory, so a relative override would point it somewhere else
        # (measured 2026-09-23: a launch with `../projects/x` never heartbeat);
        # links are kept as typed (macOS's /var is one) so paths compare as given
        root = Path(override).expanduser().absolute()
        if not _is_project(root):
            raise FileNotFoundError(
                f"no project at {root} (no {MANIFEST_FILE}): ${PROJECT_ENV} names "
                "a directory that is not a project; create it with `create_project`"
            )
        return Project(root).use()
    chosen = session_project() or remembered_project()
    if chosen is not None:
        return Project(chosen).use()
    in_checkout = checkout_projects()
    if in_checkout is not None and _is_project(in_checkout / DEFAULT_PROJECT):
        return Project(in_checkout / DEFAULT_PROJECT).use()
    raise FileNotFoundError(NO_PROJECT)


def use_project(target: str | Path) -> Project:
    """Choose the current project by name or path; remembered for later
    calls and later sessions. Refused when the target is not a project."""
    root = resolve_project_path(target)
    if not _is_project(root):
        raise FileNotFoundError(
            f"no project at {root} (no {MANIFEST_FILE}); `list_projects` lists "
            "the projects, `create_project` makes one"
        )
    remember_project(root)
    return Project(root).use()


def create_project(
    root: Path, name: str, description: str = "", loop: str = ""
) -> Project:
    """Make a project: the manifest and every kind's folder. Refuses to
    overwrite an existing manifest — a project is never silently reset."""
    root = Path(root)
    manifest_path = root / MANIFEST_FILE
    if manifest_path.exists():
        raise FileExistsError(f"a project already exists at {root}")
    if not name or "/" in name or name.startswith("."):
        raise ValueError(f"project names are plain words, got {name!r}")
    root.mkdir(parents=True, exist_ok=True)
    for folder in FOLDERS:
        (root / folder).mkdir(exist_ok=True)
    manifest = Manifest(
        schema=SCHEMA,
        name=name,
        created=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        description=description,
        loop=_loop_or_refuse(loop),
    )
    write_json(manifest_path, asdict(manifest))
    return Project(root)


def list_projects(root: Path | None = None) -> list[Project]:
    """Every project in the projects home and, from a checkout, in its own
    `projects/` (or every project under `root` when one is named), by
    name, each directory once."""
    homes = [Path(root)] if root is not None else [projects_home()]
    if root is None and (in_checkout := checkout_projects()) is not None:
        homes.append(in_checkout)
    seen: set[Path] = set()
    found: list[Project] = []
    for home in homes:
        if not home.is_dir():
            continue
        for path in sorted(home.iterdir()):
            key = path.resolve()
            if _is_project(path) and key not in seen:
                seen.add(key)
                found.append(Project(path))
    in_checkout = checkout_projects()
    sample = (in_checkout / SAMPLE_PROJECT).resolve() if in_checkout else None
    if sample is not None and any(p.root.resolve() != sample for p in found):
        # the checkout's empty sample is for a fresh clone with nothing
        # else; beside the user's own it read as a broken project (review)
        found = [p for p in found if p.root.resolve() != sample]
    return sorted(found, key=lambda p: p.root.name)


# The project a fresh checkout carries (`projects/sample`), listed only
# when there is no other.
SAMPLE_PROJECT = "sample"
# A name becomes a folder on every OS: no separators (`/`, `\`), no
# version mark (`@`), no drive mark (`:`; `D:evil` escapes on Windows), and
# not one of Windows' reserved device names (security review, 2026-10-04).
NAME_FORBIDDEN = "/\\@:"
WINDOWS_DEVICE_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)


def plain_name(name: str, what: str = "name") -> str:
    """An artifact's name as a folder: one plain word, no separators, no
    path tricks. Returns it; refuses anything else by name."""
    if (
        not name
        or name != name.strip()
        or any(c in name for c in NAME_FORBIDDEN)
        or any(not c.isprintable() for c in name)
        or name in (".", "..")
        or name.startswith(".")
        or name.split(".", 1)[0].lower() in WINDOWS_DEVICE_NAMES
    ):
        raise ValueError(f"{what}: one plain word, no separators; got {name!r}")
    return name


def _loop_or_refuse(loop: str) -> str:
    if loop and loop not in LOOPS:
        raise ValueError(f"loop is one of {', '.join(LOOPS)} (or unset), got {loop!r}")
    return loop
