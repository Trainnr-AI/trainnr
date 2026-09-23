"""Which project, and where its parts are — one answer, overridable, loud.

The same shape as `bundles/locate.py` for robot bundles: an environment
variable names the project directory, the checkout has a default, and a
missing manifest is refused with the variable's name in the message
instead of a stack trace from whatever tried to write into it.

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

from rq_pipeline.paths import checkout
from rq_pipeline.project.files import read_json, write_json

PROJECT_ENV = "TRAINNR_PROJECT"
MANIFEST_FILE = "project.json"
INDEX_DIR = ".index"
INDEX_FILE = "project.json"
# The MCP job table (`rq_pipeline.mcp_jobs`) lives beside the kinds.
JOBS_DIR = "mcp-jobs"
PROJECTS_DIR_NAME = "projects"
DEFAULT_PROJECT = "default"
SCHEMA = "trainnr-project/1"

# The one folder per artifact kind, each named once here and reached
# through `Project.folder`. The names are the kinds' plural forms so a
# directory listing reads as an inventory.
ROBOTS_FOLDER = "robots"
RECORDINGS_FOLDER = "recordings"
FITS_FOLDER = "fits"
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
        from rq_pipeline.bundles.locate import add_search_root  # noqa: PLC0415

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


def projects_dir() -> Path:
    """The checkout's `projects/` — the default home."""
    return checkout() / PROJECTS_DIR_NAME


def current_project() -> Project:
    """`$TRAINNR_PROJECT`, else `projects/default`; refused by name when
    the directory has no manifest."""
    override = os.environ.get(PROJECT_ENV, "").strip()
    # absolute: the Studio is spawned with the checkout as its working
    # directory, so a relative override would point it somewhere else
    # (measured 2026-09-23: a launch with `../projects/x` never heartbeat);
    # links are kept as typed (macOS's /var is one) so paths compare as given
    root = (
        Path(override).expanduser().absolute()
        if override
        else projects_dir() / DEFAULT_PROJECT
    )
    project = Project(root)
    if not project.manifest_path.is_file():
        raise FileNotFoundError(
            f"no project at {root} (no {MANIFEST_FILE}) — set {PROJECT_ENV} to a "
            f"project directory, or create one with `create_project`"
        )
    project.use()
    return project


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
    """Every project directory under `projects/` (or `root`), by name."""
    home = Path(root) if root is not None else projects_dir()
    if not home.is_dir():
        return []
    return sorted(
        (Project(p) for p in home.iterdir() if (p / MANIFEST_FILE).is_file()),
        key=lambda p: p.root.name,
    )


NAME_FORBIDDEN = "/\\@"


def plain_name(name: str, what: str = "name") -> str:
    """An artifact's name as a folder: one plain word, no separators, no
    path tricks. Returns it; refuses anything else by name."""
    if (
        not name
        or name != name.strip()
        or any(c in name for c in NAME_FORBIDDEN)
        or name in (".", "..")
        or name.startswith(".")
    ):
        raise ValueError(f"{what}: one plain word, no separators; got {name!r}")
    return name


def _loop_or_refuse(loop: str) -> str:
    if loop and loop not in LOOPS:
        raise ValueError(f"loop is one of {', '.join(LOOPS)} (or unset), got {loop!r}")
    return loop
