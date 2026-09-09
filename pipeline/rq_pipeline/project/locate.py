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

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ENV = "TRAINNR_PROJECT"
MANIFEST_FILE = "project.json"
INDEX_DIR = ".index"
INDEX_FILE = "project.json"
# The MCP job table (`rq_pipeline.mcp_jobs`) lives beside the kinds.
JOBS_DIR = "mcp-jobs"
PROJECTS_DIR_NAME = "projects"
DEFAULT_PROJECT = "default"
SCHEMA = "trainnr-project/1"

# The one folder per artifact kind. The names are the kinds' plural forms
# so a directory listing reads as an inventory.
FOLDERS = (
    "robots",
    "recordings",
    "fits",
    "tasks",
    "batches",
    "datasets",
    "runs",
    "policies",
    "certificates",
    "deploy",
    "findings",
)

_CHECKOUT = Path(__file__).resolve().parents[3]


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
        raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
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

        add_search_root(self.folder("robots"))
        return self

    def folder(self, kind_folder: str) -> Path:
        if kind_folder not in FOLDERS:
            raise KeyError(f"no folder {kind_folder!r} in a project; one of {FOLDERS}")
        return self.root / kind_folder

    @property
    def robots(self) -> Path:
        return self.root / "robots"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def findings(self) -> Path:
        return self.root / "findings"

    @property
    def index_path(self) -> Path:
        return self.root / INDEX_DIR / INDEX_FILE


def projects_dir() -> Path:
    """The checkout's `projects/` — the default home."""
    return _CHECKOUT / PROJECTS_DIR_NAME


def current_project() -> Project:
    """`$TRAINNR_PROJECT`, else `projects/default`; refused by name when
    the directory has no manifest."""
    override = os.environ.get(PROJECT_ENV, "").strip()
    root = Path(override).expanduser() if override else projects_dir() / DEFAULT_PROJECT
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
    manifest_path.write_text(json.dumps(asdict(manifest), indent=1) + "\n")
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
