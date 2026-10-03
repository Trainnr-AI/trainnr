"""A project: one directory where everything about one effort lives.

Until 2026-09-09 the pipeline's artifacts were scattered by type — robot
bundles at the checkout's top level, the chain's runs under `trainnr/runs`,
findings in the docs tree — and nothing gathered one effort into one
thing an agent could point at or a window could open. A project is that
thing (docs/76): a self-contained directory with a manifest, one folder
per artifact kind, and a rebuildable index the Studio reads.

Three modules, in the order a caller meets them:

- `locate`: which project — `$TRAINNR_PROJECT`, else the checkout's
  `projects/<default>`; the roots every tool writes into derive from it.
- `kinds`: what an artifact IS, decided by the marker file at its root,
  and stamped through the one hashing door — never guessed.
- `index`: walk a project, stamp every artifact, read its manifest for
  lineage, decide where the effort stands in the loop, and write it all
  to `.index/project.json`. The files are the truth; the index is a
  cache and deleting it costs nothing.
"""

from trainnr.project.index import ProjectIndex, index_project, write_index
from trainnr.project.kinds import Kind, detect, stamp_kind
from trainnr.project.locate import (
    PROJECT_ENV,
    PROJECTS_ENV,
    Project,
    create_project,
    current_project,
    list_projects,
    projects_home,
    remember_project,
    use_project,
)
from trainnr.project.task_ref import write_task_reference

__all__ = [
    "PROJECTS_ENV",
    "PROJECT_ENV",
    "Kind",
    "Project",
    "ProjectIndex",
    "create_project",
    "current_project",
    "detect",
    "index_project",
    "list_projects",
    "projects_home",
    "remember_project",
    "stamp_kind",
    "use_project",
    "write_index",
    "write_task_reference",
]
