"""The suite is a regular package on purpose.

Tests import shared fixtures as `tests.<module>` (test_envs borrows
test_mujoco_backend's pendulum). Without this file `tests` is a
namespace package, and a dependency that ships its own top-level
`tests` package (draccus 0.11.6, in the train venv) wins the name:
`ModuleNotFoundError: No module named 'tests.test_mujoco_backend'`
under `unittest discover` from `.venv-train`, green from `.venv`
(2026-08-26). A regular package on sys.path[0] is found first.
"""

import os
import tempfile

# The suite's own projects home. A test that creates or opens a project
# makes it the current one (`remember_project` writes `.current` in the
# projects home), and without this that was the developer's own: running
# the suite left ~/trainnr/projects/.current naming a deleted temporary
# folder (2026-10-30). Tests that check the default home set
# TRAINNR_HOME and drop TRAINNR_PROJECTS themselves.
_PROJECTS = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
os.environ["TRAINNR_PROJECTS"] = _PROJECTS.name
