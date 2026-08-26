"""The suite is a regular package on purpose.

Tests import shared fixtures as `tests.<module>` (test_envs borrows
test_mujoco_backend's pendulum). Without this file `tests` is a
namespace package, and a dependency that ships its own top-level
`tests` package (draccus 0.11.6, in the train venv) wins the name:
`ModuleNotFoundError: No module named 'tests.test_mujoco_backend'`
under `unittest discover` from `.venv-train`, green from `.venv`
(2026-08-26). A regular package on sys.path[0] is found first.
"""
