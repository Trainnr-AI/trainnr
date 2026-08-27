"""Which optional extras THIS interpreter has — the suite's skip guards,
named once.

The two venvs are two instruments (tests/_instruments): the locked sim
venv has mujoco and no lerobot, the train venv has both, and the mjx and
remote extras are opt-in on either. A test class or method wears the
decorator for the extra it needs and skips with the install line, so a
green run on a venv without the extra says "skipped", never "passed".
"""

from __future__ import annotations

import importlib.util
import unittest


def installed(*modules: str) -> bool:
    """True when every module resolves — list a parent before its
    submodule, `find_spec` of a dotted name imports the parent."""
    return all(importlib.util.find_spec(name) is not None for name in modules)


SIM = installed("mujoco")
ENVS = installed("mujoco", "gymnasium")
MJX = installed("mujoco", "mujoco.mjx")
NUMPY = installed("numpy")
TRAIN = installed("lerobot")
REMOTE = installed("openpi_client", "mujoco")

SIM_LINE = "sim extra not installed (uv sync --extra sim)"
TRAIN_LINE = "train extra not installed (use .venv-train)"
needs_sim = unittest.skipUnless(SIM, SIM_LINE)
needs_envs = unittest.skipUnless(ENVS, SIM_LINE)
needs_numpy = unittest.skipUnless(NUMPY, SIM_LINE)
needs_mjx = unittest.skipUnless(MJX, "mjx extra not installed (uv sync --extra mjx)")
needs_train = unittest.skipUnless(TRAIN, TRAIN_LINE)
needs_train_sim = unittest.skipUnless(TRAIN and SIM, TRAIN_LINE)
needs_remote = unittest.skipUnless(REMOTE, "needs the remote and sim extras")
