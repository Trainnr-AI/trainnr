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
import os
import unittest


def installed(*modules: str) -> bool:
    """True when every module resolves — list a parent before its
    submodule, `find_spec` of a dotted name imports the parent."""
    return all(importlib.util.find_spec(name) is not None for name in modules)


SIM = installed("mujoco")
ENVS = installed("mujoco", "gymnasium")
MJX = installed("mujoco", "mujoco.mjx")
# The MJX tests compile their scenes through XLA. On an accelerator
# (the WSL box's CUDA jax — detected by the plugin package, no jax
# import needed) that is seconds; on CPU jax the microduck bundle's
# mesh-heavy scene measured 10+ MINUTES on the Mac (2026-09-01, the
# commit gate's mystery hang). The instrument doctrine applies: mjx
# tests run where the mjx instrument lives, or on explicit opt-in.
MJX_ACCELERATED = installed("jax_cuda12_plugin") or installed("jax_plugins")
MJX_CPU_OPT_IN = os.environ.get("RQ_MJX_CPU_TESTS") == "1"
NUMPY = installed("numpy")
TRAIN = installed("lerobot")
REMOTE = installed("openpi_client", "mujoco")
SCENE = installed("open3d")
SCENE_LINE = "scene extra not installed (uv sync --extra scene)"
needs_scene = unittest.skipUnless(SCENE, SCENE_LINE)

SIM_LINE = "sim extra not installed (uv sync --extra sim)"
TRAIN_LINE = "train extra not installed (use .venv-train)"
needs_sim = unittest.skipUnless(SIM, SIM_LINE)
needs_envs = unittest.skipUnless(ENVS, SIM_LINE)
needs_numpy = unittest.skipUnless(NUMPY, SIM_LINE)
needs_mjx = unittest.skipUnless(
    MJX and (MJX_ACCELERATED or MJX_CPU_OPT_IN),
    "mjx tests need the mjx extra AND an accelerator (CPU-jax XLA compiles "
    "measured 10+ min on the mesh-heavy scenes, 2026-09-01) — opt in with "
    "RQ_MJX_CPU_TESTS=1",
)
needs_train = unittest.skipUnless(TRAIN, TRAIN_LINE)
needs_train_sim = unittest.skipUnless(TRAIN and SIM, TRAIN_LINE)
needs_remote = unittest.skipUnless(REMOTE, "needs the remote and sim extras")
MCP = installed("mcp")
needs_mcp = unittest.skipUnless(MCP, "mcp extra not installed (uv sync --extra mcp)")
