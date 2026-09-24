"""The layer chain, pinned: a module may import only from its tier or below.

    stats <- bundles, protocol, robot.model_checks <- evaluate
          <- physics, tasks, collect, robot (sysid) <- envs <- tools

Written down once (docs/22 §3, docs/32) and enforced here over every
import in the package — top-level AND lazy — because the one violation
that slipped in before this test existed hid behind a lazy import.
"""

import ast
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "rq_pipeline"
TIER = {
    "stats": 0,
    "bundles": 1,
    "protocol": 1,
    "plugins": 1,
    "cloud": 1,  # rented machines: stdlib HTTP over the plugin door, nothing above
    "robot": 3,  # sysid fits over recordings; the census gate is placed below
    "evaluate": 2,
    "rl": 2,  # SmoothRL's pure and torch halves (docs/71); nothing above stats
    # the sim-to-sim gate: plain MuJoCo + ONNX, judged by evaluate's rule; since
    # E2 it walks on captured scenes (the contact-site gap, the mirror's splat and
    # cameras), so it sits beside scenes, which stages deployments in return
    "deploy": 3,
    "fleet": 3,  # drift: the identifier's method over a recording, judged
    "scenes": 3,  # captured scenes: numpy splats, the proxy audit; robot's code version
    "physics": 3,
    "tasks": 3,
    "collect": 3,
    "envs": 4,
    "project": 4,  # the index over every kind: imports their marker constants
    "robots": 4,  # the ingest seam: reads collect's wire, writes project artifacts
}
# Modules placed below their package: stdlib gates the lower tiers import.
# And above it: the legged fit and the two public-log readers consume the
# ingest seam's `Recording` (2026-09-24), so they sit with `robots`; the
# method registry finds the fit by name at run time, not by import.
MODULE_TIER = {
    "rq_pipeline.robot.model_checks": 1,
    "rq_pipeline.robot.legged_fit": 4,
    "rq_pipeline.robot.quadruped_synth": 4,
    "rq_pipeline.robot.rosbag_sqlite": 4,
    "rq_pipeline.robot.pt_dict": 4,
}


def _package_of(module: str) -> str:
    parts = module.split(".")
    return parts[1] if len(parts) > 1 and parts[0] == "rq_pipeline" else ""


def _tier_of(module: str) -> int | None:
    if module in MODULE_TIER:
        return MODULE_TIER[module]
    return TIER.get(_package_of(module))


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return {name for name in found if name.startswith("rq_pipeline")}


class LayerChain(unittest.TestCase):
    def test_no_module_imports_above_its_tier(self) -> None:
        violations = []
        for path in sorted(PACKAGE.rglob("*.py")):
            module = ".".join(path.relative_to(PACKAGE.parent).with_suffix("").parts)
            own = _tier_of(module)
            if own is None:
                continue
            for imported in _imports(path):
                other = _tier_of(imported)
                if other is not None and other > own:
                    violations.append(f"{module} -> {imported}")
        self.assertEqual(violations, [], "\n".join(violations))

    def test_every_package_is_placed(self) -> None:
        packages = {
            p.name
            for p in PACKAGE.iterdir()
            if p.is_dir() and not p.name.startswith("_")
        }
        self.assertEqual(packages - set(TIER), set(), "unplaced packages")


if __name__ == "__main__":
    unittest.main()
