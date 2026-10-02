#!/usr/bin/env python3
"""The package layers, pinned (docs/80 §4): a lower layer never imports a higher one.

    L0  trainnr          the loop's library, the doors, the CLI
    L1  trainnr-mjlab    the mjlab trainer; imports L0
    L2  trainnr-desktop  the app; talks to L0 by files and processes only

Walks every Python file of L0 and L1 (top-level AND lazy imports, as the
in-package layer test does) and fails on an import that points up: L0
importing `trainnr_mjlab` (or its old name), either package importing
the desktop crate's name. After Rerun's crate-layer check; stdlib only,
so it runs anywhere.

    python3 tools/check-layers.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# layer -> (its source tree, the module names it must never import)
LAYERS: dict[str, tuple[Path, frozenset[str]]] = {
    "L0 trainnr": (
        ROOT / "trainnr" / "trainnr",
        frozenset({"trainnr_mjlab", "rq_mjlab", "trainnr_desktop", "mjlab"}),
    ),
    "L1 trainnr-mjlab": (
        ROOT / "trainnr-mjlab" / "src" / "trainnr_mjlab",
        frozenset({"trainnr_desktop"}),
    ),
}


def imported_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def main() -> int:
    problems: list[str] = []
    for layer, (tree_root, forbidden) in LAYERS.items():
        for path in sorted(tree_root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            hits = sorted(imported_roots(tree) & forbidden)
            if hits:
                rel = path.relative_to(ROOT)
                problems.append(f"{layer}: {rel} imports {', '.join(hits)}")
    for line in problems:
        print(line)
    print(f"layer check: {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
