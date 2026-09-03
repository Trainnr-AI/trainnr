"""Findings as records: round-trip, the ledger's provenance lines, the
dirty-tree mark — and the study spec's refusals."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rq_pipeline.evaluate.findings import (
    Finding,
    load_findings,
    render_ledger,
    repo_commit,
)

REPO = Path(__file__).resolve().parents[2]


def _finding(**over) -> Finding:
    base = dict(
        id="demo-count-lift-2026-09-04",
        claim="8 to 128 demos: success rises 10/80 to 60/80",
        date="2026-09-04",
        repo_commit="abc1234",
        argv=["study.py", "finding", "spec.json", "out"],
        instrument="mujoco-3.11.0+x86_64",
        outcome={"arms": {"n8": {"successes": 10, "trials": 80}}},
        inputs={"spec_hash": "deadbeef"},
        artifacts={"verdict": "runs/x/verdict.json"},
        sources={"mjlab": "1.6.0"},
        protocol="docs/e2e-research/62",
        caveats=("one task",),
    )
    base.update(over)
    return Finding(**base)


class TheRecord(unittest.TestCase):
    def test_round_trips_and_lists_newest_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _finding().write(_finding().path(root))
            older = _finding(id="older-2026-09-01", date="2026-09-01")
            older.write(older.path(root))
            found = load_findings(root)
        self.assertEqual(
            [f.id for f in found], ["demo-count-lift-2026-09-04", "older-2026-09-01"]
        )
        self.assertEqual(found[0].sources, {"mjlab": "1.6.0"})
        self.assertEqual(found[0].caveats, ("one task",))

    def test_the_ledger_states_every_provenance_field(self) -> None:
        page = render_ledger([_finding()])
        for needle in (
            "abc1234",
            "mujoco-3.11.0+x86_64",
            "study.py finding",
            "deadbeef",
            "mjlab: 1.6.0",
            "one task",
            "docs/e2e-research/62",
        ):
            self.assertIn(needle, page)

    def test_a_dirty_tree_is_marked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["git", "init", "-q", tmp], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    tmp,
                    "-c",
                    "user.email=t@t",
                    "-c",
                    "user.name=t",
                    "commit",
                    "-q",
                    "--allow-empty",
                    "-m",
                    "x",
                ],
                check=True,
            )
            self.assertNotIn("dirty", repo_commit(Path(tmp)))
            (Path(tmp) / "f").write_text("x")
            self.assertTrue(repo_commit(Path(tmp)).endswith("-dirty"))


def _load_study():
    if str(REPO / "tools") not in sys.path:
        sys.path.insert(0, str(REPO / "tools"))
    spec = importlib.util.spec_from_file_location(
        "study_tool", REPO / "tools" / "study.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TheSpec(unittest.TestCase):
    def test_every_committed_study_spec_loads(self) -> None:
        study = _load_study()
        specs = sorted((REPO / "docs" / "studies").glob("*.json"))
        self.assertTrue(specs)
        for path in specs:
            spec = study.load_spec(path)
            self.assertTrue(spec["arms"])

    def test_visuals_without_a_basis_are_refused(self) -> None:
        study = _load_study()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.json"
            path.write_text(
                json.dumps(
                    {
                        "id": "x",
                        "claim": "c",
                        "truth": {},
                        "train": {},
                        "eval": {},
                        "arms": {
                            "a": {
                                "episodes": 1,
                                "dr": {},
                                "basis": "b",
                                "visuals": ["headlight.diffuse_scale=0.5:1.5"],
                            }
                        },
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "basis"):
                study.load_spec(path)


if __name__ == "__main__":
    unittest.main()
