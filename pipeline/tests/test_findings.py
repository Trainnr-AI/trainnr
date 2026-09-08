"""Findings as records: round-trip, the ledger's provenance lines, the
dirty-tree mark — and the study spec's refusals."""

from __future__ import annotations

import csv
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

    def test_a_tree_without_git_reads_its_sync_stamp_or_refuses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "rq-commit"):
                repo_commit(Path(tmp))
            (Path(tmp) / ".rq-commit").write_text("a81974b\n")
            self.assertEqual(repo_commit(Path(tmp)), "a81974b-archive")

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


class TheFigure(unittest.TestCase):
    """A figure is a function of the record: the arms' intervals, the
    provenance on the canvas, the numbers in a CSV beside it."""

    def _curve(self) -> Finding:
        arms = {}
        for n, k in ((8, 10), (16, 30), (32, 50), (64, 62), (128, 68)):
            arms[f"n{n}"] = {
                "successes": k,
                "trials": 80,
                "ci95": [k / 80 - 0.1, k / 80 + 0.1],
                "episodes": n,
            }
        return _finding(id="demo-count-lift-2026-09-04", outcome={"arms": arms})

    def test_a_curve_and_a_categorical_row_render_with_provenance(self) -> None:
        try:
            import matplotlib  # noqa: F401, PLC0415
        except ImportError:
            self.skipTest("matplotlib (train extra)")
        from rq_pipeline.evaluate.figures import (  # noqa: PLC0415
            arm_rows,
            is_curve,
            render,
        )

        curve = self._curve()
        self.assertTrue(is_curve(arm_rows(curve)))
        pair = _finding(
            id="visual-dr-lift-2026-09-04",
            outcome={
                "arms": {
                    "fixed": {
                        "successes": 40,
                        "trials": 80,
                        "ci95": [0.39, 0.61],
                        "episodes": 64,
                    },
                    "visual": {
                        "successes": 55,
                        "trials": 80,
                        "ci95": [0.57, 0.79],
                        "episodes": 64,
                    },
                }
            },
        )
        self.assertFalse(is_curve(arm_rows(pair)))
        with tempfile.TemporaryDirectory() as tmp:
            for record in (curve, pair):
                written = render(record, Path(tmp))
                for fmt in ("svg", "pdf", "png", "csv"):
                    self.assertTrue((Path(tmp) / written[fmt]).exists(), fmt)
                svg = (Path(tmp) / written["svg"]).read_text()
                # The canvas is clean since 2026-09-08 (the paper's figures carry no
        # repository footer); the provenance line still exists for the tools
        # that print it, and names the commit.
        self.assertNotIn("abc1234", svg)
        self.assertIn("abc1234", fx.footer(finding))
                self.assertIn("mujoco-3.11.0+x86_64", svg)
                rows = list(csv.DictReader((Path(tmp) / written["csv"]).open()))
                self.assertEqual([r["arm"] for r in rows], list(record.outcome["arms"]))


class TheReplicates(unittest.TestCase):
    """A spec's `replicates` expands each declared arm into N pressed
    arms; comparisons pool them; the figure groups them."""

    def test_expand_and_declared_arm(self) -> None:
        study = _load_study()
        spec = {
            "replicates": 3,
            "arms": {"n8": {"episodes": 8}, "n16": {"episodes": 16}},
        }
        arms = study.expand_arms(spec)
        self.assertEqual(
            list(arms), ["n8#1", "n8#2", "n8#3", "n16#1", "n16#2", "n16#3"]
        )
        self.assertEqual(arms["n8#2"]["replicate_of"], "n8")
        self.assertEqual(study.declared_arm("n16#3"), "n16")
        self.assertEqual(study.declared_arm("fixed"), "fixed")
        self.assertEqual(study.expand_arms({"arms": {"a": {}}}), {"a": {}})

    def test_replicated_curve_renders_grouped(self) -> None:
        try:
            import matplotlib  # noqa: F401, PLC0415
        except ImportError:
            self.skipTest("matplotlib (train extra)")
        from rq_pipeline.evaluate.figures import (  # noqa: PLC0415
            arm_rows,
            is_curve,
            render,
        )

        arms = {}
        for n in (8, 32, 128):
            for r in (1, 2):
                k = 60 + n // 8 + r
                arms[f"n{n}#{r}"] = {
                    "successes": k,
                    "trials": 80,
                    "ci95": [k / 80 - 0.1, k / 80 + 0.1],
                    "episodes": n,
                    "replicate_of": f"n{n}",
                }
        record = _finding(
            id="demo-count-lift-cliff-rep-2026-09-04", outcome={"arms": arms}
        )
        rows = arm_rows(record)
        self.assertTrue(is_curve(rows))
        self.assertEqual({r["group"] for r in rows}, {"n8", "n32", "n128"})
        with tempfile.TemporaryDirectory() as tmp:
            written = render(record, Path(tmp))
            self.assertTrue((Path(tmp) / written["svg"]).exists())


class TheWalkFold(unittest.TestCase):
    """The walk C1 fold pools replicates per arm: run 1 at `<arm>`,
    replicates at `<arm>#k`, trials concatenated in replicate order."""

    def _walk_fold(self):
        if str(REPO / "tools") not in sys.path:
            sys.path.insert(0, str(REPO / "tools"))
        spec = importlib.util.spec_from_file_location(
            "walk_fold", REPO / "tools" / "walk-c1-fold.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _write_run(self, root: Path, name: str, successes: int, seed: int) -> None:
        verdict = root / name / "train" / "verdict"
        verdict.mkdir(parents=True)
        cert = {
            "successes": successes,
            "trials": 4,
            "ci95": [0.1, 0.9],
            "funnel": {"survived": 4, "tracked": successes},
            "median_err_ratio": 0.4,
            "identity": {"trained_dr_basis": "b", "seed": seed},
            "protocol": {"judged_at": "fit", "dr_basis": "b"},
            "instrument": "mjlab-test",
        }
        (verdict / "walk-verdict-at-fit-cuda.json").write_text(json.dumps(cert))
        rows = [
            {
                "source": "microduck-walk@0123456789ab",
                "policy": "p",
                "trial": t,
                "success": t < successes,
                "steps": 1,
                "instrument": "mjlab-test",
                "protocol": {},
                "seed": 1000,
                "events": [],
                "variations": {},
            }
            for t in range(4)
        ]
        (verdict / "records-at-fit-cuda.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n"
        )

    def test_replicates_pool_in_order_and_keep_per_run_counts(self) -> None:
        fold = self._walk_fold()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_run(root, "narrow", 3, 42)
            self._write_run(root, "narrow#2", 4, 43)
            self._write_run(root, "narrow#3", 2, 44)
            self._write_run(root, "wide", 1, 42)
            self.assertEqual(
                [p.name for p in fold.replicate_dirs(root, "narrow")],
                ["narrow", "narrow#2", "narrow#3"],
            )
            self.assertEqual(
                fold.outcomes(root, "narrow"),
                [True, True, True, False] + [True] * 4 + [True, True, False, False],
            )
            result = fold.fold(root, ("narrow", "wide"), alpha=0.05, delta=0.15)
            self.assertEqual(
                (
                    result["arms"]["narrow"]["successes"],
                    result["arms"]["narrow"]["trials"],
                ),
                (9, 12),
            )
            self.assertEqual(result["arms"]["narrow"]["per_run"], [3, 4, 2])
            self.assertEqual(
                [r["seed"] for r in result["replicates"]["narrow"]], [42, 43, 44]
            )
            self.assertEqual(result["arms"]["wide"]["runs"], 1)
            with self.assertRaisesRegex(FileNotFoundError, "point"):
                fold.replicate_dirs(root, "point")
