"""Drift (docs/76 §9.1): the judgment rules on constructed records, then
the whole path on the rig's committed sweeps — synthetic wear injected
into a copy of a recording must name that wheel's gear and no other,
and the unmodified recording must come back within."""

from __future__ import annotations

import math
import os
import re
import shutil
import tempfile
import unittest
from pathlib import Path

from rq_pipeline.fleet.drift import (
    ANCHORED,
    DRIFT_FILE,
    DRIFT_SCHEMA,
    LEFT,
    RECOMMEND_CLEAN,
    RECOMMEND_DRIFTED,
    UNRESOLVED,
    WITHIN,
    DriftRecord,
    judge_parameters,
    load_drift_record,
    recommendation_for,
    reference_intervals,
)
from rq_pipeline.project import PROJECT_ENV, create_project, index_project
from rq_pipeline.project.ingest import ingest
from rq_pipeline.project.kinds import Kind
from rq_pipeline.robot.fit_record import FitRecord
from rq_pipeline.robot.identify import IdentificationResult, IdentifiedParameter

REPO = Path(__file__).resolve().parents[2]
BUNDLE = REPO / "robots" / "rig-drivetrain"
SWEEP_B = REPO / "recordings" / "sweep-2026-08-24-b.wire"
SWEEP_C = REPO / "recordings" / "sweep-2026-08-24-c.wire"
# One wheel's encoder scaled: a worn or swapped gear as the encoder sees it.
WEAR = 1.4
LEFT_TICKS = re.compile(r"\bL=(-?\d+)")


def _record(*params: tuple[str, float, float, bool], when: str = "t") -> FitRecord:
    return FitRecord(
        robot="r",
        recording=f"rec-{when}@000000000000",
        anchor="damping FIXED",
        confidence=0.95,
        parameters=tuple(
            IdentifiedParameter(name, est, half, allowed_range=1.0, pinned=pinned)
            for name, est, half, pinned in params
        ),
        created_utc=when,
        units={"gear": "N*m per duty"},
    )


def _fresh(*params: tuple[str, float, float, bool]) -> IdentificationResult:
    return IdentificationResult(
        parameters=tuple(
            IdentifiedParameter(name, est, half, allowed_range=1.0, pinned=pinned)
            for name, est, half, pinned in params
        ),
        confidence=0.95,
    )


class TheRule(unittest.TestCase):
    def test_the_reference_is_the_union_across_records(self) -> None:
        ref = reference_intervals(
            (_record(("gear", 1.0, 0.1, True)), _record(("gear", 1.5, 0.1, True)))
        )
        self.assertEqual(ref["gear"], (0.9, 1.6))

    def test_within_left_unresolved_anchored(self) -> None:
        records = (_record(("gear", 1.0, 0.1, True), ("damp", 1.0, 0.0, False)),)
        judged = {
            p.name: p
            for p in judge_parameters(
                records,
                _fresh(
                    ("gear", 1.05, 0.02, True),
                    ("damp", 1.0, 0.0, False),
                    ("new", 3.0, 0.1, True),
                ),
                anchored=("damp",),
            )
        }
        self.assertEqual(judged["gear"].verdict, WITHIN)
        self.assertAlmostEqual(judged["gear"].shift or 0.0, 0.5)
        self.assertEqual(judged["gear"].unit, "N*m per duty")
        self.assertEqual(judged["damp"].verdict, ANCHORED)
        # A parameter the reference never identified cannot be judged.
        self.assertEqual(judged["new"].verdict, UNRESOLVED)
        self.assertIsNone(judged["new"].shift)
        self.assertTrue(math.isnan(judged["new"].reference_lower))
        left = judge_parameters(records, _fresh(("gear", 1.3, 0.05, True)))[0]
        self.assertEqual(left.verdict, LEFT)
        self.assertGreater(left.shift or 0.0, 1.0)
        wide = judge_parameters(records, _fresh(("gear", 1.3, 0.5, False)))[0]
        self.assertEqual(wide.verdict, UNRESOLVED, "not pinned: no verdict")

    def test_the_edge_touching_counts_as_within(self) -> None:
        records = (_record(("gear", 1.0, 0.1, True)),)
        touching = judge_parameters(records, _fresh(("gear", 1.2, 0.1, True)))[0]
        self.assertEqual(touching.verdict, WITHIN)

    def test_the_recommendation_names_the_case(self) -> None:
        self.assertEqual(recommendation_for((), ()), RECOMMEND_CLEAN)
        self.assertEqual(recommendation_for(("gear",), ("x",)), RECOMMEND_DRIFTED)
        self.assertIn("x, y", recommendation_for((), ("x", "y")))

    def test_a_record_round_trips_and_refuses_another_schema(self) -> None:
        judged = judge_parameters(
            (_record(("gear", 1.0, 0.1, True)),), _fresh(("gear", 1.3, 0.05, True))
        )
        record = DriftRecord(
            robot="r@1",
            recording="x@2",
            method="m",
            fit=("rec-t@000000000000",),
            references=1,
            parameters=judged,
            drifted=True,
            left=("gear",),
            unresolved=(),
            recommendation=RECOMMEND_DRIFTED,
            anchor="damping FIXED",
            created_utc="now",
            code="abc",
            instrument="mujoco-x",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = record.write(Path(tmp) / DRIFT_FILE)
            back = load_drift_record(path)
            self.assertEqual(back, record)
            self.assertEqual(back.schema, DRIFT_SCHEMA)
            path.write_text(path.read_text().replace(DRIFT_SCHEMA, "trainnr-drift/9"))
            with self.assertRaisesRegex(ValueError, "schema"):
                load_drift_record(path)


def _worn(source: Path, out: Path, scale: float) -> Path:
    """The recording with the left wheel's encoder ticks scaled."""
    lines = source.read_text().splitlines(keepends=True)
    out.write_text(
        "".join(
            LEFT_TICKS.sub(lambda m: f"L={round(int(m.group(1)) * scale)}", line)
            for line in lines
        )
    )
    return out


def _project_with_rig(tmp: Path):
    project = create_project(tmp / "p", "p")
    bundle = project.folder("robots") / "rig-drivetrain"
    bundle.mkdir(parents=True)
    for name in ("model.xml", "profile.json", "README.md"):
        shutil.copy2(BUNDLE / name, bundle / name)
    return project


@unittest.skipUnless(SWEEP_B.is_file() and SWEEP_C.is_file(), "the rig's sweeps")
class TheWholePath(unittest.TestCase):
    """Two real sweeps identify the rig; a copy with one wheel worn is
    checked and must name that wheel; the unworn one must come back within."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.project = _project_with_rig(Path(self.tmp.name))
        os.environ[PROJECT_ENV] = str(self.project.root)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(os.environ.pop, PROJECT_ENV, None)

    def _identified(self) -> str:
        """The rig identified from both sweeps; the robot's version after."""
        from rq_pipeline.mcp_server import check_drift, identify_system  # noqa: PLC0415

        robot = index_project(self.project).by_kind(Kind.ROBOT)[0].stamp
        rec_b = ingest(self.project, SWEEP_B, name="sweep-b")
        # No fit record yet: refused by name.
        early = check_drift(robot, rec_b["stamp"])
        self.assertEqual(early["status"], "refused")
        self.assertIn("identify", early["reason"])
        robot = identify_system(robot, rec_b["stamp"])["robot"]
        rec_c = ingest(self.project, SWEEP_C, name="sweep-c")
        return identify_system(robot, rec_c["stamp"])["robot"]

    def test_synthetic_wear_is_named_and_nothing_else(self) -> None:
        from rq_pipeline.mcp_server import check_drift  # noqa: PLC0415
        from rq_pipeline.project.details import _drift  # noqa: PLC0415
        from rq_pipeline.project.previews import _render_drift  # noqa: PLC0415

        project, tmp = self.project, Path(self.tmp.name)
        robot = self._identified()
        worn_file = _worn(SWEEP_B, tmp / "sweep-b-worn.wire", WEAR)
        worn = ingest(project, worn_file, name="sweep-b-worn")
        out = check_drift(robot, worn["stamp"])
        self.assertEqual(out["status"], "done", out)
        self.assertTrue(out["drifted"])
        self.assertIn("left_gear_per_damp", out["left"])
        verdicts = {p["name"]: p["verdict"] for p in out["parameters"]}
        self.assertEqual(verdicts["right_gear_per_damp"], WITHIN)
        self.assertEqual(verdicts["scale_ref_damping"], ANCHORED)
        self.assertEqual(out["references"], 2)
        self.assertEqual(out["recommendation"], RECOMMEND_DRIFTED)
        self.assertTrue(out["state"]["drift monitored"])
        self.assertTrue(out["check"].startswith("sweep-b-worn-check@"))
        record_path = project.root / out["record"]
        self.assertTrue(record_path.is_file())
        self.assertEqual(record_path.parent.parent.name, "monitoring")
        # The check wrote no fit record into the bundle.
        fits = project.folder("robots") / "rig-drivetrain" / "fits"
        self.assertEqual(
            sorted(p.name for p in fits.glob("*.json") if p.name != "SPREAD.json"),
            sorted(f"{r}.json" for r in load_drift_record(record_path).fit),
        )
        # The index and the Studio read it.
        index = index_project(project)
        card = next(a for a in index.artifacts if a.kind == "drift")
        self.assertEqual(card.summary["verdict"], "drifted")
        self.assertIn("left_gear_per_damp", card.summary["left"])
        self.assertEqual(card.cites["robot"], robot)
        self.assertEqual(card.cites["recording"], worn["stamp"])
        sections = _drift(project, record_path.parent, card)
        self.assertEqual(sections[0]["title"], "Drift check")
        rows = {r[0]: r[1] for r in sections[1]["rows"]}
        self.assertEqual(rows["left_gear_per_damp"], LEFT)
        tile = tmp / "tile.png"
        self.assertTrue(_render_drift(project, record_path.parent, tile, {}))
        self.assertTrue(tile.is_file())
        # A second check under the same name is refused.
        again = check_drift(robot, worn["stamp"])
        self.assertEqual(again["status"], "refused")
        self.assertIn("never overwritten", again["reason"])

    def test_the_unworn_recording_comes_back_within(self) -> None:
        from rq_pipeline.mcp_server import check_drift  # noqa: PLC0415

        project, tmp = self.project, Path(self.tmp.name)
        robot = self._identified()
        clean_file = tmp / "sweep-b-again.wire"
        shutil.copy2(SWEEP_B, clean_file)
        clean = ingest(project, clean_file, name="sweep-b-again")
        ok = check_drift(robot, clean["stamp"], name="clean-check")
        self.assertEqual(ok["status"], "done", ok)
        self.assertFalse(ok["drifted"])
        self.assertEqual(ok["left"], [])
        self.assertEqual(
            {p["verdict"] for p in ok["parameters"]} - {UNRESOLVED},
            {WITHIN, ANCHORED},
        )


if __name__ == "__main__":
    unittest.main()
