"""Drift (docs/76 §9.1): the judgment rules on constructed records, then
the whole path on the rig's committed sweeps — synthetic wear injected
into a copy of a recording must name that wheel's gear and no other,
and the unmodified recording must come back within."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
import unittest
from pathlib import Path

from tests._fixtures import rig_project
from trainnr.fleet.drift import (
    ANCHORED,
    DRIFT_FILE,
    DRIFT_SCHEMA,
    FRESH_NOT_FINITE,
    FRESH_NOT_PINNED,
    LEFT,
    MISSING_IN_FRESH,
    NO_PINNED_REFERENCE,
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
from trainnr.project import PROJECT_ENV, index_project
from trainnr.project.ingest import ingest
from trainnr.project.kinds import Kind
from trainnr.robot.fit_record import FitRecord
from trainnr.robot.identify import IdentificationResult, IdentifiedParameter

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


class LikeWithLike(unittest.TestCase):
    """Our robot's drift is judged against fits of our robot: a public log
    of another lab's Go2 is left out and named (2026-09-25)."""

    def test_only_fits_of_the_recording_s_basis_are_the_reference(self) -> None:
        from dataclasses import replace  # noqa: PLC0415

        from trainnr.fleet.drift import same_basis  # noqa: PLC0415

        ours = replace(_record(("a", 1.0, 0.1, True), when="ours"), basis="own robot")
        public = replace(_record(("a", 5.0, 0.1, True), when="iit"), basis="public log")
        legacy = _record(("a", 1.05, 0.1, True), when="legacy")  # no basis field
        kept, left_out = same_basis((ours, public, legacy), "own robot")
        self.assertEqual(
            [r.recording for r in kept], [ours.recording, legacy.recording]
        )
        self.assertEqual(left_out, (f"{public.recording} (public log)",))
        # the union no longer stretches to the other lab's 5.0
        low, high = reference_intervals(kept)["a"]
        self.assertAlmostEqual(low, 0.9)
        self.assertAlmostEqual(high, 1.15)

    def test_no_fit_of_the_basis_is_refused_by_name(self) -> None:
        from dataclasses import replace  # noqa: PLC0415

        from trainnr.fleet.drift import judge  # noqa: PLC0415
        from trainnr.robot.fit_record import FITS_DIRECTORY  # noqa: PLC0415

        public = replace(_record(("a", 5.0, 0.1, True), when="iit"), basis="public log")
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "b"
            (bundle / FITS_DIRECTORY).mkdir(parents=True)
            (bundle / FITS_DIRECTORY / f"{public.recording}.json").write_text(
                public.to_json()
            )
            with self.assertRaisesRegex(ValueError, r"own robot.*public log"):
                judge(
                    bundle, Path(tmp), None, robot="r", recording="x", basis="own robot"
                )  # type: ignore[arg-type]


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
        # A parameter the reference never pinned cannot be judged.
        self.assertEqual(judged["new"].verdict, UNRESOLVED)
        self.assertEqual(judged["new"].note, NO_PINNED_REFERENCE)
        self.assertIsNone(judged["new"].shift)
        self.assertIsNone(judged["new"].reference_lower)
        left = judge_parameters(records, _fresh(("gear", 1.3, 0.05, True)))[0]
        self.assertEqual(left.verdict, LEFT)
        self.assertGreater(left.shift or 0.0, 1.0)
        wide = judge_parameters(records, _fresh(("gear", 1.3, 0.5, False)))[0]
        self.assertEqual(wide.verdict, UNRESOLVED, "not pinned: no verdict")
        self.assertEqual(wide.note, FRESH_NOT_PINNED)

    def test_the_reference_is_built_from_pinned_records_only(self) -> None:
        """One record that never pinned a parameter (an unbounded interval)
        must not make every later check `within`."""
        unbounded = _record(("gear", 1.0, math.inf, False))
        self.assertEqual(reference_intervals((unbounded,)), {})
        far = judge_parameters((unbounded,), _fresh(("gear", 99.0, 0.01, True)))[0]
        self.assertEqual(far.verdict, UNRESOLVED)
        self.assertEqual(far.note, NO_PINNED_REFERENCE)
        pinned = _record(("gear", 1.0, 0.1, True))
        far = judge_parameters((unbounded, pinned), _fresh(("gear", 99.0, 0.01, True)))[
            0
        ]
        self.assertEqual(far.verdict, LEFT)

    def test_a_diverged_fit_and_a_missing_parameter_are_unresolved(self) -> None:
        records = (_record(("gear", 1.0, 0.1, True), ("fric", 0.5, 0.05, True)),)
        judged = {
            p.name: p
            for p in judge_parameters(records, _fresh(("gear", math.nan, 0.1, True)))
        }
        self.assertEqual(judged["gear"].verdict, UNRESOLVED)
        self.assertEqual(judged["gear"].note, FRESH_NOT_FINITE)
        self.assertIsNone(judged["gear"].fresh_estimate)
        # The reference names fric; the fresh fit did not return it.
        self.assertEqual(judged["fric"].verdict, UNRESOLVED)
        self.assertEqual(judged["fric"].note, MISSING_IN_FRESH)
        self.assertIsNone(judged["fric"].fresh_lower)
        self.assertEqual(judged["fric"].reference_lower, 0.45)

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
            self.assertEqual(DriftRecord.read(path), record, "one reader, not two")
            self.assertEqual(back.schema, DRIFT_SCHEMA)
            self.assertEqual(back.references, 1)
            # Strict JSON on disk: a strict parser reads it, no NaN, no Infinity.
            strict = json.loads(
                path.read_text(),
                parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)),
            )
            self.assertEqual(strict["parameters"][0]["reference_lower"], 0.9)
            # A row with no reference writes null, never NaN.
            unknown = judge_parameters((), _fresh(("x", 1.0, 0.1, True)))
            loose = DriftRecord(**{**record.__dict__, "parameters": unknown})
            path2 = loose.write(Path(tmp) / "b" / DRIFT_FILE)
            self.assertIn("null", path2.read_text())
            self.assertNotIn("NaN", path2.read_text())
            self.assertIsNone(load_drift_record(path2).parameters[0].reference_lower)
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
    return rig_project(tmp)


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
        from trainnr.mcp_server import check_drift, identify_system  # noqa: PLC0415

        robot = index_project(self.project).by_kind(Kind.ROBOT)[0].stamp
        rec_b = ingest(self.project, SWEEP_B, name="sweep-b")
        # No fit record yet: refused by name.
        early = check_drift(robot, rec_b["stamp"])
        self.assertEqual(early["status"], "refused")
        self.assertIn("identify", early["reason"])
        robot = identify_system(robot, rec_b["stamp"])["robot"]
        rec_c = ingest(self.project, SWEEP_C, name="sweep-c")
        return identify_system(robot, rec_c["stamp"])["robot"]

    def test_the_fits_own_recording_is_refused(self) -> None:
        """Judged against the fit made from it, a recording shifts by zero:
        "within" proves nothing (an agent ran it, 2026-10-04)."""
        from trainnr.mcp_server import check_drift  # noqa: PLC0415

        robot = self._identified()
        source = next(
            a.stamp
            for a in index_project(self.project).by_kind(Kind.RECORDING)
            if a.stamp.startswith("sweep-b@")
        )
        out = check_drift(robot, source)
        self.assertEqual(out["status"], "refused", out)
        self.assertIn("fresh telemetry", out["reason"])
        nothing = ingest(
            self.project, _worn(SWEEP_B, Path(self.tmp.name) / "w.wire", WEAR), name="w"
        )
        out = check_drift(robot, nothing["stamp"], against="no-such-deployment")
        self.assertEqual(out["status"], "refused", out)
        self.assertIn("names no deployment", out["reason"])

    def test_synthetic_wear_is_named_and_nothing_else(self) -> None:
        from trainnr.mcp_server import check_drift  # noqa: PLC0415
        from trainnr.project.details import _drift  # noqa: PLC0415
        from trainnr.project.previews import _render_drift  # noqa: PLC0415

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
        self.assertIn("left_gear_per_damp", card.summary["out of interval"])
        self.assertEqual(card.cites["robot"], robot)
        self.assertEqual(card.cites["recording"], worn["stamp"])
        sections = _drift(project, record_path.parent, card)
        self.assertEqual(sections[0]["title"], "Drift check")
        rows = {r[0]: r[1] for r in sections[1]["rows"]}
        self.assertEqual(rows["left_gear_per_damp"], "out of interval")
        tile = tmp / "tile.png"
        self.assertTrue(_render_drift(project, record_path.parent, tile, {}))
        self.assertTrue(tile.is_file())
        # A second check under the same name is refused.
        again = check_drift(robot, worn["stamp"])
        self.assertEqual(again["status"], "refused")
        self.assertIn("never overwritten", again["reason"])

    def test_the_unworn_recording_comes_back_within(self) -> None:
        from trainnr.mcp_server import check_drift  # noqa: PLC0415

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
