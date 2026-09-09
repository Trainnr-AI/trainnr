"""System identification as a door (docs/76 §6): a method registry with
the drivetrain ratio fit built in; ingest keeps the raw file; the door
fits a robot from a recording in a project, writes the record, and the
loop's 'system identified' state is proved by it; refusals by name."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from rq_pipeline.project import PROJECT_ENV, Kind, create_project, index_project
from rq_pipeline.robot.methods import detect, methods, raw_files, resolve
from rq_pipeline.robots.ingest import ingest
from tests._extras import needs_sim

REPO = Path(__file__).resolve().parents[2]
BUNDLE = REPO / "robots" / "rig-drivetrain"
SWEEP = REPO / "recordings" / "sweep-2026-08-24-b.wire"
SWEEP_C = REPO / "recordings" / "sweep-2026-08-24-c.wire"


def _project_with_rig(tmp: Path):
    project = create_project(tmp / "p", "p")
    bundle = project.folder("robots") / "rig-drivetrain"
    bundle.mkdir(parents=True)
    for name in ("model.xml", "profile.json", "README.md"):
        shutil.copy2(BUNDLE / name, bundle / name)
    return project


class Registry(unittest.TestCase):
    def test_the_drivetrain_method_is_registered_and_refuses_by_name(self) -> None:
        self.assertIn("drivetrain-ratio", methods())
        self.assertEqual(resolve("drivetrain-ratio").name, "drivetrain-ratio")
        with self.assertRaisesRegex(KeyError, "no identification method"):
            resolve("teleport")
        with tempfile.TemporaryDirectory() as tmp:
            project = _project_with_rig(Path(tmp))
            bundle = project.folder("robots") / "rig-drivetrain"
            empty = project.folder("recordings") / "nothing"
            empty.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "no raw .wire"):
                detect(bundle, empty)
            not_a_bundle = project.folder("robots") / "bare"
            not_a_bundle.mkdir()
            with self.assertRaisesRegex(ValueError, "profile"):
                detect(not_a_bundle, empty)


@needs_sim
class Door(unittest.TestCase):
    def test_ingest_keeps_the_raw_file_and_the_door_writes_a_record(self) -> None:
        from rq_pipeline.mcp_server import (  # noqa: PLC0415
            describe_identification,
            identify_system,
        )

        with tempfile.TemporaryDirectory() as tmp:
            project = _project_with_rig(Path(tmp))
            os.environ[PROJECT_ENV] = str(project.root)
            try:
                rec = ingest(project, SWEEP, name="sweep-b")
                self.assertEqual(rec["raw"], "raw/sweep-2026-08-24-b.wire")
                self.assertTrue(raw_files(project.root / rec["path"], ".wire"))
                index = index_project(project)
                robot = index.by_kind(Kind.ROBOT)[0].stamp
                before = next(s for s in index.states if s.name == "system identified")
                self.assertFalse(before.present)
                answer = identify_system(robot, rec["stamp"])
                self.assertEqual(answer["method"], "drivetrain-ratio")
                self.assertTrue(
                    answer["record"].startswith("robots/rig-drivetrain/fits/")
                )
                self.assertIn("RATIO FIT", answer["anchor"])
                names = {p["name"] for p in answer["parameters"]}
                self.assertIn("left_gear_per_damp", names)
                self.assertTrue(any(p["identified"] for p in answer["parameters"]))
                self.assertTrue(answer["state"]["system identified"])
                self.assertIsNone(answer["spread"], "one record: no spread yet")
                # The record is inside the bundle: the robot's version moved.
                self.assertNotEqual(answer["robot"], robot)
                self.assertEqual(answer["robot_before"], robot)
                robot = answer["robot"]
                described = describe_identification(robot)
                self.assertEqual(len(described["records"]), 1)
                # A second recording: the spread verdict appears.
                rec2 = ingest(project, SWEEP_C, name="sweep-c")
                again = identify_system(robot, rec2["stamp"])
                self.assertEqual(again["records"], 2)
                self.assertIsNotNone(again["spread"])
                self.assertIn("verdict", again["spread"]["left_gear_per_damp"])
                robot = again["robot"]
                self.assertIsNotNone(describe_identification(robot)["spread"])
                self.assertEqual(
                    described["records"][0]["recording"].split("@")[0],
                    "sweep-2026-08-24-b",
                )
                with self.assertRaisesRegex(KeyError, "no artifact"):
                    identify_system("nobody@000000000000", rec["stamp"])
                with self.assertRaisesRegex(ValueError, "not a recording"):
                    identify_system(robot, robot)
            finally:
                os.environ.pop(PROJECT_ENV, None)


if __name__ == "__main__":
    unittest.main()
