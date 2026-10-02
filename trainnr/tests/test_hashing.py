"""Bundle identity: bytes and names in, location out."""

import tempfile
import unittest
from pathlib import Path

from trainnr.bundles.hashing import bundle_hash, stamp


class BundleHash(unittest.TestCase):
    def _make_bundle(self, root: Path) -> None:
        (root / "model").mkdir()
        (root / "model" / "robot.xml").write_text("<mujoco/>", encoding="utf-8")
        (root / "calibration.json").write_text('{"shoulder": 12}', encoding="utf-8")

    def test_stable_across_locations(self) -> None:
        with (
            tempfile.TemporaryDirectory() as first,
            tempfile.TemporaryDirectory() as second,
        ):
            self._make_bundle(Path(first))
            self._make_bundle(Path(second))
            self.assertEqual(bundle_hash(Path(first)), bundle_hash(Path(second)))

    def test_content_change_changes_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_bundle(root)
            before = bundle_hash(root)
            # One recalibrated joint must produce a different artifact:
            # calibration is device state.
            (root / "calibration.json").write_text('{"shoulder": 13}', encoding="utf-8")
            self.assertNotEqual(before, bundle_hash(root))

    def test_rename_changes_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_bundle(root)
            before = bundle_hash(root)
            (root / "calibration.json").rename(root / "calib.json")
            self.assertNotEqual(before, bundle_hash(root))

    def test_empty_bundle_refused(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaises(ValueError),
        ):
            bundle_hash(Path(directory))

    def test_stamp_format(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_bundle(root)
            identity = stamp("scene-lab", root)
            name, _, digest = identity.partition("@")
            self.assertEqual(name, "scene-lab")
            self.assertEqual(len(digest), 12)
            with self.assertRaises(ValueError):
                stamp("bad@name", root)


class RecordsAboutABundle(unittest.TestCase):
    """Identifying a robot or auditing its import is a record ABOUT the
    bundle, not its content: writing one never moves its stamp. Until
    2026-09-25 it did, and every checkpoint certified on the old stamp
    was refused by the identity gate (go2@5003bf617b5f -> c699dc1b0772)."""

    def _bundle(self, root: Path) -> None:
        (root / "robot.xml").write_text("<mujoco/>")
        (root / "bundle.json").write_text('{\n "name": "r"\n}\n')

    def test_a_fit_record_and_an_audit_leave_the_stamp_alone(self) -> None:
        from trainnr.bundles.bundle import write_audit  # noqa: PLC0415
        from trainnr.bundles.hashing import FITS_DIR  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._bundle(root)
            before = stamp("r", root)
            (root / FITS_DIR).mkdir()
            (root / FITS_DIR / "rec@000000000000.json").write_text("{}")
            write_audit(root, {"summary": "nothing"})
            self.assertEqual(stamp("r", root), before)
            (root / "robot.xml").write_text("<mujoco><worldbody/></mujoco>")
            self.assertNotEqual(stamp("r", root), before)  # content still counts

    def test_the_robot_kind_hashes_what_stamp_hashes(self) -> None:
        from trainnr.project.kinds import Kind, artifact_hash  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._bundle(root)
            (root / "audit.json").write_text("{}")
            self.assertEqual(
                artifact_hash(Kind.ROBOT, root)[:12], stamp("r", root).split("@")[1]
            )

    def test_a_2026_09_24_audit_migrates_out_and_the_old_stamp_returns(self) -> None:
        import json  # noqa: PLC0415

        from trainnr.bundles.bundle import (  # noqa: PLC0415
            AUDIT_KEY,
            migrate_audit,
            read_audit,
            read_bundle_record,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._bundle(root)
            original = stamp("r", root)
            record = read_bundle_record(root)
            record[AUDIT_KEY] = {"summary": "7 explained"}
            (root / "bundle.json").write_text(
                json.dumps(record, indent=1, sort_keys=True) + "\n"
            )
            self.assertNotEqual(stamp("r", root), original)
            self.assertEqual(
                read_audit(root), {"summary": "7 explained"}
            )  # legacy read
            self.assertTrue(migrate_audit(root))
            self.assertEqual(stamp("r", root), original)
            self.assertEqual(read_audit(root), {"summary": "7 explained"})
            self.assertNotIn(AUDIT_KEY, read_bundle_record(root))
            self.assertFalse(migrate_audit(root))  # once only


class StampRule(unittest.TestCase):
    def test_require_stamp_is_the_one_rule(self) -> None:
        from trainnr.bundles.hashing import is_stamp, require_stamp  # noqa: PLC0415

        self.assertTrue(is_stamp("robot@000000000000"))
        self.assertEqual(require_stamp("robot@000000000000"), "robot@000000000000")
        with self.assertRaises(ValueError) as caught:
            require_stamp("robot", "robot bundle identity")
        self.assertIn("robot bundle identity", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
