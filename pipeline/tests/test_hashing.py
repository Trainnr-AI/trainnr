"""Bundle identity: bytes and names in, location out."""

import tempfile
import unittest
from pathlib import Path

from rq_pipeline.bundles.hashing import bundle_hash, stamp


class BundleHash(unittest.TestCase):
    def _make_bundle(self, root: Path) -> None:
        (root / "model").mkdir()
        (root / "model" / "robot.xml").write_text("<mujoco/>")
        (root / "calibration.json").write_text('{"shoulder": 12}')

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
            (root / "calibration.json").write_text('{"shoulder": 13}')
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


if __name__ == "__main__":
    unittest.main()
