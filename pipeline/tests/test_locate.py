"""Where bundles live: the environment override, the checkout default,
and a missing file refused by the variable's name."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rq_pipeline.bundles.locate import (
    ROBOTS_DIR_ENV,
    bundle_file,
    require_bundle_file,
    robots_dir,
)


class TheBundleDoor(unittest.TestCase):
    def test_the_checkout_default_and_the_override(self) -> None:
        with mock.patch.dict(os.environ, {ROBOTS_DIR_ENV: ""}):
            os.environ.pop(ROBOTS_DIR_ENV)
            self.assertEqual(robots_dir().name, "robots")
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.dict(os.environ, {ROBOTS_DIR_ENV: tmp}),
        ):
            self.assertEqual(robots_dir(), Path(tmp))
            self.assertEqual(
                bundle_file("aloha2", "scene.xml"), Path(tmp) / "aloha2" / "scene.xml"
            )

    def test_a_missing_file_names_the_variable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            present = Path(tmp) / "x.xml"
            present.write_text("<mujoco/>", encoding="utf-8")
            self.assertEqual(require_bundle_file(present), present)
            with self.assertRaisesRegex(FileNotFoundError, ROBOTS_DIR_ENV):
                require_bundle_file(Path(tmp) / "missing.xml")


if __name__ == "__main__":
    unittest.main()
