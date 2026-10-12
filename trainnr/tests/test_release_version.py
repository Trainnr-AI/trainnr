"""The release's two spellings of one version (tools/release.py): the
product's (`0.2.0-rc.1`, in the tag and the Studio's archives) and the one
the Python packages' files carry (PEP 440's normal form, `0.2.0rc1`). The
release looked for `trainnr-0.2.0-rc.1.tar.gz` and found nothing
(v0.2.0-rc.1, 2026-10-12)."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "release_tool", REPO / "tools" / "release.py"
)
assert _SPEC is not None and _SPEC.loader is not None
release = importlib.util.module_from_spec(_SPEC)
sys.modules["release_tool"] = release  # its dataclasses look themselves up
_SPEC.loader.exec_module(release)


class PythonSpelling(unittest.TestCase):
    def test_it_is_what_packaging_writes(self) -> None:
        # The right column is `str(packaging.version.Version(left))`.
        for version, normal in (
            ("0.1.0", "0.1.0"),
            ("0.2.0-rc.1", "0.2.0rc1"),
            ("0.2.0-rc.12", "0.2.0rc12"),
            ("1.0.0-beta.2", "1.0.0b2"),
            ("1.0.0a1", "1.0.0a1"),
        ):
            with self.subTest(version=version):
                self.assertEqual(release.python_version(version), normal)

    def test_a_version_it_cannot_read_is_refused(self) -> None:
        for version in ("", "v0.2.0", "0.2.0-nightly.1"):
            with self.subTest(version=version), self.assertRaises(ValueError):
                release.python_version(version)


if __name__ == "__main__":
    unittest.main()
