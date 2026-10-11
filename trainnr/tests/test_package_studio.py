"""The Studio laid out as an app to double-click (tools/package-studio.py,
docs/85 phase 4): a macOS bundle Finder opens, a Linux AppDir that
appimagetool packs, each with the licences inside."""

from __future__ import annotations

import importlib.util
import os
import plistlib
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _tool():
    spec = importlib.util.spec_from_file_location(
        "package_studio", REPO / "tools" / "package-studio.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TheLayouts(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = _tool()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.out = Path(tmp.name)
        self.binary = self.out / "trainnr-studio"
        self.binary.write_bytes(b"\x7fELF")
        self.notices = self.out / "THIRD_PARTY_LICENSES.md"
        self.notices.write_text("crates")

    def test_the_macos_bundle_names_its_executable_and_carries_the_licences(
        self,
    ) -> None:
        app = self.tool.macos_app(self.binary, self.out, "0.2.0-rc.1", self.notices)
        with (app / "Contents" / "Info.plist").open("rb") as source:
            info = plistlib.load(source)
        self.assertEqual(info["CFBundleExecutable"], "trainnr-studio")
        self.assertEqual(info["CFBundleShortVersionString"], "0.2.0")
        self.assertEqual(info["CFBundleVersion"], "0.2.0-rc.1")
        executable = app / "Contents" / "MacOS" / "trainnr-studio"
        self.assertTrue(os.access(executable, os.X_OK))
        resources = app / "Contents" / "Resources"
        for name in ("LICENSE", "NOTICE", "THIRD_PARTY_LICENSES.md", "trainnr.png"):
            with self.subTest(name=name):
                self.assertTrue((resources / name).is_file())

    def test_the_appdir_runs_the_studio_and_names_its_icon(self) -> None:
        appdir = self.tool.linux_appdir(self.binary, self.out, self.notices)
        self.assertTrue(os.access(appdir / "AppRun", os.X_OK))
        self.assertIn("usr/bin/trainnr-studio", (appdir / "AppRun").read_text())
        desktop = (appdir / "trainnr.desktop").read_text()
        self.assertIn("Exec=trainnr-studio", desktop)
        self.assertIn("Icon=trainnr", desktop)
        self.assertTrue((appdir / "trainnr.png").is_file())
        self.assertTrue(
            (appdir / "usr" / "share" / "doc" / "trainnr" / "LICENSE").is_file()
        )


if __name__ == "__main__":
    unittest.main()
