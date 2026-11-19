"""The Studio laid out as an app to double-click (tools/package-studio.py,
docs/85 phase 4): a macOS bundle Finder opens, a Linux AppDir that
appimagetool packs, each with the licences inside."""

from __future__ import annotations

import importlib.util
import os
import plistlib
import re
import struct
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

    def test_the_windows_installer_script_installs_per_user_with_its_files(
        self,
    ) -> None:
        exe = self.out / "trainnr-studio.exe"
        exe.write_bytes(b"MZ")
        iss = self.tool.windows_installer(exe, self.out, "0.2.0-rc.1", self.notices)
        script = iss.read_text(encoding="utf-8")
        # One identity for every version, so an install replaces the last.
        self.assertIn("AppId={{" + self.tool.WINDOWS_APP_ID + "}", script)
        self.assertIn("AppVersion=0.2.0-rc.1", script)
        self.assertIn("VersionInfoVersion=0.2.0", script, "Windows wants numbers")
        self.assertIn("PrivilegesRequired=lowest", script, "no administrator prompt")
        self.assertIn("{localappdata}\\Programs\\trainnr", script)
        self.assertNotIn("$", script, "every template word filled in")
        stage = iss.parent
        for name in (
            "trainnr-studio.exe",
            "trainnr.ico",
            "LICENSE",
            "NOTICE",
            "THIRD_PARTY_LICENSES.md",
        ):
            with self.subTest(name=name):
                self.assertTrue((stage / name).is_file())
                self.assertIn(f'"{name}"', script)
        # Inno Setup refuses a Source that matches nothing.
        for source in re.findall(r'^Source: "([^"]+)"', script, re.M):
            with self.subTest(source=source):
                self.assertTrue(list(stage.glob(source.replace("\\", "/"))))

    def test_the_icon_wraps_the_png_as_windows_reads_it(self) -> None:
        png = self.tool.ICON.read_bytes()
        ico = self.tool.png_to_ico(png)
        self.assertEqual(ico[:6], bytes([0, 0, 1, 0, 1, 0]), "an icon, one image")
        self.assertEqual(ico[22:], png)
        with self.assertRaises(ValueError):
            self.tool.png_to_ico(png[:16] + (32).to_bytes(4, "big") * 2 + png[24:])

    def test_an_exe_that_needs_the_visual_cpp_runtime_is_caught(self) -> None:
        linked = self.out / "dynamic.exe"
        linked.write_bytes(b"MZ....VCRUNTIME140.dll\x00KERNEL32.dll")
        self.assertEqual(self.tool.needs_vc_runtime(linked), ["vcruntime140.dll"])
        static = self.out / "static.exe"
        static.write_bytes(b"MZ....KERNEL32.dll\x00api-ms-win-crt-heap-l1-1-0.dll")
        self.assertEqual(self.tool.needs_vc_runtime(static), [])
        self.assertEqual(self.tool.main(["windows-crt-check", str(linked)]), 1)
        self.assertEqual(self.tool.main(["windows-crt-check", str(static)]), 0)

    def _exe(self, name: str, numbers: tuple[int, int, int], text: str) -> Path:
        """An exe whose version resource gives `numbers` and `text`, laid
        out as VS_FIXEDFILEINFO and a StringFileInfo value store them."""
        major, minor, patch = numbers
        fixed = self.tool.FIXED_FILE_INFO + struct.pack(
            "<II", major << 16 | minor, patch << 16
        )
        exe = self.out / name
        exe.write_bytes(b"MZ...." + fixed + b"\0\0" + text.encode("utf-16-le"))
        return exe

    def test_an_exe_without_its_version_is_caught(self) -> None:
        # v0.2.0's exe showed no version in Properties (Windows 11,
        # 2026-11-18).
        bare = self.out / "bare.exe"
        bare.write_bytes(b"MZ....KERNEL32.dll")
        rc = self._exe("rc.exe", (0, 2, 0), "0.2.0-rc.1")
        cases = {
            (bare, "0.2.0"): "no version resource",
            (rc, "0.2.0-rc.1"): None,
            (rc, "0.2.1"): "file version 0.2.0.0, not 0.2.1.0",
            (rc, "0.2.0-rc.2"): "no product version 0.2.0-rc.2",
        }
        for (exe, version), problem in cases.items():
            with self.subTest(exe=exe.name, version=version):
                self.assertEqual(
                    self.tool.windows_version_problem(exe, version), problem
                )
        check = ["windows-version-check", str(rc), "--version"]
        self.assertEqual(self.tool.main([*check, "0.2.0-rc.1"]), 0)
        self.assertEqual(self.tool.main([*check, "0.2.1"]), 1)


if __name__ == "__main__":
    unittest.main()
