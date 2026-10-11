#!/usr/bin/env python3
"""Lay the Studio out as an app a person installs by double-clicking:
a macOS `.app` bundle, and a Linux AppDir that `appimagetool` turns into
an AppImage (the desktop app plan, docs/85, phase 4, 2026-10-11).

    python3 tools/package-studio.py macos-app  <binary> <out> --version 0.1.0
    python3 tools/package-studio.py linux-appdir <binary> <out> --version 0.1.0

The release workflow runs it on each platform's runner and finishes the
job with the platform's own tools: `iconutil` and `hdiutil` make the
macOS icon and the disk image (with an Applications link to drag onto);
`appimagetool` (pinned, its SHA-256 checked) makes the AppImage. Windows
keeps its zip: `trainnr-studio.exe` runs where it is unpacked.

Nothing here is signed: an Apple Developer ID and a Windows signing
certificate are the maintainer's to obtain (docs/85 §7). Until then,
macOS asks the person to open the app from System Settings, Privacy &
Security, the first time.

The licences travel inside each layout (LICENSE, NOTICE, LICENSES/, and
the third-party list when the release passes it). Stdlib only.
"""

from __future__ import annotations

import argparse
import plistlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ICON = ROOT / "crates" / "trainnr-studio" / "assets" / "icon-preview.png"
NAME = "trainnr"
BUNDLE_ID = "ai.trainnr.studio"
EXECUTABLE = "trainnr-studio"
# The oldest macOS the release's Apple Silicon build is tested against.
MACOS_MINIMUM = "13.0"
DESKTOP_ENTRY = """[Desktop Entry]
Type=Application
Name=trainnr
Comment=The robotics platform on your desktop
Exec=trainnr-studio
Icon=trainnr
Categories=Development;Science;Engineering;
Terminal=false
"""
APP_RUN = """#!/bin/sh
# The AppImage's entry: the Studio, from where the image is mounted.
here="$(dirname "$(readlink -f "$0")")"
exec "$here/usr/bin/trainnr-studio" "$@"
"""


def copy_licences(into: Path, third_party: Path | None) -> None:
    """LICENSE, NOTICE, LICENSES/ and the third-party list, beside the app."""
    into.mkdir(parents=True, exist_ok=True)
    for name in ("LICENSE", "NOTICE"):
        shutil.copy2(ROOT / name, into / name)
    shutil.copytree(ROOT / "LICENSES", into / "LICENSES", dirs_exist_ok=True)
    if third_party is not None:
        shutil.copy2(third_party, into / third_party.name)


def info_plist(version: str) -> dict[str, object]:
    """The bundle's Info.plist: the name Finder shows, the executable, the
    icon, and the version on its About line."""
    return {
        "CFBundleName": NAME,
        "CFBundleDisplayName": NAME,
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleExecutable": EXECUTABLE,
        "CFBundleIconFile": NAME,
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": version.split("-", maxsplit=1)[0],
        "CFBundleVersion": version,
        "LSMinimumSystemVersion": MACOS_MINIMUM,
        "NSHighResolutionCapable": True,
        "LSApplicationCategoryType": "public.app-category.developer-tools",
    }


def macos_app(binary: Path, out: Path, version: str, third_party: Path | None) -> Path:
    """`<out>/trainnr.app`, with the PNG the release turns into `.icns`."""
    app = out / f"{NAME}.app"
    if app.exists():
        shutil.rmtree(app)
    macos = app / "Contents" / "MacOS"
    resources = app / "Contents" / "Resources"
    macos.mkdir(parents=True)
    resources.mkdir(parents=True)
    shutil.copy2(binary, macos / EXECUTABLE)
    (macos / EXECUTABLE).chmod(0o755)
    with (app / "Contents" / "Info.plist").open("wb") as sink:
        plistlib.dump(info_plist(version), sink)
    shutil.copy2(ICON, resources / f"{NAME}.png")  # iconutil makes trainnr.icns
    copy_licences(resources, third_party)
    return app


def linux_appdir(binary: Path, out: Path, third_party: Path | None) -> Path:
    """`<out>/trainnr.AppDir`, what appimagetool packs."""
    appdir = out / f"{NAME}.AppDir"
    if appdir.exists():
        shutil.rmtree(appdir)
    bin_dir = appdir / "usr" / "bin"
    bin_dir.mkdir(parents=True)
    shutil.copy2(binary, bin_dir / EXECUTABLE)
    (bin_dir / EXECUTABLE).chmod(0o755)
    (appdir / "AppRun").write_text(APP_RUN)
    (appdir / "AppRun").chmod(0o755)
    (appdir / f"{NAME}.desktop").write_text(DESKTOP_ENTRY)
    shutil.copy2(ICON, appdir / f"{NAME}.png")
    copy_licences(appdir / "usr" / "share" / "doc" / NAME, third_party)
    return appdir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("layout", choices=("macos-app", "linux-appdir"))
    parser.add_argument("binary", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--third-party", type=Path, help="THIRD_PARTY_LICENSES.md")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.layout == "macos-app":
        made = macos_app(args.binary, args.out, args.version, args.third_party)
    else:
        made = linux_appdir(args.binary, args.out, args.third_party)
    print(made)
    return 0


if __name__ == "__main__":
    sys.exit(main())
