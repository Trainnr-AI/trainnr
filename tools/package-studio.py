#!/usr/bin/env python3
"""Lay the Studio out as an app a person installs by double-clicking:
a macOS `.app` bundle, and a Linux AppDir that `appimagetool` turns into
an AppImage (the desktop app plan, docs/85, phase 4, 2026-10-11).

    python3 tools/package-studio.py macos-app  <binary> <out> --version 0.1.0
    python3 tools/package-studio.py linux-appdir <binary> <out> --version 0.1.0
    python3 tools/package-studio.py windows-installer <binary.exe> <out> --version 0.1.0
    python3 tools/package-studio.py windows-crt-check <binary.exe>

The release workflow runs it on each platform's runner and finishes the
job with the platform's own tools: `iconutil` and `hdiutil` make the
macOS icon and the disk image (with an Applications link to drag onto);
`appimagetool` (pinned, its SHA-256 checked) makes the AppImage; Inno
Setup (on GitHub's Windows runners) compiles the installer script written
here into one `setup.exe` (2026-10-12). Windows keeps its zip too, for
the plugin's installer and `trainnr studio`.

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
import struct
import sys
from pathlib import Path
from string import Template

ROOT = Path(__file__).resolve().parents[1]
ICON = ROOT / "crates" / "trainnr-studio" / "assets" / "icon-preview.png"
NAME = "trainnr"
BUNDLE_ID = "ai.trainnr.studio"
EXECUTABLE = "trainnr-studio"
# The oldest macOS the release's Apple Silicon build is tested against.
MACOS_MINIMUM = "13.0"
# The Windows installer's identity: fixed for good, so each version
# installs over the last and Windows lists one trainnr, not one per
# version.
WINDOWS_APP_ID = "64828157-76CA-42B9-919D-AFFE13E0B257"
PUBLISHER = "Prakhar Aggarwal"
HOMEPAGE = "https://github.com/Trainnr-AI/trainnr"
# The installer script Inno Setup compiles (a template). Per user, under
# %LOCALAPPDATA%\Programs, so it asks for no administrator rights; a
# Start-menu entry, an optional desktop one, an uninstaller; the projects
# and the engine under ~/trainnr are the user's and stay on uninstall.
INNO_SCRIPT = ROOT / "tools" / "windows-installer.iss"
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


def png_to_ico(png: bytes) -> bytes:
    """An icon file holding one 256-pixel PNG, as Windows Vista and later
    read it: the 6-byte header, one 16-byte entry (0 means 256), the PNG."""
    width, height = struct.unpack(">II", png[16:24])
    if (width, height) != (256, 256):
        raise ValueError(f"the icon is {width}x{height}; Windows wants 256x256")
    header = struct.pack("<HHH", 0, 1, 1)
    entry = struct.pack("<BBBBHHII", 0, 0, 0, 0, 1, 32, len(png), 6 + 16)
    return header + entry + png


def windows_installer(
    binary: Path, out: Path, version: str, third_party: Path | None
) -> Path:
    """`<out>/trainnr-windows/trainnr.iss`, with everything it installs
    beside it: the Studio, its icon and the licences."""
    stage = out / f"{NAME}-windows"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    exe = f"{EXECUTABLE}.exe"
    shutil.copy2(binary, stage / exe)
    (stage / f"{NAME}.ico").write_bytes(png_to_ico(ICON.read_bytes()))
    copy_licences(stage, third_party)
    extra = (
        f'Source: "{third_party.name}"; DestDir: "{{app}}"; Flags: ignoreversion\n'
        if third_party is not None
        else ""
    )
    script = Template(INNO_SCRIPT.read_text(encoding="utf-8")).substitute(
        app_id=WINDOWS_APP_ID,
        name=NAME,
        version=version,
        numeric=version.split("-", maxsplit=1)[0],
        publisher=PUBLISHER,
        homepage=HOMEPAGE,
        exe=exe,
        third_party=extra,
    )
    iss = stage / f"{NAME}.iss"
    iss.write_text(script, encoding="utf-8")
    return iss


#: The Visual C++ runtime DLLs a dynamically linked exe imports; with the
#: runtime linked in (`crates/trainnr-studio/.cargo/config.toml`) the
#: Studio names none of them.
VC_RUNTIME_DLLS = (b"vcruntime140.dll", b"vcruntime140_1.dll", b"msvcp140.dll")


def needs_vc_runtime(exe: Path) -> list[str]:
    """The Visual C++ runtime DLLs `exe` imports by name (an import names
    its DLL as a plain string in the file)."""
    data = exe.read_bytes().lower()
    return [dll.decode() for dll in VC_RUNTIME_DLLS if dll in data]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "layout",
        choices=("macos-app", "linux-appdir", "windows-installer", "windows-crt-check"),
    )
    parser.add_argument("binary", type=Path)
    parser.add_argument("out", type=Path, nargs="?", help="where the layout is written")
    parser.add_argument("--version")
    parser.add_argument("--third-party", type=Path, help="THIRD_PARTY_LICENSES.md")
    args = parser.parse_args(argv)
    if args.layout != "windows-crt-check" and (
        args.out is None or args.version is None
    ):
        parser.error(f"{args.layout} needs <out> and --version")
    if args.layout == "windows-crt-check":
        missing = needs_vc_runtime(args.binary)
        if missing:
            print(f"{args.binary} needs the Visual C++ runtime: {', '.join(missing)}")
            return 1
        print(f"{args.binary}: no Visual C++ runtime DLL needed")
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    if args.layout == "macos-app":
        made = macos_app(args.binary, args.out, args.version, args.third_party)
    elif args.layout == "windows-installer":
        made = windows_installer(args.binary, args.out, args.version, args.third_party)
    else:
        made = linux_appdir(args.binary, args.out, args.third_party)
    print(made)
    return 0


if __name__ == "__main__":
    sys.exit(main())
