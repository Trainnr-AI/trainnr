#!/usr/bin/env python3
"""Install Brush's prebuilt binary for this machine, no root needed.

    python3 tools/install-brush.py [--version v0.3.0] [--into ~/.local/bin]

Brush (ArthurBrussee/brush, Apache-2.0/MIT) is the splat trainer the
capture chain runs (`rq_pipeline.scenes.capture`, docs/78 §8.6). It
ships release binaries for macOS (arm64), Linux (x86_64) and Windows
(x86_64); this fetches the one for the running platform, checks its
published SHA-256, unpacks it into a user bin directory and prints
where `brush_app` landed. The chain finds it on PATH or by `--brush`.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import platform
import shutil
import stat
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

REPO = "ArthurBrussee/brush"
DEFAULT_VERSION = "v0.3.0"
BINARY = "brush_app"
# (system, machine) -> the release asset's platform word.
ASSETS = {
    ("Darwin", "arm64"): "aarch64-apple-darwin.tar.xz",
    ("Linux", "x86_64"): "x86_64-unknown-linux-gnu.tar.xz",
    ("Windows", "AMD64"): "x86_64-pc-windows-msvc.zip",
}


def asset_for(system: str, machine: str) -> str:
    try:
        return f"brush-app-{ASSETS[(system, machine)]}"
    except KeyError as missing:
        raise SystemExit(
            f"Brush publishes no binary for {system}/{machine}; build it from "
            f"source (https://github.com/{REPO}: cargo build --release) and pass "
            "the binary's path to the chain"
        ) from missing


def default_bin_dir(system: str) -> Path:
    if system == "Windows":
        return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Programs" / "brush"
    return Path.home() / ".local" / "bin"


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--into", type=Path, default=None, help="the bin directory")
    args = parser.parse_args()
    system, machine = platform.system(), platform.machine()
    asset = asset_for(system, machine)
    base = f"https://github.com/{REPO}/releases/download/{args.version}"
    print(f"[brush] {asset} {args.version} for {system}/{machine}")
    archive = fetch(f"{base}/{asset}")
    published = fetch(f"{base}/{asset}.sha256").decode().split()[0]
    digest = hashlib.sha256(archive).hexdigest()
    if digest != published:
        raise SystemExit(
            f"[brush] sha256 mismatch: got {digest}, published {published}"
        )
    into = args.into or default_bin_dir(system)
    into.mkdir(parents=True, exist_ok=True)
    name = BINARY + (".exe" if system == "Windows" else "")
    target = into / name
    if asset.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            member = next(m for m in z.namelist() if m.endswith(name))
            target.write_bytes(z.read(member))
    else:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:xz") as t:
            member = next(m for m in t.getmembers() if m.name.endswith(name))
            extracted = t.extractfile(member)
            assert extracted is not None
            target.write_bytes(extracted.read())
    target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"[brush] {target} ({len(archive) // 1_000_000} MB, sha256 {digest[:12]})")
    if shutil.which(name) is None:
        print(f"[brush] {into} is not on PATH: add it, or pass --brush {target}")
        sys.exit(2)


if __name__ == "__main__":
    main()
