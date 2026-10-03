"""The Studio, downloaded: the prebuilt binary for this platform from the
GitHub release that matches this package's version, verified and cached.

    python -m trainnr.studio_install            # install if missing
    python -m trainnr.studio_install --quiet    # the plugin's hook

A checkout that built the Studio (`cargo build --release`) never needs
this; the plugin, a wheel or a fresh clone without Rust does. The release
workflow (`.github/workflows/studio-release.yml`) publishes one archive
per platform, `trainnr-studio-<tag>-<target>.tar.gz` (`.zip` on Windows),
each with a `.sha256` beside it. The archive is fetched, its checksum
checked, unpacked into the user's cache, and only then moved into place,
so a broken download never leaves a half-written binary behind.

Standard library only: this runs before any extra is installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

from trainnr import __version__

REPOSITORY = "Trainnr-AI/trainnr"
RELEASE_ENV = "TRAINNR_STUDIO_RELEASE"  # a tag other than `v<version>`
TOKEN_ENVS = ("GH_TOKEN", "GITHUB_TOKEN")  # a private repository's releases
EXE = "trainnr-studio.exe" if sys.platform.startswith("win") else "trainnr-studio"
CHUNK = 1 << 20
TIMEOUT_S = 60

# The platforms the release workflow builds, by (system, machine).
TARGETS: dict[tuple[str, str], str] = {
    ("linux", "x86_64"): "x86_64-unknown-linux-gnu",
    ("linux", "amd64"): "x86_64-unknown-linux-gnu",
    ("darwin", "arm64"): "aarch64-apple-darwin",
    ("windows", "amd64"): "x86_64-pc-windows-msvc",
    ("windows", "x86_64"): "x86_64-pc-windows-msvc",
}


class StudioInstallError(RuntimeError):
    """Why the Studio could not be installed, in a sentence a user can act on."""


def release_tag() -> str:
    return os.environ.get(RELEASE_ENV) or f"v{__version__}"


def target() -> str | None:
    """This machine's release target, or None when no build is published."""
    return TARGETS.get((platform.system().lower(), platform.machine().lower()))


def asset_name(tag: str, triple: str) -> str:
    suffix = ".zip" if "windows" in triple else ".tar.gz"
    return f"trainnr-studio-{tag}-{triple}{suffix}"


def cache_root() -> Path:
    """The per-user cache: XDG on Linux, Library/Caches on macOS,
    LOCALAPPDATA on Windows."""
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    elif sys.platform.startswith("win"):
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "trainnr" / "studio"


def installed_binary(tag: str | None = None) -> Path | None:
    """The cached Studio for this version and platform, when it is there."""
    triple = target()
    if triple is None:
        return None
    path = cache_root() / (tag or release_tag()) / triple / EXE
    return path if path.is_file() else None


def _token() -> str | None:
    return next((os.environ[name] for name in TOKEN_ENVS if os.environ.get(name)), None)


def _request(url: str, accept: str | None = None) -> urllib.request.Request:
    headers = {"User-Agent": f"trainnr/{__version__}"}
    if accept:
        headers["Accept"] = accept
    token = _token()
    if token and url.startswith("https://api.github.com/"):
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, headers=headers)


def _asset_urls(tag: str, names: list[str]) -> dict[str, str]:
    """Download URLs for the named assets. A public release serves them at
    their browser URLs; with a token they come through the API (which is
    how a private repository's release is read)."""
    if _token() is None:
        base = f"https://github.com/{REPOSITORY}/releases/download/{tag}/"
        return {name: base + name for name in names}
    api = f"https://api.github.com/repos/{REPOSITORY}/releases/tags/{tag}"
    try:
        with urllib.request.urlopen(_request(api), timeout=TIMEOUT_S) as reply:
            release = json.load(reply)
    except OSError as why:
        raise StudioInstallError(f"no release {tag} on {REPOSITORY} ({why})") from why
    found = {a["name"]: a["url"] for a in release.get("assets", [])}
    missing = [name for name in names if name not in found]
    if missing:
        raise StudioInstallError(f"release {tag} has no {', '.join(missing)}")
    return found


def _fetch(url: str, out: Path, say: Callable[[str], None]) -> None:
    accept = "application/octet-stream" if url.startswith("https://api.") else None
    try:
        with urllib.request.urlopen(_request(url, accept), timeout=TIMEOUT_S) as reply:
            total = int(reply.headers.get("Content-Length") or 0)
            done = 0
            with out.open("wb") as sink:
                while chunk := reply.read(CHUNK):
                    sink.write(chunk)
                    done += len(chunk)
                    if total and done % (16 * CHUNK) < CHUNK:
                        say(f"  {done // CHUNK} of {total // CHUNK} MB")
    except OSError as why:
        raise StudioInstallError(f"download failed: {url} ({why})") from why


def install(say: Callable[[str], None] = print, tag: str | None = None) -> Path:
    """Download, verify and unpack the Studio; return the binary's path.
    A no-op when this version is already installed."""
    tag = tag or release_tag()
    already = installed_binary(tag)
    if already is not None:
        return already
    triple = target()
    if triple is None:
        raise StudioInstallError(
            f"no prebuilt Studio for {platform.system()} {platform.machine()}; "
            "build it from a checkout: "
            "cd crates/trainnr-studio && cargo build --release"
        )
    name = asset_name(tag, triple)
    urls = _asset_urls(tag, [name, name + ".sha256"])
    destination = cache_root() / tag / triple
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as scratch:
        work = Path(scratch)
        say(f"downloading the Studio {tag} for {triple}")
        _fetch(urls[name + ".sha256"], work / "sum", say)
        _fetch(urls[name], work / name, say)
        expected = (work / "sum").read_text().split()[0].strip().lower()
        digest = hashlib.sha256()
        with (work / name).open("rb") as archive:
            while chunk := archive.read(CHUNK):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise StudioInstallError(f"{name}: checksum mismatch, nothing installed")
        unpacked = work / "unpacked"
        if name.endswith(".zip"):
            with zipfile.ZipFile(work / name) as archive:
                archive.extractall(unpacked)
        else:
            with tarfile.open(work / name) as archive:
                archive.extractall(unpacked, filter="data")
        binary = next(unpacked.rglob(EXE), None)
        if binary is None:
            raise StudioInstallError(f"{name} holds no {EXE}")
        binary.chmod(0o755)
        staged = work / "staged"
        staged.mkdir()
        shutil.move(str(binary), staged / EXE)
        if destination.exists():
            shutil.rmtree(destination)
        staged.rename(destination)
    path = destination / EXE
    say(f"the Studio is installed: {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quiet", action="store_true", help="print nothing on success")
    args = parser.parse_args(argv)

    def say(line: str) -> None:
        if not args.quiet:
            print(line, file=sys.stderr)

    try:
        install(say)
    except StudioInstallError as why:
        print(f"trainnr: {why}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
