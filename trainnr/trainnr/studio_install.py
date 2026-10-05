"""The Studio, downloaded: the prebuilt binary for this platform from the
GitHub release that matches this package's version, verified and cached.

    python -m trainnr.studio_install            # install if missing
    python -m trainnr.studio_install --quiet    # the plugin's hook

A checkout that built the Studio (`cargo build --release`) never needs
this; the plugin, a wheel or a fresh clone without Rust does. The release
workflow (`.github/workflows/release.yml`) publishes one archive
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
import time
import urllib.error
import urllib.parse
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


RELEASES_URL = f"https://github.com/{REPOSITORY}/releases"
BUILD_FROM_SOURCE = "cd crates/trainnr-studio && cargo build --release"
# Two installs of one version race only through this lock: the plugin's
# session hook and a `launch_studio` call can start within a second.
LOCK_WAIT_S = 900.0  # a slow download of 72 MB still finishes inside this
LOCK_STALE_S = 1800.0  # a lock older than this was left by a dead process
LOCK_POLL_S = 0.5


class _NoTokenAcrossHosts(urllib.request.HTTPRedirectHandler):
    """Follow redirects, but never carry the GitHub token to another host:
    an API asset download redirects to a storage CDN."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]  # noqa: PLR0913, PLR0917
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise urllib.error.HTTPError(
                newurl, code, "refusing a redirect away from https", headers, fp
            )
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None:
            old_host = urllib.parse.urlsplit(req.full_url).hostname
            if urllib.parse.urlsplit(newurl).hostname != old_host:
                new.headers.pop("Authorization", None)
                new.unredirected_hdrs.pop("Authorization", None)
        return new


# Honours HTTPS_PROXY and friends (urllib's ProxyHandler is in the defaults).
_OPENER = urllib.request.build_opener(_NoTokenAcrossHosts())


def _request(url: str, accept: str | None = None) -> urllib.request.Request:
    headers = {"User-Agent": f"trainnr/{__version__}"}
    if accept:
        headers["Accept"] = accept
    token = _token()
    if token and url.startswith("https://api.github.com/"):
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, headers=headers)


def _not_found(tag: str, what: str) -> StudioInstallError:
    return StudioInstallError(
        f"{what}: the Studio release {tag} is not published yet, or this "
        f"platform has no build in it ({RELEASES_URL}). Until it is, build "
        f"the Studio from a checkout: {BUILD_FROM_SOURCE}"
    )


def _asset_urls(tag: str, names: list[str]) -> dict[str, str]:
    """Download URLs for the named assets. A public release serves them at
    their browser URLs; with a token they come through the API (which is
    how a private repository's release is read)."""
    if _token() is None:
        base = f"https://github.com/{REPOSITORY}/releases/download/{tag}/"
        return {name: base + name for name in names}
    api = f"https://api.github.com/repos/{REPOSITORY}/releases/tags/{tag}"
    try:
        with _OPENER.open(_request(api), timeout=TIMEOUT_S) as reply:
            release = json.load(reply)
    except urllib.error.HTTPError as why:
        if why.code == HTTP_NOT_FOUND:
            raise _not_found(tag, f"no release {tag} on {REPOSITORY}") from why
        raise StudioInstallError(f"no release {tag} on {REPOSITORY} ({why})") from why
    except (OSError, ValueError) as why:
        raise StudioInstallError(f"no release {tag} on {REPOSITORY} ({why})") from why
    found = {a["name"]: a["url"] for a in release.get("assets", [])}
    missing = [name for name in names if name not in found]
    if missing:
        raise _not_found(tag, f"release {tag} has no {', '.join(missing)}")
    return found


HTTP_NOT_FOUND = 404


def _fetch(url: str, out: Path, say: Callable[[str], None], tag: str) -> None:
    accept = "application/octet-stream" if url.startswith("https://api.") else None
    try:
        with _OPENER.open(_request(url, accept), timeout=TIMEOUT_S) as reply:
            total = int(reply.headers.get("Content-Length") or 0)
            done = 0
            with out.open("wb") as sink:
                while chunk := reply.read(CHUNK):
                    sink.write(chunk)
                    done += len(chunk)
                    if total and done % (16 * CHUNK) < CHUNK:
                        say(f"  {done // CHUNK} of {total // CHUNK} MB")
    except urllib.error.HTTPError as why:
        if why.code == HTTP_NOT_FOUND:
            raise _not_found(tag, f"download failed: {url}") from why
        raise StudioInstallError(f"download failed: {url} ({why})") from why
    except OSError as why:
        raise StudioInstallError(f"download failed: {url} ({why})") from why
    if total and done != total:
        raise StudioInstallError(f"download cut short: {url} ({done} of {total} bytes)")


def _safe_extract_tar(archive: tarfile.TarFile, into: Path) -> None:
    """Unpack with the standard library's `data` filter where it has one;
    on an older Python, refuse members that leave `into` or are links."""
    if hasattr(tarfile, "data_filter"):
        archive.extractall(into, filter="data")
        return
    root = into.resolve()
    for member in archive.getmembers():
        target = (into / member.name).resolve()
        if (
            not (target == root or root in target.parents)
            or member.issym()
            or member.islnk()
        ):
            raise StudioInstallError(f"unsafe path in the archive: {member.name}")
    archive.extractall(into)


class _InstallLock:
    """One installer per (version, platform) at a time, across processes:
    an exclusively created lock file, waited on while another holds it,
    taken over when it is older than any download could be."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.held = False

    def __enter__(self) -> _InstallLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + LOCK_WAIT_S
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    age = time.time() - self.path.stat().st_mtime
                except FileNotFoundError:
                    continue  # released between the two calls: try again
                if age > LOCK_STALE_S:
                    self.path.unlink(missing_ok=True)
                    continue
                if time.monotonic() > deadline:
                    raise StudioInstallError(
                        f"another install of the Studio holds {self.path}; "
                        "delete that file if no install is running"
                    ) from None
                time.sleep(LOCK_POLL_S)
                continue
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            self.held = True
            return self

    def __exit__(self, *_exc: object) -> None:
        if self.held:
            self.path.unlink(missing_ok=True)


def install(say: Callable[[str], None] = print, tag: str | None = None) -> Path:
    """Download, verify and unpack the Studio; return the binary's path.
    A no-op when this version is already installed; safe to run from two
    processes at once (the second waits and then finds it installed)."""
    tag = tag or release_tag()
    already = installed_binary(tag)
    if already is not None:
        return already
    triple = target()
    if triple is None:
        raise StudioInstallError(
            f"no prebuilt Studio for {platform.system()} {platform.machine()}; "
            f"build it from a checkout: {BUILD_FROM_SOURCE}"
        )
    destination = cache_root() / tag / triple
    try:
        with _InstallLock(destination.parent / f".{triple}.lock"):
            already = installed_binary(tag)
            if already is not None:
                return already
            return _install_locked(tag, triple, destination, say)
    except StudioInstallError:
        raise
    except (
        OSError,
        IndexError,
        TypeError,
        ValueError,
        tarfile.TarError,
        zipfile.BadZipFile,
    ) as why:
        raise StudioInstallError(
            f"the Studio could not be installed into {destination}: "
            f"{type(why).__name__}: {why}"
        ) from why


def _install_locked(
    tag: str, triple: str, destination: Path, say: Callable[[str], None]
) -> Path:
    name = asset_name(tag, triple)
    urls = _asset_urls(tag, [name, name + ".sha256"])
    with tempfile.TemporaryDirectory(dir=destination.parent) as scratch:
        work = Path(scratch)
        say(f"downloading the Studio {tag} for {triple}")
        _fetch(urls[name + ".sha256"], work / "sum", say, tag)
        _fetch(urls[name], work / name, say, tag)
        words = (work / "sum").read_text().split()
        if not words:
            raise StudioInstallError(f"{name}.sha256 is empty; nothing installed")
        expected = words[0].strip().lower()
        digest = hashlib.sha256()
        with (work / name).open("rb") as archive:
            while chunk := archive.read(CHUNK):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise StudioInstallError(f"{name}: checksum mismatch, nothing installed")
        unpacked = work / "unpacked"
        if name.endswith(".zip"):
            with zipfile.ZipFile(work / name) as archive:
                for member in archive.namelist():
                    if member.startswith(("/", "\\")) or ".." in Path(member).parts:
                        raise StudioInstallError(
                            f"unsafe path in the archive: {member}"
                        )
                archive.extractall(unpacked)
        else:
            with tarfile.open(work / name) as archive:
                _safe_extract_tar(archive, unpacked)
        binary = next(unpacked.rglob(EXE), None)
        if binary is None:
            raise StudioInstallError(f"{name} holds no {EXE}")
        binary.chmod(0o755)
        staged = work / "staged"
        staged.mkdir()
        shutil.move(str(binary), staged / EXE)
        # A complete install is never removed: an incomplete leftover (no
        # binary inside) is cleared, then the staged folder is renamed into
        # place in one step.
        if destination.exists() and not (destination / EXE).is_file():
            shutil.rmtree(destination)
        if not destination.exists():
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
    except (OSError, ValueError) as why:  # anything install() did not word itself
        print(f"trainnr: the Studio could not be installed: {why}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
