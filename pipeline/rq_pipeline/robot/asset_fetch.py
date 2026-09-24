"""A public robot asset fetched from GitHub at a pinned commit, whole,
into a local cache — the way a USD folder (a root layer, its sublayers,
payloads, meshes and licence) is obtained before it is read.

Stdlib only. Git LFS is handled: a "raw" download of an LFS-tracked
file is a pointer text, and the same path on the media host is the
file; an empty download is refused by name, because an empty sublayer
composes as an error that refuses the whole stage (2026-09-24). The
cache carries a marker naming the repository and commit, which the
bundle's provenance reads back.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

API_TREE = "https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1"
RAW_FILE = "https://raw.githubusercontent.com/{repo}/{commit}/{path}"
MEDIA_FILE = "https://media.githubusercontent.com/media/{repo}/{commit}/{path}"
LFS_POINTER_HEAD = b"version https://git-lfs.github.com/spec/v1"
MARKER_FILE = "fetched.json"
MARKER_SCHEMA = "trainnr-asset-fetch/1"
USER_AGENT = "rq-pipeline asset fetch"
TIMEOUT_S = 60.0
# Where fetched assets live: `$RQ_ASSET_CACHE`, else `runs/assets` beside
# the checkout's `robots/` (runs/ is never committed).
ASSET_CACHE_ENV = "RQ_ASSET_CACHE"


class AssetFetchError(OSError):
    """The asset could not be fetched whole; the message names the file."""


@dataclass(frozen=True)
class FetchedAsset:
    """A fetched tree: where it landed and what it is."""

    repository: str
    commit: str
    path: str
    root: Path
    files: tuple[str, ...]

    @property
    def marker(self) -> dict[str, object]:
        return {
            "schema": MARKER_SCHEMA,
            "repository": self.repository,
            "commit": self.commit,
            "path": self.path,
            "files": list(self.files),
        }


def asset_cache() -> Path:
    override = os.environ.get(ASSET_CACHE_ENV)
    if override:
        return Path(override).expanduser()
    from rq_pipeline.bundles.locate import robots_dir  # noqa: PLC0415

    return robots_dir().parent / "runs" / "assets"


def cache_slot(repository: str, commit: str, cache: Path | None = None) -> Path:
    """`<cache>/<owner>-<repo>-<commit7>`: one folder per pinned tree."""
    return (cache or asset_cache()) / f"{repository.replace('/', '-')}-{commit[:7]}"


def read_marker(folder: Path) -> dict[str, object] | None:
    """The fetch marker at or above `folder` (an asset's subfolder is
    inside a fetched tree), else None."""
    for candidate in (folder, *folder.parents):
        path = candidate / MARKER_FILE
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                return None
    return None


def _get(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_file(
    repository: str, commit: str, path: str, timeout: float = TIMEOUT_S
) -> bytes:
    """One file at the commit: raw first, then the LFS media host when
    the raw answer is a pointer or empty; refuses an empty file by name."""
    raw = _get(RAW_FILE.format(repo=repository, commit=commit, path=path), timeout)
    if raw and not raw.startswith(LFS_POINTER_HEAD):
        return raw
    media = _get(MEDIA_FILE.format(repo=repository, commit=commit, path=path), timeout)
    if not media:
        raise AssetFetchError(
            f"{path} is empty at {repository}@{commit[:7]} (raw and LFS)"
        )
    return media


def fetch_tree(
    repository: str,
    commit: str,
    path: str,
    *,
    cache: Path | None = None,
    timeout: float = TIMEOUT_S,
) -> FetchedAsset:
    """Every file under `path` at `commit`, into the cache; a slot whose
    marker names the same tree is returned without a request."""
    slot = cache_slot(repository, commit, cache)
    marker = read_marker(slot)
    if marker and marker.get("commit") == commit and marker.get("path") == path:
        listed = marker.get("files")
        files = tuple(str(f) for f in listed) if isinstance(listed, list) else ()
        return FetchedAsset(repository, commit, path, slot / path, files)
    try:
        listing = json.loads(
            _get(API_TREE.format(repo=repository, commit=commit), timeout)
        )
    except (urllib.error.URLError, TimeoutError, ValueError) as why:
        raise AssetFetchError(f"listing {repository}@{commit[:7]}: {why}") from why
    prefix = path.rstrip("/") + "/"
    files = tuple(
        entry["path"]
        for entry in listing.get("tree", [])
        if entry.get("type") == "blob" and entry["path"].startswith(prefix)
    )
    if not files:
        raise AssetFetchError(f"nothing under {path} at {repository}@{commit[:7]}")
    for relative in files:
        target = slot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.write_bytes(fetch_file(repository, commit, relative, timeout))
        except (urllib.error.URLError, TimeoutError) as why:
            raise AssetFetchError(
                f"{relative} at {repository}@{commit[:7]}: {why}"
            ) from why
    fetched = FetchedAsset(repository, commit, path, slot / path, files)
    (slot / MARKER_FILE).write_text(
        json.dumps(fetched.marker, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    return fetched
