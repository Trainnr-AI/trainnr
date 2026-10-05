"""A public robot asset fetched from GitHub at a pinned commit, whole,
into a local cache — the way a USD folder (a root layer, its sublayers,
payloads, meshes and licence) is obtained before it is read.

Stdlib only. Git LFS is handled: a "raw" download of an LFS-tracked
file is a pointer text, and the same path on the media host is the
file; an empty download is refused by name, because an empty sublayer
composes as an error that refuses the whole stage (2026-09-24). Every
file is checked before it is kept: its bytes against the git blob id
the tree lists, and an LFS file's media against the sha256 and size its
pointer names; a listing GitHub truncated is refused, never taken as
the whole asset (review 2026-09-24). The cache carries a marker naming
the repository and commit, written last, which the bundle's provenance
reads back.
"""

from __future__ import annotations

import hashlib
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
USER_AGENT = "trainnr asset fetch"
TIMEOUT_S = 60.0
# Where fetched assets live: `$TRAINNR_ASSETS_DIR`, else `runs/assets` in the
# checkout (runs/ is never committed) — found the way the public logs'
# cache is (`robots.public_logs`: `$TRAINNR_PUBLIC_LOGS_DIR`, `runs/public-logs`).
ASSET_CACHE_ENV = "TRAINNR_ASSETS_DIR"
# The repository's own licence and notice files, fetched with any asset
# under it: a dual-licensed repository states its root licence there
# (robotiq/isaacsim_assets: BSD-3 at the root, CC-BY beside the USD), and
# an asset fetched without it was labelled with half its licence
# (review 2026-10-03).
ROOT_LEGAL_FILES = frozenset(
    {"LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING", "NOTICE", "NOTICE.txt"}
)
LEGACY_ASSET_CACHE_ENV = "TRAINNR_ASSET_CACHE"  # the first spelling, still read
CACHE_RELATIVE = ("runs", "assets")
LFS_OID = "oid sha256:"
LFS_SIZE = "size "


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
    for env in (ASSET_CACHE_ENV, LEGACY_ASSET_CACHE_ENV):
        override = os.environ.get(env, "").strip()
        if override:
            return Path(override).expanduser()
    from trainnr.paths import user_cache  # noqa: PLC0415

    return user_cache("assets", CACHE_RELATIVE)


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


def git_blob_id(data: bytes) -> str:
    """The id git gives a file's bytes: sha1 of `blob <size>\\0<bytes>`."""
    return hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()


def _lfs_pointer(raw: bytes) -> tuple[str, int]:
    """(sha256, size) an LFS pointer names; refused by name when it names
    neither."""
    oid, size = "", -1
    for line in raw.decode("utf-8", errors="replace").splitlines():
        if line.startswith(LFS_OID):
            oid = line[len(LFS_OID) :].strip()
        elif line.startswith(LFS_SIZE):
            size = int(line[len(LFS_SIZE) :].strip())
    if not oid or size < 0:
        raise AssetFetchError("an LFS pointer without an oid and a size")
    return oid, size


def fetch_file(
    repository: str,
    commit: str,
    path: str,
    timeout: float = TIMEOUT_S,
    *,
    blob: str | None = None,
) -> bytes:
    """One file at the commit: raw first, then the LFS media host when
    the raw answer is a pointer or empty; refuses an empty file by name.
    With `blob` (the git id the tree lists), the raw answer must hash to
    it, and an LFS file's media must match its pointer's sha256 and size;
    a mismatch is refused by name and nothing is kept."""
    where = f"{path} at {repository}@{commit[:7]}"
    raw = _get(RAW_FILE.format(repo=repository, commit=commit, path=path), timeout)
    if blob is not None and raw and git_blob_id(raw) != blob:
        raise AssetFetchError(f"{where}: its bytes are not the blob the tree lists")
    if raw and not raw.startswith(LFS_POINTER_HEAD):
        return raw
    media = _get(MEDIA_FILE.format(repo=repository, commit=commit, path=path), timeout)
    if not media:
        raise AssetFetchError(f"{where} is empty (raw and LFS)")
    if raw.startswith(LFS_POINTER_HEAD):
        oid, size = _lfs_pointer(raw)
        if len(media) != size or hashlib.sha256(media).hexdigest() != oid:
            raise AssetFetchError(
                f"{where}: the LFS media is {len(media)} bytes, not the pointer's "
                f"{size} with sha256 {oid[:12]}…; nothing was kept"
            )
    elif blob is not None and git_blob_id(media) != blob:
        # no raw answer to check: the media itself must be the listed blob
        raise AssetFetchError(f"{where}: its bytes are not the blob the tree lists")
    return media


def cached_tree(
    repository: str, commit: str, path: str, *, cache: Path | None = None
) -> FetchedAsset | None:
    """The tree when it is already in the cache (its marker names this
    commit and path), else None — never a request (a test reads this and
    skips by name rather than fetch)."""
    slot = cache_slot(repository, commit, cache)
    marker = read_marker(slot)
    if not (marker and marker.get("commit") == commit and marker.get("path") == path):
        return None
    listed = marker.get("files")
    files = tuple(str(f) for f in listed) if isinstance(listed, list) else ()
    return FetchedAsset(repository, commit, path, slot / path, files)


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
    cached = cached_tree(repository, commit, path, cache=cache)
    if cached is not None:
        return cached
    try:
        listing = json.loads(
            _get(API_TREE.format(repo=repository, commit=commit), timeout)
        )
    except (urllib.error.URLError, TimeoutError, ValueError) as why:
        raise AssetFetchError(f"listing {repository}@{commit[:7]}: {why}") from why
    if listing.get("truncated"):
        raise AssetFetchError(
            f"GitHub truncated the tree of {repository}@{commit[:7]} (too large for "
            "one listing); the asset would be fetched partly — clone the repository "
            "at the commit and read it from disk instead"
        )
    prefix = path.rstrip("/") + "/"
    blobs = {
        entry["path"]: entry.get("sha")
        for entry in listing.get("tree", [])
        if entry.get("type") == "blob"
        and (entry["path"].startswith(prefix) or entry["path"] in ROOT_LEGAL_FILES)
    }
    files = tuple(blobs)
    if not any(relative.startswith(prefix) for relative in files):
        raise AssetFetchError(f"nothing under {path} at {repository}@{commit[:7]}")
    for relative in files:
        target = slot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = fetch_file(
                repository, commit, relative, timeout, blob=blobs[relative]
            )
        except (urllib.error.URLError, TimeoutError) as why:
            raise AssetFetchError(
                f"{relative} at {repository}@{commit[:7]}: {why}"
            ) from why
        part = target.with_name(target.name + ".part")
        part.write_bytes(data)
        part.replace(target)
    fetched = FetchedAsset(repository, commit, path, slot / path, files)
    (slot / MARKER_FILE).write_text(
        json.dumps(fetched.marker, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    return fetched
