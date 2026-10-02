"""The asset fetcher (`robot/asset_fetch.py`) with no network: a fake
GitHub answers the tree listing and the raw and media hosts, so what is
kept, checked and refused is pinned on any machine (review 2026-09-24:
a truncated listing was taken as the whole asset, and no file was
checked against the id the tree lists)."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from trainnr.robot import asset_fetch
from trainnr.robot.asset_fetch import (
    LFS_POINTER_HEAD,
    AssetFetchError,
    cached_tree,
    fetch_tree,
    git_blob_id,
)

REPO, COMMIT, PATH = "owner/assets", "0123456789abcdef", "robot"
LAYER = b"#usda 1.0\n(defaultPrim = 'robot')\n"
MESH = b"\x00\x01mesh bytes" * 64


def lfs_pointer(data: bytes) -> bytes:
    return (
        LFS_POINTER_HEAD
        + f"\noid sha256:{hashlib.sha256(data).hexdigest()}\n".encode()
        + f"size {len(data)}\n".encode()
    )


def fake_github(
    files: dict[str, bytes], *, lfs: dict[str, bytes] | None = None, **listing: Any
) -> Any:
    """`_get` over a fake GitHub: the tree lists `files` (an LFS file lists
    its pointer, the way git stores it), raw serves them, media serves
    the LFS bytes."""
    lfs = lfs or {}
    tree = [
        {"path": f"{PATH}/{name}", "type": "blob", "sha": git_blob_id(raw)}
        for name, raw in files.items()
    ]

    def get(url: str, _timeout: float) -> bytes:
        if "/git/trees/" in url:
            return json.dumps({"tree": tree, **listing}).encode()
        name = url.rsplit("/", 1)[1]
        if "media.githubusercontent" in url:
            return lfs[name]
        return files[name]

    return get


class TheTreeIsWholeAndChecked(unittest.TestCase):
    def test_every_file_lands_and_the_marker_is_written_last(self) -> None:
        mesh_pointer = lfs_pointer(MESH)
        get = fake_github(
            {"robot.usda": LAYER, "mesh.bin": mesh_pointer}, lfs={"mesh.bin": MESH}
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(asset_fetch, "_get", get),
        ):
            self.assertIsNone(cached_tree(REPO, COMMIT, PATH, cache=Path(tmp)))
            fetched = fetch_tree(REPO, COMMIT, PATH, cache=Path(tmp))
            self.assertEqual((fetched.root / "robot.usda").read_bytes(), LAYER)
            self.assertEqual((fetched.root / "mesh.bin").read_bytes(), MESH)
            self.assertIsNotNone(cached_tree(REPO, COMMIT, PATH, cache=Path(tmp)))

    def test_a_truncated_listing_is_refused_by_name(self) -> None:
        get = fake_github({"robot.usda": LAYER}, truncated=True)
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(asset_fetch, "_get", get),
            self.assertRaisesRegex(AssetFetchError, "truncated"),
        ):
            fetch_tree(REPO, COMMIT, PATH, cache=Path(tmp))

    def test_bytes_that_are_not_the_listed_blob_are_refused(self) -> None:
        get = fake_github({"robot.usda": LAYER})

        def tampered(url: str, timeout: float) -> bytes:
            data = get(url, timeout)
            return data if "/git/trees/" in url else data + b"x"

        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(asset_fetch, "_get", tampered),
        ):
            with self.assertRaisesRegex(AssetFetchError, "not the blob"):
                fetch_tree(REPO, COMMIT, PATH, cache=Path(tmp))
            self.assertIsNone(cached_tree(REPO, COMMIT, PATH, cache=Path(tmp)))

    def test_lfs_media_that_is_not_the_pointers_file_is_refused(self) -> None:
        get = fake_github({"mesh.bin": lfs_pointer(MESH)}, lfs={"mesh.bin": MESH[:-1]})
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(asset_fetch, "_get", get),
            self.assertRaisesRegex(AssetFetchError, "LFS media"),
        ):
            fetch_tree(REPO, COMMIT, PATH, cache=Path(tmp))


if __name__ == "__main__":
    unittest.main()
