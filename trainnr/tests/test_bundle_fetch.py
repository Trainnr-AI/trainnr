"""A bundle that does not carry some files fetches them on first use,
each checked by blob id, and its stamp is the one taken over the carried
files (the microduck's meshes, 2026-10-05: their licence keeps them out)."""

import json
import os
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from unittest import mock

from trainnr.bundles import fetch
from trainnr.bundles.bundle import model_file_of
from trainnr.bundles.hashing import stamp
from trainnr.bundles.locate import require_bundle_file
from trainnr.robot.asset_fetch import git_blob_id

MESH = b"solid mesh\nendsolid\n"


def _bundle(root: Path, *, carried: bool) -> Path:
    bundle = root / "bot"
    (bundle / "assets").mkdir(parents=True)
    (bundle / "bot.xml").write_text("<mujoco/>")
    if carried:
        (bundle / "assets" / "a.stl").write_bytes(MESH)
    return bundle


def _manifest(bundle: Path) -> None:
    (bundle / fetch.FETCH_FILE).write_text(
        json.dumps(
            {
                "schema": fetch.FETCH_SCHEMA,
                "repository": "owner/repo",
                "commit": "0" * 40,
                "licence": "test",
                "files": {
                    "assets/a.stl": {"upstream": "m/a.stl", "blob": git_blob_id(MESH)}
                },
            }
        )
    )


def _upstream(calls: list[tuple[str, str, object]]) -> Callable[..., bytes]:
    """A stand-in for `asset_fetch.fetch_file` that records each call."""

    def fake(repo: str, commit: str, path: str, **kw: object) -> bytes:
        calls.append((repo, path, kw.get("blob")))
        return MESH

    return fake


class FetchingWhatABundleLacks(unittest.TestCase):
    def test_the_stamp_is_the_carried_bundles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            carried = _bundle(Path(tmp) / "carried", carried=True)
            fetched = _bundle(Path(tmp) / "fetched", carried=False)
            _manifest(fetched)
            calls: list[tuple[str, str, object]] = []
            with mock.patch("trainnr.robot.asset_fetch.fetch_file", _upstream(calls)):
                self.assertEqual(fetch.missing_files(fetched), ["assets/a.stl"])
                require_bundle_file(fetched / "bot.xml")  # the first use fetches
            self.assertEqual(calls, [("owner/repo", "m/a.stl", git_blob_id(MESH))])
            self.assertEqual(fetch.missing_files(fetched), [])
            self.assertEqual(stamp("bot", fetched), stamp("bot", carried))

    def test_resolving_the_model_fetches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = _bundle(Path(tmp), carried=False)
            _manifest(bundle)
            calls: list[tuple[str, str, object]] = []
            with mock.patch("trainnr.robot.asset_fetch.fetch_file", _upstream(calls)):
                self.assertEqual(model_file_of(bundle), bundle / "bot.xml")
            self.assertEqual(len(calls), 1)
            self.assertEqual(fetch.missing_files(bundle), [])

    def test_a_refused_fetch_names_the_way_out(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = _bundle(Path(tmp), carried=False)
            _manifest(bundle)
            with (
                mock.patch.dict(os.environ, {fetch.NO_FETCH_ENV: "1"}),
                self.assertRaisesRegex(FileNotFoundError, "trainnr.bundles.fetch"),
            ):
                fetch.ensure_fetched(bundle)

    def test_a_bundle_without_a_manifest_fetches_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = _bundle(Path(tmp), carried=True)
            self.assertEqual(fetch.ensure_fetched(bundle), [])


if __name__ == "__main__":
    unittest.main()
