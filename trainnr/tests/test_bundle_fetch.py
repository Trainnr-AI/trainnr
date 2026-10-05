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
from trainnr.bundles.bundle import model_file_for_use, model_file_of
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


def _manifest(bundle: Path, rel: str = "assets/a.stl", **fields: str) -> None:
    (bundle / fetch.FETCH_FILE).write_text(
        json.dumps(
            {
                "schema": fetch.FETCH_SCHEMA,
                "repository": "owner/repo",
                "commit": "0" * 40,
                "licence": "test",
                "files": {rel: {"upstream": "m/a.stl", "blob": git_blob_id(MESH)}},
                **fields,
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
    def setUp(self) -> None:
        # the user's own TRAINNR_NO_ROBOT_FETCH must not decide these tests
        patch = mock.patch.dict(os.environ)
        patch.start()
        self.addCleanup(patch.stop)
        os.environ.pop(fetch.NO_FETCH_ENV, None)

    def test_a_missing_file_keeps_the_recorded_stamp(self) -> None:
        """The stamp is one hash over every file, so without the fetched
        ones it cannot be computed: the manifest records it, and a change
        to a carried file makes it computed again (review, 2026-10-05)."""
        from trainnr.bundles.hashing import carried_hash  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            whole = _bundle(Path(tmp) / "whole", carried=True)
            want = stamp("bot", whole)
            bundle = _bundle(Path(tmp) / "fetched", carried=False)
            _manifest(bundle)
            manifest = json.loads((bundle / fetch.FETCH_FILE).read_text())
            manifest["stamp"] = want
            manifest["carried"] = carried_hash(bundle, tuple(manifest["files"]))
            (bundle / fetch.FETCH_FILE).write_text(json.dumps(manifest))
            self.assertEqual(stamp("bot", bundle), want)
            (bundle / "bot.xml").write_text("<mujoco><!-- edited --></mujoco>")
            self.assertNotEqual(stamp("bot", bundle), want)

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
            self.assertEqual(model_file_of(bundle), bundle / "bot.xml")  # names only
            self.assertEqual(calls, [])
            with mock.patch("trainnr.robot.asset_fetch.fetch_file", _upstream(calls)):
                self.assertEqual(model_file_for_use(bundle), bundle / "bot.xml")
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

    def test_a_manifest_cannot_write_outside_its_bundle(self) -> None:
        """A bundle from someone else's project named `../../.bashrc` and
        the file was written there (2026-10-05)."""
        for rel in (
            "../escape.stl",
            "/tmp/escape.stl",
            "assets/../../x.stl",
            "a\\b.stl",
            "C:/x.stl",
            "",
        ):
            with self.subTest(rel=rel), tempfile.TemporaryDirectory() as tmp:
                bundle = _bundle(Path(tmp), carried=False)
                _manifest(bundle, rel)
                with self.assertRaisesRegex(ValueError, "plain path"):
                    fetch.ensure_fetched(bundle)

    def test_a_link_out_of_the_bundle_is_not_written_through(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outside = Path(tmp) / "outside"
            outside.mkdir()
            bundle = Path(tmp) / "bot"
            bundle.mkdir()
            (bundle / "assets").symlink_to(outside, target_is_directory=True)
            _manifest(bundle)
            calls: list[tuple[str, str, object]] = []
            with (
                mock.patch("trainnr.robot.asset_fetch.fetch_file", _upstream(calls)),
                self.assertRaisesRegex(ValueError, "outside the bundle"),
            ):
                fetch.ensure_fetched(bundle)
            self.assertEqual(calls, [])
            self.assertEqual(list(outside.iterdir()), [])

    def test_the_source_is_named_by_its_shape(self) -> None:
        for field, value in (("repository", "owner/repo/../x"), ("commit", "main")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                bundle = _bundle(Path(tmp), carried=False)
                _manifest(bundle, **{field: value})
                with self.assertRaises(ValueError):
                    fetch.ensure_fetched(bundle)

    def test_a_bundle_without_a_manifest_fetches_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = _bundle(Path(tmp), carried=True)
            self.assertEqual(fetch.ensure_fetched(bundle), [])


if __name__ == "__main__":
    unittest.main()
