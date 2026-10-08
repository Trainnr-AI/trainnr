"""A sample project is opened checked, once, and never over a user's
folder (the onboarding work, 2026-10-09); offline, with the download
served from memory."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from trainnr import samples


def _archive(folder: str, extra: dict[str, bytes] | None = None) -> bytes:
    """A tar.gz holding `<folder>/project.json` and any extra members."""
    members = {f"{folder}/project.json": json.dumps({"name": folder}).encode()}
    members.update(extra or {})
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


class _Reply(io.BytesIO):
    def __enter__(self) -> _Reply:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@contextmanager
def _served(body: bytes, *, listed: bytes | None = None):
    """SAMPLES holds one sample listing `listed` (default: `body`), and
    every download answers `body`."""
    listed = body if listed is None else listed
    sample = samples.Sample(
        name="demo",
        title="Demo",
        summary="",
        path="samples/demo.tar.gz",
        project="demo-sample",
        sha256=hashlib.sha256(listed).hexdigest(),
        bytes=len(listed),
    )
    opener = mock.Mock()
    opener.open = lambda *_a, **_k: _Reply(body)
    with (
        mock.patch.dict(samples.SAMPLES, {"demo": sample}, clear=True),
        mock.patch.object(samples, "_OPENER", opener),
    ):
        yield sample


class OpeningASample(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.home = Path(self._tmp.name) / "projects"

    def test_it_unpacks_into_the_projects_home_once(self) -> None:
        with _served(_archive("demo-sample")):
            root = samples.install("demo", self.home)
            self.assertEqual(root, self.home / "demo-sample")
            self.assertTrue((root / "project.json").is_file())
            self.assertEqual(samples.install("demo", self.home), root)
        self.assertEqual(sorted(p.name for p in self.home.iterdir()), ["demo-sample"])

    def test_a_download_that_is_not_the_listed_one_is_refused(self) -> None:
        listed = _archive("demo-sample")
        with (
            _served(
                _archive("demo-sample", {"demo-sample/x": b"tampered"}), listed=listed
            ),
            self.assertRaisesRegex(samples.SampleError, "listed"),
        ):
            samples.install("demo", self.home)
        self.assertFalse((self.home / "demo-sample").exists())

    def test_a_folder_of_the_users_with_the_samples_name_is_left_alone(self) -> None:
        mine = self.home / "demo-sample"
        mine.mkdir(parents=True)
        (mine / "notes.txt").write_text("mine")
        with (
            _served(_archive("demo-sample")),
            self.assertRaisesRegex(samples.SampleError, "is not the sample"),
        ):
            samples.install("demo", self.home)
        self.assertEqual((mine / "notes.txt").read_text(), "mine")

    def test_a_member_that_leaves_the_folder_is_refused(self) -> None:
        escaping = _archive("demo-sample", {"../outside.txt": b"x"})
        with (
            _served(escaping),
            self.assertRaises((samples.SampleError, tarfile.TarError)),
        ):
            samples.install("demo", self.home)
        self.assertFalse((self.home.parent / "outside.txt").exists())

    def test_an_unknown_name_is_refused_by_name(self) -> None:
        with self.assertRaisesRegex(KeyError, "no sample 'nope'"):
            samples.resolve("nope")


if __name__ == "__main__":
    unittest.main()
