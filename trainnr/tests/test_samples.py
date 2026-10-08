"""A sample project is opened checked, once, and never over a user's
folder (the onboarding work, 2026-10-09); offline, with the download
served from memory."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
import tempfile
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from trainnr import cli, samples
from trainnr.project import locate


def _archive(folder: str, extra: dict[str, bytes] | None = None) -> bytes:
    """A tar.gz holding `<folder>/project.json` and any extra members."""
    manifest = {"schema": "trainnr-project/1", "name": folder, "created": ""}
    members = {f"{folder}/project.json": json.dumps(manifest).encode()}
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


class TheStudiosOpenButton(unittest.TestCase):
    """`trainnr sample open` is what the Studio's Open runs: it prints the
    folder (the Studio switches to it) and makes it current (a Studio on
    its Welcome page follows that), or names why on its last line."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.home = Path(self._tmp.name) / "projects"
        env = mock.patch.dict(os.environ, {"TRAINNR_PROJECTS": str(self.home)})
        env.start()
        self.addCleanup(env.stop)
        session = mock.patch.object(locate, "_session_root", None)
        session.start()
        self.addCleanup(session.stop)

    def test_it_prints_the_folder_and_makes_it_current(self) -> None:
        out = io.StringIO()
        with _served(_archive("demo-sample")), redirect_stdout(out):
            self.assertEqual(cli.main(["sample", "open", "demo"]), 0)
        root = self.home / "demo-sample"
        self.assertEqual(out.getvalue().strip(), str(root))
        self.assertEqual(locate.remembered_project(), root)
        self.assertTrue((root / ".index" / "project.json").is_file())

    def test_a_failure_is_one_line_naming_why(self) -> None:
        err = io.StringIO()
        with redirect_stderr(err):
            self.assertEqual(cli.main(["sample", "open", "nope"]), 1)
        self.assertEqual(
            err.getvalue().strip(),
            "trainnr sample open: no sample 'nope'; the samples are ['go2-walk']",
        )

    def test_list_names_each_sample_and_where_it_is(self) -> None:
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(cli.main(["sample", "list"]), 0)
        rows = json.loads(out.getvalue())
        self.assertEqual([r["name"] for r in rows], sorted(samples.SAMPLES))
        self.assertIsNone(rows[0]["installed"])


if __name__ == "__main__":
    unittest.main()
