"""A preview redraws when its artifact's files are newer than the
picture: a run keeps one identity stamp for life, so the stamp cannot
say the curve changed (2026-09-12)."""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

from trainnr.project.previews import stale_preview


class StalePreview(unittest.TestCase):
    def test_newer_source_files_make_the_picture_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "runs" / "r"
            run.mkdir(parents=True)
            record = run / "training.json"
            record.write_text("{}")
            picture = Path(tmp) / "r.png"
            picture.write_bytes(b"png")
            now = time.time()
            os.utime(record, (now - 100, now - 100))
            os.utime(picture, (now - 50, now - 50))
            self.assertFalse(stale_preview(run, picture))
            os.utime(record, (now, now))  # the record grew after the picture
            self.assertTrue(stale_preview(run, picture))
            (run / ".index").mkdir()
            (run / ".index" / "x").write_text("hidden files do not count")
            os.utime(record, (now - 100, now - 100))
            self.assertFalse(stale_preview(run, picture))

    def test_an_empty_artifact_is_never_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "empty"
            empty.mkdir()
            picture = Path(tmp) / "e.png"
            picture.write_bytes(b"png")
            self.assertFalse(stale_preview(empty, picture))


if __name__ == "__main__":
    unittest.main()
