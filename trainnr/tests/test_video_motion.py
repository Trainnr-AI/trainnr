"""Video to a G1 motion, as the tools refuse it (docs/86 §5.3): nothing
starts without GEM-X and the retargeter installed, without the footage,
or over an existing recording; tracking takes only a reference."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests._extras import needs_mcp
from trainnr.project import create_project
from trainnr.project.locate import PROJECT_ENV
from trainnr.video import nvidia


class TheInstalls(unittest.TestCase):
    def test_a_missing_install_names_the_file_and_the_setup_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = {nvidia.GEMX_ENV: tmp, nvidia.SOMA_ENV: tmp}
            with mock.patch.dict(os.environ, env):
                why = nvidia.missing()
        assert why is not None
        self.assertIn("GEM-X is not installed", why)
        self.assertIn(nvidia.SETUP, why)

    def test_both_installs_complete_are_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for root, needs in (("g", nvidia.GEMX_NEEDS), ("s", nvidia.SOMA_NEEDS)):
                for need in needs:
                    (Path(tmp) / root / need).parent.mkdir(parents=True, exist_ok=True)
                    (Path(tmp) / root / need).touch()
            env = {nvidia.GEMX_ENV: f"{tmp}/g", nvidia.SOMA_ENV: f"{tmp}/s"}
            with mock.patch.dict(os.environ, env):
                self.assertIsNone(nvidia.missing())


@needs_mcp
class TheTools(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.project = create_project(Path(self._tmp.name) / "p", "p")
        env = mock.patch.dict(os.environ, {PROJECT_ENV: str(self.project.root)})
        env.start()
        self.addCleanup(env.stop)

    def test_unknown_footage_is_refused_by_name(self) -> None:
        from trainnr import mcp_server  # noqa: PLC0415

        answer = mcp_server.retarget_video("no-such-clip")
        self.assertEqual(answer["status"], "refused")
        self.assertIn("no-such-clip", answer["reason"])

    def test_tracking_takes_a_reference_not_any_recording(self) -> None:
        from trainnr import mcp_server  # noqa: PLC0415

        folder = self.project.recordings / "plain"
        folder.mkdir(parents=True)
        (folder / "recording.json").write_text(
            json.dumps(
                {
                    "schema": "trainnr-recording/1",
                    "source": "x",
                    "adapter": "wire",
                    "duration_s": 0.0,
                    "channels": [],
                }
            )
        )
        answer = mcp_server.train_tracking("plain")
        self.assertEqual(answer["status"], "refused")
        self.assertIn("retarget_video", answer["reason"])
        self.assertEqual(
            mcp_server.train_tracking("plain", iterations=0)["status"], "refused"
        )


if __name__ == "__main__":
    unittest.main()
