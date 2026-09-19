"""The headless window (docs/76 §10.5): every feed opens through one seam
that saves its stream inside the artifact it narrates, outside the
artifact's hash; the saved file is read back with no window; Show in
viewer replays it."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from rq_pipeline.bundles.hashing import bundle_hash
from rq_pipeline.project import PROJECT_ENV, create_project, index_project
from rq_pipeline.project.kinds import Kind
from rq_pipeline.project.viewer import (
    COMPONENTS,
    ENTITIES,
    NEEDS_QUERY,
    SIZE,
    TIMELINES,
    describe,
    parse_stats,
    query_available,
)
from rq_pipeline.viz import (
    STUDIO_ADDRESS,
    VIEWER_DIR,
    VIEWER_FILE_ENV,
    open_stream,
    sinks,
    studio_listening,
    viewer_file,
    viewer_file_wanted,
    viewer_files,
)

try:
    import rerun as rr
except ImportError:  # pragma: no cover - the viz extra is absent
    rr = None

REPO = Path(__file__).resolve().parents[2]
BUNDLE = REPO / "robots" / "rig-drivetrain"

STATS_PAGE = """Overview
--------
num_chunks = 9
num_rows = 101

Num chunks per entity
---------------------
/gate/command: 1
/gate/measured: 1

Num chunks per index
--------------------
log_time: 7
tick: 3

Num chunks per component
------------------------
Scalars:scalars: 2

Size (schema + data, compressed)
--------------------------------
ipc_size_bytes_total = 13.1 KiB
ipc_size_bytes_min = 910 B
"""


class _FakeRerun(types.ModuleType):
    """Enough of the SDK to see what the seam does with it."""

    def __init__(self) -> None:
        super().__init__("rerun")
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def init(self, app_id: str, **kwargs: object) -> None:
        self.calls.append(("init", (app_id,), kwargs))

    def connect_grpc(self, url: str) -> None:
        self.calls.append(("connect_grpc", (url,), {}))

    def set_sinks(self, *sinks: object) -> None:
        self.calls.append(("set_sinks", sinks, {}))

    def disconnect(self) -> None:
        self.calls.append(("disconnect", (), {}))

    class GrpcSink:
        def __init__(self, url: str) -> None:
            self.url = url

    class FileSink:
        def __init__(self, path: str) -> None:
            self.path = path


class TheSeam(unittest.TestCase):
    def test_the_file_is_hidden_inside_the_artifact_and_outside_its_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "art"
            folder.mkdir()
            (folder / "record.json").write_text("{}")
            before = bundle_hash(folder)
            path = viewer_file(folder, "train")
            self.assertEqual(path, folder / VIEWER_DIR / "train.rrd")
            path.parent.mkdir()
            path.write_bytes(b"\x00" * 64)
            self.assertEqual(
                bundle_hash(folder), before, "a picture never moves a version"
            )
            self.assertEqual(viewer_files(folder), [path])
            self.assertEqual(viewer_files(Path(tmp) / "nowhere"), [])

    def test_both_sinks_when_a_file_is_named_and_wanted(self) -> None:
        fake = _FakeRerun()
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "a" / "b" / "x.rrd"
            with mock.patch.dict(os.environ, {VIEWER_FILE_ENV: "1"}):
                chosen = sinks(fake, file=file, listening=True)
            self.assertEqual(
                [type(s).__name__ for s in chosen], ["GrpcSink", "FileSink"]
            )
            self.assertEqual(chosen[0].url, STUDIO_ADDRESS)
            self.assertEqual(chosen[1].path, str(file))
            self.assertTrue(file.parent.is_dir(), "the folder is made for the file")
            with mock.patch.dict(os.environ, {VIEWER_FILE_ENV: "off"}):
                self.assertFalse(viewer_file_wanted())
                self.assertEqual(len(sinks(fake, file=file, listening=True)), 1)
            self.assertEqual(len(sinks(fake, file=None, listening=True)), 1)

    def test_no_studio_and_a_file_means_the_file_alone(self) -> None:
        """A feed toward a viewer that never answers fills a bounded queue
        and the shutdown flush waits forever (2026-09-13); with a file to
        save to, the server sink is left out. Without a file the server
        sink stays, as every feed behaved before."""
        fake = _FakeRerun()
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "x.rrd"
            with mock.patch.dict(os.environ, {VIEWER_FILE_ENV: "1"}):
                only = sinks(fake, file=file, listening=False)
                self.assertEqual([type(s).__name__ for s in only], ["FileSink"])
                self.assertEqual(
                    [type(s).__name__ for s in sinks(fake, file=None, listening=False)],
                    ["GrpcSink"],
                )

    def test_listening_is_a_real_socket_question(self) -> None:
        import socket  # noqa: PLC0415

        with socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            port = server.getsockname()[1]
            self.assertTrue(studio_listening(f"rerun+http://127.0.0.1:{port}/proxy"))
        self.assertFalse(studio_listening(f"rerun+http://127.0.0.1:{port}/proxy"))

    def test_open_stream_connects_or_sets_both_sinks(self) -> None:
        fake = _FakeRerun()
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.dict(sys.modules, {"rerun": fake}),
            mock.patch.dict(os.environ, {VIEWER_FILE_ENV: "1"}),
        ):
            with mock.patch("rq_pipeline.viz.studio_listening", return_value=True):
                got = open_stream("app", file=None, on_term=False)
                self.assertIs(got, fake)
                self.assertEqual(fake.calls[-1][0], "connect_grpc")
                file = Path(tmp) / ".viewer" / "s.rrd"
                open_stream("app", file=file, recording_id="r1", on_term=False)
            self.assertEqual(
                fake.calls[-2],
                ("init", ("app",), {"spawn": False, "recording_id": "r1"}),
            )
            name, chosen, _ = fake.calls[-1]
            self.assertEqual(name, "set_sinks")
            self.assertEqual(len(chosen), 2)


class TheReader(unittest.TestCase):
    def test_the_stats_page_parses_by_section(self) -> None:
        sections = parse_stats(STATS_PAGE)
        self.assertEqual(
            sections[ENTITIES], {"/gate/command": "1", "/gate/measured": "1"}
        )
        self.assertEqual(sections[TIMELINES], {"log_time": "7", "tick": "3"})
        self.assertEqual(sections[COMPONENTS], {"Scalars:scalars": "2"})
        self.assertEqual(
            sections[SIZE],
            {"ipc_size_bytes_total": "13.1 KiB", "ipc_size_bytes_min": "910 B"},
        )
        self.assertEqual(sections["Overview"], {"num_chunks": "9", "num_rows": "101"})

    @unittest.skipUnless(rr is not None, "the viz extra")
    def test_a_saved_stream_is_described_with_no_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gate.rrd"
            stream = rr.RecordingStream("trainnr-test", recording_id="t1")
            stream.set_sinks(rr.FileSink(str(path)))
            for i in range(20):
                stream.set_time("tick", sequence=i)
                stream.log("gate/measured", rr.Scalars([0.1 * i]))
                stream.log("gate/pair", rr.Scalars([0.1 * i, 5.0]))
            stream.log("gate/reading", rr.TextDocument("# r"), static=True)
            stream.flush(timeout_sec=5)
            stream.disconnect()
            self.assertGreater(path.stat().st_size, 0)
            d = describe(path)
            self.assertIn("/gate/measured", d.entities)
            self.assertIn("tick", d.timelines)
            self.assertIn("Scalars:scalars", d.components)
            self.assertIn("ipc_size_bytes_total", d.size)
            if query_available():
                self.assertEqual(d.rows_per_timeline["tick"], 20)
                [series] = [s for s in d.series if s.entity == "/gate/measured"]
                self.assertEqual(
                    (series.timeline, series.rows, series.width), ("tick", 20, 1)
                )
                self.assertAlmostEqual((series.last or (0.0,))[0], 1.9)
                # Every component counted, not the first alone.
                [pair] = [s for s in d.series if s.entity == "/gate/pair"]
                self.assertEqual(pair.width, 2)
                self.assertEqual(pair.maximum, 5.0)
                self.assertEqual(d.notes, ())
            else:
                self.assertIn(NEEDS_QUERY, d.notes)
            with self.assertRaisesRegex(FileNotFoundError, "no such"):
                describe(Path(tmp) / "none.rrd")


@unittest.skipUnless(rr is not None, "the viz extra")
class TheDoorAndTheWindow(unittest.TestCase):
    def _project(self, tmp: Path):
        project = create_project(tmp / "p", "p")
        bundle = project.folder("robots") / "rig-drivetrain"
        bundle.mkdir(parents=True)
        for name in ("model.xml", "profile.json", "README.md"):
            shutil.copy2(BUNDLE / name, bundle / name)
        path = viewer_file(bundle, "session")
        stream = rr.RecordingStream("trainnr-test", recording_id="t2")
        path.parent.mkdir()
        stream.set_sinks(rr.FileSink(str(path)))
        stream.log("rig/note", rr.TextDocument("hi"), static=True)
        stream.flush(timeout_sec=5)
        stream.disconnect()
        return project, path

    def test_the_door_lists_and_describes_the_files(self) -> None:
        from rq_pipeline.mcp_server import describe_viewer_recording  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            project, path = self._project(Path(tmp))
            os.environ[PROJECT_ENV] = str(project.root)
            try:
                robot = index_project(project).by_kind(Kind.ROBOT)[0].stamp
                out = describe_viewer_recording(robot, values=False)
                self.assertEqual(out["status"], "done")
                self.assertEqual(out["kind"], "robot")
                [rec] = out["recordings"]
                self.assertEqual(rec["path"], str(path))
                self.assertIn("/rig/note", rec["entities"])
                self.assertIn("1 saved stream", out["note"])
                self.assertEqual(
                    describe_viewer_recording("nobody@000000000000")["status"],
                    "refused",
                )
            finally:
                os.environ.pop(PROJECT_ENV, None)

    def test_show_in_viewer_replays_the_saved_stream_first(self) -> None:
        from rq_pipeline.project import present as presenting  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            project, path = self._project(Path(tmp))
            robot = index_project(project).by_kind(Kind.ROBOT)[0].stamp
            replayed: list[str] = []
            real = rr.RecordingStream

            class _Stream(real):  # type: ignore[misc, valid-type]
                def connect_grpc(self, *a: object, **k: object) -> None:
                    self.memory_recording()  # no Studio under test

                def log_file_from_path(
                    self, p: object, *a: object, **k: object
                ) -> None:
                    replayed.append(str(p))

            with mock.patch.object(rr, "RecordingStream", _Stream):
                shown = presenting.present(project, robot)
            self.assertEqual(replayed, [str(path)])
            self.assertEqual(shown["viewer_files"], [str(path)])


if __name__ == "__main__":
    unittest.main()
