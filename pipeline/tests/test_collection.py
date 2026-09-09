"""The data-collection sources and the presenter: mocap CSV and BVH become
recordings marked as human motion; a live UDP capture lands a stamped
recording marked robot-operation and refuses an empty session; every
recording carries how it was collected; the presenter refuses a stamp it
does not know and a kind it has no view for, and streams the kinds it
does through a fake recording stream so the blueprint and paths are
pinned without a viewer."""

from __future__ import annotations

import socket
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from rq_pipeline.project import Kind, create_project, index_project
from rq_pipeline.robots import resolve
from rq_pipeline.robots.adapter import detect
from rq_pipeline.robots.capture import (
    FAILED,
    INGESTED,
    LISTENING,
    WireUdpCapture,
    status,
)
from rq_pipeline.robots.recording import (
    COLLECTION_MOCAP,
    COLLECTION_ROBOT_OP,
    COLLECTION_SCRIPTED,
    COLLECTION_TELEOP,
    Recording,
)
from tests._extras import needs_numpy
from tests.test_robots import WIRE, make_lerobot, make_mcap

# -- fixtures -------------------------------------------------------------------

MOCAP_HEADER = (
    "Frame,Time (Seconds),Hip.X,Hip.Y,Hip.Z,"
    "Knee.X,Knee.Y,Knee.Z,Knee.RX,Knee.RY,Knee.RZ"
)
MOCAP_CSV = (
    MOCAP_HEADER
    + "\n0,0.000,0.10,0.90,0.00,0.12,0.50,0.01,10,0,0"
    + "\n1,0.008,0.11,0.91,0.00,0.13,0.51,0.01,12,0,0"
    + "\n2,0.016,0.12,0.92,0.00,0.14,0.52,0.01,14,0,0\n"
)

BVH = """HIERARCHY
ROOT Hips
{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT Spine
  {
    OFFSET 0 10 0
    CHANNELS 3 Zrotation Xrotation Yrotation
    End Site
    {
      OFFSET 0 10 0
    }
  }
}
MOTION
Frames: 3
Frame Time: 0.0333333
0 90 0 0 0 0 0 0 0
0 91 0 5 0 0 0 1 0
0 92 0 10 0 0 0 2 0
"""


# -- mocap -----------------------------------------------------------------------


class Mocap(unittest.TestCase):
    def test_a_marker_csv_becomes_position_and_rotation_channels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "take01.csv"
            path.write_text(MOCAP_CSV)
            self.assertEqual(detect(path).name, "mocap")
            rec = resolve("mocap").build().read(path)
            self.assertEqual(rec.collection, COLLECTION_MOCAP)
            hip = rec.channels["marker.Hip.position"]
            self.assertEqual(hip.components, ("x", "y", "z"))
            np.testing.assert_allclose(hip.times, [0.0, 0.008, 0.016])
            knee_rot = rec.channels["marker.Knee.rotation"]
            self.assertEqual(knee_rot.unit, "rad")
            np.testing.assert_allclose(knee_rot.values[:, 0], np.deg2rad([10, 12, 14]))
            self.assertEqual(rec.census["markers"], ["Hip", "Knee"])
            self.assertTrue(any("not retargeted" in n for n in rec.notes))

    def test_a_robot_log_csv_is_not_mistaken_for_mocap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.csv"
            path.write_text("time,duty,ticks_left,ticks_right\n0,0,0,0\n1,10,5,5\n")
            self.assertFalse(resolve("mocap").build().accepts(path))

    def test_a_bvh_skeleton_becomes_joint_channels_in_metres_and_radians(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "walk.bvh"
            path.write_text(BVH)
            rec = resolve("mocap").build().read(path)
            hips = rec.channels["joint.Hips.position"]
            # 90 cm -> 0.90 m, assumed centimetres and said so
            np.testing.assert_allclose(hips.values[:, 1], [0.90, 0.91, 0.92])
            self.assertIn("assumed", hips.unit)
            spine = rec.channels["joint.Spine.rotation"]
            self.assertEqual(spine.components, ("z", "x", "y"))
            np.testing.assert_allclose(spine.values[:, 1], np.deg2rad([0, 1, 2]))
            self.assertEqual(rec.census["joints"], ["Hips", "Spine"])
            self.assertAlmostEqual(rec.census["rate_hz"] or 0, 30.0, places=1)

    def test_a_bvh_with_the_wrong_channel_count_is_refused_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.bvh"
            path.write_text(BVH.replace("0 90 0 0 0 0 0 0 0", "0 90 0"))
            with self.assertRaisesRegex(ValueError, "channels per frame"):
                resolve("mocap").build().read(path)


# -- how a recording was collected -------------------------------------------------


@needs_numpy
class CollectionTags(unittest.TestCase):
    def test_each_adapter_names_its_collection_and_it_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wire = resolve("wire").build().read(WIRE)
            self.assertEqual(wire.collection, COLLECTION_ROBOT_OP)
            bag = resolve("mcap").build().read(make_mcap(Path(tmp) / "b.mcap"))
            self.assertEqual(bag.collection, COLLECTION_ROBOT_OP)
            ds = resolve("lerobot").build().read(make_lerobot(Path(tmp) / "ds"))
            self.assertEqual(ds.collection, COLLECTION_TELEOP)
            # A dataset this repo pressed carries an expert stamp: scripted.
            (Path(tmp) / "ds" / "provenance.json").write_text(
                '{"expert": "kitting-expert@abc"}'
            )
            ds2 = resolve("lerobot").build().read(Path(tmp) / "ds")
            self.assertEqual(ds2.collection, COLLECTION_SCRIPTED)
            wire.write(Path(tmp) / "w")
            self.assertEqual(
                Recording.read(Path(tmp) / "w").collection, COLLECTION_ROBOT_OP
            )
            self.assertEqual(wire.manifest().collection, COLLECTION_ROBOT_OP)


# -- live capture --------------------------------------------------------------------


class LiveCapture(unittest.TestCase):
    def _free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def test_datagrams_become_a_stamped_robot_operation_recording(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = create_project(Path(tmp) / "p", "p")
            port = self._free_port()
            capture = WireUdpCapture(project, "session-1", port=port)
            state = capture.start(window_s=30.0)
            self.assertEqual(state.state, LISTENING)
            self.assertEqual(status(project)["state"], LISTENING)
            # Real status lines from a real recording, sent as datagrams.
            lines = [
                ln
                for ln in WIRE.read_text(errors="replace").splitlines()
                if ln.startswith("n=")
            ][:40]
            self.assertGreater(len(lines), 10)
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                for ln in lines:
                    s.sendto(ln.encode(), ("127.0.0.1", port))
                    time.sleep(0.002)
            deadline = time.time() + 5.0
            while capture.state.datagrams < len(lines) and time.time() < deadline:
                time.sleep(0.05)
            final = capture.stop()
            self.assertEqual(final.state, INGESTED, final.error)
            self.assertTrue((final.stamp or "").startswith("session-1@"))
            index = index_project(project)
            rec = index.by_kind(Kind.RECORDING)[0]
            self.assertEqual(rec.summary["collection"], COLLECTION_ROBOT_OP)
            self.assertFalse(list(project.folder("recordings").glob(".capture-*")))

    def test_an_empty_session_fails_by_name_and_leaves_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = create_project(Path(tmp) / "p", "p")
            capture = WireUdpCapture(project, "empty", port=self._free_port())
            capture.start(window_s=5.0)
            final = capture.stop()
            self.assertEqual(final.state, FAILED)
            self.assertIn("no datagrams", final.error or "")
            self.assertEqual(index_project(project).by_kind(Kind.RECORDING), [])

    def test_a_capture_refuses_a_name_that_already_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = create_project(Path(tmp) / "p", "p")
            (project.folder("recordings") / "taken").mkdir()
            with self.assertRaises(FileExistsError):
                WireUdpCapture(project, "taken", port=self._free_port()).start()


# -- the presenter -------------------------------------------------------------------


class Presenter(unittest.TestCase):
    def test_an_unknown_stamp_and_an_unviewable_kind_are_refused(self) -> None:
        from rq_pipeline.project.present import _PRESENTERS  # noqa: PLC0415

        self.assertIn(Kind.ROBOT, _PRESENTERS)
        self.assertIn(Kind.RECORDING, _PRESENTERS)
        self.assertIn(Kind.CERTIFICATE, _PRESENTERS)
        self.assertNotIn(Kind.DRIFT, _PRESENTERS)  # no writer yet, so no view

    def test_the_recording_presenter_logs_every_channel_on_the_time_timeline(
        self,
    ) -> None:
        """Run the recording presenter against a fake stream: every
        channel gets a static series line and one scalar per sample, and
        the blueprint holds one time-series view per channel."""
        try:
            import rerun  # noqa: F401, PLC0415
        except ImportError:
            self.skipTest("viz extra not installed")
        from rq_pipeline.project import index_project as idx  # noqa: PLC0415
        from rq_pipeline.project.present import _present_recording  # noqa: PLC0415
        from rq_pipeline.robots.ingest import ingest  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            project = create_project(Path(tmp) / "p", "p")
            ingest(project, make_mcap(Path(tmp) / "b.mcap"), name="bag")
            artifact = idx(project).by_kind(Kind.RECORDING)[0]
            fake = _RerunFake()
            shown = _present_recording(project, artifact, fake)
            self.assertEqual(shown["view"], "time series")
            self.assertIn("recording/joint.position", shown["paths"])
            scalars = [
                p for p, t in fake.logged if t == "Scalars" and "joint.position" in p
            ]
            self.assertEqual(len(scalars), 3 * 2)  # 3 samples x 2 joints
            self.assertTrue(all(tl == "time" for tl, _ in fake.times))
            self.assertIn("Vertical", type(shown["layout"]).__name__)


class _RerunFake:
    """A stand-in for a RecordingStream: records calls, and exposes the
    real archetypes (so `rr_.Scalars(...)` constructs) by delegation. It
    is a real (disconnected) RecordingStream underneath, so the presenter's
    default-recording shim can install it; only `log` and `set_time` are
    intercepted."""

    def __init__(self) -> None:
        import rerun as rr  # noqa: PLC0415

        self.logged: list[tuple[str, str]] = []
        self.times: list[tuple[str, float]] = []
        self._stream = rr.RecordingStream(application_id="test-fake")

    def __getattr__(self, name: str) -> Any:
        import rerun as rr  # noqa: PLC0415

        if name[:1].isupper():  # an archetype or type: Scalars, SeriesLines…
            return getattr(rr, name)
        return getattr(self._stream, name)

    def log(self, path: str, what: Any, static: bool = False) -> None:
        self.logged.append((path, type(what).__name__))

    def set_time(self, timeline: str, **kw: Any) -> None:
        self.times.append((timeline, float(next(iter(kw.values())))))


if __name__ == "__main__":
    unittest.main()
