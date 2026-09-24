"""The live capture from Unitree's bus (`robots/dds_capture.py`):
`rt/lowstate` and `rt/lowcmd` as one recording, landed as a rosbag2
store and read back by the bag adapter's layouts.

Three rings. The first needs nothing: a fake bus hands the listener
messages whose bytes are written by the adapter's own layouts, so the
store, the ingest, the basis, the command's lead and the Studio's
in-progress line are pinned on any Linux box. The second needs the SDK
(the `dds` extra): the LowCmd layout is checked against the bytes the
SDK itself serializes. The third runs Unitree's simulator: skipped
unless `RQ_DDS_STANDIN_TESTS=1` and their binaries are built.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any

from rq_pipeline.bundles.basis import BASIS_OWN, BASIS_SIMULATION
from rq_pipeline.robots import capture, dds_capture
from rq_pipeline.robots.adapters import rosbag2
from rq_pipeline.robots.cdr import SEQUENCE, Field, Layout, Reader, Writer
from rq_pipeline.robots.dds_capture import STANDIN_NETWORK, TOPICS
from rq_pipeline.robots.recording import (
    JOINT_COMMAND,
    JOINT_KP,
    JOINT_POSITION,
    Recording,
)
from tests._extras import installed

LINUX = sys.platform == "linux"
SDK = installed("unitree_sdk2py", "cyclonedds")
STATE_HZ = 500.0
COMMAND_LEAD_S = 0.001  # the fake controller's command goes out 1 ms before the state


def zeros(layout: Layout, layouts: dict[str, Layout]) -> dict[str, Any]:
    """A message of zeros for any layout: the shape, not the values."""

    def one(kind: str) -> Any:
        return zeros(layouts[kind], layouts) if kind in layouts else 0

    out: dict[str, Any] = {}
    for f in layout:
        if f.count is None:
            out[f.name] = one(f.type)
        elif f.count == SEQUENCE:
            out[f.name] = []
        else:
            out[f.name] = [one(f.type) for _ in range(f.count)]
    return out


class Msg:
    """What the SDK hands a handler: an object that serializes to CDR."""

    def __init__(self, data: bytes) -> None:
        self.data = data

    def serialize(self) -> bytes:
        return self.data


def low_state(q: float) -> bytes:
    msg = zeros(rosbag2.LAYOUTS[rosbag2.LOW_STATE], rosbag2.LAYOUTS)
    for m in msg["motor_state"]:
        m["q"] = q
    msg["imu_state"]["quaternion"] = [1.0, 0.0, 0.0, 0.0]
    return Writer().message(rosbag2.LAYOUTS[rosbag2.LOW_STATE], rosbag2.LAYOUTS, msg)


def low_cmd(q: float, kp: float = 20.0) -> bytes:
    msg = zeros(rosbag2.LAYOUTS[rosbag2.LOW_CMD], rosbag2.LAYOUTS)
    for m in msg["motor_cmd"]:
        m["q"], m["kp"], m["kd"] = q, kp, 0.5
    return Writer().message(rosbag2.LAYOUTS[rosbag2.LOW_CMD], rosbag2.LAYOUTS, msg)


class FakeBus:
    """A controller at STATE_HZ: each tick a command, then the state 1 ms
    later, handed to whichever topic's handler the listener subscribed."""

    def __init__(self) -> None:
        self.handlers: dict[str, Any] = {}
        self._stop = threading.Event()
        self.closed: list[str] = []

    def subscribe(self, topic: str, handler: Any, network: str, domain_id: int) -> Any:
        self.handlers[topic] = handler
        return lambda: self.closed.append(topic)

    def run(self, ticks: int) -> None:
        for i in range(ticks):
            q = 0.01 * i
            self.handlers[rosbag2.TOPIC_LOW_CMD](Msg(low_cmd(q)))
            time.sleep(COMMAND_LEAD_S)
            self.handlers[rosbag2.TOPIC_LOW_STATE](Msg(low_state(q)))
            time.sleep(1.0 / STATE_HZ - COMMAND_LEAD_S)


def seam_ingest(recordings: Path, source: Path, **kw: Any) -> dict[str, Any]:
    from rq_pipeline.robots.ingest import ingest  # noqa: PLC0415

    return ingest(recordings, source, **kw)


@unittest.skipUnless(LINUX, "Unitree's bus is Linux only; the refusal is pinned below")
class TheListener(unittest.TestCase):
    def capture(self, root: Path, bus: FakeBus, **options: Any) -> Any:
        return capture.open_capture(
            "dds",
            root / "recordings",
            root / "state",
            "session-1",
            subscribe=bus.subscribe,
            **{"network": STANDIN_NETWORK, "ingest": seam_ingest, **options},
        )

    def test_the_pair_lands_as_one_recording_with_the_commands_lead(self) -> None:
        bus = FakeBus()
        ticks = 200
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            listener = self.capture(root, bus)
            started = listener.start(window_s=30.0)
            self.assertEqual(started.state, capture.LISTENING)
            self.assertEqual(started.source, "dds:lo")
            bus.run(ticks)
            final = listener.stop()
            self.assertEqual(final.state, capture.INGESTED, final.error)
            self.assertEqual(
                final.topics,
                {rosbag2.TOPIC_LOW_STATE: ticks, rosbag2.TOPIC_LOW_CMD: ticks},
            )
            self.assertEqual(sorted(bus.closed), sorted(TOPICS))
            out = root / "recordings" / "session-1"
            rec = Recording.read(out)
            self.assertEqual(rec.basis, BASIS_OWN)
            self.assertEqual(rec.provenance["topics"], ["rt/lowstate", "rt/lowcmd"])
            for channel in (JOINT_POSITION, JOINT_COMMAND, JOINT_KP):
                self.assertEqual(len(rec.channels[channel].times), ticks, channel)
            self.assertAlmostEqual(float(rec.channels[JOINT_KP].values[0, 0]), 20.0)
            lead = rec.census["quality"]["command_lead"]
            self.assertEqual(lead["pairs"], ticks)
            self.assertGreater(lead["lead_p50_ms"], 0.0)
            clock = rec.census["quality"]["clock"][JOINT_POSITION]
            # measured off the receive times, never the nominal rate: the fake
            # bus paces in Python, so it runs at or below STATE_HZ
            self.assertGreater(clock["rate_hz"], 0.0)
            self.assertLessEqual(clock["rate_hz"], STATE_HZ * 1.05)
            # the store travels with the recording; nothing hidden is left
            self.assertTrue((out / "raw" / "capture" / rosbag2.STORE_FILE).is_file())
            self.assertFalse(list((root / "recordings").glob(".capture-*")))

    def test_the_stand_in_is_declared_and_said(self) -> None:
        bus = FakeBus()
        with tempfile.TemporaryDirectory() as tmp:
            listener = self.capture(Path(tmp), bus, basis=BASIS_SIMULATION)
            listener.start(window_s=30.0)
            bus.run(20)
            final = listener.stop()
            rec = Recording.read(Path(tmp) / "recordings" / "session-1")
        self.assertEqual(rec.basis, BASIS_SIMULATION)
        self.assertTrue(any("simulator" in n for n in final.notes))

    def test_a_silent_bus_fails_by_name_and_leaves_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            listener = self.capture(Path(tmp), FakeBus())
            listener.start(window_s=30.0)
            time.sleep(0.3)
            final = listener.stop()
            self.assertEqual(final.state, capture.FAILED)
            self.assertIn("silent", final.error or "")
            self.assertFalse(list((Path(tmp) / "recordings").iterdir()))

    def test_the_stamp_is_taken_after_the_store_lands(self) -> None:
        """Through the STAMPING ingest: the stamp the capture reports is
        the folder's own hash, with the kept store inside it (it was taken
        before the store moved in; review 2026-09-24)."""
        from rq_pipeline.bundles.hashing import bundle_hash  # noqa: PLC0415
        from rq_pipeline.project import create_project  # noqa: PLC0415
        from rq_pipeline.project.ingest import ingest as stamping  # noqa: PLC0415

        bus = FakeBus()
        with tempfile.TemporaryDirectory() as tmp:
            project = create_project(Path(tmp) / "p", "p")

            def land(_recordings: Path, source: Path, **kw: Any) -> dict[str, Any]:
                return stamping(project, source, **kw)

            listener = capture.open_capture(
                "dds",
                project.recordings,
                Path(tmp) / "state",
                "session-1",
                subscribe=bus.subscribe,
                network=STANDIN_NETWORK,
                ingest=land,
            )
            listener.start(window_s=30.0)
            bus.run(20)
            final = listener.stop()
            self.assertEqual(final.state, capture.INGESTED, final.error)
            out = project.recordings / "session-1"
            self.assertTrue((out / "raw" / "capture" / rosbag2.STORE_FILE).is_file())
            digest = (final.stamp or "").split("@", 1)[1]
            self.assertTrue(bundle_hash(out).startswith(digest))

    def test_a_capture_names_its_interface(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            self.assertRaisesRegex(ValueError, "name the interface"),
        ):
            capture.open_capture(
                "dds",
                Path(tmp) / "r",
                Path(tmp) / "s",
                "x",
                subscribe=FakeBus().subscribe,
            )

    def test_no_state_in_time_is_refused_by_name_at_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            listener = self.capture(Path(tmp), FakeBus(), first_state_timeout_s=0.3)
            listener.start(window_s=30.0)
            deadline = time.time() + 5.0
            while listener.state.state != capture.FAILED and time.time() < deadline:
                time.sleep(0.05)
            self.assertEqual(listener.state.state, capture.FAILED)
            self.assertIn("no rt/lowstate", listener.state.error or "")
            final = listener.stop()
            self.assertEqual(final.state, capture.FAILED)
            self.assertFalse(list((Path(tmp) / "recordings").iterdir()))

    def test_an_unknown_basis_is_refused_by_name(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            self.assertRaisesRegex(ValueError, "hearsay"),
        ):
            self.capture(Path(tmp), FakeBus(), basis="hearsay")


class OneInterfacePerProcess(unittest.TestCase):
    """Unitree's SDK binds one interface per process and silently keeps it
    on a second Init; the capture refuses a second interface by name."""

    def setUp(self) -> None:
        self.saved = dict(dds_capture._BOUND)
        dds_capture._BOUND.clear()

    def tearDown(self) -> None:
        dds_capture._BOUND.clear()
        dds_capture._BOUND.update(self.saved)

    def test_the_same_interface_rebinds_and_another_is_refused(self) -> None:
        self.assertEqual(dds_capture.bind_interface("lo", 0), ("lo", 0))
        self.assertEqual(dds_capture.bind_interface("lo", 0), ("lo", 0))
        with self.assertRaisesRegex(RuntimeError, "bound to 'lo'.*'eth0'"):
            dds_capture.bind_interface("eth0", 0)


class TheRegistry(unittest.TestCase):
    def test_two_sources_and_their_options(self) -> None:
        names = {s["name"]: s for s in capture.sources()}
        self.assertEqual(set(names), {"udp", "dds"})
        self.assertEqual(names["dds"]["platforms"], ["linux"])
        self.assertNotIn("ingest", names["dds"]["options"])

    def test_an_unknown_source_and_a_foreign_option_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "zigbee"):
                capture.open_capture("zigbee", root, root, "x")
            with self.assertRaisesRegex(ValueError, "network"):
                capture.open_capture("udp", root, root, "x", network="lo")

    @unittest.skipIf(LINUX, "the source refuses only off Linux")
    def test_off_linux_the_dds_source_refuses_by_name(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            self.assertRaisesRegex(RuntimeError, "linux"),
        ):
            capture.open_capture("dds", Path(tmp), Path(tmp), "x")


class TheLowCmdLayout(unittest.TestCase):
    def test_the_store_round_trips_a_command_through_the_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = rosbag2.Store(Path(tmp) / "bag")
            now = time.time_ns()
            store.append(
                [
                    (rosbag2.TOPIC_LOW_CMD, now, low_cmd(0.3, kp=33.0)),
                    (rosbag2.TOPIC_LOW_STATE, now + 1_000_000, low_state(0.3)),
                ]
            )
            with self.assertRaisesRegex(ValueError, "rt/unknown"):
                store.append([("rt/unknown", now, b"")])
            bag = store.close()
            rec = rosbag2.Rosbag2Adapter().read(bag)
        self.assertAlmostEqual(float(rec.channels[JOINT_KP].values[0, 0]), 33.0)
        self.assertAlmostEqual(float(rec.channels[JOINT_COMMAND].values[0, 0]), 0.3, 6)

    @unittest.skipUnless(SDK, "Unitree's SDK is the dds extra")
    def test_the_layout_reads_what_the_sdk_itself_writes(self) -> None:
        """The layout is Unitree's .msg; the SDK's serializer is the other
        end: a command it writes decodes field for field, every byte used."""
        from unitree_sdk2py.idl.default import (  # noqa: PLC0415
            unitree_go_msg_dds__LowCmd_ as default_cmd,
        )

        cmd = default_cmd()
        cmd.motor_cmd[4].q = 0.25
        cmd.motor_cmd[4].kp = 40.0
        cmd.level_flag = 0xFF
        data = bytes(cmd.serialize())
        reader = Reader(data)
        decoded = reader.message(rosbag2.LAYOUTS[rosbag2.LOW_CMD], rosbag2.LAYOUTS)
        self.assertAlmostEqual(decoded["motor_cmd"][4]["q"], 0.25, 6)
        self.assertAlmostEqual(decoded["motor_cmd"][4]["kp"], 40.0, 6)
        self.assertEqual(decoded["level_flag"], 0xFF)
        self.assertEqual(reader.pos, len(data))


class TheStudioSeesItFilling(unittest.TestCase):
    def test_a_listening_capture_is_in_progress_on_the_recordings_page(self) -> None:
        from rq_pipeline.project import create_project, index_project  # noqa: PLC0415
        from rq_pipeline.project.locate import INDEX_DIR  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            project = create_project(Path(tmp) / "p", "p", "test")
            capture.CaptureState(
                state=capture.LISTENING,
                name="walk-1",
                source="dds:eth0",
                topics={"rt/lowstate": 500, "rt/lowcmd": 499},
            ).write(project.root / INDEX_DIR)
            index = index_project(project)
        live = [p for p in index.in_progress if p["kind"] == "recording"]
        self.assertEqual(len(live), 1)
        self.assertEqual(live[0]["path"], "recordings/walk-1")
        self.assertIn("rt/lowcmd 499", live[0]["stage"])


@unittest.skipUnless(
    LINUX and SDK and os.environ.get("RQ_DDS_STANDIN_DEPLOYMENT"),
    "the stand-in run: RQ_DDS_STANDIN_DEPLOYMENT=<a deployment folder whose "
    "manifest names a Unitree stack>, their simulator and controller built "
    "(docs/77 §7); tools/capture-telemetry.py --standin does the same by hand",
)
class TheStandIn(unittest.TestCase):
    def test_their_simulator_and_controller_record_as_one_pair(self) -> None:
        from rq_pipeline.robots.dds_capture import standin_capture  # noqa: PLC0415

        deployment = Path(os.environ["RQ_DDS_STANDIN_DEPLOYMENT"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final = standin_capture(
                deployment,
                root / "recordings",
                root / "state",
                "standin",
                seconds=4.0,
                ingest=seam_ingest,
            )
            self.assertEqual(final.state, capture.INGESTED, final.error)
            rec = Recording.read(root / "recordings" / "standin")
        self.assertEqual(rec.basis, BASIS_SIMULATION)
        self.assertGreater(final.topics["rt/lowstate"], 100)
        self.assertGreater(final.topics["rt/lowcmd"], 100)
        self.assertIn(JOINT_COMMAND, rec.channels)
        self.assertGreater(rec.census["quality"]["command_lead"]["pairs"], 100)


__all__ = ["Field"]
