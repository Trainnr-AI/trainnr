"""The rosbag2 adapter, the table-driven CDR reader, the measured quality
block, the public-log registry and the provenance that rides with a
recording into the project's loop state.

The synthetic bag is written by the SAME layout tables the adapter reads
(`cdr.Writer`), so a layout is pinned from both ends; the real Go2 bag
(YibinWu/leg-odometry, 132.6 MB) is read only when it is in the cache and
the test says so by name when it is not.
"""

from __future__ import annotations

import dataclasses
import io
import sqlite3
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from rq_pipeline.project.index import index_project
from rq_pipeline.project.ingest import ingest as ingest_into_project
from rq_pipeline.project.kinds import Kind
from rq_pipeline.project.locate import create_project
from rq_pipeline.robots import public_logs, quality
from rq_pipeline.robots.adapter import detect
from rq_pipeline.robots.adapters import rosbag2
from rq_pipeline.robots.cdr import Reader, Writer, parse_msg
from rq_pipeline.robots.ingest import ingest
from rq_pipeline.robots.recording import (
    BASIS_OWN,
    BASIS_PUBLIC,
    IMU_ORIENTATION,
    JOINT_EFFORT,
    JOINT_POSITION,
    JOINT_VELOCITY,
    Recording,
)

NS = 1_000_000_000
RATE_HZ = 500
MOTOR_DEGC = (20, 45)  # the bag's motors, indoors, warm not hot
LOW_STATE_MSG = """\
uint8[2] head
uint8 level_flag
uint8 frame_reserve
uint32[2] sn
uint32[2] version
uint16 bandwidth
IMUState imu_state
MotorState[20] motor_state
BmsState bms_state
int16[4] foot_force
int16[4] foot_force_est
uint32 tick
uint8[40] wireless_remote
uint8 bit_flag
float32 adc_reel
int8 temperature_ntc1
int8 temperature_ntc2
float32 power_v
float32 power_a
uint16[4] fan_frequency
uint32 reserve
uint32 crc
"""
MOTOR_STATE_MSG = """\
uint8 mode
float32 q
float32 dq
float32 ddq
float32 tau_est
float32 q_raw
float32 dq_raw
float32 ddq_raw
int8 temperature
uint32 lost
uint32[2] reserve
"""


def _motor(i: int, t: float) -> dict:
    return {
        "mode": 1,
        "q": 0.5 * np.sin(t + i),
        "dq": 0.5 * np.cos(t + i),
        "ddq": 0.0,
        "tau_est": float(i),
        "q_raw": 0.0,
        "dq_raw": 0.0,
        "ddq_raw": 0.0,
        "temperature": 30 + i,
        "lost": 0,
        "reserve": [0, 0],
    }


def _low_state(t: float) -> dict:
    return {
        "head": [0xFE, 0xEF],
        "level_flag": 0xFF,
        "frame_reserve": 0,
        "sn": [1, 2],
        "version": [3, 4],
        "bandwidth": 0,
        "imu_state": {
            "quaternion": [1.0, 0.0, 0.0, 0.0],
            "gyroscope": [0.0, 0.0, 0.1],
            "accelerometer": [0.0, 0.0, 9.81],
            "rpy": [0.0, 0.0, 0.0],
            "temperature": 45,
        },
        "motor_state": [_motor(i, t) for i in range(rosbag2.MOTOR_SLOTS)],
        "bms_state": {
            "version_high": 1,
            "version_low": 0,
            "status": 0,
            "soc": 80,
            "current": -1500,
            "cycle": 10,
            "bq_ntc": [25, 25],
            "mcu_ntc": [30, 30],
            "cell_vol": [3900] * 15,
        },
        "foot_force": [10, 20, 30, 40],
        "foot_force_est": [11, 21, 31, 41],
        "tick": int(t * 1000),
        "wireless_remote": [0] * 40,
        "bit_flag": 0,
        "adc_reel": 0.0,
        "temperature_ntc1": 20,
        "temperature_ntc2": 21,
        "power_v": 28.5,
        "power_a": 3.2,
        "fan_frequency": [0, 0, 0, 0],
        "reserve": 0,
        "crc": 0,
    }


def make_bag(root: Path, *, seconds: float = 1.0, gap_at: int | None = None) -> Path:
    """A rosbag2 sqlite3 directory with /lowstate at RATE_HZ, written by
    the adapter's own layouts; `gap_at` drops ten samples after that
    index so a dropout exists to be measured."""
    root.mkdir(parents=True, exist_ok=True)
    start = 1_700_000_000 * NS
    db = root / "bag_0.db3"
    with sqlite3.connect(db) as con:
        con.execute(
            "CREATE TABLE topics(id INTEGER PRIMARY KEY, name TEXT NOT NULL, "
            "type TEXT NOT NULL, serialization_format TEXT NOT NULL, "
            "offered_qos_profiles TEXT NOT NULL)"
        )
        con.execute(
            "CREATE TABLE messages(id INTEGER PRIMARY KEY, topic_id INTEGER NOT NULL, "
            "timestamp INTEGER NOT NULL, data BLOB NOT NULL)"
        )
        con.execute(
            "INSERT INTO topics VALUES (1, '/lowstate', ?, 'cdr', '')",
            (rosbag2.LOW_STATE,),
        )
        con.execute("INSERT INTO topics VALUES (2, '/other', 'x/msg/Y', 'cdr', '')")
        rows = []
        for i in range(int(seconds * RATE_HZ)):
            if gap_at is not None and gap_at < i <= gap_at + 10:
                continue
            t = i / RATE_HZ
            data = Writer().message(
                rosbag2.LAYOUTS[rosbag2.LOW_STATE], rosbag2.LAYOUTS, _low_state(t)
            )
            rows.append((1, start + int(t * NS), data))
        rows.append((2, start, b"\x00\x01\x00\x00"))
        con.executemany(
            "INSERT INTO messages(topic_id, timestamp, data) VALUES (?, ?, ?)", rows
        )
    (root / rosbag2.METADATA_FILE).write_text(
        "rosbag2_bagfile_information:\n"
        "  version: 5\n"
        "  storage_identifier: sqlite3\n"
        "  starting_time:\n"
        f"    nanoseconds_since_epoch: {start}\n",
        encoding="utf-8",
    )
    return root


class TheLayouts(unittest.TestCase):
    def test_the_layouts_are_unitrees_msg_files(self) -> None:
        """The tables are copied, not transcribed: parsing the vendor's
        `.msg` text gives the same layout."""
        self.assertEqual(parse_msg(LOW_STATE_MSG), rosbag2.LAYOUTS[rosbag2.LOW_STATE])
        self.assertEqual(parse_msg(MOTOR_STATE_MSG), rosbag2.LAYOUTS["MotorState"])

    def test_write_then_read_round_trips_every_field(self) -> None:
        msg = _low_state(0.25)
        data = Writer().message(
            rosbag2.LAYOUTS[rosbag2.LOW_STATE], rosbag2.LAYOUTS, msg
        )
        back = Reader(data).message(rosbag2.LAYOUTS[rosbag2.LOW_STATE], rosbag2.LAYOUTS)
        self.assertEqual(back["bms_state"]["soc"], 80)
        self.assertEqual(back["foot_force"], [10, 20, 30, 40])
        self.assertAlmostEqual(
            back["motor_state"][3]["q"], msg["motor_state"][3]["q"], 6
        )
        self.assertEqual(back["motor_state"][19]["temperature"], 49)
        self.assertEqual(back["imu_state"]["temperature"], 45)

    def test_an_unknown_nested_type_is_refused_by_name(self) -> None:
        layout = (dataclasses.replace(rosbag2.LAYOUTS["IMUState"][0], type="Nope"),)
        with self.assertRaisesRegex(ValueError, "Nope"):
            Reader(b"\x00\x01\x00\x00" + b"\x00" * 64).message(layout, {})


class TheAdapter(unittest.TestCase):
    def test_a_bag_directory_is_detected_and_read_as_channels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bag = make_bag(Path(tmp) / "bag")
            self.assertEqual(detect(bag).name, rosbag2.NAME)
            rec = rosbag2.Rosbag2Adapter().read(bag)
        self.assertEqual(rec.adapter, rosbag2.NAME)
        for name in (JOINT_POSITION, JOINT_VELOCITY, JOINT_EFFORT, IMU_ORIENTATION):
            self.assertIn(name, rec.channels)
        q = rec.channels[JOINT_POSITION]
        self.assertEqual(q.values.shape, (RATE_HZ, 12))
        self.assertEqual(q.components, rosbag2.GO2_MOTORS)
        self.assertEqual(q.unit, "rad")
        self.assertEqual(rec.channels[IMU_ORIENTATION].components, ("w", "x", "y", "z"))
        self.assertEqual(rec.channels["motor.temperature"].values[0, 5], 35.0)
        self.assertEqual(
            rec.census["topics"]["/other"], {"type": "x/msg/Y", "messages": 1}
        )
        self.assertEqual(rec.census["motors_read"], 12)
        self.assertEqual(rec.census["recorded"], "2023-11-14T22:13:20+00:00")
        self.assertIn(rosbag2.NOTE_TAU_EST, rec.notes)
        self.assertAlmostEqual(q.rate_hz or 0, RATE_HZ, delta=1)

    def test_the_quality_block_is_measured_not_assumed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rec = rosbag2.Rosbag2Adapter().read(make_bag(Path(tmp) / "bag", gap_at=100))
        block = rec.census[quality.QUALITY_KEY]
        clock = block["clock"][JOINT_POSITION]
        self.assertEqual(clock["samples"], RATE_HZ - 10)
        self.assertEqual(clock["dropouts"], 1)
        self.assertAlmostEqual(clock["longest_gap_ms"], 22.0, delta=0.5)
        self.assertAlmostEqual(clock["interval_p50_ms"], 2.0, delta=0.05)
        self.assertLess(clock["rate_hz"], RATE_HZ)  # the gap lowers the mean rate
        lo, hi = block["joint_range"]["FR_hip"]  # 0.5*sin(t) over one second
        self.assertAlmostEqual(lo, 0.0, delta=0.01)
        self.assertAlmostEqual(hi, 0.5 * np.sin(1.0), delta=0.01)
        self.assertGreater(block["moving_fraction"], 0.9)

    def test_a_bag_with_nothing_this_adapter_decodes_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "bag"
            root.mkdir()
            with sqlite3.connect(root / "x_0.db3") as con:
                con.execute(
                    "CREATE TABLE topics(id INTEGER PRIMARY KEY, name TEXT, type TEXT, "
                    "serialization_format TEXT, offered_qos_profiles TEXT)"
                )
                con.execute(
                    "CREATE TABLE messages(id INTEGER PRIMARY KEY, topic_id INTEGER, "
                    "timestamp INTEGER, data BLOB)"
                )
                con.execute(
                    "INSERT INTO topics VALUES (1, '/tf', 'tf2_msgs/msg/TF', 'cdr', '')"
                )
                con.execute("INSERT INTO messages VALUES (1, 1, 5, X'00010000')")
            (root / rosbag2.METADATA_FILE).write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "tf2_msgs/msg/TF"):
                rosbag2.Rosbag2Adapter().read(root)


class TheProvenance(unittest.TestCase):
    def test_a_public_log_rides_into_the_recording_and_the_loop_state(self) -> None:
        """Ingested with the registry's provenance, the recording says
        whose robot it was, and the project's telemetry stage carries the
        word - present, never the operator's own."""
        entry = public_logs.resolve("go2-leg-odometry")
        with tempfile.TemporaryDirectory() as tmp:
            bag = make_bag(Path(tmp) / "bag", seconds=0.2)
            project = create_project(Path(tmp) / "p", "p", "test")
            out = ingest_into_project(
                project,
                bag,
                name="public",
                provenance=entry.provenance(),
                basis=entry.basis,
            )
            self.assertEqual(out["basis"], BASIS_PUBLIC)
            rec = Recording.read(project.root / out["path"])
            self.assertEqual(rec.provenance["robot"], "go2")
            self.assertIn("unlabelled", rec.provenance["licence"])
            index = index_project(project)
        telemetry = next(s for s in index.states if s.name == "telemetry recorded")
        self.assertTrue(telemetry.present)
        self.assertEqual(telemetry.basis, BASIS_PUBLIC)
        recording = index.by_kind(Kind.RECORDING)[0]
        self.assertEqual(recording.summary["basis"], BASIS_PUBLIC)
        self.assertEqual(recording.summary["robot"], "go2")
        self.assertAlmostEqual(recording.summary["rate_hz"], RATE_HZ, delta=1)

    def test_the_operators_own_recording_wears_no_word_on_the_chip(self) -> None:
        """A bag says nothing about whose robot it was; the operator who
        ingests their own says so, and the strip shows the stage plain."""
        with tempfile.TemporaryDirectory() as tmp:
            bag = make_bag(Path(tmp) / "bag", seconds=0.2)
            project = create_project(Path(tmp) / "p", "p", "test")
            ingest_into_project(project, bag, name="own", basis=BASIS_OWN)
            index = index_project(project)
        telemetry = next(s for s in index.states if s.name == "telemetry recorded")
        self.assertTrue(telemetry.present)
        self.assertEqual(telemetry.basis, BASIS_OWN)
        self.assertNotIn("basis", index.by_kind(Kind.RECORDING)[0].summary)

    def test_an_unknown_basis_is_refused_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bag = make_bag(Path(tmp) / "bag", seconds=0.1)
            with self.assertRaisesRegex(ValueError, "borrowed"):
                ingest(Path(tmp) / "r", bag, basis="borrowed")


class TwoReadersOneContainer(unittest.TestCase):
    """Two adapters read rosbag2's sqlite store — this one for Unitree's
    message types, `robot/rosbag_sqlite.py` for other robots' profiles —
    and they split a bag by its CONTENT, so `detect` names exactly one
    (a suffix claim by both refused every bag, found at the merge of
    2026-09-24)."""

    def test_a_unitree_bag_detects_as_this_adapter_and_not_the_other(self) -> None:
        from rq_pipeline.robot.rosbag_sqlite import Rosbag2Sqlite  # noqa: PLC0415
        from rq_pipeline.robots.adapter import detect  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            bag = make_bag(Path(tmp) / "bag", seconds=0.1)
            self.assertEqual(detect(bag).name, "rosbag2")
            db3 = next(bag.glob("*.db3"))
            self.assertEqual(detect(db3).name, "rosbag2")
            self.assertFalse(Rosbag2Sqlite().accepts(db3))

    def test_a_file_that_is_no_bag_is_claimed_by_neither(self) -> None:
        from rq_pipeline.robot.rosbag_sqlite import Rosbag2Sqlite  # noqa: PLC0415
        from rq_pipeline.robots.adapters.rosbag2 import Rosbag2Adapter  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            junk = Path(tmp) / "x.db3"
            junk.write_bytes(b"not a database")
            self.assertFalse(Rosbag2Adapter().accepts(junk))
            self.assertFalse(Rosbag2Sqlite().accepts(junk))


class TheRegistry(unittest.TestCase):
    def test_every_entry_names_its_bytes_digest_licence_and_robot(self) -> None:
        for entry in public_logs.PUBLIC_LOGS.values():
            self.assertGreater(entry.bytes, 0)
            self.assertEqual(len(entry.sha256), 64)
            self.assertTrue(entry.licence and entry.robot and entry.recorded)
            self.assertEqual(entry.basis, BASIS_PUBLIC)
        self.assertIn("go2-leg-odometry", [r["name"] for r in public_logs.listing()])

    def test_a_download_of_the_wrong_size_is_refused_and_deleted(self) -> None:
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w") as z:
            z.writestr("rosbag2_2025_02_19-22_46_51/metadata.yaml", "x")
        bad = payload.getvalue()

        class Response(io.BytesIO):
            def __enter__(self) -> Response:
                return self

            def __exit__(self, *_: object) -> None:
                self.close()

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "registry expects"):
                public_logs.fetch(
                    "go2-leg-odometry",
                    Path(tmp),
                    opener=lambda *_a, **_k: Response(bad),
                )
            self.assertEqual(list((Path(tmp) / "go2-leg-odometry").glob("*.zip")), [])
        with self.assertRaisesRegex(KeyError, "unknown public log"):
            public_logs.resolve("nope")


REAL = public_logs.locate("go2-leg-odometry")


@unittest.skipUnless(
    REAL is not None,
    "the real Go2 bag is not in the cache: `python3 tools/public-log.py fetch "
    "go2-leg-odometry` (132.6 MB)",
)
class TheRealGo2Bag(unittest.TestCase):
    """What the one public Go2 LowState bag actually holds, measured on
    2026-09-24: pinned loosely so a re-fetch of the same release passes
    and a different bag does not."""

    def test_the_bag_reads_and_its_numbers_are_the_measured_ones(self) -> None:
        assert REAL is not None
        rec = rosbag2.Rosbag2Adapter().read(REAL)
        q = rec.census[quality.QUALITY_KEY]
        clock = q["clock"][JOINT_POSITION]
        self.assertEqual(clock["samples"], 368_113)
        self.assertAlmostEqual(clock["rate_hz"], 499.8, delta=0.5)
        self.assertLess(clock["interval_p99_ms"], 5.0)
        self.assertLess(clock["dropouts"], 50)
        self.assertGreater(q["moving_fraction"], 0.9)
        hips = [q["joint_range"][f"{leg}_hip"] for leg in ("FR", "FL", "RR", "RL")]
        self.assertLess(max(abs(v) for lo, hi in hips for v in (lo, hi)), 0.5)
        temps = rec.channels["motor.temperature"].values
        self.assertTrue(temps.min() >= MOTOR_DEGC[0] and temps.max() <= MOTOR_DEGC[1])
        self.assertLess(np.abs(rec.channels[JOINT_EFFORT].values).max(), 40)
        self.assertEqual(rec.census["recorded"][:10], "2025-02-19")


if __name__ == "__main__":
    unittest.main()
