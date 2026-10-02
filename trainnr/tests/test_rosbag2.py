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
from typing import Any
from unittest import mock

import numpy as np

from trainnr.project.index import index_project
from trainnr.project.ingest import ingest as ingest_into_project
from trainnr.project.kinds import Kind
from trainnr.project.locate import create_project
from trainnr.robots import public_logs, quality
from trainnr.robots.adapter import detect
from trainnr.robots.adapters import rosbag2
from trainnr.robots.cdr import Reader, Writer, parse_msg
from trainnr.robots.ingest import ingest
from trainnr.robots.recording import (
    BASIS_OWN,
    BASIS_PUBLIC,
    BASIS_UNKNOWN,
    IMU_ORIENTATION,
    JOINT_COMMAND,
    JOINT_EFFORT,
    JOINT_FEEDFORWARD,
    JOINT_KD,
    JOINT_KP,
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


# Unitree's own files (unitree_ros2 master, unitree_go/msg, read 2026-09-24).
LOW_CMD_MSG = """\
uint8[2] head
uint8 level_flag
uint8 frame_reserve
uint32[2] sn
uint32[2] version
uint16 bandwidth
MotorCmd[20] motor_cmd
BmsCmd bms_cmd
uint8[40] wireless_remote
uint8[12] led
uint8[2] fan
uint8 gpio
uint32 reserve
uint32 crc
"""
MOTOR_CMD_MSG = """\
uint8 mode
float32 q
float32 dq
float32 tau
float32 kp
float32 kd
uint32[3] reserve
"""
BMS_CMD_MSG = """\
uint8 off
uint8[3] reserve
"""


def _low_cmd(t: float) -> dict:
    return {
        "head": [0xFE, 0xEF],
        "level_flag": 0xFF,
        "frame_reserve": 0,
        "sn": [0, 0],
        "version": [0, 0],
        "bandwidth": 0,
        "motor_cmd": [
            {
                "mode": 1,
                "q": 0.1 * i + t,
                "dq": 0.0,
                "tau": 0.5,
                "kp": 25.0,
                "kd": 0.5,
                "reserve": [0, 0, 0],
            }
            for i in range(rosbag2.MOTOR_SLOTS)
        ],
        "bms_cmd": {"off": 0, "reserve": [0, 0, 0]},
        "wireless_remote": [0] * 40,
        "led": [0] * 12,
        "fan": [0, 0],
        "gpio": 0,
        "reserve": 0,
        "crc": 0,
    }


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


def make_bag(
    root: Path,
    *,
    seconds: float = 1.0,
    gap_at: int | None = None,
    with_cmd: bool = False,
) -> Path:
    """A rosbag2 sqlite3 directory with /lowstate at RATE_HZ, written by
    the adapter's own layouts; `gap_at` drops ten samples after that
    index so a dropout exists to be measured; `with_cmd` adds /lowcmd at
    the same rate, the pair a deployed controller would record."""
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
        if with_cmd:
            con.execute(
                "INSERT INTO topics VALUES (3, '/lowcmd', ?, 'cdr', '')",
                (rosbag2.LOW_CMD,),
            )
        rows = []
        for i in range(int(seconds * RATE_HZ)):
            if gap_at is not None and gap_at < i <= gap_at + 10:
                continue
            t = i / RATE_HZ
            data = Writer().message(
                rosbag2.LAYOUTS[rosbag2.LOW_STATE], rosbag2.LAYOUTS, _low_state(t)
            )
            rows.append((1, start + int(t * NS), data))
            if with_cmd:
                cmd = Writer().message(
                    rosbag2.LAYOUTS[rosbag2.LOW_CMD], rosbag2.LAYOUTS, _low_cmd(t)
                )
                rows.append((3, start + int(t * NS) + 1000, cmd))
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
        self.assertEqual(parse_msg(LOW_CMD_MSG), rosbag2.LAYOUTS[rosbag2.LOW_CMD])
        self.assertEqual(parse_msg(MOTOR_CMD_MSG), rosbag2.LAYOUTS["MotorCmd"])
        self.assertEqual(parse_msg(BMS_CMD_MSG), rosbag2.LAYOUTS["BmsCmd"])

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
            # the entry's facts about THIS log travel with it (they were
            # listed on the entry and never reached the recording)
            self.assertEqual(rec.provenance["notes"], list(entry.notes))
            index = index_project(project)
        telemetry = next(s for s in index.states if s.name == "telemetry recorded")
        self.assertTrue(telemetry.present)
        self.assertEqual(telemetry.basis, BASIS_PUBLIC)
        recording = index.by_kind(Kind.RECORDING)[0]
        self.assertEqual(recording.summary["basis"], BASIS_PUBLIC)
        self.assertEqual(recording.summary["robot"], "go2")
        self.assertAlmostEqual(
            float(recording.summary["rate"].split()[0]), RATE_HZ, delta=1
        )

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


class TheStoreIsABag(unittest.TestCase):
    """The store a live capture writes carries the metadata ROS tools read
    (review 2026-09-24: it wrote only the start, and an empty store's 0
    read back as 1970)."""

    def test_the_metadata_names_counts_duration_and_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = rosbag2.Store(Path(tmp) / "s")
            state = Writer().message(
                rosbag2.LAYOUTS[rosbag2.LOW_STATE], rosbag2.LAYOUTS, _low_state(0.0)
            )
            cmd = Writer().message(
                rosbag2.LAYOUTS[rosbag2.LOW_CMD], rosbag2.LAYOUTS, _low_cmd(0.0)
            )
            start = 1_700_000_000 * NS
            store.append(
                [
                    (rosbag2.TOPIC_LOW_CMD, start, cmd),
                    (rosbag2.TOPIC_LOW_STATE, start + 2_000_000, state),
                    (rosbag2.TOPIC_LOW_STATE, start + 4_000_000, state),
                ]
            )
            root = store.close()
            text = (root / rosbag2.METADATA_FILE).read_text(encoding="utf-8")
            recorded = rosbag2._recorded_at(root)
            empty = rosbag2.Store(Path(tmp) / "e").close()
            recorded_empty = rosbag2._recorded_at(empty)
        self.assertIn("  message_count: 3\n", text)
        self.assertIn("    nanoseconds: 4000000\n", text)
        self.assertIn(f"        name: {rosbag2.TOPIC_LOW_STATE}\n", text)
        self.assertIn("      message_count: 2\n", text)
        self.assertIn(f"    - {rosbag2.STORE_FILE}\n", text)
        self.assertTrue((recorded or "").startswith("2023-11-14"))
        self.assertIsNone(recorded_empty)


class SplitBags(unittest.TestCase):
    def test_a_bag_split_past_ten_files_keeps_every_sample(self) -> None:
        """rosbag2 names splits `_0 ... _10`; read as text `_10` came before
        `_2` and every sample of `_2` was dropped as out of order (review
        2026-09-24). Split one bag into `_2` and `_10`: nothing is lost."""
        import shutil  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            bag = make_bag(Path(tmp) / "bag", seconds=0.2)
            whole = rosbag2.Rosbag2Adapter().read(bag)
            n = len(whole.channels[JOINT_POSITION].times)
            one = bag / "bag_0.db3"
            with sqlite3.connect(one) as con:
                (mid,) = con.execute(
                    "SELECT timestamp FROM messages WHERE topic_id=1 "
                    "ORDER BY timestamp LIMIT 1 OFFSET ?",
                    (n // 2,),
                ).fetchone()
            for name, keep in (("bag_2.db3", "<"), ("bag_10.db3", ">=")):
                shutil.copy(one, bag / name)
                with sqlite3.connect(bag / name) as con:
                    con.execute(
                        f"DELETE FROM messages WHERE NOT (timestamp {keep} ?)", (mid,)
                    )
            one.unlink()
            split = rosbag2.Rosbag2Adapter().read(bag)
        self.assertEqual(split.census["files"], ["bag_2.db3", "bag_10.db3"])
        self.assertEqual(len(split.channels[JOINT_POSITION].times), n)
        self.assertEqual(split.census["topics"]["/lowstate"]["dropped"], 0)


class OneAdapterManyProfiles(unittest.TestCase):
    """One adapter reads rosbag2's sqlite store; which robot's bag it is
    comes from the PROFILE its message types anchor (2026-09-24: two
    readers had each claimed every `.db3` and refused every bag)."""

    def test_a_unitree_bag_reads_through_its_profile_with_basis_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bag = make_bag(Path(tmp) / "bag", seconds=0.1)
            self.assertEqual(detect(bag).name, rosbag2.NAME)
            db3 = next(bag.glob("*.db3"))
            self.assertEqual(detect(db3).name, rosbag2.NAME)
            rec = rosbag2.Rosbag2Adapter().read(bag)
        self.assertEqual(rec.census["profile"], "unitree-go2")
        self.assertEqual(rec.census["motors_read"], 12)
        # A Unitree bag may be the operator's own robot: the caller says whose.
        self.assertEqual(rec.basis, BASIS_UNKNOWN)

    def test_lowcmd_beside_lowstate_ingests_the_command_and_the_gains(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rec = rosbag2.Rosbag2Adapter().read(
                make_bag(Path(tmp) / "bag", seconds=0.2, with_cmd=True)
            )
        for name in (JOINT_COMMAND, JOINT_FEEDFORWARD, JOINT_KP, JOINT_KD):
            self.assertIn(name, rec.channels)
        self.assertEqual(rec.channels[JOINT_KP].values.shape, (100, 12))
        self.assertEqual(rec.channels[JOINT_KP].values[0, 0], 25.0)
        self.assertAlmostEqual(rec.channels[JOINT_COMMAND].values[0, 3], 0.3, 6)
        self.assertEqual(rec.channels[JOINT_COMMAND].components, rosbag2.GO2_MOTORS)
        self.assertIn(rosbag2.NOTE_LOW_CMD, rec.notes)
        # the state is still there, on its own clock
        self.assertEqual(rec.channels[JOINT_POSITION].values.shape, (100, 12))

    def test_a_bag_that_is_two_robots_or_two_imus_is_refused_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bag = make_bag(Path(tmp) / "bag", seconds=0.05)
            db3 = next(bag.glob("*.db3"))
            with sqlite3.connect(db3) as con:
                con.execute(
                    "INSERT INTO topics VALUES (7, '/joint_states', ?, 'cdr', '')",
                    (rosbag2.DFKI_JOINT_STATE,),
                )
            with self.assertRaisesRegex(ValueError, "one bag is one robot"):
                rosbag2.Rosbag2Adapter().read(bag)
            with sqlite3.connect(db3) as con:
                con.execute("DELETE FROM topics WHERE id = 7")
                con.execute(
                    "INSERT INTO topics VALUES (8, '/lowstate2', ?, 'cdr', '')",
                    (rosbag2.LOW_STATE,),
                )
            with self.assertRaisesRegex(ValueError, "which is the robot's"):
                rosbag2.Rosbag2Adapter().read(bag)

    def test_a_file_that_is_no_bag_is_not_claimed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            junk = Path(tmp) / "x.db3"
            junk.write_bytes(b"not a database")
            self.assertFalse(rosbag2.Rosbag2Adapter().accepts(junk))

    def test_a_layout_short_of_fields_or_bytes_is_refused_with_the_type(
        self,
    ) -> None:
        """The checked decode refuses bytes the layout leaves unread (beyond
        the trailing alignment) and a read past the end, naming the type
        (review 2026-09-24: SportModeState's layout lacked its path points
        and every real message left 280 bytes unread, silently)."""
        full = Writer().message(
            rosbag2.LAYOUTS[rosbag2.LOW_STATE], rosbag2.LAYOUTS, _low_state(0.0)
        )
        self.assertIn(
            "motor_state", Reader(full).decode(rosbag2.LOW_STATE, rosbag2.LAYOUTS)
        )
        short = dict(rosbag2.LAYOUTS)
        short[rosbag2.LOW_STATE] = rosbag2.LAYOUTS[rosbag2.LOW_STATE][:-2]
        with self.assertRaisesRegex(ValueError, "LowState: .* bytes left unread"):
            Reader(full).decode(rosbag2.LOW_STATE, short)
        with self.assertRaisesRegex(ValueError, "LowState: the bytes end before"):
            Reader(full[:100]).decode(rosbag2.LOW_STATE, rosbag2.LAYOUTS)
        with self.assertRaisesRegex(ValueError, "runs past"):
            Reader(b"\x00\x01\x00\x00" + (1000).to_bytes(4, "little")).string()

    def test_big_endian_cdr_is_refused_by_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "big-endian"):
            Reader(b"\x00\x00\x00\x00" + b"\x00" * 8)


class _Response(io.BytesIO):
    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _serving(payload: bytes) -> Any:
    """An opener that serves `payload`, honouring a Range header the way
    Zenodo does — so the zip-member fetcher is pinned with no network."""

    def opener(request: Any, **_k: Any) -> _Response:
        header = getattr(request, "headers", {}).get("Range")
        if header is None:
            return _Response(payload)
        first, last = (int(v) for v in header.split("=", 1)[1].split("-"))
        return _Response(payload[first : last + 1])

    return opener


class TheRegistry(unittest.TestCase):
    def test_every_entry_names_its_pieces_digests_licence_and_robot(self) -> None:
        for entry in public_logs.PUBLIC_LOGS.values():
            self.assertIn(entry.fetch, public_logs.FETCHERS)
            self.assertGreater(entry.bytes, 0)
            for piece in entry.pieces:
                self.assertGreater(piece.bytes, 0)
                self.assertEqual(len(piece.sha256), 64)
                if entry.fetch == public_logs.FETCH_ZIP_MEMBERS:
                    self.assertIsNotNone(piece.span)
            self.assertTrue(entry.licence and entry.robot and entry.recorded)
            self.assertEqual(entry.basis, BASIS_PUBLIC)
        listed = {r["name"]: r for r in public_logs.listing()}
        for name in ("go2-leg-odometry", "dfki-go2-field201", "iit-go2-chirp"):
            self.assertTrue(listed[name]["readable"])
            self.assertTrue(listed[name]["licence"])
        for name in ("quadslam", "doglegs", "legkilo"):
            self.assertFalse(listed[name]["readable"])
            self.assertIn("ROS 1", listed[name]["why"])

    def test_a_log_no_adapter_reads_is_refused_with_the_reason(self) -> None:
        with self.assertRaisesRegex(KeyError, "ROS 1"):
            public_logs.resolve("quadslam")
        with self.assertRaisesRegex(KeyError, "unknown public log"):
            public_logs.resolve("nope")

    def test_a_download_of_the_wrong_size_is_refused_and_deleted(self) -> None:
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w") as z:
            z.writestr("rosbag2_2025_02_19-22_46_51/metadata.yaml", "x")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "registry expects"):
                public_logs.fetch(
                    "go2-leg-odometry",
                    Path(tmp),
                    opener=_serving(payload.getvalue()),
                )
            self.assertEqual(list((Path(tmp) / "go2-leg-odometry").glob("*.zip")), [])

    def test_zip_members_are_read_by_range_inflated_and_checked(self) -> None:
        """Two members out of a larger remote zip: only their bytes are
        asked for, each inflated and checked against its digest; a
        member that differs is refused and nothing lands."""
        import dataclasses  # noqa: PLC0415
        import hashlib  # noqa: PLC0415

        bag_bytes = b"SQLite format 3\x00" + bytes(range(256)) * 400
        meta = b"rosbag2_bagfile_information:\n  version: 5\n"
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("other/huge.db3", b"\x01" * 50_000)
            z.writestr("run/bag/bag_0.db3", bag_bytes)
            z.writestr("run/bag/metadata.yaml", meta)
        archive = payload.getvalue()
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            info = {i.filename: i for i in z.infolist()}

        def piece(name: str, data: bytes) -> public_logs.Piece:
            i = info[name]
            return public_logs.Piece(
                name,
                len(data),
                hashlib.sha256(data).hexdigest(),
                public_logs.ZipSpan(i.header_offset, i.compress_size, i.compress_type),
            )

        entry = public_logs.PublicLog(
            name="toy",
            robot="go2",
            url="https://example.invalid/toy.zip",
            fetch=public_logs.FETCH_ZIP_MEMBERS,
            pieces=(
                piece("run/bag/bag_0.db3", bag_bytes),
                piece("run/bag/metadata.yaml", meta),
            ),
            member="run/bag",
            adapter="rosbag2",
            source="a test",
            recorded="2026-09-24",
            licence="test",
        )
        asked: list[str] = []
        serve = _serving(archive)

        def opener(request: Any, **k: Any) -> _Response:
            asked.append(request.headers["Range"])
            return serve(request, **k)

        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.dict(public_logs.PUBLIC_LOGS, {"toy": entry}),
        ):
            out = public_logs.fetch("toy", Path(tmp), opener=opener)
            self.assertEqual((out / "bag_0.db3").read_bytes(), bag_bytes)
            self.assertEqual((out / "metadata.yaml").read_bytes(), meta)
            self.assertEqual(len(asked), 4)  # a header and a body per member
            self.assertEqual(public_logs.fetch("toy", Path(tmp), opener=None), out)
            broken = dataclasses.replace(
                entry,
                name="broken",
                pieces=(dataclasses.replace(entry.pieces[0], sha256="0" * 64),),
            )
            with mock.patch.dict(public_logs.PUBLIC_LOGS, {"broken": broken}):
                with self.assertRaisesRegex(ValueError, "registry expects"):
                    public_logs.fetch("broken", Path(tmp), opener=serve)
                self.assertIsNone(public_logs.locate("broken", Path(tmp)))

    def test_a_download_that_breaks_leaves_nothing_that_is_served(self) -> None:
        """A stream that drops midway, and a fetch whose second piece
        fails after the first landed, leave no file and no folder that a
        later `fetch` or `locate` would serve (review 2026-09-24: the
        truncated file was returned unchecked the next time)."""
        import hashlib  # noqa: PLC0415

        first, second = b"first piece" * 100, b"second piece" * 50
        entry = public_logs.PublicLog(
            name="toy-files",
            robot="go2",
            url="https://example.invalid/x",
            fetch=public_logs.FETCH_FILE,
            pieces=(
                public_logs.Piece(
                    "a.bin", len(first), hashlib.sha256(first).hexdigest()
                ),
                public_logs.Piece(
                    "b.bin", len(second), hashlib.sha256(second).hexdigest()
                ),
            ),
            member="a.bin",
            adapter="pt-dict",
            source="a test",
            recorded="2026-09-24",
            licence="test",
        )

        class Drops(io.BytesIO):
            def read(self, size: int = -1) -> bytes:
                raise ConnectionResetError("network dropped")

            def __enter__(self) -> Drops:
                return self

            def __exit__(self, *_: object) -> None:
                self.close()

        served = iter([_Response(first), Drops()])
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.dict(public_logs.PUBLIC_LOGS, {"toy-files": entry}),
        ):
            with self.assertRaises(ConnectionResetError):
                public_logs.fetch(
                    "toy-files", Path(tmp), opener=lambda *a, **k: next(served)
                )
            self.assertIsNone(public_logs.locate("toy-files", Path(tmp)))
            self.assertEqual([p.name for p in Path(tmp).rglob("*") if p.is_file()], [])
            # a later fetch starts clean and serves only what checks
            again = iter([_Response(first), _Response(second)])
            out = public_logs.fetch(
                "toy-files", Path(tmp), opener=lambda *a, **k: next(again)
            )
            self.assertEqual(out.read_bytes(), first)
            self.assertEqual(public_logs.locate("toy-files", Path(tmp)), out)

    def test_a_server_that_ignores_the_range_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "ignores ranges"):
            public_logs._range(
                "https://example.invalid/x.zip",
                10,
                4,
                lambda *_a, **_k: _Response(b"the whole archive, not four bytes"),
            )


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
