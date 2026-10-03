"""The robot seam: a recording's shape and round trip; the registry
detecting one adapter per source and refusing ambiguity; the wire
adapter over a real recording; the LeRobot adapter over a dataset laid
out exactly as v3 does; the MCAP adapter over a bag this test WRITES to
the spec (CDR-encoded JointState and Imu), so both ends of the format
are pinned; ingest landing a stamped `recording` in a project."""

from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests._extras import installed, needs_numpy
from trainnr.project import Kind, create_project, index_project
from trainnr.project.ingest import ingest
from trainnr.robots import Channel, Recording, list_adapters, resolve
from trainnr.robots.adapter import detect
from trainnr.robots.adapters import mcap as mcap_mod
from trainnr.robots.recording import JOINT_COMMAND, JOINT_POSITION

REPO = Path(__file__).resolve().parents[2]
WIRE = REPO / "recordings" / "chase-arm-2026-08-17.wire"


# -- helpers: a v3 LeRobot layout and an MCAP bag, written by the tests ----


def make_lerobot(root: Path) -> Path:
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    names = ["shoulder", "elbow"]
    info = {
        "codebase_version": "v3.0",
        "fps": 50,
        "total_episodes": 2,
        "robot_type": "so101",
        "features": {
            "observation.images.top": {"dtype": "video", "shape": [480, 640, 3]},
            "observation.state": {"dtype": "float32", "shape": [2], "names": names},
            "action": {"dtype": "float32", "shape": [2], "names": names},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    # Two episodes of 3 frames; each episode's clock restarts at 0.
    rows = {
        "timestamp": [0.0, 0.02, 0.04, 0.0, 0.02, 0.04],
        "episode_index": [0, 0, 0, 1, 1, 1],
        "observation.state": [
            [0.1, 0.2],
            [0.11, 0.21],
            [0.12, 0.22],
            [0.5, 0.6],
            [0.51, 0.61],
            [0.52, 0.62],
        ],
        "action": [
            [0.1, 0.2],
            [0.12, 0.22],
            [0.14, 0.24],
            [0.5, 0.6],
            [0.52, 0.62],
            [0.54, 0.64],
        ],
    }
    pq.write_table(pa.table(rows), root / "data" / "chunk-000" / "file-000.parquet")
    return root


def _cdr_string(s: str) -> bytes:
    raw = s.encode() + b"\x00"
    return struct.pack("<I", len(raw)) + raw


class _CdrWriter:
    """Little-endian CDR with the 4-byte encapsulation header; alignment
    is relative to the body start, as the reader assumes."""

    def __init__(self) -> None:
        self.buf = bytearray(b"\x00\x01\x00\x00")

    def _align(self, n: int) -> None:
        rel = len(self.buf) - 4
        self.buf += b"\x00" * ((-rel) % n)

    def u32(self, v: int) -> None:
        self._align(4)
        self.buf += struct.pack("<I", v)

    def i32(self, v: int) -> None:
        self._align(4)
        self.buf += struct.pack("<i", v)

    def f64(self, v: float) -> None:
        self._align(8)
        self.buf += struct.pack("<d", v)

    def string(self, s: str) -> None:
        self._align(4)
        self.buf += _cdr_string(s)

    def header(self, t: float) -> None:
        self.i32(int(t))
        self.u32(round((t - int(t)) * 1e9))
        self.string("base")


def joint_state_cdr(
    t: float, names: list[str], pos: list[float], vel: list[float], eff: list[float]
) -> bytes:
    w = _CdrWriter()
    w.header(t)
    w.u32(len(names))
    for n in names:
        w.string(n)
    for arr in (pos, vel, eff):
        w.u32(len(arr))
        for v in arr:
            w.f64(v)
    return bytes(w.buf)


def imu_cdr(t: float, gyro: list[float]) -> bytes:
    w = _CdrWriter()
    w.header(t)
    for v in (0.0, 0.0, 0.0, 1.0):
        w.f64(v)
    for _ in range(9):
        w.f64(0.0)
    for v in gyro:
        w.f64(v)
    for _ in range(9):
        w.f64(0.0)
    for v in (0.0, 0.0, 9.81):
        w.f64(v)
    for _ in range(9):
        w.f64(0.0)
    return bytes(w.buf)


def _record(op: int, body: bytes) -> bytes:
    return bytes([op]) + struct.pack("<Q", len(body)) + body


def make_mcap(path: Path, *, chunked: bool = False, compression: str = "") -> Path:
    """A bag with /joint_states (3 msgs) and /imu (2 msgs), uncompressed."""
    schema_js = _record(
        mcap_mod.OP_SCHEMA,
        struct.pack("<H", 1)
        + _mcap_str(mcap_mod.JOINT_STATE)
        + _mcap_str("ros2msg")
        + _mcap_bytes(b"string[] name\nfloat64[] position\n"),
    )
    schema_imu = _record(
        mcap_mod.OP_SCHEMA,
        struct.pack("<H", 2)
        + _mcap_str(mcap_mod.IMU)
        + _mcap_str("ros2msg")
        + _mcap_bytes(b""),
    )
    chan_js = _record(
        mcap_mod.OP_CHANNEL,
        struct.pack("<HH", 1, 1)
        + _mcap_str("/joint_states")
        + _mcap_str("cdr")
        + struct.pack("<I", 0),
    )
    chan_imu = _record(
        mcap_mod.OP_CHANNEL,
        struct.pack("<HH", 2, 2)
        + _mcap_str("/imu")
        + _mcap_str("cdr")
        + struct.pack("<I", 0),
    )
    names = ["shoulder", "elbow"]
    msgs = b""
    for i, t in enumerate((10.0, 10.01, 10.02)):
        data = joint_state_cdr(t, names, [0.1 * i, 0.2 * i], [1.0, 2.0], [0.5, 0.6])
        msgs += _record(
            mcap_mod.OP_MESSAGE,
            struct.pack("<HIQQ", 1, i, int(t * 1e9), int(t * 1e9)) + data,
        )
    for i, t in enumerate((10.0, 10.02)):
        data = imu_cdr(t, [0.01, 0.02, 0.03])
        msgs += _record(
            mcap_mod.OP_MESSAGE,
            struct.pack("<HIQQ", 2, i, int(t * 1e9), int(t * 1e9)) + data,
        )
    records = schema_js + schema_imu + chan_js + chan_imu + msgs
    if chunked:
        inner = records
        if compression:
            inner = (
                b"\x00" * 8
            )  # pretend-compressed bytes: the reader must refuse by name
        chunk_body = (
            struct.pack("<QQQI", 0, 0, len(records), 0)
            + _mcap_str(compression)
            + struct.pack("<Q", len(inner))
            + inner
        )
        records = _record(mcap_mod.OP_CHUNK, chunk_body)
    header = _record(mcap_mod.OP_HEADER, _mcap_str("ros2") + _mcap_str("test"))
    data_end = _record(mcap_mod.OP_DATA_END, struct.pack("<I", 0))
    footer = _record(mcap_mod.OP_FOOTER, struct.pack("<QQI", 0, 0, 0))
    path.write_bytes(
        mcap_mod.MAGIC + header + records + data_end + footer + mcap_mod.MAGIC
    )
    return path


def _mcap_str(s: str) -> bytes:
    raw = s.encode()
    return struct.pack("<I", len(raw)) + raw


def _mcap_bytes(b: bytes) -> bytes:
    return struct.pack("<I", len(b)) + b


# -- the recording -----------------------------------------------------------


class TheRecording(unittest.TestCase):
    def test_a_channel_refuses_misaligned_or_unordered_times(self) -> None:
        with self.assertRaisesRegex(ValueError, "times vs"):
            Channel("x", np.array([0.0, 1.0]), np.zeros((3, 1)))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            Channel("x", np.array([0.0, 0.0]), np.zeros((2, 1)))
        with self.assertRaisesRegex(ValueError, "component names"):
            Channel("x", np.array([0.0, 1.0]), np.zeros((2, 2)), components=("a",))

    def test_write_then_read_round_trips_and_rates_are_honest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            times = np.linspace(0.0, 1.0, 51)
            rec = Recording(
                source="s",
                adapter="test",
                channels={
                    JOINT_POSITION: Channel(
                        JOINT_POSITION, times, np.ones((51, 2)), "rad", ("a", "b")
                    ),
                    "single": Channel("single", np.array([0.0]), np.zeros((1, 1))),
                },
                census={"joints": 2},
                notes=["a note"],
            )
            rec.write(Path(tmp) / "r")
            back = Recording.read(Path(tmp) / "r")
            self.assertEqual(back.channels[JOINT_POSITION].components, ("a", "b"))
            self.assertAlmostEqual(back.channels[JOINT_POSITION].rate_hz or 0, 50.0)
            self.assertIsNone(back.channels["single"].rate_hz)
            self.assertEqual(back.census, {"joints": 2})
            self.assertEqual(back.notes, ["a note"])
            self.assertAlmostEqual(back.duration_s, 1.0)

    def test_excitation_holds_the_command_at_each_measurement(self) -> None:
        cmd = Channel("cmd", np.array([0.0, 1.0]), np.array([[10.0], [20.0]]))
        meas = Channel(
            "m", np.array([0.0, 0.5, 1.0, 1.5]), np.array([[1.0], [2.0], [3.0], [4.0]])
        )
        rec = Recording("s", "t", {"cmd": cmd, "m": meas})
        t, u, y = rec.excitation(controls="cmd", measurements="m")
        self.assertEqual(list(u.ravel()), [10.0, 10.0, 20.0, 20.0])
        self.assertEqual(len(t), 4)
        self.assertEqual(y.shape, (4, 1))


# -- the registry -------------------------------------------------------------


class TheRegistry(unittest.TestCase):
    def test_the_builtins_register_and_resolve(self) -> None:
        names = set(list_adapters())
        self.assertTrue({"wire", "lerobot", "mcap", "mocap"} <= names, names)
        self.assertEqual(resolve("wire").name, "wire")
        with self.assertRaisesRegex(KeyError, "unknown robot adapter"):
            resolve("carrier-pigeon")

    def test_detect_picks_exactly_one_or_refuses_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bag = make_mcap(Path(tmp) / "a.mcap")
            self.assertEqual(detect(bag).name, "mcap")
            self.assertEqual(detect(WIRE).name, "wire")
            with self.assertRaisesRegex(ValueError, "no robot adapter reads"):
                detect(Path(tmp) / "notes.txt")


# -- the adapters -------------------------------------------------------------


class Wire(unittest.TestCase):
    def test_a_real_rig_recording_becomes_channels_on_the_50hz_clock(self) -> None:
        rec = resolve("wire").build().read(WIRE)
        self.assertEqual(rec.adapter, "wire")
        self.assertIn(JOINT_COMMAND, rec.channels)
        ticks = rec.channels["wheel.ticks"]
        self.assertEqual(ticks.components, ("left", "right"))
        self.assertGreater(len(ticks.times), 10)
        self.assertEqual(rec.census["status_frames"], len(ticks.times))
        self.assertEqual(rec.channels["servo.pulse_us"].values.shape[1], 3)
        self.assertTrue(any("50 Hz" in n for n in rec.notes))

    def test_the_arm_joint_stream_is_refused_by_name_not_silently(self) -> None:
        with self.assertRaisesRegex(ValueError, "J"):
            resolve("wire").build().read(REPO / "recordings" / "bench-two-joint.wire")


@needs_numpy
@unittest.skipUnless(installed("pyarrow"), "needs pyarrow (the lerobot extra)")
class LeRobot(unittest.TestCase):
    def test_a_v3_dataset_yields_state_and_action_with_joint_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_lerobot(Path(tmp) / "ds")
            self.assertEqual(detect(root).name, "lerobot")
            rec = resolve("lerobot").build().read(root)
            pos = rec.channels[JOINT_POSITION]
            self.assertEqual(pos.components, ("shoulder", "elbow"))
            self.assertEqual(pos.values.shape, (6, 2))
            # Two episodes stitched onto one strictly increasing clock.
            self.assertTrue(np.all(np.diff(pos.times) > 0))
            self.assertEqual(rec.census["episodes"], 2)
            self.assertEqual(rec.census["cameras"], ["observation.images.top"])
            self.assertIn("assumed", pos.unit)


class Mcap(unittest.TestCase):
    def test_a_bag_written_to_the_spec_decodes_joint_state_and_imu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bag = make_mcap(Path(tmp) / "walk.mcap")
            rec = resolve("mcap").build().read(bag)
            pos = rec.channels[JOINT_POSITION]
            self.assertEqual(pos.components, ("shoulder", "elbow"))
            np.testing.assert_allclose(pos.values[:, 0], [0.0, 0.1, 0.2])
            np.testing.assert_allclose(pos.times, [10.0, 10.01, 10.02], atol=1e-6)
            self.assertIn("N*m", rec.channels["joint.effort"].unit)
            gyro = rec.channels["imu.angular_velocity"]
            np.testing.assert_allclose(gyro.values[0], [0.01, 0.02, 0.03])
            # sensor_msgs sends x, y, z, w; the channel is w first, MuJoCo's,
            # as every adapter's is (review 2026-09-24: this one was x-first)
            orientation = rec.channels["imu.orientation"]
            self.assertEqual(orientation.components, ("w", "x", "y", "z"))
            np.testing.assert_allclose(orientation.values[0], [1.0, 0.0, 0.0, 0.0])
            self.assertEqual(
                rec.census["topics"]["/joint_states"]["type"], mcap_mod.JOINT_STATE
            )
            self.assertEqual(rec.census["topics"]["/imu"]["messages"], 2)

    def test_an_uncompressed_chunk_reads_and_a_compressed_one_is_refused_by_name(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plain = make_mcap(Path(tmp) / "plain.mcap", chunked=True)
            rec = resolve("mcap").build().read(plain)
            self.assertEqual(rec.channels[JOINT_POSITION].values.shape, (3, 2))
            zstd = make_mcap(Path(tmp) / "zstd.mcap", chunked=True, compression="zstd")
            try:
                import zstandard  # noqa: F401, PLC0415

                self.skipTest("zstandard installed: the refusal path is not reachable")
            except ImportError:
                pass
            with self.assertRaisesRegex(mcap_mod.UnsupportedBagError, "zstd"):
                resolve("mcap").build().read(zstd)

    def test_a_non_mcap_file_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "x.mcap"
            bad.write_bytes(b"not a bag")
            with self.assertRaisesRegex(ValueError, "bad magic"):
                resolve("mcap").build().read(bad)


# -- ingest ---------------------------------------------------------------------


class Ingest(unittest.TestCase):
    def test_a_bag_lands_as_a_stamped_recording_and_lights_the_first_state(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = create_project(Path(tmp) / "p", "p")
            bag = make_mcap(Path(tmp) / "go2-walk.mcap")
            record = ingest(project, bag)
            self.assertTrue(record["stamp"].startswith("go2-walk@"))
            self.assertEqual(record["adapter"], "mcap")
            self.assertEqual(record["path"], "recordings/go2-walk")
            self.assertEqual(
                {c["name"] for c in record["channels"]} >= {JOINT_POSITION}, True
            )
            index = index_project(project)
            rec = index.by_kind(Kind.RECORDING)[0]
            self.assertEqual(rec.stamp, record["stamp"])
            self.assertEqual(rec.summary["format"], "mcap")
            self.assertEqual(rec.summary["channels"], len(record["channels"]))
            self.assertIn(
                "telemetry recorded", {s.name for s in index.states if s.present}
            )
            with self.assertRaises(FileExistsError):
                ingest(project, bag)

    def test_a_named_adapter_and_a_bad_name_are_honoured_and_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = create_project(Path(tmp) / "p", "p")
            record = ingest(project, WIRE, name="bench", adapter="wire")
            self.assertTrue(record["stamp"].startswith("bench@"))
            with self.assertRaises(ValueError):
                ingest(project, WIRE, name="a/b")


if __name__ == "__main__":
    unittest.main()
