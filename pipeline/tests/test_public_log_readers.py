"""The two readers the public Go2 logs needed: a rosbag2 SQLite bag
decoded by declared CDR layouts, and a torch-saved dict read without
torch. Both build their fixtures by hand from the formats' own rules,
so the tests need no ROS and no torch."""

from __future__ import annotations

import io
import pickle
import sqlite3
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np


def _cdr_joint_state(values: np.ndarray, frame_id: str = "") -> bytes:
    """One `interfaces/msg/JointState` in CDR_LE: header (Time, string),
    then four float64[12] arrays, aligned per the CDR rules."""
    body = struct.pack("<iI", 12, 345)  # stamp
    encoded = frame_id.encode() + b"\x00"
    body += struct.pack("<I", len(encoded)) + encoded
    body += b"\x00" * (-len(body) % 8)  # align the doubles
    for k in range(4):
        body += struct.pack("<12d", *(values + k))
    return b"\x00\x01\x00\x00" + body


class Rosbag2Sqlite(unittest.TestCase):
    def test_decodes_by_layout_with_cdr_alignment(self) -> None:
        from rq_pipeline.robot.rosbag_sqlite import decode  # noqa: PLC0415

        values = np.linspace(-1.0, 1.0, 12)
        for frame in ("", "base", "a_longer_frame_id"):
            message = decode(
                "interfaces/msg/JointState", _cdr_joint_state(values, frame)
            )
            self.assertEqual(message["header"]["frame_id"], frame)
            np.testing.assert_allclose(message["position"], values)
            np.testing.assert_allclose(message["acceleration"], values + 3)

    def test_reads_a_bag_through_the_dfki_profile_and_refuses_an_unknown_one(
        self,
    ) -> None:
        from rq_pipeline.robot.rosbag_sqlite import (  # noqa: PLC0415
            Rosbag2Sqlite,
            read_bag,
        )
        from rq_pipeline.robots.adapter import detect  # noqa: PLC0415
        from rq_pipeline.robots.recording import BASIS_PUBLIC  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            bag = Path(tmp) / "run_0.db3"
            with sqlite3.connect(bag) as db:
                db.execute(
                    "CREATE TABLE topics(id INTEGER PRIMARY KEY, name TEXT, type TEXT, "
                    "serialization_format TEXT, offered_qos_profiles TEXT)"
                )
                db.execute(
                    "CREATE TABLE messages(id INTEGER PRIMARY KEY, topic_id INTEGER, "
                    "timestamp INTEGER, data BLOB)"
                )
                db.execute(
                    "INSERT INTO topics VALUES "
                    "(1, '/joint_states', 'interfaces/msg/JointState', 'cdr', '')"
                )
                for i in range(50):
                    db.execute(
                        "INSERT INTO messages(topic_id, timestamp, data) "
                        "VALUES (1, ?, ?)",
                        (1_000_000 * i, _cdr_joint_state(np.full(12, 0.01 * i))),
                    )
            self.assertEqual(detect(bag).name, "db3")
            recording = Rosbag2Sqlite().read(bag)
            self.assertEqual(recording.basis, BASIS_PUBLIC)
            self.assertEqual(recording.census["profile"], "dfki-go2")
            self.assertEqual(
                recording.channels["joint.position"].components[0], "FL_hip_joint"
            )
            self.assertAlmostEqual(
                recording.channels["joint.position"].rate_hz or 0, 1000.0, delta=1
            )
            with sqlite3.connect(bag) as db:
                db.execute("UPDATE topics SET name = '/somewhere_else'")
            with self.assertRaisesRegex(ValueError, "no rosbag2 profile matches"):
                read_bag(bag)


def _torch_zip(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """A `torch.save`-shaped zip: `data.pkl` referring to storages by
    key, each storage's bytes under `data/<key>`. Written with a fake
    `torch` in `sys.modules` for the pickle's global references only."""
    import sys  # noqa: PLC0415
    import types  # noqa: PLC0415

    torch = types.ModuleType("torch")
    utils = types.ModuleType("torch._utils")

    class FloatStorage:  # the class the pickle names for the dtype
        pass

    def _rebuild_tensor_v2(*args: object) -> object:  # never called here
        return args

    torch.FloatStorage = FloatStorage  # type: ignore[attr-defined]
    utils._rebuild_tensor_v2 = _rebuild_tensor_v2  # type: ignore[attr-defined]
    _rebuild_tensor_v2.__module__ = "torch._utils"
    _rebuild_tensor_v2.__qualname__ = "_rebuild_tensor_v2"
    FloatStorage.__module__ = "torch"
    FloatStorage.__qualname__ = "FloatStorage"

    class Storage:
        def __init__(self, key: str, numel: int) -> None:
            self.key, self.numel = key, numel

    class Tensor:
        def __init__(self, storage: Storage, shape: tuple[int, ...]) -> None:
            self.storage, self.shape = storage, shape

        def __reduce__(self) -> object:
            return (
                _rebuild_tensor_v2,
                (self.storage, 0, self.shape, _strides(self.shape), False, {}),
            )

    class Pickler(pickle.Pickler):
        def persistent_id(self, obj: object) -> object:
            if isinstance(obj, Storage):
                return ("storage", FloatStorage, obj.key, "cpu", obj.numel)
            return None

    saved = dict(sys.modules)
    sys.modules["torch"] = torch
    sys.modules["torch._utils"] = utils
    try:
        storages: dict[str, bytes] = {}
        payload = {}
        for k, (name, array) in enumerate(arrays.items()):
            flat = np.ascontiguousarray(array, dtype="<f4")
            storages[str(k)] = flat.tobytes()
            payload[name] = Tensor(Storage(str(k), flat.size), flat.shape)
        stream = io.BytesIO()
        Pickler(stream, protocol=2).dump(payload)
    finally:
        sys.modules.clear()
        sys.modules.update(saved)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("archive/data.pkl", stream.getvalue())
        for key, raw in storages.items():
            z.writestr(f"archive/data/{key}", raw)


def _strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    out, acc = [], 1
    for n in reversed(shape):
        out.append(acc)
        acc *= n
    return tuple(reversed(out))


class TorchPickle(unittest.TestCase):
    def test_reads_tensors_without_torch_and_refuses_other_globals(self) -> None:
        from rq_pipeline.robot.torch_pickle import load_tensors  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.pt"
            arrays = {
                "time": np.arange(6, dtype=np.float32),
                "dof_pos": np.arange(12, dtype=np.float32).reshape(6, 2),
            }
            _torch_zip(path, arrays)
            loaded = load_tensors(path)
            np.testing.assert_allclose(loaded["time"], arrays["time"])
            np.testing.assert_allclose(loaded["dof_pos"], arrays["dof_pos"])
            evil = Path(tmp) / "evil.pt"
            with zipfile.ZipFile(evil, "w") as z:
                z.writestr(
                    "archive/data.pkl", pickle.dumps({"x": pickle.dumps}, protocol=2)
                )
            with self.assertRaisesRegex(pickle.UnpicklingError, "refused"):
                load_tensors(evil)

    def test_reads_numpy_arrays_saved_in_the_dict(self) -> None:
        from rq_pipeline.robot.torch_pickle import load_tensors  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "n.pt"
            with zipfile.ZipFile(path, "w") as z:
                z.writestr(
                    "archive/data.pkl",
                    pickle.dumps({"kp": np.full(3, 20.0)}, protocol=2),
                )
            np.testing.assert_allclose(load_tensors(path)["kp"], 20.0)


class PtDict(unittest.TestCase):
    def test_the_iit_layout_becomes_a_recording_with_pd_gains(self) -> None:
        from rq_pipeline.robot.pt_dict import PtDict  # noqa: PLC0415
        from rq_pipeline.robots.recording import BASIS_PUBLIC  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "traj_0.pt"
            n = 40
            saved = {
                "time": np.arange(n) * 0.005,
                "dof_pos": np.zeros((n, 12)),
                "dof_vel": np.zeros((n, 12)),
                "des_dof_pos": np.zeros((n, 12)),
                "des_dof_vel": np.zeros((n, 12)),
                "kp": np.full(12, 20.0),
                "kd": np.full(12, 1.5),
            }
            with zipfile.ZipFile(path, "w") as z:
                z.writestr("archive/data.pkl", pickle.dumps(saved, protocol=2))
            recording = PtDict().read(path)
            self.assertEqual(recording.basis, BASIS_PUBLIC)
            self.assertEqual(recording.channels["joint.kp"].values.shape, (n, 12))
            self.assertEqual(
                recording.channels["joint.position"].components[3], "FR_hip_joint"
            )
            with zipfile.ZipFile(path, "w") as z:
                z.writestr(
                    "archive/data.pkl", pickle.dumps({"other": np.zeros(3)}, protocol=2)
                )
            with self.assertRaisesRegex(ValueError, "no .pt layout matches"):
                PtDict().read(path)


class RecordingBasis(unittest.TestCase):
    def test_the_basis_travels_through_the_manifest_and_is_checked(self) -> None:
        from rq_pipeline.robots.recording import (  # noqa: PLC0415
            BASIS_PUBLIC,
            Channel,
            Recording,
        )

        times = np.arange(0, 1.0, 0.1)
        channel = Channel("joint.position", times, np.zeros((10, 1)), "rad", ("j",))
        with self.assertRaisesRegex(ValueError, "basis must be one of"):
            Recording("s", "a", {"joint.position": channel}, basis="hearsay")
        with tempfile.TemporaryDirectory() as tmp:
            Recording("s", "a", {"joint.position": channel}, basis=BASIS_PUBLIC).write(
                Path(tmp) / "r"
            )
            self.assertEqual(Recording.read(Path(tmp) / "r").basis, BASIS_PUBLIC)
