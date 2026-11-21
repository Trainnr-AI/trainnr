"""The `servo-bench` identification method (#101): a servo's actuator
model from a bench recording. A known servo drives a pendulum through
steps and a chirp; its positions are quantized as a hobby servo's encoder
reports them; the fit recovers the servo, writes an actuator bundle in the
library's format, and a walk can name it."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tests._extras import needs_sim, needs_slow

BENCH_XML = """<mujoco><compiler angle="radian"/>
<worldbody><body name="arm"><joint name="servo" axis="0 1 0"/>
<geom type="capsule" fromto="0 0 0 0.08 0 0" size="0.005" mass="0.01"/>
<geom type="sphere" pos="0.08 0 0" size="0.01" mass="0.05"/></body></worldbody>
</mujoco>"""
# stall torque, duty per rad, back-EMF, Coulomb friction, armature, delay
TRUE = np.array([0.25, 40.0, 0.004, 0.008, 2e-5, 0.012])


def synthetic_recording(arm: object, bench: object, seconds: float = 12.0) -> object:
    from trainnr.robot.servo_bench import Block, simulate  # noqa: PLC0415
    from trainnr.robots.recording import (  # noqa: PLC0415
        JOINT_COMMAND,
        JOINT_POSITION,
        Channel,
        Recording,
    )

    t = np.arange(0.0, seconds, 1 / 50)
    half = seconds / 2
    command = np.where((t % 2) < 1, 0.6, -0.6) * (t < half) + (
        t >= half
    ) * 0.5 * np.sin(2 * np.pi * (0.2 + 0.25 * (t - half)) * (t - half))
    q = simulate(arm, bench, TRUE, Block(t, np.zeros_like(t), t, command))
    rng = np.random.default_rng(1)
    step = math.radians(300 / 1024)  # an SCS0009's encoder: 1024 counts on 300 degrees
    measured = np.round((q + rng.normal(0, 0.002, len(q))) / step) * step
    return Recording(
        source="synthetic bench",
        adapter="test",
        basis="simulation",
        channels={
            JOINT_POSITION: Channel(
                JOINT_POSITION, t, measured[:, None], "rad", ("servo",)
            ),
            JOINT_COMMAND: Channel(
                JOINT_COMMAND, t, command[:, None], "rad", ("servo",)
            ),
        },
    )


@needs_sim
class TheServoBench(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.bench_dir = self.root / "bench"
        self.bench_dir.mkdir()
        (self.bench_dir / "bench.xml").write_text(BENCH_XML)
        (self.bench_dir / "bundle.json").write_text(
            json.dumps(
                {
                    "schema": "trainnr-robot/2",
                    "name": "bench",
                    "model_file": "bench.xml",
                }
            )
        )
        (self.bench_dir / "bench.json").write_text(
            json.dumps({"servo": "scs0009", "joint": "servo", "vin": 6.0, "kp": 32})
        )

    def test_the_fit_writes_a_verified_bundle_with_its_firmware(self) -> None:
        from trainnr.robot import servo_bench  # noqa: PLC0415
        from trainnr.robot.actuator_bundle import verify  # noqa: PLC0415

        bench = servo_bench.Bench.load(self.bench_dir)
        arm = servo_bench.Arm.of(self.bench_dir / "bench.xml", "servo")
        recording = self.root / "rec"
        synthetic_recording(arm, bench).write(recording)
        known = servo_bench.BenchFit(
            estimate=TRUE,
            replicates=np.array([TRUE, TRUE]),
            asymptotic=TRUE * 0.01,
            rmse_rad=0.003,
            held_out_rmse_rad=0.005,
            blocks=6,
        )
        with mock.patch.object(servo_bench, "fit_bench", lambda *a, **k: known):
            result, record = servo_bench.ServoBench().fit(self.bench_dir, recording)
        assert record is not None
        self.assertEqual(len(result.pinned), len(servo_bench.PARAMETERS))
        bundle = json.loads(
            (
                self.bench_dir / servo_bench.ACTUATORS_DIR / "scs0009.bundle.json"
            ).read_text()
        )
        self.assertEqual(verify(bundle), [])
        params = bundle["params"]
        # S = kt vin / R and B = kt^2 / R give kt and R back
        self.assertAlmostEqual(params["kt"] * 6.0 / params["R"], TRUE[0])
        self.assertAlmostEqual(params["kt"] ** 2 / params["R"], TRUE[2])
        self.assertAlmostEqual(
            bundle["context"]["firmware"]["error_gain"] * 32, TRUE[1]
        )
        self.assertEqual(bundle["metrics"]["held_out_position_rmse_rad"], 0.005)

    @needs_slow
    def test_the_fit_recovers_the_servo_and_writes_its_bundle(self) -> None:
        from trainnr.robot import servo_bench  # noqa: PLC0415
        from trainnr.robot.actuator_bundle import verify  # noqa: PLC0415

        bench = servo_bench.Bench.load(self.bench_dir)
        arm = servo_bench.Arm.of(self.bench_dir / "bench.xml", "servo")
        recording = self.root / "rec"
        synthetic_recording(arm, bench).write(recording)
        method = servo_bench.ServoBench()
        self.assertIsNone(method.accepts(self.bench_dir, recording))
        # the method's own search on 12 s of steps and a chirp (about a
        # minute); a shorter search misses the gain's valley
        with mock.patch.object(servo_bench, "REPLICATES", 2):
            result, record = method.fit(self.bench_dir, recording)
        found = {p.name: p.estimate for p in result.parameters}
        for name, true in zip(
            [p[0] for p in servo_bench.PARAMETERS], TRUE, strict=True
        ):
            if name == "armature":  # small beside the arm's 3.4e-4 kg m^2
                self.assertLess(abs(found[name] - true), 2e-5, name)
            elif name == "command_delay":
                self.assertLess(abs(found[name] - true), 0.003, name)
            else:
                self.assertLess(abs(found[name] / true - 1), 0.1, name)
        assert record is not None
        bundle_file = self.bench_dir / servo_bench.ACTUATORS_DIR / "scs0009.bundle.json"
        bundle = json.loads(bundle_file.read_text())
        self.assertEqual(verify(bundle), [])  # in the library's format, verified
        self.assertEqual(bundle["params"]["model"], "m1")
        firmware = bundle["context"]["firmware"]
        self.assertAlmostEqual(firmware["error_gain"] * 32, found["duty_per_rad"])
        self.assertLess(bundle["metrics"]["held_out_position_rmse_rad"], 0.02)

    def test_what_the_bench_lacks_is_said(self) -> None:
        from trainnr.robot.servo_bench import ServoBench  # noqa: PLC0415

        (self.bench_dir / "bench.json").unlink()
        self.assertIn(
            "bench.json", ServoBench().accepts(self.bench_dir, self.root) or ""
        )


if __name__ == "__main__":
    unittest.main()
