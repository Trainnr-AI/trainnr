"""The wire→identification bridge, and the full Paper 0 rehearsal.

The rehearsal is the point: true drivetrain → synthetic sweep → integer
tick quantization (the rig's actual noise) → the real StatusFrame path →
identify() → truth recovered with honest intervals. When the rig session
happens, only the data source changes.
"""

import importlib.util
import unittest
from math import tau
from pathlib import Path

from rq_pipeline.bundles.profile import load_profile
from rq_pipeline.collect.excitation import drivetrain_excitation
from rq_pipeline.collect.frames import STATUS_HZ
from rq_pipeline.collect.wire import Recording, StatusFrame, parse_recording

MUJOCO_PRESENT = importlib.util.find_spec("mujoco") is not None

REPO_ROOT = Path(__file__).resolve().parents[2]
RIG_BUNDLE = REPO_ROOT / "robots" / "rig-drivetrain"
CHASE_RECORDING = REPO_ROOT / "recordings" / "chase-arm-2026-08-17.wire"

RIG = load_profile(RIG_BUNDLE)
DRIVETRAIN_XML = RIG_BUNDLE / RIG.model_file

# Ground truth for the LEFT wheel, distinct from the XML nominals. The
# RIGHT wheel keeps the XML nominals exactly, so its sensor contributes
# only quantization noise to the residual.
TRUE_LEFT_GEAR = 0.00023
TRUE_LEFT_DAMPING = 0.0009
TRUE_LEFT_FRICTION = 0.0012


def _set_left_gear(spec, parameter):
    spec.actuator("left_motor").gear[0] = parameter.value[0]


def _set_left_damping(spec, parameter):
    spec.joint("left").damping[0] = parameter.value[0]


def _set_left_friction(spec, parameter):
    spec.joint("left").frictionloss = parameter.value[0]


class AdapterOnTheChaseFixture(unittest.TestCase):
    def test_shapes_origins_and_clock(self) -> None:
        # Structure only: chase data is signal-poor for fitting (one
        # scalar duty, per-wheel split unobserved during turns), which is
        # why Paper 0 uses the sweep. The adapter itself must still work.
        recording = parse_recording(CHASE_RECORDING)
        data = drivetrain_excitation(recording, RIG)
        count = len(recording.statuses)
        self.assertEqual(data.controls.shape, (count, 2))
        self.assertEqual(data.measurements.shape, (count, 2))
        self.assertEqual(data.times[0], 0.0)
        self.assertTrue((data.measurements[0] == 0.0).all())
        # The 50 Hz status counter is the clock; sequence gaps are real
        # elapsed time and must be preserved, not compacted.
        self.assertAlmostEqual(
            data.times[-1],
            (recording.statuses[-1].seq - recording.statuses[0].seq) / STATUS_HZ,
        )

    def test_rejects_empty_recording(self) -> None:
        # A nonsense tick scale can no longer reach this function at all:
        # RobotProfile refuses it at construction (see test_profile).
        with self.assertRaises(ValueError):
            drivetrain_excitation(Recording(), RIG)


@unittest.skipUnless(MUJOCO_PRESENT, "sim extra not installed (uv sync --extra sim)")
class PaperZeroRehearsal(unittest.TestCase):
    def _synthesize_sweep(self):
        """Simulate the bench sweep on a TRUE drivetrain, then degrade to
        exactly what the wire carries: integer ticks, integer duty."""
        import mujoco  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        from mujoco import rollout  # noqa: PLC0415

        from rq_pipeline.robot.identify import staged_excitation  # noqa: PLC0415

        spec = mujoco.MjSpec.from_file(str(DRIVETRAIN_XML))
        spec.actuator("left_motor").gear[0] = TRUE_LEFT_GEAR
        spec.joint("left").damping[0] = TRUE_LEFT_DAMPING
        spec.joint("left").frictionloss = TRUE_LEFT_FRICTION
        model = spec.compile()

        times = np.arange(0.0, 24.0, model.opt.timestep)
        wave = staged_excitation(
            times, frequencies_hz=[0.3, 0.9, 2.1, 4.3], peak_amplitude=30.0
        )
        duty = np.rint(45.0 + wave)  # positive, integer — as the wire has it
        controls = np.column_stack([duty, duty])[None, :-1, :]

        data = mujoco.MjData(model)
        initial = np.zeros(
            (1, mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS))
        )
        _state, sensordata = rollout.rollout(model, data, initial, controls)
        radians = sensordata[0]  # (n, 2): left, right

        radians_per_tick = tau / RIG.ticks_per_revolution
        ticks = np.rint(radians / radians_per_tick).astype(int)
        statuses = [
            StatusFrame(
                seq=1000 + step,
                x=0.0,
                y=0.0,
                heading=0.0,
                ticks_left=int(ticks[step, 0]),
                ticks_right=int(ticks[step, 1]),
                errors_left=0,
                errors_right=0,
                duty_percent=int(duty[step]),
                stalled=False,
            )
            for step in range(radians.shape[0])
        ]
        recording = Recording(statuses=statuses)
        return drivetrain_excitation(recording, RIG)

    def test_recovers_left_wheel_from_quantized_ticks(self) -> None:
        from rq_pipeline.robot.identify import (  # noqa: PLC0415
            ParameterSpec,
            identify,
        )

        result = identify(
            DRIVETRAIN_XML.read_text(),
            self._synthesize_sweep(),
            parameters=[
                ParameterSpec(
                    "left_gear",
                    TRUE_LEFT_GEAR,
                    0.00002,
                    0.001,
                    _set_left_gear,
                    initial_guess=0.0001,
                ),
                ParameterSpec(
                    "left_damping",
                    TRUE_LEFT_DAMPING,
                    0.00005,
                    0.01,
                    _set_left_damping,
                    initial_guess=0.0005,
                ),
                ParameterSpec(
                    "left_friction",
                    TRUE_LEFT_FRICTION,
                    0.0,
                    0.01,
                    _set_left_friction,
                    initial_guess=0.0005,
                ),
            ],
        )
        by_name = {p.name: p for p in result.parameters}
        truths = {
            "left_gear": TRUE_LEFT_GEAR,
            "left_damping": TRUE_LEFT_DAMPING,
            "left_friction": TRUE_LEFT_FRICTION,
        }
        for name, truth in truths.items():
            fitted = by_name[name]
            self.assertTrue(
                fitted.pinned, f"{name} should be pinned:\n{result.summary()}"
            )
            # Measured by this test's own runs, and now pinned as the
            # rehearsal's finding: tick quantization of an integrated
            # signal is CORRELATED noise, giving a SYSTEMATIC single-run
            # bias (~2% on gear, ~5% on damping at 12 s; barely improved
            # at 24 s) that iid-assuming intervals cannot cover. So the
            # honest gate is 8% relative accuracy with pinned verdicts —
            # and Paper 0's protocol repeats runs rather than trusting
            # one interval, with these numbers as the ceiling to beat.
            self.assertLessEqual(
                abs(fitted.estimate - truth),
                max(0.08 * truth, 3.0 * fitted.half_width),
                f"{name} estimate {fitted.estimate} vs truth {truth}",
            )


if __name__ == "__main__":
    unittest.main()
