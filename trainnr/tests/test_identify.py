"""The wedge module against ground truth: recover known parameters, and
say honestly when a parameter cannot be recovered.

Runs only with the `sim` extra (mujoco[sysid]).
"""

import unittest

from tests._extras import needs_sim

# A pendulum whose damping we will pretend not to know. The jointpos +
# jointvel sensors are the measurements a real robot's encoder provides.
TRUE_DAMPING = 0.15
PENDULUM = f"""
<mujoco>
  <option timestep="0.005"/>
  <worldbody>
    <body>
      <joint name="hinge" type="hinge" axis="0 1 0" damping="{TRUE_DAMPING}"/>
      <geom name="rod" type="capsule" fromto="0 0 0 0 0 -0.4" size="0.02"
            mass="0.8"/>
    </body>
  </worldbody>
  <actuator><motor joint="hinge"/></actuator>
  <sensor>
    <jointpos joint="hinge"/>
    <jointvel joint="hinge"/>
  </sensor>
</mujoco>
"""


def _set_damping(spec, parameter):
    # Spec fields are arrays; in-place writes survive compile.
    spec.joint("hinge").damping[0] = parameter.value[0]


def _set_unused_friction(spec, parameter):
    # The rod never touches anything, so its sliding friction cannot
    # influence any measurement: unidentifiable BY CONSTRUCTION.
    spec.geom("rod").friction[0] = parameter.value[0]


@needs_sim
class StagedExcitation(unittest.TestCase):
    def test_amplitude_stages_rise_and_respect_peak(self) -> None:
        import numpy as np  # noqa: PLC0415

        from trainnr.robot.identify import staged_excitation  # noqa: PLC0415

        times = np.arange(0.0, 6.0, 0.005)
        signal = staged_excitation(
            times, frequencies_hz=[0.7, 1.9, 3.3], peak_amplitude=2.0, stages=3
        )
        self.assertEqual(times.shape, signal.shape)
        thirds = np.array_split(signal, 3)
        peaks = [float(np.max(np.abs(third))) for third in thirds]
        self.assertLess(peaks[0], peaks[1])
        self.assertLess(peaks[1], peaks[2])
        self.assertLessEqual(peaks[2], 2.0 + 1e-9)


@needs_sim
class Identify(unittest.TestCase):
    def _synthesize(self):
        """Roll out the TRUE model to manufacture 'measured' data."""
        import mujoco  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        from mujoco import rollout  # noqa: PLC0415

        from trainnr.robot.identify import staged_excitation  # noqa: PLC0415

        model = mujoco.MjModel.from_xml_string(PENDULUM)
        data = mujoco.MjData(model)
        signal = staged_excitation(
            np.arange(0.0, 4.0, model.opt.timestep),
            frequencies_hz=[0.5, 1.3, 2.9],
            peak_amplitude=0.6,
            stages=3,
        )
        controls = signal.reshape(1, -1, 1)
        initial = np.zeros(
            (1, mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS))
        )
        state, sensordata = rollout.rollout(model, data, initial, controls[:, :-1, :])
        measured_times = state[0, :, 0]
        # Noiseless synthetic data makes intervals shrink below the
        # optimizer's numerical error, so "truth inside interval" would
        # be an unfair test. Real sensors are noisy (the rig's encoders
        # quantize); seeded noise makes the interval meaningful.
        noise = np.random.default_rng(7).normal(
            0.0, [0.002, 0.02], size=sensordata[0].shape
        )
        measurements = sensordata[0] + noise
        from trainnr.robot.identify import ExcitationData  # noqa: PLC0415

        return ExcitationData(
            times=measured_times,
            controls=signal[:-1].reshape(-1, 1),
            measurements=measurements,
        )

    def test_recovers_true_damping_with_honest_interval(self) -> None:
        from trainnr.robot.identify import (  # noqa: PLC0415
            ParameterSpec,
            identify,
        )

        result = identify(
            PENDULUM,
            self._synthesize(),
            parameters=[
                ParameterSpec(
                    name="damping",
                    nominal=TRUE_DAMPING,
                    min_value=0.01,
                    max_value=1.0,
                    apply=_set_damping,
                    initial_guess=0.5,  # start far from the truth
                )
            ],
        )
        damping = result.parameters[0]
        self.assertAlmostEqual(damping.estimate, TRUE_DAMPING, delta=0.02)
        self.assertTrue(damping.pinned)
        self.assertLessEqual(damping.lower, TRUE_DAMPING)
        self.assertGreaterEqual(damping.upper, TRUE_DAMPING)
        self.assertIn("pinned", result.summary())

    def test_unidentifiable_parameter_reported_not_pretended(self) -> None:
        # THE point of the identifiability report: a parameter the data
        # cannot constrain must come back NOT PINNED, loudly — never a
        # confident number.
        from trainnr.robot.identify import (  # noqa: PLC0415
            ParameterSpec,
            identify,
        )

        result = identify(
            PENDULUM,
            self._synthesize(),
            parameters=[
                ParameterSpec(
                    name="damping",
                    nominal=TRUE_DAMPING,
                    min_value=0.01,
                    max_value=1.0,
                    apply=_set_damping,
                    initial_guess=0.4,
                ),
                ParameterSpec(
                    name="phantom_friction",
                    nominal=0.5,
                    min_value=0.0,
                    max_value=2.0,
                    apply=_set_unused_friction,
                ),
            ],
        )
        by_name = {p.name: p for p in result.parameters}
        self.assertTrue(by_name["damping"].pinned)
        self.assertFalse(by_name["phantom_friction"].pinned)
        self.assertIn("NOT PINNED", result.summary())

    def test_refuses_empty_parameter_list(self) -> None:
        from trainnr.robot.identify import identify  # noqa: PLC0415

        with self.assertRaises(ValueError):
            identify(PENDULUM, self._synthesize(), parameters=[])


if __name__ == "__main__":
    unittest.main()
