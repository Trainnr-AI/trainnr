"""The MuJoCo backend against real physics, plus the census gate wired live.

Runs only with the `sim` extra; the fast gate skips it loudly.
"""

import unittest

from rq_pipeline.robot.model_checks import DeadModelError, assert_model_alive
from tests._extras import needs_sim

# A minimal but complete robot: one hinge pendulum with a motor and a
# joint-position sensor. Enough to satisfy the census and to fall under
# gravity when uncontrolled.
PENDULUM = """
<mujoco>
  <option timestep="0.01"/>
  <worldbody>
    <body>
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom name="rod" type="capsule" fromto="0 0 0 0 0 -0.5" size="0.02"
            mass="1"/>
    </body>
  </worldbody>
  <actuator><motor joint="hinge"/></actuator>
  <sensor><jointpos joint="hinge"/></sensor>
</mujoco>
"""

# A plausible file that is not a robot: geometry, no actuators, no
# sensors — the USD-import failure shape, as MJCF.
INERT_SCENE = """
<mujoco>
  <worldbody>
    <body><geom name="box" type="box" size="0.1 0.1 0.1" mass="1"/></body>
  </worldbody>
</mujoco>
"""


@needs_sim
class Census(unittest.TestCase):
    def _backend(self, xml: str):
        from rq_pipeline.physics.mujoco_backend import (  # noqa: PLC0415
            MuJoCoBackend,
        )

        backend = MuJoCoBackend()
        backend.load_mjcf_string(xml)
        return backend

    def test_live_census_passes_the_gate(self) -> None:
        counts = self._backend(PENDULUM).counts()
        self.assertEqual(counts.actuators, 1)
        self.assertEqual(counts.sensors, 1)
        assert_model_alive(
            counts.actuators, counts.sensors, counts.geoms, source="pendulum"
        )

    def test_inert_model_refused_by_the_gate(self) -> None:
        # This is the whole point of the census: the model LOADS without
        # error, and the gate still refuses it.
        counts = self._backend(INERT_SCENE).counts()
        with self.assertRaises(DeadModelError):
            assert_model_alive(
                counts.actuators, counts.sensors, counts.geoms, source="inert"
            )

    def test_rollout_batches_and_gravity_acts(self) -> None:
        import numpy as np  # noqa: PLC0415

        backend = self._backend(PENDULUM)
        home = backend.default_initial_state()
        batch = np.stack([home, home])
        steps = 100
        controls = np.zeros((2, steps, 1))
        states = backend.rollout(batch, controls)
        self.assertEqual(states.shape[0], 2)
        self.assertEqual(states.shape[1], steps)
        # Identical inputs, identical trajectories — determinism.
        self.assertTrue(np.array_equal(states[0], states[1]))
        # The pendulum starts horizontal (fromto along -z is vertical at
        # qpos 0 — so instead assert it MOVED under gravity if displaced,
        # and stayed put at exact equilibrium otherwise. Displace batch 1:
        displaced = home.copy()
        displaced[1] += 0.3  # qpos follows time in FULLPHYSICS layout
        states = backend.rollout(np.stack([home, displaced]), np.zeros((2, steps, 1)))
        motion_home = np.ptp(states[0, :, 1])
        motion_displaced = np.ptp(states[1, :, 1])
        self.assertGreater(motion_displaced, motion_home)
        self.assertGreater(motion_displaced, 0.05)

    def test_shape_errors_are_loud(self) -> None:
        import numpy as np  # noqa: PLC0415

        backend = self._backend(PENDULUM)
        home = backend.default_initial_state()
        with self.assertRaises(ValueError):
            backend.rollout(home, np.zeros((1, 10, 1)))  # 1-D initial
        with self.assertRaises(ValueError):
            backend.rollout(
                np.stack([home]),
                np.zeros((2, 10, 1)),  # nbatch mismatch
            )


@needs_sim
class SatisfiesTheHarness(unittest.TestCase):
    def test_the_cpu_backend_is_an_engine(self) -> None:
        """The harness declares `Engine`; the metrology instrument must
        satisfy it structurally, checked here rather than trusted."""
        from rq_pipeline.evaluate.harness import Engine  # noqa: PLC0415
        from rq_pipeline.physics.mujoco_backend import MuJoCoBackend  # noqa: PLC0415

        self.assertIsInstance(MuJoCoBackend(), Engine)

    def test_its_stepper_is_a_stepper(self) -> None:
        from rq_pipeline.evaluate.harness import Stepper  # noqa: PLC0415
        from rq_pipeline.physics.mujoco_backend import MuJoCoBackend  # noqa: PLC0415

        backend = MuJoCoBackend()
        backend.load_mjcf_string(PENDULUM)
        stepper = backend.stepper(backend.default_initial_state(), 3)
        self.assertIsInstance(stepper, Stepper)
        self.assertEqual(stepper.sensordata.shape, (backend.model.nsensordata,))
        self.assertEqual(dict(stepper.extras), {})

    def test_a_protocol_needing_what_the_engine_lacks_is_refused_first(self) -> None:
        """`observables` is the seam's hook for engines that expose more
        than the row (a deformable's particles): a protocol declares
        what it needs, the engine what it has, and a mismatch is refused
        naming both — before a trial is spent."""
        from rq_pipeline.evaluate.harness import (  # noqa: PLC0415
            SimPolicy,
            evaluate_policies,
        )
        from rq_pipeline.physics.mujoco_backend import MuJoCoBackend  # noqa: PLC0415
        from rq_pipeline.protocol import (  # noqa: PLC0415
            EpisodeProtocol,
            protocol_fields,
        )

        backend = MuJoCoBackend()
        backend.load_mjcf_string(PENDULUM)
        width = backend.model.nu

        def protocol(**extra):
            return EpisodeProtocol(
                trials=1,
                steps=5,
                control_interval=1,
                perturb=lambda _trial, home: home,
                success=lambda _states, _sensors: True,
                **extra,
            )

        self.assertNotIn("observables", protocol_fields(protocol()))
        needy = protocol(observables=("particle_q",))
        self.assertEqual(protocol_fields(needy)["observables"], ["particle_q"])
        with self.assertRaises(ValueError) as caught:
            evaluate_policies(
                backend,
                [SimPolicy("limp", lambda _step, _sensors: [0.0] * width)],
                needy,
                source="pendulum@000000000000",
            )
        self.assertIn("particle_q", str(caught.exception))
        self.assertIn(backend.instrument, str(caught.exception))


if __name__ == "__main__":
    unittest.main()
