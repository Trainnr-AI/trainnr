"""The so101-nominal bundle: census, keyframes, and a live hold test.

Not a physics validation — every number in the bundle is nominal, and
its README says so. What these tests pin is the CONTRACT: the model
loads, the census matches the real arm's topology, the wrapper's added
sensors report, and a closed-loop hold through the harness path holds.
"""

import unittest
from pathlib import Path

from tests._extras import needs_sim

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE = REPO_ROOT / "robots" / "so101-nominal"

JOINTS = 6
SENSORS = 12  # jointpos + jointvel per joint, added by our wrapper
HOME_QPOS = [0.0, -1.57, 1.57, 1.57, -1.57, 0.0]


@needs_sim
class BundleContract(unittest.TestCase):
    def _backend(self):
        from rq_pipeline.physics.mujoco_backend import MuJoCoBackend  # noqa: PLC0415

        backend = MuJoCoBackend()
        backend.load_mjcf(BUNDLE / "so101.xml")
        return backend

    def test_census_matches_the_real_arm(self) -> None:
        counts = self._backend().counts()
        self.assertEqual(counts.actuators, JOINTS)
        self.assertEqual(counts.sensors, SENSORS)
        self.assertGreater(counts.geoms, 20)  # meshes + collision pads

    def test_upstream_file_is_untouched_by_the_wrapper(self) -> None:
        # The wrapper adds sensors and nothing else: loading the raw
        # upstream file must differ from the bundle model ONLY in sensor
        # count. If this fails, someone edited Menagerie's file in place
        # and the provenance story is broken.
        from rq_pipeline.physics.mujoco_backend import MuJoCoBackend  # noqa: PLC0415

        upstream = MuJoCoBackend()
        upstream.load_mjcf(BUNDLE / "so_arm100.xml")
        ours = self._backend().counts()
        theirs = upstream.counts()
        self.assertEqual(theirs.sensors, 0)
        self.assertEqual((ours.actuators, ours.geoms), (theirs.actuators, theirs.geoms))

    def test_closed_loop_hold_at_home_through_the_harness_path(self) -> None:
        # Position actuators at the home keyframe's ctrl: from the zero
        # state, the arm must travel to home and stay there. Exercises
        # the exact stepping path the task suite uses.
        import numpy as np  # noqa: PLC0415

        from rq_pipeline.evaluate.harness import run_sensor_episode  # noqa: PLC0415

        backend = self._backend()

        def hold_home(step, sensordata):
            return HOME_QPOS

        _states, sensors = run_sensor_episode(
            backend,
            hold_home,
            backend.default_initial_state(),
            steps=600,
            control_interval=5,
        )
        final_positions = sensors[-1, :JOINTS]
        self.assertLess(
            float(np.max(np.abs(final_positions - np.array(HOME_QPOS)))), 0.05
        )
        final_velocities = sensors[-1, JOINTS:]
        self.assertLess(float(np.max(np.abs(final_velocities))), 0.05)


if __name__ == "__main__":
    unittest.main()
