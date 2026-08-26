"""The aloha2-nominal bundle: census, keyframe, and a live hold test.

Same contract shape as the so101 bundle test, one rung up: two arms,
fourteen servos, six cameras. What these tests pin is the CONTRACT —
the model loads, the census matches the real rig's topology (the
cameras included, since a vision policy on a camera-less model fails
silently), the wrapper adds sensors and nothing else, and a closed-loop
hold at the upstream keyframe holds through the harness path.
"""

import importlib.util
import unittest
from pathlib import Path

MUJOCO_PRESENT = importlib.util.find_spec("mujoco") is not None

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE = REPO_ROOT / "robots" / "aloha2-nominal"

ARMS = 2
SERVOS_PER_ARM = 7  # six joints + the gripper drive; the second finger mimics
ACTUATORS = ARMS * SERVOS_PER_ARM
SENSORS = 2 * ACTUATORS  # jointpos + jointvel per servo, added by our wrapper
CAMERAS = 6  # two wrist + overhead + worm's-eye (D405-matched) + two POV
# Upstream keyframe `neutral_pose`, ctrl half (14): left arm then right.
NEUTRAL_KEYFRAME = "neutral_pose"
NEUTRAL_CTRL = [0, -0.96, 1.16, 0, -0.3, 0, 0.0084] * ARMS


@unittest.skipUnless(MUJOCO_PRESENT, "sim extra not installed (uv sync --extra sim)")
class BundleContract(unittest.TestCase):
    def _backend(self):
        from rq_pipeline.physics.mujoco_backend import MuJoCoBackend  # noqa: PLC0415

        backend = MuJoCoBackend()
        backend.load_mjcf(BUNDLE / "aloha2.xml")
        return backend

    def test_census_matches_the_real_rig(self) -> None:
        counts = self._backend().counts()
        self.assertEqual(counts.actuators, ACTUATORS)
        self.assertEqual(counts.sensors, SENSORS)
        self.assertEqual(counts.cameras, CAMERAS)
        self.assertGreater(counts.geoms, 80)  # two arms + the whole frame (95)

    def test_upstream_files_are_untouched_by_the_wrapper(self) -> None:
        # The wrapper adds sensors and nothing else: loading the raw
        # upstream scene must differ from the bundle model ONLY in sensor
        # count. If this fails, someone edited Menagerie's files in place
        # and the provenance story is broken.
        from rq_pipeline.physics.mujoco_backend import MuJoCoBackend  # noqa: PLC0415

        upstream = MuJoCoBackend()
        upstream.load_mjcf(BUNDLE / "scene.xml")
        ours = self._backend().counts()
        theirs = upstream.counts()
        self.assertEqual(theirs.sensors, 0)
        self.assertEqual(
            (ours.actuators, ours.geoms, ours.cameras),
            (theirs.actuators, theirs.geoms, theirs.cameras),
        )

    def test_the_arm_alone_has_no_cameras(self) -> None:
        # The operator's question, pinned: cameras are rig content. The
        # arm file inside the scene carries the two wrist cameras and the
        # two viewpoint cameras; the frame-mounted overhead and worm's-eye
        # cameras live in scene.xml only.
        from rq_pipeline.physics.mujoco_backend import MuJoCoBackend  # noqa: PLC0415

        arms_only = MuJoCoBackend()
        arms_only.load_mjcf(BUNDLE / "aloha.xml")
        self.assertEqual(arms_only.counts().cameras, CAMERAS - 2)

    def test_the_neutral_keyframe_is_where_episodes_start(self) -> None:
        # The reset state (all zeros) has both arms straight up, and the
        # path from there to neutral crosses: the grippers meet at the
        # top centre and jam at over 1 kN — measured. So ALOHA protocols
        # name `neutral_pose` as home, and the backend must serve it.
        import numpy as np  # noqa: PLC0415

        backend = self._backend()
        zeros = backend.default_initial_state()
        neutral = backend.keyframe_state(NEUTRAL_KEYFRAME)
        joints = slice(1, 1 + 2 * (SERVOS_PER_ARM + 1))  # 16 joints follow time
        self.assertTrue(np.allclose(zeros[joints], 0.0))
        self.assertAlmostEqual(float(neutral[joints][1]), -0.96, places=3)
        self.assertAlmostEqual(float(neutral[joints][2]), 1.16, places=3)
        with self.assertRaises(KeyError):
            backend.keyframe_state("no_such_pose")

    def test_closed_loop_hold_at_neutral_through_the_harness_path(self) -> None:
        # Position actuators at the neutral keyframe's ctrl, from the
        # keyframe state: both arms must stay put under gravity — the
        # identified gains against the identified masses. Exercises the
        # exact stepping path the task suite uses.
        import numpy as np  # noqa: PLC0415

        from rq_pipeline.evaluate.harness import run_sensor_episode  # noqa: PLC0415

        backend = self._backend()

        def hold_neutral(step, sensordata):
            return NEUTRAL_CTRL

        _states, sensors = run_sensor_episode(
            backend,
            hold_neutral,
            backend.keyframe_state(NEUTRAL_KEYFRAME),
            steps=1000,
            control_interval=10,
        )
        final_positions = sensors[-1, :ACTUATORS]
        self.assertLess(
            float(np.max(np.abs(final_positions - np.array(NEUTRAL_CTRL)))), 0.05
        )
        final_velocities = sensors[-1, ACTUATORS:]
        self.assertLess(float(np.max(np.abs(final_velocities))), 0.05)

    def test_closed_loop_travel_from_neutral(self) -> None:
        # The controller must also MOVE: from neutral, command both arms
        # to a nearby pose (shoulders forward, elbows opened) that does
        # not cross the centre line, and arrive.
        import numpy as np  # noqa: PLC0415

        from rq_pipeline.evaluate.harness import run_sensor_episode  # noqa: PLC0415

        backend = self._backend()
        target = [0.2, -0.7, 0.9, 0.0, -0.5, 0.3, 0.02] * ARMS

        def go(step, sensordata):
            return target

        _states, sensors = run_sensor_episode(
            backend,
            go,
            backend.keyframe_state(NEUTRAL_KEYFRAME),
            steps=1000,
            control_interval=10,
        )
        self.assertLess(
            float(np.max(np.abs(sensors[-1, :ACTUATORS] - np.array(target)))), 0.05
        )


if __name__ == "__main__":
    unittest.main()
