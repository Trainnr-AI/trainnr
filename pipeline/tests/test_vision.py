"""The vision rollout: pixels in, controls out, same pairing rules."""

import importlib.util
import unittest

MUJOCO_PRESENT = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(MUJOCO_PRESENT, "sim extra not installed (uv sync --extra sim)")
class VisionRollout(unittest.TestCase):
    def _task_and_backend(self):
        from rq_pipeline.physics.mujoco_backend import MuJoCoBackend  # noqa: PLC0415
        from rq_pipeline.tasks.so101 import build_reach  # noqa: PLC0415

        task = build_reach()
        backend = MuJoCoBackend()
        backend.load_spec(task.spec)
        return task, backend

    def test_observation_matches_the_armnetbench_contract(self) -> None:
        import numpy as np  # noqa: PLC0415

        from rq_pipeline.evaluate.vision import ARMNETBENCH_CAMERAS  # noqa: PLC0415

        _, backend = self._task_and_backend()
        seen: dict = {}

        def probe(step, observation):
            if step == 0:
                seen.update(observation)
            return np.zeros(6)

        backend.closed_loop_vision_rollout(
            backend.default_initial_state(), probe, 4, 2, ARMNETBENCH_CAMERAS
        )
        self.assertEqual(seen["observation.images.front"].shape, (576, 1024, 3))
        self.assertEqual(seen["observation.images.top"].shape, (576, 1024, 3))
        self.assertEqual(seen["observation.images.wrist"].shape, (720, 1280, 3))
        self.assertEqual(seen["observation.images.front"].dtype, np.uint8)
        self.assertEqual(seen["observation.state"].shape, (6,))
        self.assertEqual(seen["observation.state"].dtype, np.float32)
        # The render must not be a black frame — a wrongly-aimed camera
        # scores every policy 0% with no error, the silent-failure shape
        # the census gate exists to catch.
        self.assertGreater(int(seen["observation.images.front"].max()), 20)
        self.assertGreater(int(seen["observation.images.wrist"].max()), 20)

    def test_paired_trials_render_identically(self) -> None:
        import numpy as np  # noqa: PLC0415

        from rq_pipeline.evaluate.vision import ARMNETBENCH_CAMERAS  # noqa: PLC0415

        _, backend = self._task_and_backend()
        frames = []

        def grab(step, observation):
            if step == 0:
                frames.append(observation["observation.images.front"].copy())
            return np.zeros(6)

        for _ in range(2):
            backend.closed_loop_vision_rollout(
                backend.default_initial_state(), grab, 2, 1, ARMNETBENCH_CAMERAS
            )
        self.assertTrue(np.array_equal(frames[0], frames[1]))

    def test_vision_evaluation_reuses_the_harness_rules(self) -> None:
        import numpy as np  # noqa: PLC0415

        from rq_pipeline.evaluate.vision import (  # noqa: PLC0415
            VisionPolicy,
            evaluate_vision_policies,
        )

        task, backend = self._task_and_backend()
        limp = VisionPolicy(name="limp", act=lambda step, obs: np.zeros(6))
        with self.assertRaises(ValueError):
            evaluate_vision_policies(backend, [limp], task.protocol, source="unstamped")
        scores = evaluate_vision_policies(
            backend, [limp], task.protocol, source="so101-reach@testhash"
        )
        self.assertEqual(scores[0].trials, task.protocol.trials)


if __name__ == "__main__":
    unittest.main()
