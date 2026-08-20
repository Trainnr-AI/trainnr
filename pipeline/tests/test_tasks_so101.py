"""The reach task end-to-end: composition, census, and graded policies."""

import importlib.util
import unittest

MUJOCO_PRESENT = importlib.util.find_spec("mujoco") is not None

HOME_CTRL = [0.0, -1.57, 1.57, 1.57, -1.57, 0.0]
REST_CTRL = [0.0, -3.32, 3.11, 1.18, 0.0, -0.174]


@unittest.skipUnless(MUJOCO_PRESENT, "sim extra not installed (uv sync --extra sim)")
class ReachTask(unittest.TestCase):
    def _task_and_backend(self):
        from rq_pipeline.physics.mujoco_backend import MuJoCoBackend  # noqa: PLC0415
        from rq_pipeline.tasks.so101 import build_reach  # noqa: PLC0415

        task = build_reach()
        backend = MuJoCoBackend()
        backend.load_spec(task.spec)
        return task, backend

    def test_composition_census(self) -> None:
        from rq_pipeline.tasks.so101 import FINGERTIP_SLICE  # noqa: PLC0415

        task, backend = self._task_and_backend()
        counts = backend.counts()
        self.assertEqual(counts.actuators, 6)
        self.assertEqual(counts.sensors, 13)  # 12 arm + 1 referee framepos
        self.assertGreater(counts.geoms, 30)  # arm meshes + the table
        self.assertEqual(FINGERTIP_SLICE, slice(12, 15))
        # The computed target is a real point in the workspace, not the
        # origin and not garbage.
        self.assertNotAlmostEqual(sum(abs(v) for v in task.target), 0.0)

    def test_graded_policies_score_in_order(self) -> None:
        from rq_pipeline.evaluate.harness import (  # noqa: PLC0415
            SimPolicy,
            evaluate_policies,
        )

        task, backend = self._task_and_backend()
        policies = [
            SimPolicy("go-home", lambda step, sense: HOME_CTRL),
            SimPolicy("go-rest", lambda step, sense: REST_CTRL),
            SimPolicy("limp", lambda step, sense: [0.0] * 6),
        ]
        scores = evaluate_policies(
            backend, policies, task.protocol, source="so101-reach@000000000000"
        )
        by_name = {score.name: score.score for score in scores}
        # The policy that commands the target posture succeeds from every
        # perturbed start; the two that command other postures never do.
        self.assertEqual(by_name["go-home"], 1.0)
        self.assertEqual(by_name["go-rest"], 0.0)
        self.assertEqual(by_name["limp"], 0.0)


if __name__ == "__main__":
    unittest.main()
