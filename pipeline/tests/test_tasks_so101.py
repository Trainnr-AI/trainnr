"""The reach task end-to-end: composition, census, and graded policies."""

import unittest

from tests._extras import needs_sim
from tests._instruments import expected_expert_rate

HOME_CTRL = [0.0, -1.57, 1.57, 1.57, -1.57, 0.0]
REST_CTRL = [0.0, -3.32, 3.11, 1.18, 0.0, -0.174]


@needs_sim
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


@needs_sim
class LiftTask(unittest.TestCase):
    def test_graded_pick_ladder(self) -> None:
        import numpy as np  # noqa: PLC0415

        from rq_pipeline.evaluate.harness import (  # noqa: PLC0415
            SimPolicy,
            evaluate_policies,
        )
        from rq_pipeline.physics.mujoco_backend import MuJoCoBackend  # noqa: PLC0415
        from rq_pipeline.tasks.so101 import (  # noqa: PLC0415
            CUBE_Z_STATE_INDEX,
            build_lift,
            scripted_no_close,
            scripted_pick,
        )

        task = build_lift()
        backend = MuJoCoBackend()
        backend.load_spec(task.spec)
        # Pin the privileged-state layout the success predicate reads:
        # cube freejoint first, so its height is state index 3 and the
        # home state has it resting at half-height on the table.
        home = backend.default_initial_state()
        self.assertAlmostEqual(float(np.asarray(home)[CUBE_Z_STATE_INDEX]), 0.015)

        scores = evaluate_policies(
            backend,
            [
                SimPolicy("pick", scripted_pick),
                SimPolicy("no-close", scripted_no_close),
                SimPolicy("limp", lambda step, sense: [0.0] * 6),
            ],
            task.protocol,
            source="so101-lift@000000000000",
        )
        by_name = {score.name: score.score for score in scores}
        # The measured jitter sweep says the pick succeeds from every
        # perturbed pocket position; without the squeeze, nothing lifts.
        self.assertEqual(by_name["pick"], 1.0)
        self.assertEqual(by_name["no-close"], 0.0)
        self.assertEqual(by_name["limp"], 0.0)


@needs_sim
class StackTask(unittest.TestCase):
    def test_graded_stack_ladder(self) -> None:
        from rq_pipeline.evaluate.harness import (  # noqa: PLC0415
            SimPolicy,
            evaluate_policies,
        )
        from rq_pipeline.physics.mujoco_backend import MuJoCoBackend  # noqa: PLC0415
        from rq_pipeline.tasks.so101 import (  # noqa: PLC0415
            build_stack,
            scripted_no_close,
            scripted_stack,
            scripted_stack_no_release,
        )

        task = build_stack()
        backend = MuJoCoBackend()
        backend.load_spec(task.spec)
        scores = evaluate_policies(
            backend,
            [
                SimPolicy("stack", scripted_stack),
                SimPolicy("no-release", scripted_stack_no_release),
                SimPolicy("no-close", scripted_no_close),
            ],
            task.protocol,
            source="so101-stack@000000000000",
        )
        by_name = {score.name: score.score for score in scores}
        # The measured catch basin covers the whole +-2 mm jitter grid
        # (9/9) on the locked MuJoCo, so the scripted stacker goes clear
        # there; the rate is pinned per instrument (tests/_instruments).
        # Carrying without releasing, or never gripping, must never
        # count as a stack.
        self.assertEqual(
            by_name["stack"],
            expected_expert_rate(task.name, backend.instrument, self),
        )
        self.assertEqual(by_name["no-release"], 0.0)
        self.assertEqual(by_name["no-close"], 0.0)


@needs_sim
class InsertTask(unittest.TestCase):
    def test_graded_insert_ladder(self) -> None:
        from rq_pipeline.evaluate.harness import (  # noqa: PLC0415
            SimPolicy,
            evaluate_policies,
        )
        from rq_pipeline.physics.mujoco_backend import MuJoCoBackend  # noqa: PLC0415
        from rq_pipeline.tasks.so101 import (  # noqa: PLC0415
            build_insert,
            scripted_insert,
            scripted_insert_no_release,
            scripted_no_close,
        )

        task = build_insert()
        backend = MuJoCoBackend()
        backend.load_spec(task.spec)
        scores = evaluate_policies(
            backend,
            [
                SimPolicy("insert", scripted_insert),
                SimPolicy("no-release", scripted_insert_no_release),
                SimPolicy("no-close", scripted_no_close),
            ],
            task.protocol,
            source="so101-insert@000000000000",
        )
        by_name = {score.name: score.score for score in scores}
        # 9/9 across the jitter grid in the sizing probes on the locked
        # MuJoCo; the pocket's +-6 mm clearance is the task's precision
        # axis, and the rate is pinned per instrument (tests/_instruments).
        self.assertEqual(
            by_name["insert"],
            expected_expert_rate(task.name, backend.instrument, self),
        )
        self.assertEqual(by_name["no-release"], 0.0)
        self.assertEqual(by_name["no-close"], 0.0)


if __name__ == "__main__":
    unittest.main()
