"""Paper 2's whole shape, executed: tasks → harness → certify → pool.

Three tasks (lift, block_stack, tool_insert), four NAMED policies run on
every task — the same identity evaluated across a suite, exactly how a
real policy family is — then per-task certificates and the cross-task
pooled correlation. Everything downstream of the sim scores uses the
production statistics; the "real" outcomes are hand-authored but ordered
like reality (expert >> grip-only > no-grip > limp).

The dry run's second job is to make the POWER arithmetic executable:
with n=4 policies per task and 3 tasks, the pooled lower bound cannot
clear a 0.5 gate even at near-perfect agreement — the same arithmetic
that says Paper 2 needs 6-7 retrained policies per task, now asserted
by a test instead of a register row.
"""

import unittest

from tests._extras import needs_sim
from tests._instruments import expected_expert_rate

# Hand-authored real outcomes, one row per task, ordered like reality.
REAL = {
    "lift": {
        "expert": (43, 50),
        "grip-only": (14, 50),
        "no-grip": (5, 50),
        "limp": (1, 50),
    },
    "block_stack": {
        "expert": (21, 50),
        "grip-only": (9, 50),
        "no-grip": (4, 50),
        "limp": (1, 50),
    },
    "tool_insert": {
        "expert": (26, 50),
        "grip-only": (11, 50),
        "no-grip": (3, 50),
        "limp": (2, 50),
    },
}


@needs_sim
class GateADryRun(unittest.TestCase):
    def test_the_full_paper_two_shape(self) -> None:
        from rq_pipeline.evaluate.certificate import certify  # noqa: PLC0415
        from rq_pipeline.evaluate.harness import (  # noqa: PLC0415
            SimPolicy,
            evaluate_policies,
            join_with_real,
        )
        from rq_pipeline.physics.mujoco_backend import MuJoCoBackend  # noqa: PLC0415
        from rq_pipeline.stats.pooling import pool_rank_correlations  # noqa: PLC0415
        from rq_pipeline.tasks.so101 import (  # noqa: PLC0415
            build_insert,
            build_lift,
            build_stack,
            scripted_insert,
            scripted_no_close,
            scripted_pick,
            scripted_stack,
        )

        def grip_only(step, sensordata):
            # Closes on the cube and just holds — never lifts or carries.
            control = list(scripted_pick(min(step, 1399), sensordata))
            return control

        def limp(step, sensordata):
            return [0.0] * 6

        suites = {
            "lift": (build_lift(), scripted_pick),
            "block_stack": (build_stack(), scripted_stack),
            "tool_insert": (build_insert(), scripted_insert),
        }
        certificates = {}
        pooling_input = {}
        for task_name, (task, expert) in suites.items():
            backend = MuJoCoBackend()
            backend.load_spec(task.spec)
            scores = evaluate_policies(
                backend,
                [
                    SimPolicy("expert", expert),
                    SimPolicy("grip-only", grip_only),
                    SimPolicy("no-grip", scripted_no_close),
                    SimPolicy("limp", limp),
                ],
                task.protocol,
                source=f"so101-{task_name}@000000000000",
            )
            by_name = {s.name: s.score for s in scores}
            # The expert must own each task at the rate measured on this
            # instrument (tests/_instruments); the ablations must not.
            self.assertEqual(
                by_name["expert"],
                expected_expert_rate(task_name, backend.instrument, self),
                task_name,
            )
            self.assertEqual(by_name["no-grip"], 0.0, task_name)
            self.assertEqual(by_name["limp"], 0.0, task_name)

            outcomes = join_with_real(scores, REAL[task_name])
            certificates[task_name] = certify(
                robot_bundle="so101-nominal@000000000000",
                scene_bundle=f"so101-{task_name}@000000000000",
                outcomes=outcomes,
                gate_threshold=0.5,
                instrument="mujoco-cpu",
            )
            pooling_input[task_name] = (
                [outcome.sim_score for outcome in outcomes],
                [o.real_successes / o.real_trials for o in outcomes],
            )

        # Per-task: n=4 gives every certificate an exact p, and none can
        # PASS the interval gate — small-n honesty, per task.
        for task_name, certificate in certificates.items():
            self.assertIsNotNone(certificate.exact_p_value, task_name)
            self.assertFalse(certificate.gate_passed, task_name)

        pooled = pool_rank_correlations(pooling_input)
        # Correlation is real and consistent across tasks...
        self.assertGreater(pooled.rho, 0.6)
        self.assertGreater(pooled.heterogeneity_p, 0.5)
        self.assertIsNotNone(pooled.combined_exact_p)
        # ...and the POWER lesson, asserted: 3 tasks x n=4 policies
        # cannot clear a 0.5 pooled gate even with this agreement. This
        # is why Paper 2's design needs 6-7 retrained policies per task.
        self.assertLess(pooled.lower, 0.5)


if __name__ == "__main__":
    unittest.main()
