"""Task acceptance: the shipped kitting task is a test; a variant the
expert cannot do is rejected with the funnel as the reason; a variant
that rewards doing nothing is rejected as no test at all."""

import importlib.util
import unittest
from dataclasses import replace

MUJOCO_PRESENT = importlib.util.find_spec("mujoco") is not None
STAMP = "aloha2-nominal@000000000000"
OUT_OF_REACH_Y = 0.9  # the tray a metre from the arms' base line
ANYWHERE_M = 10.0  # an in-slot radius that covers the whole table
# Measured 2026-08-27 on mujoco 3.11.0 and 3.12.0 alike: the shipped band's
# paired corners 2 and 3 sit at its near-base edge, where the expert's
# closing-plane orientation is IK-nullspace-random (docs/31's known edge);
# trimming that edge by 6 cm (y_high 0.44 -> 0.38 before the ACT shift)
# makes every corner doable.
SHIPPED_EXPERT_SUCCESSES = 2
SHIPPED_FUNNEL = [4, 3, 3, 2]
ACCEPTED_Y_HIGH = 0.38


@unittest.skipUnless(MUJOCO_PRESENT, "sim extra not installed (uv sync --extra sim)")
class TheCriticLoop(unittest.TestCase):
    def _accept(self, spec):
        from rq_pipeline.tasks.acceptance import accept  # noqa: PLC0415
        from rq_pipeline.tasks.aloha2 import (  # noqa: PLC0415
            build_kitting,
            scripted_kitting_episode,
        )

        task = build_kitting(spec=spec)
        return task, accept(
            task,
            lambda model, initial: scripted_kitting_episode(model, initial, spec=spec),
            source=STAMP,
        )

    def test_the_shipped_kitting_task_fails_its_own_review(self) -> None:
        """The loop's first real verdict: the shipped spec is NOT
        accepted — its expert passes two of the four paired corners,
        and the funnel says where the other two stop. Pinned as the
        measured fact it is; the decision on the band is the
        operator's (docs/07 2026-08-27)."""
        from rq_pipeline.tasks.aloha2 import KITTING_SPEC  # noqa: PLC0415

        task, verdict = self._accept(KITTING_SPEC)
        self.assertFalse(verdict.accepted)
        self.assertEqual(verdict.expert_successes, SHIPPED_EXPERT_SUCCESSES)
        self.assertEqual(verdict.floor_successes, 0)
        self.assertEqual(verdict.refusals, ())
        self.assertEqual(verdict.funnel["expert"], SHIPPED_FUNNEL)
        self.assertEqual(verdict.task, task.stamp)
        self.assertRegex(task.stamp, r"^kitting@[0-9a-f]{12}$")
        self.assertIn("funnel", str(verdict))

    def test_the_near_base_edge_trimmed_is_accepted(self) -> None:
        """The accept path, and the measured fix: 6 cm off the band's
        near-base edge and the expert climbs the whole funnel on every
        paired corner, while the floor still passes nothing."""
        from rq_pipeline.tasks.aloha2 import (  # noqa: PLC0415
            ACT_SIM_Y_SHIFT,
            KITTING_SPEC,
        )

        trimmed = replace(
            KITTING_SPEC,
            part_spawn={
                arm: (xs, (ys[0], ACCEPTED_Y_HIGH + ACT_SIM_Y_SHIFT))
                for arm, (xs, ys) in KITTING_SPEC.part_spawn.items()
            },
        )
        task, verdict = self._accept(trimmed)
        self.assertTrue(verdict.accepted, str(verdict))
        self.assertEqual(verdict.expert_successes, verdict.trials)
        self.assertEqual(verdict.floor_successes, 0)
        self.assertEqual(verdict.funnel["expert"], [verdict.trials] * 4)
        self.assertNotEqual(task.stamp, build_stamp())

    def test_a_tray_out_of_reach_is_rejected_with_the_reason(self) -> None:
        from rq_pipeline.tasks.aloha2 import KITTING_SPEC  # noqa: PLC0415

        far = replace(KITTING_SPEC, tray_center=(0.0, OUT_OF_REACH_Y))
        task, verdict = self._accept(far)
        self.assertFalse(verdict.accepted)
        self.assertNotEqual(task.stamp, build_stamp())
        self.assertEqual(verdict.floor_successes, 0)
        text = str(verdict)
        self.assertIn("REJECTED", text)
        self.assertTrue(
            verdict.refusals or verdict.expert_successes < verdict.trials, text
        )
        self.assertTrue(any("expert" in reason for reason in verdict.reasons), text)

    def test_a_referee_that_rewards_doing_nothing_is_rejected(self) -> None:
        from rq_pipeline.tasks.aloha2 import KITTING_SPEC  # noqa: PLC0415

        lax = replace(KITTING_SPEC, in_slot_xy_m=ANYWHERE_M, in_slot_z_m=ANYWHERE_M)
        _task, verdict = self._accept(lax)
        self.assertFalse(verdict.accepted)
        self.assertEqual(verdict.floor_successes, verdict.trials)
        self.assertTrue(any("not a test" in reason for reason in verdict.reasons))


def build_stamp() -> str:
    from rq_pipeline.tasks.aloha2 import build_kitting  # noqa: PLC0415

    return build_kitting().stamp


class TheStamp(unittest.TestCase):
    def test_a_spec_is_named_by_its_content(self) -> None:
        from rq_pipeline.bundles.hashing import (  # noqa: PLC0415
            content_stamp,
            require_stamp,
        )

        a = content_stamp("kitting", {"tray": (0.0, -0.02), "steps": 14000})
        b = content_stamp("kitting", {"tray": (0.0, 0.9), "steps": 14000})
        self.assertNotEqual(a, b)
        self.assertEqual(
            a, content_stamp("kitting", {"steps": 14000, "tray": (0.0, -0.02)})
        )
        require_stamp(a)  # the same shape as a bundle's


if __name__ == "__main__":
    unittest.main()
