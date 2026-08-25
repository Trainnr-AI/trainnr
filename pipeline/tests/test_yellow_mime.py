"""The ONE arm reconstruction: pinned so the overlays stay honest.

air_mime_pose/salute_pose are what every viewer draws for the metal's
arm (the wire carries no arm telemetry). A drift here silently shifts
every sim/real overlay — the original sin was a guessed 1.2 s blend
that finished ~14 s before the metal.
"""

import importlib.util
import unittest

MUJOCO_PRESENT = importlib.util.find_spec("mujoco") is not None


class Salute(unittest.TestCase):
    def test_two_full_waves_then_none(self) -> None:
        from rq_pipeline.tasks.yellow import (  # noqa: PLC0415
            SALUTE_TOTAL_S,
            salute_pose,
        )

        self.assertAlmostEqual(SALUTE_TOTAL_S, 1.4)
        self.assertEqual(salute_pose(0.1)[4], 0.35)  # closed first
        self.assertEqual(salute_pose(0.5)[4], -0.5)  # then open
        self.assertIsNone(salute_pose(SALUTE_TOTAL_S))
        self.assertIsNone(salute_pose(-0.1))


class AirMime(unittest.TestCase):
    def test_slew_paced_schedule_and_endpoints(self) -> None:
        from rq_pipeline.tasks.yellow import (  # noqa: PLC0415
            AIR_PICK_SEQUENCE,
            AIR_TUCK,
            SLEW_RAD_PER_S,
            air_mime_pose,
        )

        self.assertEqual(air_mime_pose(0.0), list(AIR_TUCK))
        # Total = every blend (largest joint delta at the firmware's
        # slew) plus every hold; past it the pose parks at the last row.
        total, previous = 0.0, AIR_TUCK
        for pose, hold in AIR_PICK_SEQUENCE:
            delta = max(abs(a - b) for a, b in zip(previous, pose, strict=True))
            total += delta / SLEW_RAD_PER_S + hold
            previous = pose
        self.assertEqual(air_mime_pose(total + 1.0), list(AIR_PICK_SEQUENCE[-1][0]))
        # Mid-first-blend the wrist is strictly between tuck and reach.
        first_blend = (
            max(
                abs(a - b)
                for a, b in zip(AIR_TUCK, AIR_PICK_SEQUENCE[1][0], strict=True)
            )
            / SLEW_RAD_PER_S
        )
        hold0 = AIR_PICK_SEQUENCE[0][1]
        mid = air_mime_pose(hold0 + first_blend / 2)
        self.assertGreater(mid[3], AIR_TUCK[3])
        self.assertLess(mid[3], AIR_PICK_SEQUENCE[1][0][3])


if __name__ == "__main__":
    unittest.main()
