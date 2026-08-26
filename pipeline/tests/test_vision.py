"""The ArmnetBench camera rig, rendered through the gymnasium env.

What a released checkpoint expects to see — three cameras at the
dataset's resolutions, six joint positions — must come out of the env
with those shapes, uint8, and not black: a wrongly aimed camera scores
every policy 0% with no error, the silent-failure shape the census gate
exists to catch.
"""

import importlib.util
import unittest

SIM_PRESENT = (
    importlib.util.find_spec("mujoco") is not None
    and importlib.util.find_spec("gymnasium") is not None
)
SO101_JOINTS = 6
NOT_BLACK = 20


@unittest.skipUnless(SIM_PRESENT, "sim extra not installed (uv sync --extra sim)")
class ArmnetBenchRig(unittest.TestCase):
    def test_observation_matches_the_armnetbench_contract(self) -> None:
        import numpy as np  # noqa: PLC0415

        from rq_pipeline.envs.robotiq import RobotiqEnv  # noqa: PLC0415
        from rq_pipeline.tasks.so101 import build_reach  # noqa: PLC0415

        env = RobotiqEnv(
            build_reach(),
            state_width=SO101_JOINTS,
            instruction="reach the target",
            source="so101-reach@testhash",
        )
        try:
            observation, _ = env.reset(seed=0)
            pixels = observation["pixels"]
            self.assertEqual(pixels["front"].shape, (576, 1024, 3))
            self.assertEqual(pixels["top"].shape, (576, 1024, 3))
            self.assertEqual(pixels["wrist"].shape, (720, 1280, 3))
            self.assertEqual(pixels["front"].dtype, np.uint8)
            self.assertEqual(observation["agent_pos"].shape, (SO101_JOINTS,))
            self.assertEqual(observation["agent_pos"].dtype, np.float32)
            self.assertGreater(int(pixels["front"].max()), NOT_BLACK)
            self.assertGreater(int(pixels["wrist"].max()), NOT_BLACK)
            self.assertEqual(env.action_space.shape, (SO101_JOINTS,))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
