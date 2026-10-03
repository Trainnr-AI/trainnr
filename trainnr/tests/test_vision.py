"""The ArmnetBench camera rig, rendered through the gymnasium env.

What a released checkpoint expects to see — three cameras at the
dataset's resolutions, six joint positions — must come out of the env
with those shapes, uint8, and not black: a wrongly aimed camera scores
every policy 0% with no error, the silent-failure shape the census gate
exists to catch.
"""

import unittest

from tests._extras import needs_envs

SO101_JOINTS = 6
NOT_BLACK = 20


@needs_envs
class ArmnetBenchRig(unittest.TestCase):
    def test_observation_matches_the_armnetbench_contract(self) -> None:
        import numpy as np  # noqa: PLC0415

        from trainnr.envs.gymnasium_env import TrainnrEnv  # noqa: PLC0415
        from trainnr.tasks.so101 import build_reach  # noqa: PLC0415

        env = TrainnrEnv(build_reach(), source="so101-reach@testhash")
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
