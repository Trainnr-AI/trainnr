"""The LeRobot plugin: our env through LeRobot's own factory and preprocessor.

Runs only in the train venv (`lerobot` + `mujoco` present); the gymnasium
contract itself is pinned in test_envs.py with the sim extra alone.
"""

import importlib.util
import unittest

TRAIN_PRESENT = (
    importlib.util.find_spec("lerobot") is not None
    and importlib.util.find_spec("mujoco") is not None
)
SERVOS = 14
TOP_CAMERA_HW = (480, 640)
ALOHA_TICKS = 400  # 4000 physics steps / control every 10


@unittest.skipUnless(TRAIN_PRESENT, "train extra not installed (use .venv-train)")
class ThroughLeRobot(unittest.TestCase):
    def test_make_env_and_preprocess_observation(self) -> None:
        import torch  # noqa: PLC0415
        from lerobot.envs.factory import make_env  # noqa: PLC0415
        from lerobot.envs.utils import (  # noqa: PLC0415
            env_to_policy_features,
            preprocess_observation,
        )

        from rq_pipeline.envs.lerobot_plugin import RobotiqEnvConfig  # noqa: PLC0415

        cfg = RobotiqEnvConfig(task="transfer_cube")
        self.assertEqual(cfg.type, "robotiq")
        policy_features = env_to_policy_features(cfg)
        self.assertEqual(policy_features["observation.state"].shape, (SERVOS,))
        self.assertEqual(
            policy_features["observation.images.top"].shape, (3, *TOP_CAMERA_HW)
        )
        self.assertEqual(policy_features["action"].shape, (SERVOS,))

        suites = make_env(cfg, n_envs=1, use_async_envs=False)
        vec = suites["robotiq"][0]
        try:
            self.assertEqual(vec.call("_max_episode_steps")[0], ALOHA_TICKS)
            self.assertEqual(
                vec.call("task_description")[0], "transfer the cube to the left gripper"
            )
            observation, _ = vec.reset(seed=[1000])
            batch = preprocess_observation(observation)
            self.assertEqual(tuple(batch["observation.state"].shape), (1, SERVOS))
            self.assertEqual(
                tuple(batch["observation.images.top"].shape), (1, 3, *TOP_CAMERA_HW)
            )
            self.assertEqual(batch["observation.images.top"].dtype, torch.float32)
            self.assertLessEqual(float(batch["observation.images.top"].max()), 1.0)
            _, _, terminated, truncated, info = vec.step(
                vec.action_space.sample() * 0.0
            )
            self.assertFalse(terminated[0])
            self.assertFalse(truncated[0])
            self.assertIn("is_success", info)
        finally:
            vec.close()

    def test_unknown_or_missing_task_is_refused_at_config_time(self) -> None:
        from rq_pipeline.envs.lerobot_plugin import RobotiqEnvConfig  # noqa: PLC0415

        with self.assertRaises(ValueError):
            RobotiqEnvConfig(task="juggling")
        with self.assertRaises(ValueError):
            RobotiqEnvConfig()  # no silent default task

    def test_fps_comes_from_the_task(self) -> None:
        from rq_pipeline.envs.lerobot_plugin import RobotiqEnvConfig  # noqa: PLC0415

        self.assertEqual(RobotiqEnvConfig(task="kitting").fps, 50)
        self.assertEqual(RobotiqEnvConfig(task="reach").fps, 50)


if __name__ == "__main__":
    unittest.main()
