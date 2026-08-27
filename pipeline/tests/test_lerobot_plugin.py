"""The LeRobot plugin: our env through LeRobot's own factory and preprocessor.

Runs only in the train venv (`lerobot` + `mujoco` present); the gymnasium
contract itself is pinned in test_envs.py with the sim extra alone.
"""

import importlib.util
import os
import unittest
from pathlib import Path

TRAIN_PRESENT = (
    importlib.util.find_spec("lerobot") is not None
    and importlib.util.find_spec("mujoco") is not None
)
SERVOS = 14
TOP_CAMERA_HW = (480, 640)
ALOHA_TICKS = 400  # 4000 physics steps / control every 10
EXECUTED_HORIZON = 10  # 0.2 s of a 2 s chunk before the policy is asked again
# A per-machine artifact: the WSL card's T5 checkpoint. Point elsewhere with
# the env var; absent, the checkpoint-backed test skips and says so.
T5_CHECKPOINT = Path(
    os.environ.get(
        "RQ_T5_CHECKPOINT", "runs/t5-act-kitting/checkpoints/020000/pretrained_model"
    )
)


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

    @unittest.skipUnless(
        T5_CHECKPOINT.exists(), "no local T5 checkpoint (runs/ is per machine)"
    )
    def test_load_policy_acts_on_the_env_observation(self) -> None:
        """The library loader: a checkpoint acts on the env's raw
        observation and returns one command per servo. Runs only where
        the T5 checkpoint exists (the WSL card that trained it)."""
        from rq_pipeline.envs.lerobot_policy import (  # noqa: PLC0415
            best_device,
            load_policy,
        )
        from rq_pipeline.envs.robotiq import make_env  # noqa: PLC0415

        env = make_env("kitting")
        try:
            observation, _ = env.reset(seed=1000)
            loaded = load_policy(
                T5_CHECKPOINT, instruction=env.task_description, device=best_device()
            )
            loaded.reset()
            action = loaded.act(observation)
            self.assertEqual(action.shape, (SERVOS,))
            self.assertTrue(env.action_space.contains(action.astype("float32")))
            # The chunk path: the whole prediction, de-normalised, and the
            # scheduler executing it on the protocol's horizon.
            from rq_pipeline.evaluate.scheduler import ActionScheduler  # noqa: PLC0415

            chunk = loaded.predict(observation)
            self.assertEqual(chunk.shape[1], SERVOS)
            self.assertGreaterEqual(chunk.shape[0], 1)
            scheduler = ActionScheduler(
                loaded.as_chunk_policy(), executed_horizon=EXECUTED_HORIZON, nu=SERVOS
            )
            scheduler.reset()
            first = scheduler.act(observation)
            self.assertEqual(first.shape, (SERVOS,))
            self.assertTrue(env.action_space.contains(first.astype("float32")))
        finally:
            env.close()

    def test_cli_flags_spell_the_plugin_once(self) -> None:
        """A tool that drives lerobot-train/eval takes the --env.* flags
        from the plugin, never from its own string."""
        from rq_pipeline.envs.lerobot_plugin import RobotiqEnvConfig  # noqa: PLC0415

        flags = RobotiqEnvConfig.cli_flags(
            "kitting", record_to="runs/x/episodes.jsonl", policy_name="smoke"
        )
        self.assertEqual(flags[0], "--env.type=robotiq")
        self.assertEqual(flags[1], "--env.task=kitting")
        self.assertEqual(flags[2], "--env.discover_packages_path=rq_pipeline.envs")
        self.assertIn("--env.record_to=runs/x/episodes.jsonl", flags)
        self.assertIn("--env.policy_name=smoke", flags)
        self.assertEqual(len(RobotiqEnvConfig.cli_flags("kitting")), 3)

    def test_cli_flags_carry_the_trials(self) -> None:
        from rq_pipeline.envs.lerobot_plugin import RobotiqEnvConfig  # noqa: PLC0415

        flags = RobotiqEnvConfig.cli_flags("kitting", trials=10)
        self.assertIn("--env.trials=10", flags)
        self.assertNotIn(
            "--env.trials", " ".join(RobotiqEnvConfig.cli_flags("kitting"))
        )

    def test_fps_comes_from_the_task(self) -> None:
        from rq_pipeline.envs.lerobot_plugin import RobotiqEnvConfig  # noqa: PLC0415

        self.assertEqual(RobotiqEnvConfig(task="kitting").fps, 50)
        self.assertEqual(RobotiqEnvConfig(task="reach").fps, 50)


if __name__ == "__main__":
    unittest.main()
