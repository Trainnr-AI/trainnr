"""The gymnasium env: the ecosystem's contract, pinned against our rules.

Runs with the `sim` extra (mujoco + gymnasium). The LeRobot plugin's
half lives in the train venv and is pinned there.
"""

import dataclasses
import importlib.util
import unittest

SIM_PRESENT = (
    importlib.util.find_spec("mujoco") is not None
    and importlib.util.find_spec("gymnasium") is not None
)
SHORT_STEPS = 40  # four control ticks of the ALOHA protocol, enough to truncate
SOURCE = "aloha2-transfer@testhash"
SERVOS = 14
# GPU rasterizers are not bit-exact frame to frame: on the WSL box's D3D12
# path the same state renders with ±1 in ~20 of 307k pixels, and once in a
# full suite run a pixel went to ±2 (llvmpipe is exact; measured 2026-08-26,
# 30 frames). Physics pairing is exact; pixels pair to within the
# rasterizer's noise — a moved cube changes ~770 pixels by more than 1.
PIXEL_TOLERANCE = 1
NOISE_PIXELS = 32  # pixels allowed beyond the tolerance for "the same frame"


def pixels_match(a, b) -> bool:
    import numpy as np  # noqa: PLC0415

    beyond = np.abs(a.astype(int) - b.astype(int)).max(axis=-1) > PIXEL_TOLERANCE
    return bool(beyond.sum() <= NOISE_PIXELS)


@unittest.skipUnless(SIM_PRESENT, "sim extra not installed (uv sync --extra sim)")
class StepperIsTheOneLoop(unittest.TestCase):
    def test_stepper_agrees_with_mujoco_rollout(self) -> None:
        # Two integrators, one trajectory: the harness's tick-by-tick
        # Stepper against MuJoCo's own batched `rollout` (the sysid path)
        # under the same constant control, bit for bit — including the
        # short last tick when the budget is not a multiple.
        import numpy as np  # noqa: PLC0415

        from rq_pipeline.evaluate.harness import run_sensor_episode  # noqa: PLC0415
        from rq_pipeline.physics.mujoco_backend import (  # noqa: PLC0415
            MuJoCoBackend,
            Stepper,
        )
        from tests.test_mujoco_backend import PENDULUM  # noqa: PLC0415

        backend = MuJoCoBackend()
        backend.load_mjcf_string(PENDULUM)
        home = backend.default_initial_state()
        home[1] += 0.3  # displaced, so gravity acts
        control = np.array([0.2])
        steps = 7
        batched = backend.rollout(home[None], np.tile(control, (1, steps, 1)))[0]
        stepper = Stepper(backend.model, home, steps)
        while not stepper.done:
            stepper.advance(control, 3)  # the last call gets one step, not three
        self.assertEqual(stepper.step, steps)
        self.assertTrue(np.array_equal(batched, stepper.states))
        states, _sensors = run_sensor_episode(
            backend, lambda step, sense: control, home, steps=steps, control_interval=3
        )
        self.assertTrue(np.array_equal(states, stepper.states))
        with self.assertRaises(ValueError):
            stepper.advance(np.zeros(2), 3)  # wrong width is loud


@unittest.skipUnless(SIM_PRESENT, "sim extra not installed (uv sync --extra sim)")
class GymnasiumContract(unittest.TestCase):
    def _env(self, steps: int | None = None):
        from rq_pipeline.envs.robotiq import RobotiqEnv  # noqa: PLC0415
        from rq_pipeline.tasks.aloha2 import build_transfer_cube  # noqa: PLC0415

        task = build_transfer_cube()
        if steps is not None:
            task = dataclasses.replace(
                task, protocol=dataclasses.replace(task.protocol, steps=steps)
            )
        return RobotiqEnv(
            task, state_width=SERVOS, instruction="transfer the cube", source=SOURCE
        )

    def test_passes_gymnasium_env_checker(self) -> None:
        from gymnasium.utils.env_checker import check_env  # noqa: PLC0415
        from gymnasium.wrappers import FilterObservation  # noqa: PLC0415

        # The checker's determinism test compares observations to 1e-5,
        # which GPU pixels cannot promise (PIXEL_TOLERANCE); it runs on
        # the state view, and the pixel contract is pinned below.
        env = self._env(steps=SHORT_STEPS)
        check_env(FilterObservation(env, ["agent_pos"]), skip_render_check=True)
        env.close()

    def test_observation_is_lerobot_raw_shape(self) -> None:
        import numpy as np  # noqa: PLC0415

        env = self._env(steps=SHORT_STEPS)
        observation, info = env.reset(seed=0)
        self.assertEqual(observation["pixels"]["top"].shape, (480, 640, 3))
        self.assertEqual(observation["pixels"]["top"].dtype, np.uint8)
        self.assertGreater(int(observation["pixels"]["top"].max()), 20)  # not black
        self.assertEqual(observation["agent_pos"].shape, (SERVOS,))
        self.assertEqual(observation["agent_pos"].dtype, np.float32)
        self.assertIs(info["is_success"], False)
        self.assertEqual(env.metadata["render_fps"], 50)
        self.assertEqual(env._max_episode_steps, SHORT_STEPS // 10)
        self.assertEqual(env.task_description, "transfer the cube")
        env.close()

    def test_seed_is_the_trial_so_starts_pair(self) -> None:
        import numpy as np  # noqa: PLC0415

        env = self._env(steps=SHORT_STEPS)
        first, _ = env.reset(seed=1002)
        again, info = env.reset(seed=1002)
        other, _ = env.reset(seed=1003)
        self.assertTrue(np.array_equal(first["agent_pos"], again["agent_pos"]))
        self.assertTrue(pixels_match(first["pixels"]["top"], again["pixels"]["top"]))
        self.assertEqual(info["trial"], 1002 % env.protocol.trials)
        # A different trial moves the cube: the frame changes, the arms do not.
        self.assertFalse(pixels_match(first["pixels"]["top"], other["pixels"]["top"]))
        self.assertTrue(np.array_equal(first["agent_pos"], other["agent_pos"]))
        env.close()

    def test_episode_truncates_with_a_verdict_under_both_names(self) -> None:
        import numpy as np  # noqa: PLC0415

        from rq_pipeline.tasks.aloha2 import NEUTRAL_CTRL  # noqa: PLC0415

        env = self._env(steps=SHORT_STEPS)
        env.reset(seed=0)
        limp = np.asarray(NEUTRAL_CTRL, dtype=np.float32)
        for tick in range(env._max_episode_steps):
            _, reward, terminated, truncated, info = env.step(limp)
            self.assertFalse(terminated)
            if tick + 1 < env._max_episode_steps:
                self.assertFalse(truncated)
                self.assertIs(info["is_success"], False)
        self.assertTrue(truncated)
        self.assertIs(info["is_success"], False)  # limp never transfers
        self.assertIs(info["success"], info["is_success"])
        self.assertEqual(reward, 0.0)
        env.close()

    def test_unstamped_source_is_refused_before_any_episode(self) -> None:
        from rq_pipeline.envs.robotiq import RobotiqEnv  # noqa: PLC0415
        from rq_pipeline.tasks.aloha2 import build_transfer_cube  # noqa: PLC0415

        with self.assertRaises(ValueError):
            RobotiqEnv(
                build_transfer_cube(),
                state_width=SERVOS,
                instruction="transfer the cube",
                source="unstamped",
            )


if __name__ == "__main__":
    unittest.main()
