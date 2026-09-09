"""The gymnasium env: the ecosystem's contract, pinned against our rules.

Runs with the `sim` extra (mujoco + gymnasium). The LeRobot plugin's
half lives in the train venv and is pinned there.
"""

import dataclasses
import unittest

from tests._extras import needs_envs

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


@needs_envs
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


@needs_envs
class GymnasiumContract(unittest.TestCase):
    def _env(self, steps: int | None = None):
        from rq_pipeline.envs.robotiq import RobotiqEnv  # noqa: PLC0415
        from rq_pipeline.tasks.aloha2 import build_transfer_cube  # noqa: PLC0415

        task = build_transfer_cube()
        if steps is not None:
            task = dataclasses.replace(
                task, protocol=dataclasses.replace(task.protocol, steps=steps)
            )
        return RobotiqEnv(task, source=SOURCE)

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
        self.assertEqual(env.task_description, "transfer the cube to the left gripper")
        env.close()

    def test_trials_size_the_paired_starts_to_the_evaluation(self) -> None:
        """An evaluator asking for N episodes gets N distinct starts: the
        spec is rebuilt with trials=N and the stamp says so. Measured
        2026-08-27: ten lerobot-eval episodes on the four-trial spec
        were the same four starts two and a half times."""
        from rq_pipeline.envs.robotiq import InfoKeys, make_env  # noqa: PLC0415
        from rq_pipeline.protocol import protocol_fields  # noqa: PLC0415

        shipped = make_env("kitting")
        sized = make_env("kitting", trials=6)
        self.assertEqual(shipped.protocol.trials, 4)
        self.assertEqual(sized.protocol.trials, 6)
        self.assertNotEqual(
            protocol_fields(sized.protocol), protocol_fields(shipped.protocol)
        )
        _obs, info = sized.reset(seed=5)
        self.assertEqual(info[InfoKeys.TRIAL], 5)
        _obs, info = shipped.reset(seed=5)
        self.assertEqual(info[InfoKeys.TRIAL], 1)  # the wrap, when nobody sized it
        shipped.close()
        sized.close()
        with self.assertRaisesRegex(ValueError, "no spec to size"):
            make_env("transfer_cube", trials=6)  # code-only task: nothing to rebuild

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

    def test_env_writes_the_full_record_when_asked(self) -> None:
        import tempfile  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        import numpy as np  # noqa: PLC0415

        from rq_pipeline.envs.robotiq import RobotiqEnv  # noqa: PLC0415
        from rq_pipeline.evaluate.records import read_records  # noqa: PLC0415
        from rq_pipeline.tasks.aloha2 import (  # noqa: PLC0415
            NEUTRAL_CTRL,
            build_transfer_cube,
        )

        task = build_transfer_cube()
        task = dataclasses.replace(
            task, protocol=dataclasses.replace(task.protocol, steps=SHORT_STEPS)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.jsonl"
            env = RobotiqEnv(task, source=SOURCE, record_to=path, policy_name="limp")
            env.reset(seed=1001)
            limp = np.asarray(NEUTRAL_CTRL, dtype=np.float32)
            for _ in range(env._max_episode_steps):
                env.step(limp)
            env.close()
            (row,) = read_records(path)
        self.assertEqual((row.policy, row.seed, row.trial), ("limp", 1001, 1))
        self.assertIs(row.success, False)
        self.assertEqual(row.events, ())  # limp never touches the cube
        self.assertEqual(
            row.protocol["milestones"], ["cube_moved", "cube_lifted", "cube_at_left"]
        )
        self.assertTrue(row.instrument.startswith("mujoco-"))

    def test_unstamped_source_is_refused_before_any_episode(self) -> None:
        from rq_pipeline.envs.robotiq import RobotiqEnv  # noqa: PLC0415
        from rq_pipeline.tasks.aloha2 import build_transfer_cube  # noqa: PLC0415

        with self.assertRaises(ValueError):
            RobotiqEnv(build_transfer_cube(), source="unstamped")

    def test_variations_are_drawn_by_trial_applied_and_recorded(self) -> None:
        import tempfile  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        import numpy as np  # noqa: PLC0415

        from rq_pipeline.envs.robotiq import RobotiqEnv  # noqa: PLC0415
        from rq_pipeline.evaluate.records import read_records  # noqa: PLC0415
        from rq_pipeline.evaluate.variations import (  # noqa: PLC0415
            Uniform,
            Variation,
        )
        from rq_pipeline.tasks.aloha2 import (  # noqa: PLC0415
            NEUTRAL_CTRL,
            build_transfer_cube,
        )

        task = build_transfer_cube()
        task = dataclasses.replace(
            task, protocol=dataclasses.replace(task.protocol, steps=SHORT_STEPS)
        )
        sweep = (
            Variation("joints", "damping_scale", Uniform((0.7,), (1.3,))),
            Variation("actuators", "gain_scale", Uniform((0.8,), (1.2,))),
            Variation("cube", "mass_scale", Uniform((0.5,), (2.0,))),
            Variation("top", "offset_m", Uniform((-0.03,) * 3, (0.03,) * 3)),
            Variation("lights", "diffuse_scale", Uniform((0.5,), (1.5,))),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.jsonl"
            env = RobotiqEnv(task, source=SOURCE, record_to=path, variations=sweep)
            nominal_damping = env.model.dof_damping.copy()
            nominal_gain = env.model.actuator_gainprm[:, 0].copy()
            nominal_bias = env.model.actuator_biasprm[:, 1].copy()
            env.reset(seed=2)
            drawn = env.drawn
            scale = drawn["joints.damping_scale"]
            self.assertTrue(np.allclose(env.model.dof_damping, nominal_damping * scale))
            gain = drawn["actuators.gain_scale"]
            # BOTH kp terms, so the setpoint stays put (docs/07, the gain
            # that was a setpoint).
            self.assertTrue(
                np.allclose(env.model.actuator_gainprm[:, 0], nominal_gain * gain)
            )
            self.assertTrue(
                np.allclose(env.model.actuator_biasprm[:, 1], nominal_bias * gain)
            )
            env.reset(seed=3)
            self.assertNotEqual(env.drawn["joints.damping_scale"], scale)
            env.reset(seed=2)  # pairing: trial 2 draws the same again
            self.assertEqual(env.drawn, drawn)
            limp = np.asarray(NEUTRAL_CTRL, dtype=np.float32)
            for _ in range(env._max_episode_steps):
                env.step(limp)
            env.close()
            (row,) = read_records(path)
        self.assertEqual(row.variations["joints.damping_scale"], scale)
        self.assertEqual(len(row.variations["top.offset_m"]), 3)
        self.assertEqual(len(row.protocol["variations"]), len(sweep))
        with self.assertRaises(ValueError):
            RobotiqEnv(
                task,
                source=SOURCE,
                variations=(Variation("nobody", "mass_scale", Uniform((1,), (2,))),),
            )

    def test_every_registered_task_makes_by_gym_id(self) -> None:
        import gymnasium as gym  # noqa: PLC0415

        from rq_pipeline.envs.contract import gym_id  # noqa: PLC0415
        from rq_pipeline.tasks.registry import tasks  # noqa: PLC0415
        from rq_pipeline.tasks.walks import walk_robot  # noqa: PLC0415

        registry = tasks()
        self.assertEqual(
            gym_id("robotiq/kitting"), "robotiq/kitting-v0"
        )  # the one literal pin
        for task_id, entry in registry.items():
            if walk_robot(task_id) is not None:
                continue  # walks are not gymnasium environments
            env = gym.make(gym_id(task_id))
            try:
                self.assertEqual(env.unwrapped.task, entry.name)
                self.assertIn("@", env.unwrapped.source)
                self.assertGreater(env.unwrapped.state_width, 0)
            finally:
                env.close()


if __name__ == "__main__":
    unittest.main()
