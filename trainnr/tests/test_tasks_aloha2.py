"""The ALOHA 2 task suite: composition, the census, layout pins, cameras.

What these tests pin is the CONTRACT the training ladder builds on:
the transfer-cube scene composes on the bundle with its referee and
camera, the state layout the success predicate reads is where the
module says it is, the episode starts at the keyframe with the cube
resting in its spawn box, and the `top` camera a gym-aloha checkpoint
consumes renders a real frame at the checkpoint's resolution.
"""

import unittest

from tests._extras import needs_render, needs_sim

BLACK_FRAME_MEAN = 5.0


class GripperRemap(unittest.TestCase):
    def test_commands_span_the_ctrl_range_and_states_the_measured_travel(
        self,
    ) -> None:
        from trainnr.tasks.aloha2 import (  # noqa: PLC0415
            ALOHA2_GRIPPER_CTRL_CLOSE,
            ALOHA2_GRIPPER_CTRL_OPEN,
            ALOHA2_GRIPPER_JOINT_CLOSED,
            ALOHA2_GRIPPER_JOINT_OPEN,
            gripper_ctrl_from_normalized,
            gripper_normalized_from_joint,
        )

        self.assertAlmostEqual(
            gripper_ctrl_from_normalized(0.0), ALOHA2_GRIPPER_CTRL_CLOSE
        )
        self.assertAlmostEqual(
            gripper_ctrl_from_normalized(1.0), ALOHA2_GRIPPER_CTRL_OPEN
        )
        self.assertAlmostEqual(
            gripper_ctrl_from_normalized(7.0), ALOHA2_GRIPPER_CTRL_OPEN
        )
        # Closed pads read 0 like the dataset; full open reads 1; the
        # ctrl floor (unreachable by the fingers) clips to 0.
        self.assertAlmostEqual(
            gripper_normalized_from_joint(ALOHA2_GRIPPER_JOINT_CLOSED), 0.0
        )
        self.assertAlmostEqual(
            gripper_normalized_from_joint(ALOHA2_GRIPPER_JOINT_OPEN), 1.0
        )
        self.assertAlmostEqual(
            gripper_normalized_from_joint(ALOHA2_GRIPPER_CTRL_CLOSE), 0.0
        )
        self.assertAlmostEqual(gripper_normalized_from_joint(0.5), 1.0)

    def test_state_and_action_adapters_touch_only_the_gripper_channels(self) -> None:
        import numpy as np  # noqa: PLC0415

        from trainnr.tasks.aloha2 import (  # noqa: PLC0415
            ARM_SENSOR_WIDTH,
            act_sim_state,
            ctrl_from_act_sim_action,
        )

        sensordata = np.arange(ARM_SENSOR_WIDTH + 6, dtype=np.float64) * 0.01
        state = act_sim_state(sensordata)
        self.assertEqual(state.shape, (14,))
        np.testing.assert_allclose(state[:6], sensordata[:6])
        np.testing.assert_allclose(state[7:13], sensordata[7:13])
        action = np.full(14, 0.5)
        ctrl = ctrl_from_act_sim_action(action)
        np.testing.assert_allclose(ctrl[:6], 0.5)
        self.assertAlmostEqual(float(ctrl[6]), 0.0195)
        self.assertAlmostEqual(float(ctrl[13]), 0.0195)


@needs_sim
@needs_render
class TransferCubeScene(unittest.TestCase):
    def _loaded(self):
        from trainnr.physics.mujoco_backend import MuJoCoBackend  # noqa: PLC0415
        from trainnr.tasks.aloha2 import build_transfer_cube  # noqa: PLC0415

        task = build_transfer_cube()
        backend = MuJoCoBackend()
        backend.load_spec(task.spec)
        return task, backend

    def test_census_bundle_plus_referee_plus_top_camera(self) -> None:
        from trainnr.tasks.aloha2 import ARM_SENSOR_WIDTH, SERVOS  # noqa: PLC0415

        _task, backend = self._loaded()
        counts = backend.counts()
        self.assertEqual(counts.actuators, SERVOS)
        # Sensor ELEMENTS: 28 scalar joint sensors + two 3-vector referees.
        self.assertEqual(counts.sensors, ARM_SENSOR_WIDTH + 2)
        self.assertEqual(counts.cameras, 6 + 1)  # the rig's six + top

    def test_home_is_the_keyframe_with_the_cube_resting_in_its_box(self) -> None:
        import numpy as np  # noqa: PLC0415

        from trainnr.evaluate.harness import home_state  # noqa: PLC0415
        from trainnr.tasks.aloha2 import (  # noqa: PLC0415
            CUBE_HALF,
            CUBE_SPAWN_X,
            CUBE_SPAWN_Y,
            CUBE_STATE_SLICE,
        )

        task, backend = self._loaded()
        home = np.asarray(home_state(backend, task.protocol))
        self.assertAlmostEqual(float(home[2]), -0.96, places=3)  # left shoulder
        cube = home[CUBE_STATE_SLICE]
        self.assertTrue(CUBE_SPAWN_X[0] <= cube[0] <= CUBE_SPAWN_X[1])
        self.assertTrue(CUBE_SPAWN_Y[0] <= cube[1] <= CUBE_SPAWN_Y[1])
        self.assertAlmostEqual(float(cube[2]), CUBE_HALF, places=3)
        # Every paired start keeps the cube inside the spawn box.
        for trial in range(4):
            start = task.protocol.perturb(trial, home)[CUBE_STATE_SLICE]
            self.assertTrue(CUBE_SPAWN_X[0] <= start[0] <= CUBE_SPAWN_X[1])
            self.assertTrue(CUBE_SPAWN_Y[0] <= start[1] <= CUBE_SPAWN_Y[1])

    def test_cube_state_index_is_where_the_predicate_reads(self) -> None:
        # Settle from home with the arms held: the cube must stay on the
        # table, and its z must live at the pinned index.
        from trainnr.evaluate.harness import (  # noqa: PLC0415
            home_state,
            run_sensor_episode,
        )
        from trainnr.tasks.aloha2 import (  # noqa: PLC0415
            CUBE_HALF,
            CUBE_Z_STATE_INDEX,
            NEUTRAL_CTRL,
        )

        task, backend = self._loaded()
        states, sensors = run_sensor_episode(
            backend,
            lambda step, sense: NEUTRAL_CTRL,
            home_state(backend, task.protocol),
            steps=500,
            control_interval=10,
        )
        self.assertAlmostEqual(
            float(states[-1, CUBE_Z_STATE_INDEX]), CUBE_HALF, delta=0.003
        )
        self.assertFalse(task.protocol.success(states, sensors))  # nobody picked it

    def _first_frame(self, task):
        from trainnr.envs.gymnasium_env import TrainnrEnv  # noqa: PLC0415

        env = TrainnrEnv(task, source=f"{task.name}@test")
        try:
            observation, _ = env.reset(seed=0)
            return observation["pixels"]["top"]
        finally:
            env.close()

    def test_top_camera_renders_a_real_frame_at_checkpoint_resolution(self) -> None:
        task, _backend = self._loaded()
        frame = self._first_frame(task)
        self.assertEqual(frame.shape, (480, 640, 3))
        self.assertGreater(float(frame.mean()), BLACK_FRAME_MEAN)

    def test_act_sim_look_is_the_same_scene_in_grey(self) -> None:
        # Same census, same physics layout, darker frame: the cosmetic
        # twin for checkpoints trained on gym-aloha's renders.
        from trainnr.physics.mujoco_backend import MuJoCoBackend  # noqa: PLC0415
        from trainnr.tasks.aloha2 import (  # noqa: PLC0415
            ACT_SIM_LOOK,
            build_transfer_cube,
        )

        wood_task, wood = self._loaded()
        grey_task = build_transfer_cube(look=ACT_SIM_LOOK)
        grey = MuJoCoBackend()
        grey.load_spec(grey_task.spec)
        self.assertEqual(
            (wood.counts().actuators, wood.counts().sensors, wood.counts().cameras),
            (grey.counts().actuators, grey.counts().sensors, grey.counts().cameras),
        )
        wood_frame = self._first_frame(wood_task)
        grey_frame = self._first_frame(grey_task)
        self.assertGreater(float(grey_frame.mean()), BLACK_FRAME_MEAN)
        self.assertLess(float(grey_frame.mean()), float(wood_frame.mean()))
        with self.assertRaises(ValueError):
            build_transfer_cube(look="neon")


if __name__ == "__main__":
    unittest.main()
