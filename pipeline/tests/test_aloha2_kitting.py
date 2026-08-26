"""Kitting: the T5 scene, its pins, and the scripted demo's proof."""

import importlib.util
import unittest

MUJOCO_PRESENT = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(MUJOCO_PRESENT, "sim extra not installed (uv sync --extra sim)")
class KittingScene(unittest.TestCase):
    def test_census_and_state_slices(self) -> None:
        import mujoco  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        from rq_pipeline.tasks.aloha2 import (  # noqa: PLC0415
            PART_HOME,
            PART_STATE_SLICE,
            build_kitting,
        )

        task = build_kitting()
        model = task.spec.compile()
        self.assertEqual(model.nu, 14)
        self.assertEqual(model.ncam, 7)  # six D405 + the task's top
        data = mujoco.MjData(model)
        key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "neutral_pose")
        mujoco.mj_resetDataKeyframe(model, data, key)
        mujoco.mj_forward(model, data)
        size = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        state = np.empty(size)
        mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        # The slice pins: FULLPHYSICS carries time first, so the parts'
        # keyframe homes must read back through the declared slices.
        for arm in ("right", "left"):
            found = state[PART_STATE_SLICE[arm]]
            self.assertTrue(
                np.allclose(found, PART_HOME[arm], atol=1e-9),
                f"{arm}: {found} != {PART_HOME[arm]}",
            )

    def test_scripted_demo_succeeds_in_the_proven_band(self) -> None:
        """Trial 0 (the demo generator's band) must place both parts.

        The scripted choreography is the T5 demo source; this run IS
        the proof that the whole chain — chained grip-centre IK with
        tilted approach, closed-loop clamped correction, verify-and-
        retry — carries two parts into their slots. The far spawn band
        (parts near the arms' base line) is a KNOWN open edge: the
        closing-plane orientation is IK-nullspace-random there and the
        close back-drives upward. The generator samples the proven
        band and filters by this same referee.
        """
        from rq_pipeline.evaluate.harness import events_for  # noqa: PLC0415
        from rq_pipeline.physics.mujoco_backend import keyframe_state  # noqa: PLC0415
        from rq_pipeline.tasks.aloha2 import (  # noqa: PLC0415
            KittingStats,
            build_kitting,
            scripted_kitting_episode,
        )

        task = build_kitting()
        model = task.spec.compile()
        home = keyframe_state(model, task.protocol.home)
        stats = KittingStats()
        states, sensors, actions = scripted_kitting_episode(
            model, task.protocol.perturb(0, home), stats=stats
        )
        self.assertTrue(task.protocol.success(states, sensors), stats)
        # The milestone chain agrees with the verdict, in order: touched,
        # lifted, one placed, both placed — the funnel a policy is read by.
        self.assertEqual(
            [event["name"] for event in events_for(task.protocol, states, sensors)],
            ["part_moved", "part_lifted", "one_in_slot", "both_in_slot"],
        )
        # The proven band needs no retry and never truncates — if either
        # fires here, robustness regressed and the stats say where.
        self.assertEqual(stats.retries, [])
        self.assertFalse(stats.truncated)
        # The dataset contract: one 14-wide action row per control tick.
        self.assertEqual(actions.shape[1], 14)
        self.assertEqual(len(actions), task.protocol.steps // 10)

    def test_referee_slice_reads_the_left_pad(self) -> None:
        """Pin the sensor LAYOUT the success predicate assumes.

        LEFT_GRIPPER_POS_SLICE indexes sensordata by position; a sensor
        declared before the referees would shift it and transfer/kitting
        would score garbage silently (the review's E2).
        """
        import mujoco  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        from rq_pipeline.tasks.aloha2 import (  # noqa: PLC0415
            LEFT_GRIPPER_POS_SLICE,
            build_kitting,
        )

        model = build_kitting().spec.compile()
        data = mujoco.MjData(model)
        key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "neutral_pose")
        mujoco.mj_resetDataKeyframe(model, data, key)
        mujoco.mj_forward(model, data)
        pad = data.geom_xpos[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left/left_g1")
        ]
        self.assertTrue(
            np.allclose(data.sensordata[LEFT_GRIPPER_POS_SLICE], pad, atol=1e-9)
        )


if __name__ == "__main__":
    unittest.main()
