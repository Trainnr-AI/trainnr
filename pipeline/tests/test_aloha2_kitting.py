"""Kitting: the T5 scene, its pins, and the scripted demo's proof."""

import unittest
from dataclasses import replace

from tests._extras import needs_sim

SHORT_STEPS = 400  # 0.8 s: the first hover never ends
CONTROL_TICK = 10  # physics steps per control tick (aloha2.CONTROL_INTERVAL)


@needs_sim
class KittingScene(unittest.TestCase):
    def test_census_and_state_slices(self) -> None:
        import mujoco  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        from rq_pipeline.tasks.aloha2 import (  # noqa: PLC0415
            PART_HOME,
            PART_ORDER,
            PART_STATE_SLICE,
            SERVOS,
            build_kitting,
        )

        task = build_kitting()
        model = task.spec.compile()
        self.assertEqual(model.nu, SERVOS)
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
        for arm in PART_ORDER:
            found = state[PART_STATE_SLICE[arm]]
            self.assertTrue(
                np.allclose(found, PART_HOME[arm], atol=1e-9),
                f"{arm}: {found} != {PART_HOME[arm]}",
            )

    def test_a_short_clock_truncates_and_says_so(self) -> None:
        """The silent-truncation guard, and the dataset contract (one
        14-wide action row per control tick) — on a 0.8 s protocol, so
        the pin costs a second, not the 28 s the full episode costs
        (which test_acceptance runs on every paired corner anyway)."""
        from rq_pipeline.physics.mujoco_backend import keyframe_state  # noqa: PLC0415
        from rq_pipeline.tasks.aloha2 import (  # noqa: PLC0415
            KITTING_SPEC,
            SERVOS,
            KittingStats,
            build_kitting,
            scripted_kitting_episode,
        )

        spec = replace(KITTING_SPEC, steps=SHORT_STEPS)
        task = build_kitting(spec=spec)
        model = task.spec.compile()
        stats = KittingStats()
        _states, _sensors, actions = scripted_kitting_episode(
            model,
            task.protocol.perturb(0, keyframe_state(model, task.protocol.home)),
            stats=stats,
            spec=spec,
        )
        self.assertTrue(stats.truncated)
        self.assertEqual(actions.shape, (SHORT_STEPS // CONTROL_TICK, SERVOS))

    def test_the_choreography_facts(self) -> None:
        """The grasp axis mirrors with the arm; the beats track the part
        until the slot beats; the DR scale touches BOTH servo terms
        (the 2026-08-26 bug scaled one and moved every setpoint)."""
        from rq_pipeline.tasks.aloha2 import (  # noqa: PLC0415
            KITTING_SPEC,
            KittingChoreography,
            build_kitting,
            grasp_axis,
            kitting_waypoints,
            scale_dynamics,
        )

        right, left = grasp_axis("right", (0.2, -0.1)), grasp_axis("left", (-0.2, -0.1))
        self.assertEqual(right[0], -left[0])
        self.assertEqual(right[1:], left[1:])
        plan = kitting_waypoints("right", (0.2, -0.1))
        slot = KITTING_SPEC.slot_centers["right"]
        for beat in plan:
            anchored = beat.name in KittingChoreography.SLOT_ANCHORED_SEGMENTS
            self.assertEqual(beat.target[:2], slot if anchored else (0.2, -0.1), beat)
        spec = build_kitting().spec
        gains = [(a.gainprm[0], a.biasprm[1]) for a in spec.actuators]
        scale_dynamics(spec, damping_scale=1.0, gain_scale=2.0)
        for (kp, kb), actuator in zip(gains, spec.actuators, strict=True):
            self.assertAlmostEqual(actuator.gainprm[0], 2 * kp)
            self.assertAlmostEqual(actuator.biasprm[1], 2 * kb)

    def test_scripted_demo_survives_scaled_dynamics(self) -> None:
        """±30% on damping and stiffness, kp scaled on BOTH servo terms.

        Pins the 2026-08-26 correction: scaling gainprm[0] alone moved
        every setpoint by the factor and the same draw failed with both
        retries burnt. Damping x1.25, stiffness x0.80 — a far corner of
        the ±30% box the generator samples.
        """
        from rq_pipeline.evaluate.harness import events_for  # noqa: PLC0415
        from rq_pipeline.physics.mujoco_backend import keyframe_state  # noqa: PLC0415
        from rq_pipeline.tasks.aloha2 import (  # noqa: PLC0415
            KittingStats,
            build_kitting,
            scale_dynamics,
            scripted_kitting_episode,
        )

        task = build_kitting()
        spec = build_kitting().spec
        scale_dynamics(spec, damping_scale=1.25, gain_scale=0.80)
        model = spec.compile()
        home = keyframe_state(model, task.protocol.home)
        stats = KittingStats()
        states, sensors, _actions = scripted_kitting_episode(
            model, task.protocol.perturb(0, home), stats=stats
        )
        self.assertTrue(task.protocol.success(states, sensors), stats)
        # No retry, no truncation, and the milestone chain agrees with the
        # verdict in order — the funnel a policy is read by.
        self.assertEqual(stats.retries, [])
        self.assertFalse(stats.truncated)
        self.assertEqual(
            [event["name"] for event in events_for(task.protocol, states, sensors)],
            ["part_moved", "part_lifted", "one_in_slot", "both_in_slot"],
        )

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


class TheExpertStamp(unittest.TestCase):
    def test_the_expert_is_named_by_its_choreography(self) -> None:
        from unittest import mock  # noqa: PLC0415

        from rq_pipeline.tasks.aloha2 import (  # noqa: PLC0415
            KittingChoreography,
            expert_stamp,
        )

        stamp = expert_stamp()
        self.assertRegex(stamp, r"^kitting-expert@[0-9a-f]{12}$")
        self.assertEqual(stamp, expert_stamp())
        with mock.patch.object(KittingChoreography, "PARK_SECONDS", 2.0):
            self.assertNotEqual(expert_stamp(), stamp)
        self.assertIn("SEGMENTS", KittingChoreography.fields())


if __name__ == "__main__":
    unittest.main()
