"""gripper-pick: the USD-imported 2F-85 on a carriage, composed, refereed
and accepted — the census, the solver block, the layout the referee
reads, the pads' reach the expert trusts, and the ladder (pick passes
every paired trial; no-close, limp and the hold-home floor pass none)."""

from __future__ import annotations

import unittest
from dataclasses import replace

from tests._extras import needs_sim

TASK_ID = "robotiq/gripper-pick"
TEST_SOURCE = "robotiq-2f85-isaac@000000000000"


def _built(spec=None):
    from rq_pipeline.physics.mujoco_backend import MuJoCoBackend  # noqa: PLC0415
    from rq_pipeline.tasks.gripper_pick import (  # noqa: PLC0415
        GRIPPER_PICK_SPEC,
        build_gripper_pick,
    )

    task = build_gripper_pick(spec=spec or GRIPPER_PICK_SPEC)
    backend = MuJoCoBackend()
    backend.load_spec(task.spec)
    return task, backend


@needs_sim
class Composition(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.task, cls.backend = _built()

    def test_registered_as_a_reviewable_family(self) -> None:
        from rq_pipeline.tasks.experts import experts  # noqa: PLC0415
        from rq_pipeline.tasks.overlay import families  # noqa: PLC0415
        from rq_pipeline.tasks.registry import resolve  # noqa: PLC0415

        entry = resolve("gripper-pick")
        self.assertEqual(entry.task_id, TASK_ID)
        self.assertEqual(entry.rig, "robotiq-2f85-isaac")
        self.assertIn(TASK_ID, families())
        self.assertIn(TASK_ID, experts())
        self.assertEqual(self.task.bundle_dir.name, "robotiq-2f85-isaac")
        self.assertTrue(self.task.stamp.startswith("gripper-pick@"))

    def test_census(self) -> None:
        import mujoco  # noqa: PLC0415

        from rq_pipeline.tasks.gripper_pick import (  # noqa: PLC0415
            FINGER_JOINT,
            STATE_WIDTH,
        )

        model = self.backend.model
        counts = self.backend.counts()
        # three carriage drives, then the bundle's one finger drive
        self.assertEqual(counts.actuators, 4)
        names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(model.nu)
        ]
        self.assertEqual(
            names,
            [
                "carriage_x_drive",
                "carriage_y_drive",
                "carriage_z_drive",
                "gripper_finger_joint_drive",
            ],
        )
        # three carriage jointpos + the bundle's sixteen
        self.assertEqual(counts.sensors, 19)
        # agent_pos is (x, y, z, finger): the fourth sensor is the finger's
        self.assertEqual(STATE_WIDTH, 4)
        finger_sensor = model.sensor(STATE_WIDTH - 1)
        self.assertEqual(
            mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_JOINT, int(finger_sensor.objid[0])
            ),
            FINGER_JOINT,
        )
        # carriage 3 + gripper 8 + the cube's free joint (7 qpos)
        self.assertEqual(model.nq, 3 + 8 + 7)
        self.assertEqual(self.task.state_width, STATE_WIDTH)

    def test_the_full_solver_block_is_pinned(self) -> None:
        import mujoco  # noqa: PLC0415

        from rq_pipeline.tasks.scene import NominalOptions  # noqa: PLC0415

        opt = self.backend.model.opt
        self.assertEqual(opt.solver, mujoco.mjtSolver.mjSOL_NEWTON)
        self.assertEqual(opt.cone, mujoco.mjtCone.mjCONE_ELLIPTIC)
        self.assertEqual(opt.integrator, mujoco.mjtIntegrator.mjINT_EULER)
        self.assertEqual(opt.impratio, NominalOptions.IMPRATIO)
        self.assertEqual(opt.timestep, NominalOptions.TIMESTEP)

    def test_the_referee_reads_the_cube(self) -> None:
        import numpy as np  # noqa: PLC0415

        from rq_pipeline.tasks.gripper_pick import (  # noqa: PLC0415
            GRIPPER_PICK_SPEC,
            Layout,
        )

        layout = Layout.of(self.backend.model)
        home = np.asarray(self.backend.default_initial_state())
        # the cube rests on the table at its half-height, at the origin
        np.testing.assert_allclose(
            home[layout.cube_pos], (0.0, 0.0, GRIPPER_PICK_SPEC.cube_half)
        )

    def test_the_cube_meets_the_pads_with_torsion_and_their_stiffness(self) -> None:
        import mujoco  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        from rq_pipeline.tasks.gripper_pick import (  # noqa: PLC0415
            CUBE_BODY,
            GRIPPER_PREFIX,
            CubeContact,
        )

        model = self.backend.model
        cube = model.geom(f"{CUBE_BODY}_geom")
        pad = model.geom(f"{GRIPPER_PREFIX}{CubeContact.PAD_GEOM}")
        self.assertGreater(int(cube.priority[0]), int(pad.priority[0]))
        self.assertEqual(int(cube.condim[0]), 4)
        np.testing.assert_allclose(cube.solref, pad.solref)
        np.testing.assert_allclose(cube.solimp, pad.solimp)
        self.assertEqual(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(cube.bodyid[0])),
            CUBE_BODY,
        )

    def test_the_pads_reach_the_expert_trusts(self) -> None:
        """PickChoreography.PAD_REACH_M against the composed scene: fingers
        closed on nothing, the fingertip hulls' lowest vertex sits that far
        under the gripper base — a regenerated bundle that moved it fails
        here before it fails a grasp."""
        import mujoco  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        from rq_pipeline.tasks.gripper_pick import (  # noqa: PLC0415
            GRIPPER_PREFIX,
            Layout,
            PickChoreography,
        )

        model = self.backend.model
        layout = Layout.of(model)
        data = mujoco.MjData(model)
        # lift the cube out of the way; hold the carriage at home
        data.qpos[layout.cube_pos.start - 1 + 2] = -1.0
        data.ctrl[layout.finger_ctrl] = layout.finger_closed
        for _ in range(1500):
            mujoco.mj_step(model, data)
        base = data.xpos[model.body(f"{GRIPPER_PREFIX}base_link").id]
        lowest = np.inf
        for side in ("left", "right"):
            geom = model.geom(f"{GRIPPER_PREFIX}{side}_fingertip_hull")
            mesh = int(geom.dataid[0])
            start = model.mesh_vertadr[mesh]
            verts = model.mesh_vert[start : start + model.mesh_vertnum[mesh]]
            world = (
                data.geom_xpos[geom.id]
                + verts @ data.geom_xmat[geom.id].reshape(3, 3).T
            )
            lowest = min(lowest, float(world[:, 2].min()))
        self.assertAlmostEqual(
            base[2] - lowest, PickChoreography.PAD_REACH_M, delta=0.003
        )

    def test_every_paired_start_is_placed_in_the_band(self) -> None:
        from rq_pipeline.tasks.gripper_pick import Layout  # noqa: PLC0415

        protocol = self.task.protocol
        layout = Layout.of(self.backend.model)
        home = self.backend.default_initial_state()
        spec = self.task.task_spec
        corners = set()
        for trial in range(protocol.trials):
            start = protocol.perturb(trial, home)
            self.backend.validate_start(start, protocol.placements, trial=trial)
            x, y = start[layout.cube_pos][:2]
            self.assertTrue(spec.cube_spawn_x[0] <= x <= spec.cube_spawn_x[1])
            self.assertTrue(spec.cube_spawn_y[0] <= y <= spec.cube_spawn_y[1])
            corners.add((round(float(x), 6), round(float(y), 6)))
        self.assertEqual(len(corners), 4)  # four distinct corners

    def test_a_spec_change_moves_the_stamp(self) -> None:
        from rq_pipeline.tasks.gripper_pick import (  # noqa: PLC0415
            GRIPPER_PICK_SPEC,
            build_gripper_pick,
        )

        moved = build_gripper_pick(spec=replace(GRIPPER_PICK_SPEC, lift_m=0.06))
        self.assertNotEqual(moved.stamp, self.task.stamp)

    def test_a_band_out_of_reach_or_off_the_table_is_refused(self) -> None:
        from rq_pipeline.tasks.gripper_pick import (  # noqa: PLC0415
            GRIPPER_PICK_SPEC,
            build_gripper_pick,
        )

        for overlay, words in (
            ({"cube_spawn_x": (-0.05, 0.2)}, "reach"),
            ({"cube_spawn_y": (0.05, -0.05)}, "backwards"),
            ({"cube_spawn_x": (-0.05, 0.05), "cube_half": 0.2}, "off the table"),
        ):
            with self.subTest(overlay=overlay):
                with self.assertRaises(ValueError) as caught:
                    build_gripper_pick(spec=replace(GRIPPER_PICK_SPEC, **overlay))
                self.assertIn(words, str(caught.exception))

    def test_a_hold_longer_than_the_episode_is_refused(self) -> None:
        from rq_pipeline.tasks.gripper_pick import (  # noqa: PLC0415
            GRIPPER_PICK_SPEC,
            build_gripper_pick,
        )

        with self.assertRaises(ValueError) as caught:
            build_gripper_pick(spec=replace(GRIPPER_PICK_SPEC, hold_s=60.0))
        self.assertIn("hold_s", str(caught.exception))


@needs_sim
class Ladder(unittest.TestCase):
    """The acceptance critic, rung by rung, through `tasks/acceptance.accept`
    — the same call `tools/accept-task.py` and the `accept_task` door make."""

    @classmethod
    def setUpClass(cls) -> None:
        from rq_pipeline.tasks.acceptance import accept  # noqa: PLC0415
        from rq_pipeline.tasks.gripper_pick import LADDER  # noqa: PLC0415

        cls.verdicts = {}
        for rung, expert in LADDER.items():
            task, backend = _built()
            cls.verdicts[rung] = accept(
                task, expert, source=TEST_SOURCE, backend=backend
            )

    def test_the_expert_is_accepted(self) -> None:
        verdict = self.verdicts["pick"]
        self.assertTrue(verdict.accepted, str(verdict))
        self.assertEqual(verdict.expert_successes, verdict.trials)
        self.assertEqual(verdict.floor_successes, 0)

    def test_never_closing_lifts_nothing(self) -> None:
        verdict = self.verdicts["no-close"]
        self.assertEqual(verdict.expert_successes, 0, str(verdict))
        # the funnel says why: the jaw never closed
        self.assertEqual(verdict.funnel["expert"][0], 0)

    def test_a_limp_rig_lifts_nothing(self) -> None:
        verdict = self.verdicts["limp"]
        self.assertEqual(verdict.expert_successes, 0, str(verdict))

    def test_the_floor_passes_none(self) -> None:
        for rung, verdict in self.verdicts.items():
            self.assertEqual(verdict.floor_successes, 0, rung)


if __name__ == "__main__":
    unittest.main()
