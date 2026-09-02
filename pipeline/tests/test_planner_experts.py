"""The planner expert (docs/66 §3 source 2, D3): beats written from
poses, executed by chained IK, judged by each task's own referee -
measured against the scripted experts' ceiling, per instrument."""

from __future__ import annotations

import unittest
from pathlib import Path

from rq_pipeline.collect.choreography import Lift, PickPlacePlanner, Place, PlannerKnobs
from tests._extras import needs_sim
from tests._instruments import expected_planner_rate

X_LINE = (1.0, 0.0, 0.0)


class TheBeats(unittest.TestCase):
    def test_a_lift_is_hover_descend_close_lift_hold(self) -> None:
        beats = PickPlacePlanner(PlannerKnobs()).plan(
            (0.1, -0.2, 0.015), (0.012, 0.012, 0.015), Lift(0.1), X_LINE
        )
        self.assertEqual(
            [b.name for b in beats], ["hover", "descend", "close", "lift", "hold"]
        )
        hover, descend, close = beats[:3]
        self.assertAlmostEqual(hover.target[2], 0.015 + 0.015 + 0.06)  # top + clearance
        self.assertEqual((descend.jaw, close.jaw), (1.0, 0.0))
        self.assertEqual(descend.closing, X_LINE)
        self.assertEqual(
            (hover.corrected, descend.corrected, close.corrected), ("xyz", "xy", None)
        )
        self.assertAlmostEqual(beats[-1].target[2], 0.1)

    def test_a_place_carries_lowers_opens_and_retracts(self) -> None:
        beats = PickPlacePlanner(PlannerKnobs()).plan(
            (0.0, -0.25, 0.015),
            (0.012, 0.012, 0.015),
            Place((0.08, -0.2, 0.045)),
            X_LINE,
        )
        self.assertEqual(
            [b.name for b in beats][-4:], ["carry", "lower", "open", "retract"]
        )
        self.assertEqual(beats[-1].jaw, 1.0)
        self.assertAlmostEqual(
            beats[-3].target[2], 0.045 + PlannerKnobs().place_clearance_m
        )
        self.assertTrue(all(b.closing == X_LINE for b in beats))

    def test_a_place_may_release_from_its_own_height(self) -> None:
        beats = PickPlacePlanner(PlannerKnobs()).plan(
            (0.0, -0.25, 0.015),
            (0.012, 0.012, 0.015),
            Place((0.08, -0.2, 0.015), 0.018),
            X_LINE,
        )
        self.assertAlmostEqual(beats[-3].target[2], 0.015 + 0.018)


def _planned_rate(task_name: str) -> tuple[int, int]:
    import mujoco  # noqa: PLC0415

    from rq_pipeline.collect.planner_demos import planned_episode  # noqa: PLC0415
    from rq_pipeline.physics.mujoco_backend import MuJoCoBackend  # noqa: PLC0415
    from rq_pipeline.tasks.registry import resolve  # noqa: PLC0415
    from rq_pipeline.tasks.so101 import (  # noqa: PLC0415
        planner_goal,
        planner_object,
        so101_gripper,
    )

    task = resolve(task_name).build()
    model = task.spec.compile()
    gripper = so101_gripper(model)
    backend = MuJoCoBackend()
    backend.load_model(model)
    home = backend.default_initial_state()  # the so101 protocols declare no keyframe
    passes = 0
    for trial in range(task.protocol.trials):
        initial = task.protocol.perturb(trial, home)
        data = mujoco.MjData(model)
        data.qpos[:] = initial[1 : 1 + model.nq]
        mujoco.mj_forward(model, data)
        goal = planner_goal(task_name, model, data)
        states, sensors, _ = planned_episode(
            model,
            initial,
            task=task,
            gripper=gripper,
            goal=goal,
            object_body=planner_object(task_name),
        )
        passes += int(task.protocol.success(states, sensors))
    return passes, task.protocol.trials


@needs_sim
class ThePlannerOnTheSo101Tasks(unittest.TestCase):
    """The planner's rate per task, pinned per instrument (skips where
    nobody measured); the scripted experts' ceiling is the bar."""

    def _pin(self, task_name: str) -> None:
        from rq_pipeline.physics.mujoco_backend import MuJoCoBackend  # noqa: PLC0415
        from rq_pipeline.tasks.registry import resolve  # noqa: PLC0415

        backend = MuJoCoBackend()
        backend.load_spec(resolve(task_name).build().spec)
        expected = expected_planner_rate(task_name, backend.instrument, self)
        passes, trials = _planned_rate(task_name)
        print(f"\n[planner] {task_name} {passes}/{trials} on {backend.instrument}")
        self.assertEqual(passes / trials, expected)

    def test_lift(self) -> None:
        self._pin("lift")

    def test_block_stack(self) -> None:
        self._pin("block_stack")

    def test_tool_insert(self) -> None:
        self._pin("tool_insert")


@needs_sim
class TheGraspSite(unittest.TestCase):
    def test_the_grasp_site_is_where_the_scripted_lift_holds_the_cube(self) -> None:
        """`_GRASP_POS`/`_GRASP_QUAT` were measured from the scripted
        lift's hold; re-measure: the site sits on the held cube's centre,
        its z points down the finger line, its y along the closing line."""
        import mujoco  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        from rq_pipeline.physics.mujoco_backend import MuJoCoBackend  # noqa: PLC0415
        from rq_pipeline.tasks import so101  # noqa: PLC0415
        from rq_pipeline.tasks.registry import resolve  # noqa: PLC0415
        from rq_pipeline.tasks.task import CONTROL_INTERVAL  # noqa: PLC0415

        task = resolve("lift").build()
        model = task.spec.compile()
        backend = MuJoCoBackend()
        backend.load_model(model)
        gripper = so101.so101_gripper(model)
        cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, so101.CUBE_BODY)
        stepper = backend.stepper(backend.default_initial_state(), task.protocol.steps)
        while not stepper.done:
            control = so101.scripted_pick(stepper.step, stepper.data.sensordata)
            stepper.advance(np.asarray(control, dtype=float), CONTROL_INTERVAL)
        self.assertTrue(task.protocol.success(stepper.states, stepper.sensors))
        gap = gripper.grasp_point(stepper.data) - stepper.data.xpos[cube]
        self.assertLess(float(np.linalg.norm(gap)), 0.002)
        axes = np.asarray(stepper.data.site_xmat[gripper.site_id]).reshape(3, 3)
        self.assertLess(float(np.linalg.norm(axes[:, 2] - [0, 0, -1])), 0.05)
        self.assertLess(abs(abs(float(axes[:, 1] @ [1, 0, 0])) - 1), 0.01)


@needs_sim
class ThePlannerPress(unittest.TestCase):
    def test_a_planned_batch_carries_the_planner_stamp(self) -> None:
        import tempfile  # noqa: PLC0415

        from rq_pipeline.collect.kitting_export import DemoLayout  # noqa: PLC0415
        from rq_pipeline.collect.planner_demos import (  # noqa: PLC0415
            generate_planned_demos,
        )
        from rq_pipeline.collect.press import EpisodeManifest  # noqa: PLC0415
        from rq_pipeline.tasks.registry import resolve  # noqa: PLC0415
        from rq_pipeline.tasks.so101 import planner_rig  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            batch = generate_planned_demos(
                Path(tmp),
                task_factory=resolve("lift").build,
                rig=planner_rig("lift"),
                dr={"damping": (1.0, 1.0), "gain": (1.0, 1.0)},
                basis="nominal",
                episodes=2,
                seed=3,
                frame_every=0,
                say=lambda _: None,
            )
            self.assertTrue(batch.complete)
            first = Path(tmp) / DemoLayout.EPISODE_DIR.format(index=0)
            manifest = EpisodeManifest.read_from(first)
        self.assertEqual(batch.expert, PlannerKnobs().stamp)
        self.assertEqual(manifest.expert, PlannerKnobs().stamp)
        self.assertTrue(manifest.expert.startswith("planner@"))
        self.assertIn("trial", manifest.draws)


if __name__ == "__main__":
    unittest.main()
