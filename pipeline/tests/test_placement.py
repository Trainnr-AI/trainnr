"""Start validation: a task declares where its objects start, and the
engine refuses a start that is not admissible — before a trial is spent,
naming the trial, the body and the check (docs/e2e-research/42 §3)."""

import unittest

from rq_pipeline.protocol import (
    EpisodeProtocol,
    Placement,
    PlacementChecks,
    protocol_fields,
)
from tests._extras import needs_sim

KITTING_TRIALS = 4
LIFT_M = 0.03  # a part held this far above the table is not resting on it
STAMP = "test@000000000000"


def _protocol(**extra):
    return EpisodeProtocol(
        trials=1,
        steps=5,
        control_interval=1,
        perturb=lambda _trial, home: home,
        success=lambda _states, _sensors: True,
        **extra,
    )


class TheDeclaration(unittest.TestCase):
    def test_fields_carry_placements_only_when_declared(self) -> None:
        self.assertNotIn("placements", protocol_fields(_protocol()))
        declared = _protocol(placements=(Placement("cube", "table", x=(0.0, 0.2)),))
        fields = protocol_fields(declared)["placements"]
        self.assertEqual(fields[0]["body"], "cube")
        self.assertEqual(fields[0]["x"], [0.0, 0.2])
        self.assertIsNone(fields[0]["y"])

    def test_bad_declarations_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            Placement("cube", "table", x=(0.2, 0.0))  # high before low
        with self.assertRaises(ValueError):
            Placement("cube", "table", tolerance_m=0.0)
        with self.assertRaises(ValueError):  # a body placed twice
            _protocol(placements=(Placement("a", "t"), Placement("a", "t")))


@needs_sim
class TheGate(unittest.TestCase):
    def _kitting(self):
        from rq_pipeline.physics.mujoco_backend import MuJoCoBackend  # noqa: PLC0415
        from rq_pipeline.tasks.aloha2 import (  # noqa: PLC0415
            PART_STATE_SLICE,
            build_kitting,
        )

        task = build_kitting()
        backend = MuJoCoBackend()
        backend.load_spec(task.spec)
        home = backend.keyframe_state(task.protocol.home)
        return task, backend, home, PART_STATE_SLICE

    def test_every_paired_start_of_every_task_is_admissible(self) -> None:
        """The pin that matters: each task's own perturb, every trial."""
        from rq_pipeline.physics.mujoco_backend import MuJoCoBackend  # noqa: PLC0415
        from rq_pipeline.physics.placement import start_verdicts  # noqa: PLC0415
        from rq_pipeline.tasks.registry import tasks  # noqa: PLC0415
        from rq_pipeline.tasks.walks import walk_robot  # noqa: PLC0415

        for task_id, entry in tasks().items():
            if walk_robot(task_id) is not None:
                continue  # a walk has no placements to admit
            task = entry.build()
            if not task.protocol.placements:
                continue
            backend = MuJoCoBackend()
            backend.load_spec(task.spec)
            home = (
                backend.keyframe_state(task.protocol.home)
                if task.protocol.home
                else backend.default_initial_state()
            )
            for trial in range(task.protocol.trials):
                start = task.protocol.perturb(trial, home)
                verdicts = start_verdicts(
                    backend.model, start, task.protocol.placements
                )
                for body, checks in verdicts.items():
                    with self.subTest(task=task_id, trial=trial, body=body):
                        self.assertEqual(set(checks), set(PlacementChecks.ALL))
                        self.assertTrue(all(checks.values()), checks)

    def test_a_part_inside_a_wall_fails_no_overlap_and_is_refused_by_name(self) -> None:
        import mujoco  # noqa: PLC0415

        from rq_pipeline.physics.placement import (  # noqa: PLC0415
            PlacementRefusedError,
            require_start,
            start_verdicts,
        )

        task, backend, home, slices = self._kitting()
        model = backend.model
        wall = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "slot_right_north")
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        bad = task.protocol.perturb(0, home).copy()
        bad[slices["right"]] = data.geom_xpos[wall]
        verdicts = start_verdicts(model, bad, task.protocol.placements)
        self.assertFalse(verdicts["part_right"][PlacementChecks.NO_OVERLAP])
        self.assertTrue(verdicts["part_left"][PlacementChecks.NO_OVERLAP])
        with self.assertRaises(PlacementRefusedError) as caught:
            require_start(model, bad, task.protocol.placements, trial=3)
        message = str(caught.exception)
        self.assertIn("trial 3", message)
        self.assertIn("part_right", message)
        self.assertIn(PlacementChecks.NO_OVERLAP, message)
        self.assertNotIn("part_left", message)

    def test_lifted_is_not_on_support_and_out_of_band_is_not_in_limits(self) -> None:
        from rq_pipeline.physics.placement import start_verdicts  # noqa: PLC0415

        task, backend, home, slices = self._kitting()
        lifted = task.protocol.perturb(0, home).copy()
        lifted[slices["right"].start + 2] += LIFT_M
        verdicts = start_verdicts(backend.model, lifted, task.protocol.placements)
        self.assertFalse(verdicts["part_right"][PlacementChecks.ON_SUPPORT])
        self.assertTrue(verdicts["part_right"][PlacementChecks.NO_OVERLAP])
        outside = task.protocol.perturb(0, home).copy()
        outside[slices["right"].start] = 0.0  # the left arm's side of the table
        verdicts = start_verdicts(backend.model, outside, task.protocol.placements)
        self.assertFalse(verdicts["part_right"][PlacementChecks.IN_LIMITS])
        self.assertTrue(verdicts["part_right"][PlacementChecks.ON_SUPPORT])

    def test_the_harness_refuses_before_any_episode_and_records_verdicts(self) -> None:
        from rq_pipeline.evaluate.harness import (  # noqa: PLC0415
            SimPolicy,
            evaluate_policies,
        )
        from rq_pipeline.evaluate.records import read_records  # noqa: PLC0415
        from rq_pipeline.physics.mujoco_backend import MuJoCoBackend  # noqa: PLC0415
        from rq_pipeline.physics.placement import PlacementRefusedError  # noqa: PLC0415
        from rq_pipeline.tasks.so101 import build_lift  # noqa: PLC0415

        task = build_lift()
        backend = MuJoCoBackend()
        backend.load_spec(task.spec)
        width = backend.model.nu
        calls = []

        def limp(step, sensors):
            calls.append(step)
            return [0.0] * width

        def hovering(trial, home):
            start = task.protocol.perturb(trial, home).copy()
            start[9] += LIFT_M  # cube_a's z: FULLPHYSICS [time, 6 arm, x, y, z...]
            return start

        bad = EpisodeProtocol(
            trials=task.protocol.trials,
            steps=task.protocol.steps,
            control_interval=task.protocol.control_interval,
            perturb=hovering,
            success=task.protocol.success,
            home=task.protocol.home,
            placements=task.protocol.placements,
        )
        with self.assertRaises(PlacementRefusedError):
            evaluate_policies(backend, [SimPolicy("limp", limp)], bad, source=STAMP)
        self.assertEqual(calls, [])  # refused before a single control was asked for
        import tempfile  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episodes.jsonl"
            good = EpisodeProtocol(
                trials=1,
                steps=20,
                control_interval=task.protocol.control_interval,
                perturb=task.protocol.perturb,
                success=task.protocol.success,
                home=task.protocol.home,
                placements=task.protocol.placements,
            )
            evaluate_policies(
                backend, [SimPolicy("limp", limp)], good, source=STAMP, record_to=path
            )
            (record,) = read_records(path)
            (placed,) = task.protocol.placements
            self.assertEqual(
                record.placement[placed.body], dict.fromkeys(PlacementChecks.ALL, True)
            )

    def test_the_env_refuses_at_reset(self) -> None:
        from dataclasses import replace  # noqa: PLC0415

        from rq_pipeline.envs.robotiq import RobotiqEnv, bundle_source  # noqa: PLC0415
        from rq_pipeline.physics.placement import PlacementRefusedError  # noqa: PLC0415
        from rq_pipeline.tasks.aloha2 import (  # noqa: PLC0415
            PART_STATE_SLICE,
            build_kitting,
        )

        base = build_kitting()

        def hovering(trial, home):
            start = base.protocol.perturb(trial, home).copy()
            start[PART_STATE_SLICE["left"].start + 2] += LIFT_M
            return start

        task = replace(base, protocol=replace(base.protocol, perturb=hovering))
        env = RobotiqEnv(task, source=bundle_source(task.bundle_dir))
        try:
            with self.assertRaises(PlacementRefusedError) as caught:
                env.reset(seed=1002)
            self.assertIn("trial 2", str(caught.exception))
            self.assertIn("part_left", str(caught.exception))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
