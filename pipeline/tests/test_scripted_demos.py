"""The open-loop scripted press adapter: the gain rule (both kp terms),
the condition riding every manifest, the dr refusal — on the one-hinge
task, headless (frame_every=0)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._extras import needs_sim
from tests.test_press_batch import TARGET, XML, _task

DR = {"damping": (0.9, 1.1), "gain": (0.95, 1.05)}


@needs_sim
class ScaleServoDynamics(unittest.TestCase):
    def test_scales_damped_joints_and_both_kp_terms(self) -> None:
        import mujoco  # noqa: PLC0415

        from rq_pipeline.collect.scripted_demos import (  # noqa: PLC0415
            scale_servo_dynamics,
        )

        spec = mujoco.MjSpec.from_string(XML)
        free = spec.worldbody.add_body(name="loose")
        free.add_freejoint()
        free.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.01, 0.01, 0.01], mass=0.01
        )
        kp = spec.actuators[0].gainprm[0]
        damping = spec.joints[0].damping[0]
        scale_servo_dynamics(spec, damping_scale=1.5, gain_scale=0.8)
        self.assertAlmostEqual(spec.joints[0].damping[0], damping * 1.5)
        self.assertEqual(spec.joints[1].damping[0], 0.0)  # the free joint: untouched
        self.assertAlmostEqual(spec.actuators[0].gainprm[0], kp * 0.8)
        # biasprm[1] is -kp for a position servo; scaled TOGETHER or the
        # "gain" becomes a setpoint (docs/07 2026-08-26).
        self.assertAlmostEqual(spec.actuators[0].biasprm[1], -kp * 0.8)
        model = spec.compile()
        self.assertAlmostEqual(model.actuator_gainprm[0][0], kp * 0.8)


@needs_sim
class GenerateScriptedDemos(unittest.TestCase):
    def _policy(self, step: int, sensordata: object) -> list[float]:
        del step, sensordata
        return [TARGET]

    def test_the_condition_rides_every_manifest(self) -> None:
        from rq_pipeline.collect.press import EpisodeManifest  # noqa: PLC0415
        from rq_pipeline.collect.scripted_demos import (  # noqa: PLC0415
            generate_scripted_demos,
        )

        with tempfile.TemporaryDirectory() as tmp:
            batch = generate_scripted_demos(
                Path(tmp),
                task_factory=_task,
                policy=self._policy,
                expert="hold-up@test",
                dr=DR,
                basis="test-declared span",
                episodes=2,
                seed=9,
                frame_every=0,
                say=lambda _line: None,
            )
            self.assertTrue(batch.complete)
            manifests = [
                EpisodeManifest.read_from(Path(tmp) / f"episode_{i:04d}")
                for i in range(2)
            ]
        for manifest in manifests:
            self.assertEqual(manifest.dynamics_basis, "test-declared span")
            for param, (low, high) in DR.items():
                self.assertTrue(low <= manifest.dynamics[param] <= high)
            self.assertIn("trial", manifest.draws)
            self.assertEqual(manifest.expert, "hold-up@test")
        # Two episodes, two draws — the randomiser randomises.
        self.assertNotEqual(
            manifests[0].dynamics["damping"], manifests[1].dynamics["damping"]
        )

    def test_refuses_a_condition_missing_a_range(self) -> None:
        from rq_pipeline.collect.scripted_demos import (  # noqa: PLC0415
            generate_scripted_demos,
        )

        with (
            tempfile.TemporaryDirectory() as tmp,
            self.assertRaisesRegex(ValueError, "damping"),
        ):
            generate_scripted_demos(
                Path(tmp),
                task_factory=_task,
                policy=self._policy,
                expert="e",
                dr={"gain": (0.9, 1.1)},
                basis="b",
                episodes=1,
                seed=1,
                frame_every=0,
            )


if __name__ == "__main__":
    unittest.main()
