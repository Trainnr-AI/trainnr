"""The nominal solver condition, pinned on every compiled scene.

Half of these values were inherited MuJoCo defaults nobody ever chose
(Euler at 500 Hz — found by the 2026-08-27 solver review, docs/47/48).
sysid fits are fits OF this discretization, so the option block is part
of the identified artifact; this test is the guard that an upstream
MuJoCo default change (they already recommend implicitfast) cannot
silently move the nominal condition. mjSOL_NEWTON + elliptic is also a
MEASURED requirement: the only solver/cone that passes the kitting
referee on both arm64 and x86_64 (tools/solver-study.py, docs/47 §7 —
PGS passes on x86_64 alone, at 28x the iterations).
"""

import importlib.util
import unittest

MUJOCO_PRESENT = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(MUJOCO_PRESENT, "sim extra not installed (uv sync --extra sim)")
class NominalOptions(unittest.TestCase):
    def test_every_task_scene_resolves_to_the_pinned_block(self) -> None:
        import mujoco  # noqa: PLC0415

        from rq_pipeline.tasks.aloha2 import build_kitting  # noqa: PLC0415
        from rq_pipeline.tasks.components import compose  # noqa: PLC0415
        from rq_pipeline.tasks.so101 import build_lift  # noqa: PLC0415

        scenes = {
            "kitting": build_kitting().spec,
            "lift": build_lift().spec,
            "rig": compose(car=True, arm=True),
        }
        # Pin-both-ends, on purpose: the declaration lives in
        # tasks/scene.py::NominalOptions; this is the second copy, so
        # moving the nominal condition takes a decision in two places.
        for name, spec in scenes.items():
            opt = spec.compile().opt
            with self.subTest(scene=name):
                self.assertEqual(opt.solver, mujoco.mjtSolver.mjSOL_NEWTON)
                self.assertEqual(opt.integrator, mujoco.mjtIntegrator.mjINT_EULER)
                self.assertEqual(opt.cone, mujoco.mjtCone.mjCONE_ELLIPTIC)
                self.assertEqual(opt.timestep, 0.002)
                self.assertEqual(opt.impratio, 10)

    def test_the_families_refuse_an_unknown_name(self) -> None:
        from rq_pipeline.tasks.scene import option_enum  # noqa: PLC0415

        with self.assertRaises(ValueError):
            option_enum("solver", "juggling")


if __name__ == "__main__":
    unittest.main()
