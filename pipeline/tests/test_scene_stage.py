"""A many-worlds grid built on a task's own stage keeps the stage's
ground and dressing, gets a sky when it brings none, and still hides
each child's floor (2026-09-12: the walk mirror on mjlab's terrain)."""

from __future__ import annotations

import unittest

from tests._extras import needs_sim


@needs_sim
class GridOnAStage(unittest.TestCase):
    def test_the_stage_ground_survives_and_a_sky_is_added(self) -> None:
        import mujoco  # noqa: PLC0415 - the sim extra

        from rq_pipeline.tasks.scene import (  # noqa: PLC0415
            FLOOR_GEOM,
            GeomGroup,
            grid_of,
        )

        stage = mujoco.MjSpec.from_string(
            '<mujoco><asset><texture name="g" type="2d" builtin="checker" width="8" '
            'height="8"/><material name="g" texture="g"/></asset><worldbody>'
            '<light name="sun" pos="0 0 2"/><geom name="stage/terrain" type="plane" '
            'size="0 0 0.1" material="g"/></worldbody></mujoco>'
        )
        child = (
            '<mujoco><worldbody><geom name="floor" type="plane" size="0 0 0.1"/>'
            '<body name="b"><freejoint/><geom type="sphere" size="0.1"/></body>'
            "</worldbody></mujoco>"
        )
        scene, prefixes = grid_of(
            "staged",
            (mujoco.MjSpec.from_string(child) for _ in range(2)),
            pitch=0.0,
            stage=stage,
        )
        model = scene.compile()
        names = [model.geom(i).name for i in range(model.ngeom)]
        self.assertIn("stage/terrain", names)
        self.assertNotIn("ground", names)  # the plain plane is not added on a stage
        self.assertEqual(prefixes, ["w00/", "w01/"])
        floors = [i for i in range(model.ngeom) if names[i].endswith(FLOOR_GEOM)]
        self.assertEqual(len(floors), 2)
        self.assertTrue(all(model.geom_group[i] == GeomGroup.HIDDEN for i in floors))
        skies = [
            i
            for i in range(model.ntex)
            if model.tex_type[i] == mujoco.mjtTexture.mjTEXTURE_SKYBOX
        ]
        self.assertEqual(len(skies), 1)

    def test_a_stage_with_its_own_sky_keeps_it(self) -> None:
        import mujoco  # noqa: PLC0415 - the sim extra

        from rq_pipeline.tasks.scene import add_sky  # noqa: PLC0415

        spec = mujoco.MjSpec.from_string(
            '<mujoco><asset><texture name="own" type="skybox" builtin="flat" '
            'rgb1="1 0 0" width="8" height="8"/></asset><worldbody/></mujoco>'
        )
        add_sky(spec)
        self.assertEqual([t.name for t in spec.textures], ["own"])


if __name__ == "__main__":
    unittest.main()
