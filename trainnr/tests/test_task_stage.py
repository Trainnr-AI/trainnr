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

        from trainnr.tasks.scene import (  # noqa: PLC0415
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
            sky=True,
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

    def test_the_bare_grid_keeps_its_pixels(self) -> None:
        # No sky unless asked, no shadow budget unless dressed: the walk
        # press's chase camera renders training and judging frames of
        # the vision student off this grid (2026-09-13).
        import importlib.util  # noqa: PLC0415
        import sys  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        import mujoco  # noqa: PLC0415 - the sim extra

        from trainnr.tasks.scene import RenderBudget  # noqa: PLC0415

        tools = Path(__file__).resolve().parents[2] / "tools"
        sys.path.insert(0, str(tools))
        spec = importlib.util.spec_from_file_location(
            "studio_render_stream", tools / "studio-render-stream.py"
        )
        srs = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(srs)

        def skies(model: mujoco.MjModel) -> int:
            return sum(
                int(model.tex_type[i] == mujoco.mjtTexture.mjTEXTURE_SKYBOX)
                for i in range(model.ntex)
            )

        bare = srs.walk_scene("microduck", 1, 256, dressed=False)
        dressed = srs.walk_scene("microduck", 1, 256)
        self.assertEqual(skies(bare), 0)
        self.assertEqual(
            bare.vis.quality.shadowsize,
            mujoco.MjModel.from_xml_string("<mujoco/>").vis.quality.shadowsize,
        )
        self.assertEqual(skies(dressed), 1)
        self.assertEqual(dressed.vis.quality.shadowsize, RenderBudget.SHADOWSIZE)

    def test_a_stage_with_its_own_sky_keeps_it(self) -> None:
        import mujoco  # noqa: PLC0415 - the sim extra

        from trainnr.tasks.scene import add_sky  # noqa: PLC0415

        spec = mujoco.MjSpec.from_string(
            '<mujoco><asset><texture name="own" type="skybox" builtin="flat" '
            'rgb1="1 0 0" width="8" height="8"/></asset><worldbody/></mujoco>'
        )
        add_sky(spec)
        self.assertEqual([t.name for t in spec.textures], ["own"])


if __name__ == "__main__":
    unittest.main()
