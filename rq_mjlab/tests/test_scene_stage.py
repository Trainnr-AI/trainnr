"""The captured scene as the walk's stage (`rq_mjlab.scene_stage`,
docs/78 E2): the saved grid becomes an mjlab sub-terrain at the scene's
own coordinates with the spawn at the course's start; the head camera
rides on the base; the picture is an observation; the scene's
gaussians reach every render context mjlab builds."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from rq_pipeline.scenes.record import (
    DECLARED,
    SCENE_FILE,
    SPLAT_FILE,
    Alignment,
    Capture,
    Gap,
    Physics,
    SceneRecord,
    Tool,
)
from rq_pipeline.scenes.splat import Splats, write_ply
from rq_pipeline.scenes.stage import FLOOR_FRICTION, HEAD_CAMERA, HEAD_POS
from rq_pipeline.scenes.terrain import Grid, write_grid

from rq_mjlab.scene_stage import (
    CAMERA_HEIGHT,
    CAMERA_TERM,
    CAMERA_WIDTH,
    HFIELD_NAME,
    ROBOT_BASE,
    TRAIN_CELL_M,
    RendersSplats,
    SceneHeightfieldCfg,
    camera_rgb,
    head_camera_cfg,
    render_splats,
    scene_terrain_cfg,
    unrender_splats,
)
from rq_mjlab.walks import use_project, walk_spec

PROJECT = Path(__file__).resolve().parents[2] / "projects" / "go2-walk"
# A 0.5 m cell grid, 4 columns along x from x0=1, 3 rows along y from y0=-1,
# rising 0.1 m per column: the heights the fake proxy would have.
GRID = Grid(
    heights=np.array([[2.0, 2.1, 2.2, 2.3]] * 3, dtype=np.float64),
    x0=1.0,
    y0=-1.0,
    cell=0.5,
)
WAYPOINTS = [[1.5, 0.0, 2.1], [2.5, 0.0, 2.3]]  # the start is 1 m before the first


def _scene(tmp: Path, *, grid: bool = True, splats: bool = False) -> Path:
    scene = tmp / "fake"
    scene.mkdir()
    SceneRecord(
        name="fake",
        source="the test",
        capture=Capture(),
        tools=(Tool(name="test"),),
        splat={"file": SPLAT_FILE},
        proxy={"file": "proxy.obj", "vertices": 4, "faces": 2, "watertight": False},
        alignment=Alignment(
            scale=1.0,
            rotation_euler_xyz=(0, 0, 0),
            translation=(0, 0, 0),
            source="the test",
        ),
        gap=Gap(
            chamfer_m=None,
            p95_m=None,
            beyond_tolerance_fraction=None,
            hidden_fraction=None,
            tolerance_m=0.02,
            visible_samples=0,
            proxy_samples=0,
            method="none",
        ),
        physics=(
            Physics(
                name=FLOOR_FRICTION, value=[1.25, 0.3, 0.3], basis=DECLARED, span=0.2
            ),
        ),
        created_utc="2026-09-22T00:00:00+00:00",
        code="test",
        course={"waypoints": WAYPOINTS, "source": "the test"},
    ).write(scene / SCENE_FILE)
    if grid:
        write_grid(scene, GRID, 0.0, "abc")
    if splats:
        n = 5
        write_ply(
            Splats(
                means=np.zeros((n, 3), np.float32),
                quats=np.tile([1.0, 0, 0, 0], (n, 1)).astype(np.float32),
                scales=np.full((n, 3), 0.01, np.float32),
                opacities=np.array([0.9, 0.9, 0.1, 0.9, 0.1], np.float32),
                colors=np.full((n, 3), 0.5, np.float32),
            ),
            scene / SPLAT_FILE,
        )
    return scene


class TheSubTerrain(unittest.TestCase):
    def test_the_hfield_lands_at_the_scenes_coordinates(self) -> None:
        import mujoco  # noqa: PLC0415
        from mjlab.terrains.terrain_entity import TerrainEntity  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            cfg = scene_terrain_cfg(_scene(Path(tmp)), cell=GRID.cell)
            self.assertEqual(cfg.terrain_type, "generator")
            assert cfg.terrain_generator is not None
            self.assertEqual(cfg.terrain_generator.size, (1.5, 1.0))
            entity = TerrainEntity(cfg, device="cpu")
            model = entity.spec.compile()
            self.assertEqual(model.nhfield, 1)
            self.assertEqual((model.hfield_nrow[0], model.hfield_ncol[0]), (3, 4))
            geom = model.geom(HFIELD_NAME)
            # the grid's centre in the world, not mjlab's origin-centred patch
            np.testing.assert_allclose(model.geom_pos[geom.id], [1.75, -0.5, 2.0])
            # the spawn: one metre before the first waypoint, on the surface
            origin = (
                entity.terrain_origins[0, 0]
                if hasattr(entity, "terrain_origins")
                else None
            )
            if origin is None:
                origin = np.asarray(entity.env_origins[0].cpu())
            np.testing.assert_allclose(origin[:2], [0.5, 0.0], atol=1e-9)
            self.assertAlmostEqual(float(origin[2]), 2.0, places=6)
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            self.assertEqual(model.geom_type[geom.id], mujoco.mjtGeom.mjGEOM_HFIELD)
            np.testing.assert_allclose(model.geom_friction[geom.id], [1.25, 0.3, 0.3])

    def test_training_resamples_the_saved_grid_to_its_own_spacing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = scene_terrain_cfg(_scene(Path(tmp)))
            assert cfg.terrain_generator is not None
            patch = cfg.terrain_generator.sub_terrains["scene"]
            assert isinstance(patch, SceneHeightfieldCfg)
            self.assertEqual(patch.grid.cell, TRAIN_CELL_M)
            self.assertEqual(patch.grid.heights.shape, (21, 31))
            # the same footprint, the same linear surface
            self.assertEqual(cfg.terrain_generator.size, (1.5, 1.0))
            np.testing.assert_allclose(
                patch.grid.heights[0, [0, 10, 30]], [2.0, 2.1, 2.3]
            )

    def test_a_scene_without_its_grid_is_refused_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "heightfield.npz"):
                scene_terrain_cfg(_scene(Path(tmp), grid=False))

    def test_the_patch_corner_is_mjlabs(self) -> None:
        patch = SceneHeightfieldCfg.of(GRID, (0.5, 0.0), None)
        np.testing.assert_allclose(patch.corner, [-0.75, -0.5, 0.0])


class TheSceneBounds(unittest.TestCase):
    """A world that leaves the scene's footprint is truncated: past the
    heightfield there is no ground (NaN at iteration 1017, 2026-09-23)."""

    def test_the_footprint_is_the_grids_and_the_margin_is_inside_it(self) -> None:
        from rq_mjlab.scene_stage import (  # noqa: PLC0415
            BOUNDS_MARGIN_M,
            out_of_scene_bounds,
            scene_footprint,
        )

        patch = SceneHeightfieldCfg.of(GRID, (0.5, 0.0), None)
        generator = SimpleNamespace(sub_terrains={"scene": patch})
        xy = torch.tensor([[1.5, -0.5], [1.05, -0.5], [2.45, 0.45], [1.5, 0.9]])
        env: Any = SimpleNamespace(
            num_envs=4,
            device="cpu",
            scene=_SceneWithTerrain(generator, xy),
        )
        self.assertEqual(scene_footprint(env), (1.0, 2.5, -1.0, 0.0))
        out = out_of_scene_bounds(env)
        # inside; inside but within the margin of the x edge; past the y
        # edge (and within the margin of the x edge); past the y edge
        self.assertEqual(out.tolist(), [False, True, True, True])
        self.assertEqual(
            out_of_scene_bounds(env, margin=0.0).tolist(), [False, False, True, True]
        )
        self.assertEqual(BOUNDS_MARGIN_M, 0.3)

    def test_off_a_scene_nothing_is_out_of_bounds(self) -> None:
        from rq_mjlab.scene_stage import out_of_scene_bounds  # noqa: PLC0415

        env: Any = SimpleNamespace(
            num_envs=2, device="cpu", scene=SimpleNamespace(terrain=None)
        )
        self.assertEqual(out_of_scene_bounds(env).tolist(), [False, False])


class _SceneWithTerrain:
    def __init__(self, generator: Any, xy: torch.Tensor) -> None:
        self.terrain = SimpleNamespace(cfg=SimpleNamespace(terrain_generator=generator))
        self._robot = SimpleNamespace(
            data=SimpleNamespace(
                root_link_pos_w=torch.cat([xy, torch.zeros(len(xy), 1)], 1)
            )
        )

    def __getitem__(self, name: str) -> Any:
        return self._robot


class TheHeadCamera(unittest.TestCase):
    def test_the_camera_rides_on_the_base_looking_forward(self) -> None:
        head = head_camera_cfg()
        self.assertEqual((head.name, head.parent_body), (HEAD_CAMERA, ROBOT_BASE))
        self.assertEqual(head.pos, HEAD_POS)
        self.assertEqual((head.width, head.height), (CAMERA_WIDTH, CAMERA_HEIGHT))
        self.assertEqual(head.data_types, ("rgb",))
        self.assertAlmostEqual(sum(q * q for q in head.quat), 1.0, places=6)

    def test_the_picture_is_a_row_per_world_in_unit_range(self) -> None:
        rgb = torch.full((3, 4, 5, 3), 255, dtype=torch.uint8)
        env: Any = SimpleNamespace(
            scene=SimpleNamespace(
                sensors={"head": SimpleNamespace(data=SimpleNamespace(rgb=rgb))}
            )
        )
        out = camera_rgb(env, "head")
        self.assertEqual(tuple(out.shape), (3, 60))
        self.assertEqual(float(out.max()), 1.0)
        env.scene.sensors["head"].data.rgb = None
        with self.assertRaisesRegex(ValueError, "renders no rgb"):
            camera_rgb(env, "head")


class TheSplatsInTheContext(unittest.TestCase):
    def test_the_proxy_adds_the_arrays_and_forwards_the_rest(self) -> None:
        calls: list[dict[str, Any]] = []

        def create_render_context(mjm: Any, **kwargs: Any) -> str:
            calls.append({"mjm": mjm, **kwargs})
            return "context"

        module = SimpleNamespace(
            create_render_context=create_render_context, put_model=lambda m: "model"
        )
        proxy = RendersSplats(module, {"splat_position": np.zeros((2, 3), np.float32)})
        self.assertEqual(proxy.create_render_context("m", nworld=4), "context")
        self.assertEqual(calls[0]["nworld"], 4)
        self.assertEqual(calls[0]["splat_position"].shape, (2, 3))
        self.assertEqual(proxy.put_model("x"), "model")

    def test_installing_reads_the_visible_gaussians_and_uninstalling_restores(
        self,
    ) -> None:
        module = SimpleNamespace(create_render_context=lambda *a, **k: k)
        context_module = SimpleNamespace(mjwarp=module)
        with tempfile.TemporaryDirectory() as tmp:
            scene = _scene(Path(tmp), splats=True)
            n = render_splats(scene, context_module=context_module)
        self.assertEqual(n, 3)  # two of five are below the visible opacity
        kwargs = context_module.mjwarp.create_render_context("m")
        self.assertEqual(kwargs["splat_position"].shape, (3, 3))
        self.assertEqual(kwargs["splat_rgba"].shape, (3, 4))
        # installing twice wraps the module once
        with tempfile.TemporaryDirectory() as tmp:
            render_splats(_scene(Path(tmp), splats=True), context_module=context_module)
        self.assertIs(context_module.mjwarp._module, module)
        unrender_splats(context_module=context_module)
        self.assertIs(context_module.mjwarp, module)


class TheWalksTakeAScene(unittest.TestCase):
    def test_the_other_walks_refuse_a_scene_by_name(self) -> None:
        for robot in ("microduck", "go1"):
            with self.subTest(robot=robot):
                with self.assertRaisesRegex(ValueError, "no scene stage"):
                    walk_spec(robot).env_cfg(
                        dr_span=None, pin_scale=None, scene=Path("/nowhere")
                    )


@unittest.skipUnless(
    (PROJECT / "robots" / "go2" / "go2.xml").is_file(), "no go2-walk project"
)
class TheGo2OnAScene(unittest.TestCase):
    def test_the_config_stands_on_the_scene_and_sees_it(self) -> None:
        use_project(PROJECT)
        with tempfile.TemporaryDirectory() as tmp:
            scene = _scene(Path(tmp))
            cfg, identity = walk_spec("go2").env_cfg(
                dr_span=0.1, pin_scale=None, scene=scene
            )
            self.assertEqual(cfg.scene.terrain.terrain_type, "generator")
            self.assertNotIn("terrain_levels", cfg.curriculum)
            self.assertNotIn("out_of_terrain_bounds", cfg.terminations)
            sensors = {s.name for s in cfg.scene.sensors}
            self.assertIn(
                "terrain_scan", sensors
            )  # the rough recipe: the hurdles are seen
            self.assertIn(HEAD_CAMERA, sensors)
            for group in ("actor", "critic"):
                self.assertIn(CAMERA_TERM, cfg.observations[group].terms)
            self.assertTrue(identity["scene"].startswith("fake@"))
            self.assertIn("64x64", identity["cameras"])
            without, identity = walk_spec("go2").env_cfg(
                dr_span=0.1, pin_scale=None, scene=scene, cameras=False
            )
            self.assertNotIn(HEAD_CAMERA, {s.name for s in without.scene.sensors})
            self.assertNotIn(CAMERA_TERM, without.observations["actor"].terms)
            self.assertEqual(identity["cameras"], "none")


if __name__ == "__main__":
    unittest.main()
