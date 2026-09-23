"""What stands above the ground as an occupancy volume: closed around
every occupied component, a table top with air beneath it, joined with
the ground's top surface into the proxy, and staged as a heightfield
under convex parts so a walker goes under the table."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from rq_pipeline.scenes import volume
from tests._extras import needs_scene

FACES_PER_EDGE = 2  # a closed surface: every edge borders two triangles
BESIDE_LEG_X = (0.1, 0.4)  # under the table top, clear of its leg
BESIDE_LEG_Y = 0.3


def _table(rng: np.random.Generator, n: int = 30000) -> np.ndarray:
    """Centres of a floor, a table top 0.7 m up, and one leg under it."""
    floor = np.column_stack(
        [rng.uniform(-1, 1, n), rng.uniform(-1, 1, n), rng.normal(0, 0.004, n)]
    )
    k = n // 3
    top = np.column_stack(
        [
            rng.uniform(-0.5, 0.5, k),
            rng.uniform(-0.5, 0.5, k),
            rng.uniform(0.70, 0.74, k),
        ]
    )
    m = n // 10
    leg = np.column_stack(
        [
            rng.uniform(-0.03, 0.03, m),
            rng.uniform(-0.03, 0.03, m),
            rng.uniform(0.0, 0.70, m),
        ]
    )
    return np.concatenate([floor, top, leg])


class TheVolume(unittest.TestCase):
    def test_the_voxel_surface_is_closed_and_leaves_the_air_under_the_table(
        self,
    ) -> None:
        centres = _table(np.random.default_rng(0))
        vertices, faces, facts = volume.overhang_mesh(centres)
        self.assertGreater(facts["voxels"], 100)
        # closed: every edge shared by exactly two triangles
        edges = np.sort(
            np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]),
            axis=1,
        )
        _, counts = np.unique(edges, axis=0, return_counts=True)
        self.assertTrue((counts == FACES_PER_EDGE).all())
        # the leg reaches down to the clearance, the top spans its slab
        self.assertAlmostEqual(
            vertices[:, 2].min(), volume.OVERHANG_CLEARANCE_M, delta=volume.VOXEL_M
        )
        self.assertGreater(vertices[:, 2].max(), 0.7)
        # and beside the leg, under the top, there is air: no vertex below the top
        beside = vertices[
            (np.abs(vertices[:, 0]) > BESIDE_LEG_X[0])
            & (np.abs(vertices[:, 0]) < BESIDE_LEG_X[1])
            & (np.abs(vertices[:, 1]) < BESIDE_LEG_Y)
        ]
        self.assertGreater(beside[:, 2].min(), 0.6)

    def test_nothing_above_the_ground_is_a_fact_not_a_failure(self) -> None:
        flat = np.column_stack([np.linspace(0, 1, 500), np.zeros(500), np.zeros(500)])
        _, faces, facts = volume.overhang_mesh(flat)
        self.assertEqual((facts["voxels"], facts["faces"], faces.shape[0]), (0, 0, 0))
        ground, above = volume.split_by_clearance(flat)
        self.assertEqual((len(ground), len(above)), (500, 0))

    def test_a_floater_alone_in_its_voxel_is_not_a_thing(self) -> None:
        one = np.array([[0.0, 0.0, 1.0]])
        grid, _ = volume.occupancy(one)
        self.assertFalse(grid.any())
        grid, _ = volume.occupancy(np.repeat(one, volume.MIN_CENTRES, 0))
        self.assertTrue(grid.any())


@needs_scene
class TheOverhangsTerrain(unittest.TestCase):
    def test_the_stage_carries_the_ground_under_the_parts(self) -> None:
        import mujoco  # noqa: PLC0415

        from rq_pipeline.scenes import capture, terrain  # noqa: PLC0415
        from rq_pipeline.scenes.obj import write_obj  # noqa: PLC0415
        from rq_pipeline.scenes.proxy import (  # noqa: PLC0415
            OVERHANG_PARTS,
            ensure_parts,
        )
        from rq_pipeline.scenes.record import (  # noqa: PLC0415
            GROUND_FILE,
            OVERHANG_FILE,
            PROXY_FILE,
        )

        centres = _table(np.random.default_rng(1))
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp)
            ground, _ = volume.split_by_clearance(centres)
            gv, gf, _ = capture.top_surface_mesh(ground)
            write_obj(scene / GROUND_FILE, gv, gf)
            ov, of, _ = volume.overhang_mesh(centres)
            write_obj(scene / OVERHANG_FILE, ov, of)
            write_obj(
                scene / PROXY_FILE,
                np.concatenate([gv, ov]),
                np.concatenate([gf, of + len(gv)]),
            )
            parts = ensure_parts(scene, files=OVERHANG_PARTS)
            self.assertGreater(parts.parts, 0)
            spec = mujoco.MjSpec()
            facts = terrain.overhangs(spec, spec.worldbody, scene, None)
            model = spec.compile()
            self.assertEqual(facts.kind, terrain.OVERHANGS)
            self.assertEqual(facts.geoms, 1 + parts.parts)
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            group = np.ones(6, dtype=np.uint8)
            gid = np.zeros(1, dtype=np.int32)
            # under the table a ray down from the clearance reaches the floor
            d = mujoco.mj_ray(
                model,
                data,
                np.array([0.3, 0.0, 0.2]),
                np.array([0, 0, -1.0]),
                group,
                1,
                -1,
                gid,
            )
            self.assertLess(abs(0.2 - d), 0.06)
            # and from above, the table top comes first
            d = mujoco.mj_ray(
                model,
                data,
                np.array([0.3, 0.0, 2.0]),
                np.array([0, 0, -1.0]),
                group,
                1,
                -1,
                gid,
            )
            self.assertGreater(2.0 - d, 0.6)
            self.assertTrue((scene / terrain.grid_file(GROUND_FILE)).is_file())


if __name__ == "__main__":
    unittest.main()
