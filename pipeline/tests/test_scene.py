"""Captured scenes (docs/78 §3): the splat file round-trips, the frame
transform matches MuJoCo's own euler convention, the gap audit reads a
plane against a plane, and a Neverwhere folder imports as a scene the
index, the drawer, the tile and the door all read."""

from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from rq_pipeline.deploy.manifest import Key, load_manifest
from rq_pipeline.project import PROJECT_ENV, create_project, index_project
from rq_pipeline.project.kinds import Kind
from rq_pipeline.scenes import neverwhere, proxy, stage, terrain
from rq_pipeline.scenes.obj import read_obj, write_obj
from rq_pipeline.scenes.record import (
    DECLARED,
    MEASURED,
    PROXY_FILE,
    PROXY_MJCF,
    SCENE_FILE,
    SCENE_SCHEMA,
    SPLAT_FILE,
    UNRECORDED,
    Physics,
    load_scene_record,
)
from rq_pipeline.scenes.splat import (
    SH_C0,
    Splats,
    euler_xyz_matrix,
    quat_from_matrix,
    read_ply,
    read_web_splat,
    write_ply,
)
from tests._extras import SCENE, needs_scene, needs_sim
from tests.test_deploy import _manifest


def _splats(n: int = 50, seed: int = 0, sh: int = 0) -> Splats:
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return Splats(
        means=rng.uniform(-1, 1, (n, 3)).astype(np.float32),
        quats=q,
        scales=rng.uniform(0.01, 0.1, (n, 3)).astype(np.float32),
        opacities=rng.uniform(0.05, 0.95, n).astype(np.float32),
        colors=rng.uniform(0.1, 0.9, (n, 3)).astype(np.float32),
        sh_rest=rng.normal(size=(n, sh, 3)).astype(np.float32)
        if sh
        else np.zeros((0, 0, 3), np.float32),
    )


def _web_splat(path: Path, splats: Splats) -> Path:
    rows = np.zeros((splats.count, 32), np.uint8)
    rows[:, 0:12] = splats.means.astype("<f4").view(np.uint8).reshape(-1, 12)
    rows[:, 12:24] = splats.scales.astype("<f4").view(np.uint8).reshape(-1, 12)
    rows[:, 24:27] = (splats.colors * 255).astype(np.uint8)
    rows[:, 27] = (splats.opacities * 255).astype(np.uint8)
    rows[:, 28:32] = np.clip(splats.quats * 128 + 128, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.tofile(path)
    return path


def _plane_obj(path: Path, z: float = 0.0, half: float = 2.0) -> Path:
    corners = ((-half, -half), (half, -half), (half, half), (-half, half))
    lines = [f"v {x} {y} {z}" for x, y in corners] + ["f 1 2 3", "f 1 3 4"]
    path.write_text("\n".join(lines) + "\n")
    return path


class TheSplatFile(unittest.TestCase):
    def test_the_ply_round_trips_with_harmonics(self) -> None:
        s = _splats(sh=15)
        with tempfile.TemporaryDirectory() as tmp:
            path = write_ply(s, Path(tmp) / "a.ply")
            back = read_ply(path)
        self.assertEqual(back.count, s.count)
        self.assertEqual(back.sh_degree, 3)
        np.testing.assert_allclose(back.means, s.means, atol=1e-6)
        np.testing.assert_allclose(back.scales, s.scales, rtol=1e-5)
        np.testing.assert_allclose(back.opacities, s.opacities, atol=1e-5)
        np.testing.assert_allclose(back.colors, s.colors, atol=1e-5)
        np.testing.assert_allclose(back.sh_rest, s.sh_rest, atol=1e-5)
        np.testing.assert_allclose(
            np.abs((back.quats * s.quats).sum(1)), 1.0, atol=1e-5
        )

    def test_a_ply_that_is_not_a_splat_is_refused_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mesh.ply"
            path.write_bytes(
                b"ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
                b"property float x\nproperty float y\nproperty float z\nend_header\n"
                + np.zeros(3, "<f4").tobytes()
            )
            with self.assertRaisesRegex(ValueError, "not a 3DGS PLY"):
                read_ply(path)

    def test_the_web_splat_reads_its_quantised_records(self) -> None:
        s = _splats(n=8)
        with tempfile.TemporaryDirectory() as tmp:
            back = read_web_splat(_web_splat(Path(tmp) / "m.splat", s))
        self.assertEqual(back.count, 8)
        np.testing.assert_allclose(back.means, s.means, atol=1e-6)
        np.testing.assert_allclose(back.opacities, s.opacities, atol=1 / 255)
        np.testing.assert_allclose(
            np.abs((back.quats * s.quats).sum(1)), 1.0, atol=0.02
        )

    def test_colour_is_the_zeroth_harmonic(self) -> None:
        self.assertAlmostEqual(SH_C0, 0.28209479, places=6)


class TheFrame(unittest.TestCase):
    @needs_sim
    def test_euler_xyz_matches_mujocos_own_convention(self) -> None:
        import mujoco  # noqa: PLC0415

        euler = np.array([0.4, -1.1, 2.3])
        text = " ".join(map(str, euler))
        # Radians, as the scene files carry them (MuJoCo's default is degrees).
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><compiler angle="radian"/><worldbody>'
            f'<body name="b" euler="{text}"/></worldbody></mujoco>'
        )
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        theirs = data.xmat[model.body("b").id].reshape(3, 3)
        np.testing.assert_allclose(euler_xyz_matrix(euler), theirs, atol=1e-9)
        q = quat_from_matrix(theirs)
        np.testing.assert_allclose(np.abs(q @ model.body("b").quat), 1.0, atol=1e-9)

    def test_a_similarity_moves_centres_scales_and_rotations_together(self) -> None:
        s = _splats(n=5)
        rot = euler_xyz_matrix(np.array([0.0, 0.0, math.pi / 2]))
        moved = s.transformed(
            scale=2.0, rotation=rot, translation=np.array([1.0, 0.0, 0.0])
        )
        expected = 2.0 * (s.means @ rot.T) + np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(moved.means, expected, atol=1e-5)
        np.testing.assert_allclose(moved.scales, s.scales * 2.0, rtol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(moved.quats, axis=1), 1.0, atol=1e-6)


class TheRecord(unittest.TestCase):
    def test_a_physics_value_carries_its_basis(self) -> None:
        Physics(name="f", value=1.0, basis=DECLARED, span=0.2)
        Physics(name="f", value=1.0, basis=MEASURED, interval=(0.9, 1.1))
        with self.assertRaisesRegex(ValueError, "carries its span"):
            Physics(name="f", value=1.0, basis=DECLARED)
        with self.assertRaisesRegex(ValueError, "carries its interval"):
            Physics(name="f", value=1.0, basis=MEASURED)
        with self.assertRaisesRegex(ValueError, "basis is one of"):
            Physics(name="f", value=1.0, basis="guessed")


def _neverwhere_folder(
    tmp: Path, *, friction: str = "1.25 0.3 0.3", n: int = 400
) -> Path:
    """A fake Neverwhere scene: a splat plane at z=0.5 in a frame the
    transform scales by 2 and lifts by 1, over a proxy plane at z=2."""
    src = tmp / "hurdle_fake_v1"
    (src / "3dgs").mkdir(parents=True)
    (src / "geometry").mkdir()
    rng = np.random.default_rng(1)
    means = np.stack(
        [rng.uniform(-1, 1, n), rng.uniform(-1, 1, n), np.full(n, 0.5)], 1
    ).astype(np.float32)
    s = Splats(
        means=means,
        quats=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),
        scales=np.full((n, 3), 0.02, np.float32),
        opacities=np.full(n, 0.9, np.float32),
        colors=np.tile(np.array([0.2, 0.6, 0.9], np.float32), (n, 1)),
    )
    _web_splat(src / neverwhere.WEB_SPLAT, s)
    (src / neverwhere.COLLISION_TF).write_text(
        json.dumps(
            {
                "mesh_scale": 2.0,
                "mesh_pos": [0.0, 0.0, 1.0],
                "mesh_euler": [0.0, 0.0, 0.0],
            }
        )
    )
    _plane_obj(src / neverwhere.COLLISION_MESH, z=2.0, half=2.0)
    (src / "hurdle_fake_v1.xml").write_text(
        '<mujoco><worldbody><geom type="sdf" name="collision_mesh_geom" '
        f'mesh="collision_mesh" friction="{friction}"/></worldbody></mujoco>'
    )
    return src


class TheImport(unittest.TestCase):
    def test_a_folder_that_is_not_a_scene_is_refused_by_name(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            self.assertRaisesRegex(ValueError, "not a Neverwhere scene"),
        ):
            neverwhere.import_scene(Path(tmp), Path(tmp) / "out", name="x")

    def test_the_splat_lands_in_the_world_frame_and_the_record_is_honest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = _neverwhere_folder(Path(tmp))
            out = Path(tmp) / "scenes" / "fake"
            path = neverwhere.import_scene(src, out, name="fake")
            self.assertEqual(path.name, SCENE_FILE)
            for f in (SPLAT_FILE, PROXY_FILE, PROXY_MJCF):
                self.assertTrue((out / f).is_file(), f)
            moved = read_ply(out / SPLAT_FILE)
            # z = 2 * 0.5 + 1 = 2: the splat plane meets the proxy plane.
            np.testing.assert_allclose(moved.means[:, 2], 2.0, atol=1e-5)
            np.testing.assert_allclose(moved.scales, 0.04, atol=1e-6)
            rec = load_scene_record(path)
            self.assertEqual(rec.schema, SCENE_SCHEMA)
            self.assertEqual(rec.source, "neverwhere/hurdle_fake_v1")
            self.assertEqual(rec.capture.device, UNRECORDED)
            self.assertEqual(rec.alignment.scale, 2.0)
            [friction] = rec.physics
            self.assertEqual(
                (friction.name, friction.basis, friction.span),
                ("floor_friction", DECLARED, 0.2),
            )
            self.assertEqual(friction.value, [1.25, 0.3, 0.3])
            self.assertEqual(rec.declared, ("floor_friction",))
            self.assertIn('group="3"', (out / PROXY_MJCF).read_text())
            self.assertIn('friction="1.25 0.3 0.3"', (out / PROXY_MJCF).read_text())
            if SCENE:
                self.assertIsNotNone(rec.gap.p95_m)
                assert rec.gap.p95_m is not None
                self.assertLess(rec.gap.p95_m, 0.001, "the planes coincide")
                self.assertEqual(rec.gap.beyond_tolerance_fraction, 0.0)
            else:
                self.assertIsNone(rec.gap.p95_m)
                self.assertIn("scene extra", rec.gap.note)
            with self.assertRaisesRegex(ValueError, "never overwritten"):
                neverwhere.import_scene(src, out, name="fake")

    @needs_scene
    def test_the_gap_sees_a_lifted_proxy(self) -> None:
        from rq_pipeline.scenes import gap as gap_audit  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            rng = np.random.default_rng(2)
            n = 300
            s = Splats(
                means=np.stack(
                    [rng.uniform(-1, 1, n), rng.uniform(-1, 1, n), np.zeros(n)], 1
                ).astype(np.float32),
                quats=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),
                scales=np.full((n, 3), 0.02, np.float32),
                opacities=np.full(n, 0.9, np.float32),
                colors=np.full((n, 3), 0.5, np.float32),
            )
            proxy = _plane_obj(Path(tmp) / "p.obj", z=0.05, half=1.0)
            g = gap_audit.measure(s, proxy, tolerance_m=0.02)
            self.assertAlmostEqual(g.p95_m or 0.0, 0.05, places=3)
            self.assertEqual(g.beyond_tolerance_fraction, 1.0)
            self.assertEqual(g.hidden_fraction, 1.0)
            self.assertEqual(g.visible_samples, n)
            self.assertEqual(g.footprint_fraction, 1.0)


class TheProjectAndTheDoor(unittest.TestCase):
    def test_the_scene_is_indexed_read_and_shown(self) -> None:
        from rq_pipeline.mcp_server import describe_scene, import_scene  # noqa: PLC0415
        from rq_pipeline.project.details import _scene  # noqa: PLC0415
        from rq_pipeline.project.previews import _render_scene  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            project = create_project(Path(tmp) / "p", "p")
            src = _neverwhere_folder(Path(tmp))
            os.environ[PROJECT_ENV] = str(project.root)
            try:
                out = import_scene(str(src), "fake")
                self.assertEqual(out["status"], "done", out)
                self.assertTrue(out["scene"].startswith("fake@"))
                self.assertEqual(out["declared"], ["floor_friction"])
                self.assertEqual(out["splats"], 400)
                index = index_project(project)
                [card] = index.by_kind(Kind.SCENE)
                self.assertEqual(card.summary["source"], "neverwhere/hurdle_fake_v1")
                self.assertEqual(card.summary["splats"], 400)
                self.assertIn("gap p95", card.summary)
                sections = _scene(project, project.root / card.path, card)
                self.assertEqual(sections[0]["title"], "Scene")
                self.assertEqual(
                    sections[1]["title"], "Visible surface against the collision proxy"
                )
                tile = Path(tmp) / "tile.png"
                if _render_scene(project, project.root / card.path, tile, {}):
                    self.assertTrue(tile.is_file())
                described = describe_scene(card.stamp)
                self.assertEqual(described["status"], "done")
                self.assertEqual(described["alignment"]["scale"], 2.0)
                self.assertEqual(import_scene(str(src), "fake")["status"], "refused")
                self.assertEqual(
                    import_scene(str(Path(tmp) / "nowhere"), "b")["status"], "refused"
                )
                self.assertEqual(
                    describe_scene("nobody@000000000000")["status"], "refused"
                )
            finally:
                os.environ.pop(PROJECT_ENV, None)


if __name__ == "__main__":
    unittest.main()


def _l_shape_obj(path: Path) -> Path:
    """A non-convex step: a 2 m floor slab with a 0.3 m block on one half -
    one convex hull would roof the whole slab at the block's height."""
    boxes = [((-1.0, -1.0, -0.1), (1.0, 1.0, 0.0)), ((0.0, -1.0, 0.0), (1.0, 1.0, 0.3))]
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for lo, hi in boxes:
        base = len(vertices)
        for dz in (lo[2], hi[2]):
            for dy in (lo[1], hi[1]):
                for dx in (lo[0], hi[0]):
                    vertices.append([dx, dy, dz])
        quads = [
            (0, 1, 3, 2),
            (4, 6, 7, 5),
            (0, 4, 5, 1),
            (2, 3, 7, 6),
            (0, 2, 6, 4),
            (1, 5, 7, 3),
        ]
        for a, b, c, d in quads:
            faces.append([base + a, base + b, base + c])
            faces.append([base + a, base + c, base + d])
    return write_obj(path, np.array(vertices), np.array(faces))


class TheProxyParts(unittest.TestCase):
    def test_obj_round_trips_and_refuses_a_faceless_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = write_obj(Path(tmp) / "t.obj", np.eye(3), np.array([[0, 1, 2]]))
            v, f = read_obj(p)
            self.assertEqual(v.shape, (3, 3))
            self.assertEqual(f.tolist(), [[0, 1, 2]])
            (Path(tmp) / "empty.obj").write_text("v 0 0 0\n")
            with self.assertRaisesRegex(ValueError, "no faces"):
                read_obj(Path(tmp) / "empty.obj")

    @needs_scene
    def test_a_step_becomes_more_than_one_hull_and_the_record_reads_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp)
            _l_shape_obj(scene / "proxy.obj")
            first = proxy.ensure_parts(scene)
            self.assertGreaterEqual(first.parts, 2, "one hull would roof the step")
            self.assertEqual(len(first.files), first.parts)
            self.assertTrue((scene / proxy.PARTS_DIR / first.files[0]).is_file())
            self.assertIn("CoACD", first.tool)
            # the hulls of a union of boxes sit on the boxes: a small gap
            self.assertLess(first.gap["hulls_to_proxy_p95_m"], 0.06)
            self.assertLess(first.gap["proxy_to_hulls_p95_m"], 0.06)
            again = proxy.ensure_parts(scene)
            self.assertEqual(again, first, "a second call reads the record")
            with self.assertRaises(FileExistsError):
                proxy.decompose(scene / "proxy.obj", scene / proxy.PARTS_DIR)


TINY_ROBOT = """<mujoco>
  <option timestep="0.005"/>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.01"/>
    <body name="base">
      <freejoint/>
      <geom size="0.1"/>
      <body name="leg">
        <joint name="j" axis="0 1 0"/>
        <geom name="foot" size="0.05" pos="0 0 -0.2" group="3"/>
      </body>
    </body>
  </worldbody>
  <actuator><position joint="j" name="j"/></actuator>
  <keyframe><key name="home" qpos="0 0 0.3 1 0 0 0 0"/></keyframe>
</mujoco>
"""


def _staged_scene(tmp: Path, *, n: int = 400) -> Path:
    """The fake Neverwhere scene imported, with a two-waypoint course
    laid out along +x at the proxy plane's height (z=2)."""
    src = _neverwhere_folder(tmp, n=n)
    (src / "hurdle_fake_v1.xml").write_text(
        '<mujoco><worldbody><geom type="sdf" name="collision_mesh_geom" '
        'mesh="collision_mesh" friction="1.25 0.3 0.3"/>'
        '<body name="waypoint-1" mocap="true" pos="1 0 2.3"/>'
        '<body name="waypoint-0" mocap="true" pos="0 0 2.3"/>'
        "</worldbody></mujoco>"
    )
    neverwhere.import_scene(src, tmp / "fake", name="fake")
    return tmp / "fake"


@needs_scene
@needs_sim
class TheStage(unittest.TestCase):
    def test_the_course_gives_the_start_and_the_terrain_replaces_the_floor(
        self,
    ) -> None:
        import mujoco  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            scene = _staged_scene(Path(tmp))
            record = load_scene_record(scene / SCENE_FILE)
            self.assertEqual(record.course["waypoints"], [[0, 0, 2.3], [1, 0, 2.3]])
            s = stage.compose(
                TINY_ROBOT, {}, scene, scene_stamp="fake@1", terrain=terrain.HULLS
            )
            self.assertTrue(s.floor_removed)
            self.assertEqual(s.heading_deg, 0.0)
            # the hulls of a flat proxy carry CoACD's thickness: within the
            # decomposition's own recorded gap of the plane at z=2
            parts = proxy.load_decomposition(scene)
            assert parts is not None
            self.assertLess(
                abs(s.surface_z - 2.0), parts.gap["hulls_to_proxy_p95_m"] + 0.01
            )
            # one metre before the first waypoint, standing 0.3 m over the surface
            self.assertAlmostEqual(s.start[0], -1.0)
            self.assertAlmostEqual(s.start[2], s.surface_z + 0.3, places=6)
            model = mujoco.MjModel.from_xml_string(s.xml)  # self-contained: no assets
            self.assertEqual(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor"), -1
            )
            self.assertEqual(model.ncam, 2)
            self.assertGreaterEqual(model.nmesh, s.terrain.geoms)
            self.assertEqual(s.terrain.kind, terrain.HULLS)
            np.testing.assert_allclose(
                model.key_qpos[0][:3], s.start, atol=1e-5
            )  # the XML rounds
            geom = model.geom(terrain.PART_MESH.format(index=0))
            np.testing.assert_allclose(geom.friction, [1.25, 0.3, 0.3])
            self.assertEqual(int(geom.group), neverwhere.COLLISION_GROUP)

    def test_a_perturbation_moves_the_terrain_not_the_robot(self) -> None:
        import mujoco  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            scene = _staged_scene(Path(tmp))
            lifted = stage.compose(
                TINY_ROBOT,
                {},
                scene,
                scene_stamp="fake@1",
                perturbation=stage.Perturbation("z+20mm", (0, 0, 0.02)),
            )
            nominal = lifted.surface_z  # the nominal height, recorded
            self.assertAlmostEqual(
                lifted.start[2], nominal + 0.3, places=6
            )  # the robot stays
            model = mujoco.MjModel.from_xml_string(lifted.xml)
            self.assertAlmostEqual(
                stage._surface_z(model, -1.0, 0.0), nominal + 0.02, places=4
            )
            turned = stage.compose(
                TINY_ROBOT,
                {},
                scene,
                scene_stamp="fake@1",
                perturbation=stage.Perturbation("yaw+5deg", yaw_deg=5.0),
            )
            model = mujoco.MjModel.from_xml_string(turned.xml)
            # a yaw about the start leaves the ground under the start where it was
            self.assertAlmostEqual(
                stage._surface_z(model, -1.0, 0.0), nominal, places=4
            )
            self.assertEqual(turned.facts()["perturbation"]["yaw_deg"], 5.0)

    def test_a_scene_without_a_course_refuses_a_guessed_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = _neverwhere_folder(Path(tmp))
            neverwhere.import_scene(src, Path(tmp) / "scene", name="fake")
            with self.assertRaisesRegex(ValueError, "lays out no course"):
                stage.compose(TINY_ROBOT, {}, Path(tmp) / "scene", scene_stamp="fake@1")
            s = stage.compose(
                TINY_ROBOT,
                {},
                Path(tmp) / "scene",
                scene_stamp="fake@1",
                start_xy=(0.5, 0.5),
                heading_deg=90.0,
            )
            self.assertAlmostEqual(s.start[1], 0.5)

    def test_a_staged_deployment_is_a_deployment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scene = _staged_scene(Path(tmp))
            (Path(tmp) / "tiny").mkdir()
            deployment = _manifest(Path(tmp) / "tiny")
            (deployment / "scene.xml").write_text(TINY_ROBOT)
            (Path(tmp) / "assets").mkdir()
            staged = Path(tmp) / "tiny-on-fake"
            s = stage.stage_deployment(
                deployment, scene, staged, assets_dir=Path(tmp) / "assets"
            )
            m = load_manifest(staged)  # loads: the schema and keys are the same
            self.assertEqual(m.scene_path, staged / stage.STAGE_FILE)
            block = m.raw[Key.SCENE]
            self.assertTrue(block["terrain"].startswith("scene fake@"))
            self.assertEqual(block["terrain_kind"], terrain.HEIGHTFIELD)
            self.assertEqual(block["terrain_geoms"], s.terrain.geoms)
            self.assertIn("top_surface_p95_m", block["terrain_gap"])
            self.assertEqual(block["staged_from"]["deployment"], "tiny")
            # the course, verbatim, so a course gate reads the manifest alone
            self.assertEqual(block["course"]["waypoints"], [[0, 0, 2.3], [1, 0, 2.3]])
            self.assertTrue(m.raw[Key.STAMP_OF].endswith(" on fake"))
            self.assertTrue((staged / "policy.onnx").is_file())
            with self.assertRaises(FileExistsError):
                stage.stage_deployment(
                    deployment, scene, staged, assets_dir=Path(tmp) / "assets"
                )


def _slope_obj(path: Path) -> Path:
    """A 4 m square plane rising 0.1 m per metre along x, at z=2 at x=0."""
    xs = np.array([-2.0, 2.0, 2.0, -2.0])
    ys = np.array([-2.0, -2.0, 2.0, 2.0])
    vertices = np.column_stack([xs, ys, 2.0 + 0.1 * xs])
    return write_obj(path, vertices, np.array([[0, 1, 2], [0, 2, 3]]))


@needs_scene
@needs_sim
class TheHeightfield(unittest.TestCase):
    def test_the_grid_reads_the_top_surface_the_right_way_round(self) -> None:
        import mujoco  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            src = _neverwhere_folder(Path(tmp))
            _slope_obj(src / neverwhere.COLLISION_MESH)
            neverwhere.import_scene(src, Path(tmp) / "slope", name="slope")
            s = stage.compose(
                TINY_ROBOT,
                {},
                Path(tmp) / "slope",
                scene_stamp="slope@1",
                start_xy=(0.0, 0.0),
                heading_deg=0.0,
            )
            self.assertEqual(s.terrain.kind, terrain.HEIGHTFIELD)
            self.assertEqual(s.terrain.geoms, 1)
            self.assertLess(s.terrain.gap["top_surface_p95_m"], 0.005)
            self.assertEqual(s.terrain.gap["not_top_surface_fraction"], 0.0)
            self.assertAlmostEqual(s.surface_z, 2.0, places=2)
            model = mujoco.MjModel.from_xml_string(s.xml)  # data rides inline
            self.assertEqual(model.nhfield, 1)
            # x is the slope's axis: +1 m along x is 0.1 m higher, along y nothing
            self.assertAlmostEqual(stage._surface_z(model, 1.0, 0.0), 2.1, places=2)
            self.assertAlmostEqual(stage._surface_z(model, 0.0, 1.0), 2.0, places=2)
            self.assertAlmostEqual(stage._surface_z(model, -1.5, -1.5), 1.85, places=2)

    def test_a_grid_resamples_to_another_spacing_over_the_same_footprint(
        self,
    ) -> None:
        grid = terrain.Grid(
            heights=np.array([[0.0, 0.1, 0.2], [0.0, 0.1, 0.2]]),
            x0=1.0,
            y0=2.0,
            cell=0.5,
        )
        coarse = grid.resampled(1.0)
        self.assertEqual(coarse.heights.shape, (2, 2))
        np.testing.assert_allclose(coarse.heights, [[0.0, 0.2], [0.0, 0.2]])
        fine = grid.resampled(0.25)
        self.assertEqual(fine.heights.shape, (3, 5))
        np.testing.assert_allclose(fine.heights[0], [0.0, 0.05, 0.1, 0.15, 0.2])
        self.assertEqual((fine.x0, fine.y0, fine.cell), (1.0, 2.0, 0.25))

    def test_the_grid_is_saved_once_and_re_sampled_for_a_new_proxy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = _neverwhere_folder(Path(tmp))
            _slope_obj(src / neverwhere.COLLISION_MESH)
            neverwhere.import_scene(src, Path(tmp) / "slope", name="slope")
            scene = Path(tmp) / "slope"
            grid, _filled = terrain.ensure_grid(scene)
            self.assertTrue((scene / terrain.GRID_FILE).is_file())
            saved = terrain.read_grid(scene)
            assert saved is not None
            np.testing.assert_allclose(saved[0].heights, grid.heights, atol=1e-6)
            self.assertEqual(
                (saved[0].x0, saved[0].y0, saved[0].cell), (grid.x0, grid.y0, grid.cell)
            )
            self.assertEqual(saved[2], terrain.proxy_hash(scene))
            # a saved grid is read, not re-sampled
            with mock.patch.object(terrain, "sample_grid") as sampler:
                terrain.ensure_grid(scene)
                sampler.assert_not_called()
            # a new proxy is re-sampled
            (scene / terrain.PROXY_FILE).write_text(
                (scene / terrain.PROXY_FILE).read_text().replace("2.0", "3.0", 1)
            )
            with mock.patch.object(
                terrain, "sample_grid", return_value=(grid, 0.5)
            ) as sampler:
                terrain.ensure_grid(scene)
                sampler.assert_called_once()

    def test_an_unknown_terrain_is_refused_by_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "no terrain 'sand'"):
            terrain.terrain_builder("sand")
        self.assertEqual(terrain.terrain_names(), ("heightfield", "hulls"))


class _StandingRuntime:
    """A fake gate runtime: never moves, never falls, one contact under
    the base every tick - enough to drive the assay's plumbing."""

    instrument = "fake"
    command_limit = None

    def __init__(self, start: tuple[float, float, float]) -> None:
        self.command = np.zeros(3, np.float32)
        self.start = np.array(start)

    def reset(self) -> None:
        pass

    def observe(self) -> np.ndarray:
        return np.zeros(1, np.float32)

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros(1, np.float32)

    def apply(self, action: np.ndarray) -> None:
        pass

    def base_velocity_b(self) -> np.ndarray:
        return np.zeros(3)

    def fell_over(self) -> bool:
        return False

    def contact_points(self) -> np.ndarray | None:
        return self.start[None, :] - [0.0, 0.0, 0.3]

    def pose(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.start, np.array([1.0, 0, 0, 0]), np.zeros(1)


@needs_scene
@needs_sim
class TheAssayAndTheContactSites(unittest.TestCase):
    def test_the_assay_stages_every_perturbation_and_reads_the_cliff(self) -> None:
        from rq_pipeline.deploy.gate import CONTACTS_FILE  # noqa: PLC0415
        from rq_pipeline.scenes import assay  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            # dense enough that a 10 cm ball around a contact holds gaussians
            scene = _staged_scene(Path(tmp), n=20_000)
            (Path(tmp) / "tiny").mkdir()
            deployment = _manifest(Path(tmp) / "tiny")
            (deployment / "scene.xml").write_text(TINY_ROBOT)
            (Path(tmp) / "assets").mkdir()
            root = Path(tmp) / "deploy"
            root.mkdir()
            short = (
                stage.NOMINAL,
                stage.Perturbation("z-20mm", (0, 0, -0.02)),
            )
            record = assay.assay(
                deployment,
                scene,
                root,
                "tiny-on-fake",
                assets_dir=Path(tmp) / "assets",
                trials=2,
                seed=1,
                perturbations=short,
                open=lambda manifest, assets_dir=None: _StandingRuntime(
                    tuple(manifest.raw["scene"]["start"])
                ),
            )
            self.assertEqual(
                [r["deployment"] for r in record["perturbations"]],
                [
                    "tiny-on-fake",
                    "tiny-on-fake-z-20mm",
                ],
            )
            self.assertTrue((root / "tiny-on-fake-z-20mm" / stage.STAGE_FILE).is_file())
            self.assertFalse(record["cliff"]["measurable"])  # nothing walked
            self.assertIn("unmeasurable", record["cliff"]["note"])
            self.assertIsNotNone(assay.read_assay(root / "tiny-on-fake"))
            # the gate kept where the fake touched and measured the gap there
            gate = json.loads((root / "tiny-on-fake" / "gate.json").read_text())
            contacts = gate["contacts"]
            self.assertEqual(contacts["file"], CONTACTS_FILE.format(runtime="mujoco"))
            # one point per tick of the two course walks (the standing fake
            # runs out its budget every time)
            self.assertEqual(
                contacts["points"], sum(r["steps"] for r in gate["records"])
            )
            self.assertEqual(gate["protocol"]["course"]["waypoints"], 2)
            self.assertEqual(contacts["site_gap"]["radius_m"], 0.1)
            self.assertIsNotNone(contacts["site_gap"]["chamfer_m"])
            self.assertIn("contact sites", contacts["site_gap"]["method"])

    def test_the_mujoco_runtime_sees_its_contacts(self) -> None:
        import mujoco  # noqa: PLC0415

        from rq_pipeline.deploy.runtime import Runtime  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "tiny").mkdir()
            deployment = _manifest(Path(tmp) / "tiny")
            (deployment / "scene.xml").write_text(TINY_ROBOT)
            manifest = load_manifest(deployment)
            model = mujoco.MjModel.from_xml_string(TINY_ROBOT)
            data = mujoco.MjData(model)
            runtime = Runtime(manifest=manifest, model=model, data=data, session=None)
            data.qpos[2] = 0.15  # the foot sphere (r 0.05 at -0.2) into the floor
            mujoco.mj_forward(model, data)
            points = runtime.contact_points()
            assert points is not None
            self.assertEqual(points.shape[1], 3)
            self.assertGreaterEqual(points.shape[0], 1)
            self.assertLess(abs(points[0][2]), 0.06)  # at the floor

    def test_the_cameras_refuse_by_name_without_the_renderer(self) -> None:
        from rq_pipeline.scenes import cameras  # noqa: PLC0415

        found = cameras._version()
        if found is not None and found >= cameras.RENDERER_SINCE:
            self.skipTest("this instrument has the renderer")
        opened, why = cameras.open_cameras(None, _splats(), names=("head",))
        self.assertIsNone(opened)
        self.assertIn("3.13", why)
        pixel = np.array([0x0A0B0C], np.uint32)
        self.assertEqual(cameras.unpack(pixel).tolist(), [[10, 11, 12]])
