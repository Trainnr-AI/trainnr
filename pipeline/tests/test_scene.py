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

import numpy as np

from rq_pipeline.project import PROJECT_ENV, create_project, index_project
from rq_pipeline.project.kinds import Kind
from rq_pipeline.scenes import neverwhere
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


def _neverwhere_folder(tmp: Path, *, friction: str = "1.25 0.3 0.3") -> Path:
    """A fake Neverwhere scene: a splat plane at z=0.5 in a frame the
    transform scales by 2 and lifts by 1, over a proxy plane at z=2."""
    src = tmp / "hurdle_fake_v1"
    (src / "3dgs").mkdir(parents=True)
    (src / "geometry").mkdir()
    rng = np.random.default_rng(1)
    n = 400
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
