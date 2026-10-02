"""Capture to scene (docs/78 §3): the chain's plumbing under a fake
runner that writes what each tool would, the alignment on a synthetic
tilted floor, the refusals by name."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tests._extras import needs_scene
from trainnr.scenes import capture, tooling
from trainnr.scenes.record import SCENE_FILE, load_scene_record
from trainnr.scenes.splat import Splats, euler_xyz_matrix, write_ply
from trainnr.scenes.splatters import SPLATTERS


def _tilted_room(n: int = 6000, seed: int = 0) -> tuple[Splats, np.ndarray, np.ndarray]:
    """A floor of gaussians with a box on it, then tilted and shifted:
    returns the splat, the rotation applied and the shift."""
    rng = np.random.default_rng(seed)
    floor = np.column_stack([rng.uniform(-2, 2, n), rng.uniform(-2, 2, n), np.zeros(n)])
    box = np.column_stack(
        [
            rng.uniform(0, 0.6, n // 6),
            rng.uniform(0, 0.6, n // 6),
            rng.uniform(0, 0.4, n // 6),
        ]
    )
    means = np.vstack([floor, box])
    rotation = euler_xyz_matrix(np.array([0.3, -0.2, 0.7]))
    shift = np.array([1.0, -2.0, 0.5])
    moved = means @ rotation.T + shift
    m = moved.shape[0]
    splats = Splats(
        means=moved.astype(np.float32),
        quats=np.tile(np.array([1, 0, 0, 0], np.float32), (m, 1)),
        scales=np.full((m, 3), 0.01, np.float32),
        opacities=np.full(m, 0.9, np.float32),
        colors=np.full((m, 3), 0.5, np.float32),
    )
    return splats, rotation, shift


class TheTopSurface(unittest.TestCase):
    def test_a_step_keeps_its_height_and_a_hole_is_filled_low(self) -> None:
        rng = np.random.default_rng(0)
        n = 4000
        step_x, step_z, hole_lo, hole_hi = 0.5, 0.3, 0.2, 0.24
        left_of, right_of = 0.45, 0.55  # a cell clear of the step's edge
        pts = np.column_stack([rng.uniform(0, 1, n), rng.uniform(0, 1, n), np.zeros(n)])
        pts[pts[:, 0] > step_x, 2] = step_z  # a step on the right half
        hole = (
            (pts[:, 0] > hole_lo)
            & (pts[:, 0] < hole_hi)
            & (pts[:, 1] > hole_lo)
            & (pts[:, 1] < hole_hi)
        )
        v, f, facts = capture.top_surface_mesh(pts[~hole], cell=0.02)
        self.assertGreater(facts["cells_filled"], 0)
        self.assertGreater(len(f), 1000)
        left = v[v[:, 0] < left_of][:, 2]
        right = v[v[:, 0] > right_of][:, 2]
        self.assertLess(left.max(), 0.01)
        self.assertGreater(right.min(), step_z - 0.01)

    def test_floaters_above_and_below_a_lawn_are_struck(self) -> None:
        """A real capture's lawn: a few percent of centres well above the
        ground (grass tips, specks) and a few below it; the surface must
        be the lawn, not the specks (the Go2 hung on them, 2026-09-23)."""
        rng = np.random.default_rng(1)
        n = 20000
        pts = np.column_stack(
            [rng.uniform(0, 1, n), rng.uniform(0, 1, n), rng.normal(0, 0.005, n)]
        )
        floaters = rng.choice(n, 400, replace=False)
        pts[floaters[:200], 2] += 0.3
        pts[floaters[200:], 2] -= 0.3
        v, _, facts = capture.top_surface_mesh(pts, cell=0.02)
        # the percentile drops most floaters before the median sees them
        self.assertGreater(facts["cells_despiked"], 0)
        self.assertLess(np.abs(v[:, 2]).max(), 0.03)


class TheFrameMath(unittest.TestCase):
    def test_rotation_to_z_and_eulers_round_trip(self) -> None:
        normal = np.array([0.3, -0.4, 0.8])
        normal /= np.linalg.norm(normal)
        r = capture.rotation_to_z(normal)
        np.testing.assert_allclose(r @ normal, [0, 0, 1], atol=1e-9)
        np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-9)
        eul = capture.euler_xyz_of(r)
        np.testing.assert_allclose(euler_xyz_matrix(np.array(eul)), r, atol=1e-9)
        np.testing.assert_allclose(
            capture.rotation_to_z(np.array([0, 0, 1.0])), np.eye(3)
        )

    def test_a_colmap_text_model_is_counted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sparse = Path(tmp)
            (sparse / "cameras.txt").write_text(
                "# cams\n1 OPENCV 640 480 500 500 320 240 0 0 0 0\n"
            )
            (sparse / "images.txt").write_text(
                "# images\n1 1 0 0 0 0 0 0 1 a.png\n1 2 -1\n2 1 0 0 0 0 0 0 1 b.png\n\n"
            )
            (sparse / "points3D.txt").write_text(
                "# pts\n1 0 0 0 0 0 0 0 1 0\n2 1 1 1 0 0 0 0 1 0\n"
            )
            poses = capture.read_sparse(sparse)
            self.assertEqual(poses.images_registered, 2)
            self.assertEqual(poses.points, 2)
            self.assertEqual(poses.camera_model, "OPENCV")


class TheTools(unittest.TestCase):
    def test_missing_tools_are_named(self) -> None:
        with (
            mock.patch("shutil.which", return_value=None),
            self.assertRaises(capture.MissingToolError) as caught,
        ):
            capture.Tools.find(video=True, splatter="brush")
        why = str(caught.exception)
        for name in ("ffmpeg", "ffprobe", "colmap", "brush_app"):
            self.assertIn(name, why)
        with (
            mock.patch("shutil.which", return_value=None),
            self.assertRaises(capture.MissingToolError) as caught,
        ):
            capture.Tools.find(video=False, splatter="brush")
        self.assertNotIn("ffmpeg", str(caught.exception))

    def test_the_refusal_names_this_machines_install_line(self) -> None:
        # never another machine's package manager (docs/78 §8.6)
        self.assertEqual(
            tooling.install_hint("colmap", "Darwin"), "brew install colmap"
        )
        self.assertEqual(
            tooling.install_hint("colmap", "Linux"), "sudo apt install colmap"
        )
        self.assertIn("tools/install-brush.py", tooling.install_hint("brush", "Linux"))
        self.assertIn(
            "tools/install-brush.py", tooling.install_hint("brush", "Windows")
        )
        self.assertEqual(
            tooling.install_hint("ffprobe", "Linux"), "sudo apt install ffmpeg"
        )
        with (
            mock.patch("shutil.which", return_value=None),
            mock.patch("platform.system", return_value="Linux"),
            self.assertRaises(capture.MissingToolError) as caught,
        ):
            capture.Tools.find(video=True, splatter="brush")
        self.assertIn("colmap (sudo apt install colmap)", str(caught.exception))
        self.assertNotIn("brew", str(caught.exception))


@needs_scene
class TheAlignment(unittest.TestCase):
    def test_the_floor_lands_at_z0_with_up_up(self) -> None:
        splats, _rotation, _shift = _tilted_room()
        aligned, record, floor = capture.align(splats, scale=None)
        z = aligned.means[:, 2]
        self.assertGreater(floor.inliers, 5000)
        self.assertLess(abs(np.median(z)), 0.01)  # the floor at zero
        self.assertGreater(z.max(), 0.3)  # the box above it, not below
        self.assertLess(z.min(), 0.01)
        self.assertIn("unrecorded", record.source)
        self.assertEqual(record.scale, 1.0)
        scaled, record, _ = capture.align(splats, scale=2.0)
        self.assertAlmostEqual(
            float(scaled.means[:, 2].max()), 2 * float(z.max()), places=2
        )
        self.assertIn("declared", record.source)

    def test_the_proxy_comes_from_the_splat(self) -> None:
        splats, _, _ = _tilted_room()
        aligned, _, _ = capture.align(splats, scale=None)
        with tempfile.TemporaryDirectory() as tmp:
            facts, _ = capture.build_proxy(aligned, Path(tmp) / "proxy.obj")
            self.assertGreater(facts["faces"], 100)
            self.assertIn("top surface", facts["method"])
            self.assertGreater(facts["cells_seen"], 1000)
            self.assertTrue((Path(tmp) / "proxy.obj").is_file())


class _FakeChain:
    """Writes what each tool would: a COLMAP text model, Brush's PLY."""

    def __init__(self, splats: Splats) -> None:
        self.calls: list[list[str]] = []
        self.splats = splats

    def __call__(self, argv, cwd: Path, log: Path) -> None:
        words = [str(a) for a in argv]
        self.calls.append(words)
        tool = Path(words[0]).name
        if tool == "colmap" and words[1] == "mapper":
            # the walk split in two: a small model 0 and the one to use, 1
            out = Path(words[words.index("--output_path") + 1])
            for model in ("0", "1"):
                (out / model).mkdir(parents=True)
                (out / model / "cameras.bin").write_bytes(b"")
        elif tool == "colmap" and words[1] == "model_converter":
            sparse = Path(words[words.index("--output_path") + 1])
            frames = (1, 2) if sparse.name == "0" else (1, 2, 3, 4)
            (sparse / "cameras.txt").write_text("1 OPENCV 8 8 5 5 4 4 0 0 0 0\n")
            (sparse / "images.txt").write_text(
                "".join(
                    f"{i} 1 0 0 0 0 0 0 1 frame_{i:04d}.png\n1 2 -1\n" for i in frames
                )
            )
            (sparse / "points3D.txt").write_text("1 0 0 0 0 0 0 0 1 0\n")
        elif tool == "brush_app":
            out = Path(words[words.index("--export-path") + 1])
            write_ply(self.splats, out / words[words.index("--export-name") + 1])
        log.write_text(
            log.read_text() + " ".join(words) + "\n" if log.is_file() else ""
        )


@needs_scene
class TheChain(unittest.TestCase):
    def test_frames_become_a_scene_by_fake_tools(self) -> None:
        splats, _, _ = _tilted_room()
        with tempfile.TemporaryDirectory() as tmp:
            frames = Path(tmp) / "shots"
            frames.mkdir()
            for i in range(4):
                (frames / f"img{i}.png").write_bytes(b"\x89PNG")
            tools = capture.Tools(
                ffmpeg=None,
                ffprobe=None,
                colmap=Path("/fake/colmap"),
                splatter=SPLATTERS["brush"],
                trainer=Path("/fake/brush_app"),
            )
            fake = _FakeChain(splats)
            with mock.patch.object(capture, "tool_version", return_value="9.9"):
                path = capture.capture_scene(
                    frames,
                    Path(tmp) / "room",
                    name="room",
                    tools=tools,
                    run=fake,
                    steps=10,
                    floor_friction=[1.0, 0.005, 0.0001],
                    narrate=False,
                )
            record = load_scene_record(path)
            self.assertEqual(path.name, SCENE_FILE)
            self.assertEqual(
                [c[1] for c in fake.calls if Path(c[0]).name == "colmap"],
                ["feature_extractor", "exhaustive_matcher", "mapper"]
                + ["model_converter"] * 2,
            )
            brush = next(c for c in fake.calls if Path(c[0]).name == "brush_app")
            self.assertIn("--total-steps", brush)
            self.assertNotIn("--rerun-enabled", brush)
            self.assertEqual(record.capture.frames, 4)
            self.assertEqual(record.splat["poses"]["registered"], 4)
            self.assertEqual(record.splat["poses"]["models"], [2, 4])
            self.assertEqual(record.splat["poses"]["chosen"], "1")
            brush_dataset = Path(brush[brush.index(str(tools.trainer)) + 1])
            self.assertEqual((brush_dataset / "sparse" / "0").resolve().name, "1")
            self.assertEqual(
                [t.name for t in record.tools],
                ["COLMAP", "Brush", "Open3D", "trainnr.scenes"],
            )
            self.assertEqual(record.declared, ("floor_friction",))
            self.assertIn("top surface", record.proxy["method"])
            self.assertIsNotNone(record.gap.chamfer_m)
            self.assertTrue(
                (Path(tmp) / "room" / "colmap" / "dataset" / "images").exists()
            )
            self.assertEqual(record.source, "capture/shots")
            self.assertIn("captured from", record.notes[0])
            self.assertIn("visible surface itself", record.notes[1])
            # resumable: a second call over the same folder is refused as a scene
            with self.assertRaisesRegex(ValueError, "never overwritten"):
                capture.capture_scene(
                    frames,
                    Path(tmp) / "room",
                    name="room",
                    tools=tools,
                    run=fake,
                    narrate=False,
                )

    def test_too_few_frames_refuse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            frames = Path(tmp) / "shots"
            frames.mkdir()
            (frames / "a.png").write_bytes(b"\x89PNG")
            tools = capture.Tools(
                ffmpeg=None,
                ffprobe=None,
                colmap=Path("/c"),
                splatter=SPLATTERS["brush"],
                trainer=Path("/b"),
            )
            with self.assertRaisesRegex(ValueError, "at least 3"):
                capture.capture_scene(
                    frames,
                    Path(tmp) / "x",
                    name="x",
                    tools=tools,
                    run=_FakeChain(_tilted_room()[0]),
                    narrate=False,
                )


class TheDoor(unittest.TestCase):
    def test_the_door_refuses_a_missing_tool_by_name(self) -> None:
        from trainnr.mcp_server import capture_scene  # noqa: PLC0415
        from trainnr.project import PROJECT_ENV, create_project  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            create_project(Path(tmp) / "p", name="p")
            os.environ[PROJECT_ENV] = str(Path(tmp) / "p")
            try:
                with mock.patch("shutil.which", return_value=None):
                    out = capture_scene(tmp, "room")
                self.assertEqual(out["status"], "refused")
                self.assertIn("colmap", out["reason"])
                out = capture_scene(str(Path(tmp) / "nothing.mp4"), "room")
                self.assertEqual(out["status"], "refused")
                self.assertIn("no video", out["reason"])
            finally:
                os.environ.pop(PROJECT_ENV, None)
