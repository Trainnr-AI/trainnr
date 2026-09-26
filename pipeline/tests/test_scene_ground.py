"""A captured scene under a mirrored robot (`viz.scene_ground`), and the
viewport finding the scene a staged deployment stands on."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tests._extras import NUMPY, SIM

TOOLS = Path(__file__).resolve().parents[2] / "tools"


class _Recorder:
    """A stand-in `rerun`: every log call kept, archetypes as plain records."""

    class ViewCoordinates:
        RIGHT_HAND_Z_UP = "RIGHT_HAND_Z_UP"

    def __init__(self) -> None:
        self.logged: list[tuple[str, object, bool]] = []

    def log(self, path, what, static=False):
        self.logged.append((path, what, static))

    @staticmethod
    def GaussianSplats3D(**fields):  # noqa: N802 - rerun's own name
        return ("splats", fields)

    @staticmethod
    def Mesh3D(**fields):  # noqa: N802 - rerun's own name
        return ("mesh", fields)


def _scene(folder: Path, *, proxy: bool = True) -> None:
    """Five gaussians, two too faint to draw, and a two-triangle proxy."""
    import numpy as np  # noqa: PLC0415

    from rq_pipeline.scenes.record import PROXY_FILE, SPLAT_FILE  # noqa: PLC0415
    from rq_pipeline.scenes.splat import Splats, write_ply  # noqa: PLC0415

    n = 5
    splats = Splats(
        means=np.arange(n * 3, dtype=np.float32).reshape(n, 3) * 0.1,
        quats=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),
        scales=np.full((n, 3), 0.02, np.float32),
        opacities=np.array([0.9, 0.1, 0.8, 0.2, 1.0], np.float32),
        colors=np.full((n, 3), 0.5, np.float32),
    )
    write_ply(splats, folder / SPLAT_FILE)
    if proxy:
        (folder / PROXY_FILE).write_text(
            "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3\nf 1 3 4\n"
        )


@unittest.skipUnless(NUMPY, "needs numpy")
class TheSceneGround(unittest.TestCase):
    def test_the_visible_splat_and_the_proxy_go_in_once_static(self) -> None:
        from rq_pipeline.viz import PROXY_RGBA, scene_ground  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            _scene(Path(tmp))
            rr = _Recorder()
            drawn = scene_ground(rr, tmp)
        self.assertEqual(drawn, {"gaussians": 3, "of": 5, "proxy_faces": 2})
        by_path = {path: (what, static) for path, what, static in rr.logged}
        self.assertEqual(by_path["world"], ("RIGHT_HAND_Z_UP", True))
        kind, fields = by_path["world/scene/splat"][0]
        self.assertEqual(kind, "splats")
        self.assertEqual(fields["centers"].shape, (3, 3))
        self.assertTrue(by_path["world/scene/splat"][1])  # static
        kind, fields = by_path["world/scene/proxy"][0]
        self.assertEqual(kind, "mesh")
        self.assertEqual(fields["albedo_factor"], list(PROXY_RGBA))
        self.assertTrue(by_path["world/scene/proxy"][1])

    def test_no_proxy_draws_the_splat_alone(self) -> None:
        from rq_pipeline.viz import scene_ground  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            _scene(Path(tmp), proxy=False)
            rr = _Recorder()
            drawn = scene_ground(rr, tmp)
        self.assertEqual(drawn["proxy_faces"], 0)
        self.assertNotIn("world/scene/proxy", [p for p, _, _ in rr.logged])

    def test_a_scene_without_a_splat_is_refused_by_name(self) -> None:
        from rq_pipeline.viz import scene_ground  # noqa: PLC0415

        with (
            tempfile.TemporaryDirectory() as tmp,
            self.assertRaisesRegex(FileNotFoundError, Path(tmp).name),
        ):
            scene_ground(_Recorder(), tmp)

    def test_splats_already_read_are_not_read_again(self) -> None:
        from rq_pipeline.scenes.record import SPLAT_FILE  # noqa: PLC0415
        from rq_pipeline.scenes.splat import read_ply  # noqa: PLC0415
        from rq_pipeline.viz import scene_ground  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            _scene(Path(tmp))
            splats = read_ply(Path(tmp) / SPLAT_FILE)
            (Path(tmp) / SPLAT_FILE).unlink()  # gone: only the given set can serve
            drawn = scene_ground(_Recorder(), tmp, splats=splats)
        self.assertEqual(drawn["gaussians"], 3)


def _tool():
    if str(TOOLS) not in sys.path:  # the tools import their `_lab` neighbour
        sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location(
        "studio_render_stream", TOOLS / "studio-render-stream.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(SIM, "needs mujoco (the tool imports it)")
class TheStagedScene(unittest.TestCase):
    """`staged_scene_dir`: the folder a deployment's manifest names, in
    the open project; None for a plane or a non-deployment scene."""

    def test_a_staged_deployment_names_its_scene_folder(self) -> None:
        from rq_pipeline.project.locate import SCENES_FOLDER  # noqa: PLC0415
        from rq_pipeline.scenes.stage import SCENE_TERRAIN_WORD  # noqa: PLC0415

        tool = _tool()
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / SCENES_FOLDER / "garden").mkdir(parents=True)
            tool.PROJECT_ROOT = Path(tmp)
            raw = {"scene": {"terrain": f"{SCENE_TERRAIN_WORD} garden@abc"}}
            opened = SimpleNamespace(
                context=SimpleNamespace(manifest=SimpleNamespace(raw=raw))
            )
            self.assertEqual(
                tool.staged_scene_dir(opened), Path(tmp) / SCENES_FOLDER / "garden"
            )
            # named but missing: said, and the twin goes on without it
            raw["scene"]["terrain"] = f"{SCENE_TERRAIN_WORD} lost@abc"
            self.assertIsNone(tool.staged_scene_dir(opened))
            # a plane deployment, and a scene that is no deployment at all
            raw["scene"] = {"terrain": "plane"}
            self.assertIsNone(tool.staged_scene_dir(opened))
            self.assertIsNone(tool.staged_scene_dir(object()))


if __name__ == "__main__":
    unittest.main()
