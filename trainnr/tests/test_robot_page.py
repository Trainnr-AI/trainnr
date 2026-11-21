"""A robot as onboarding takes it and as its page shows it (#120, #122,
#123): only the files the model reaches enter the bundle; the page and
`describe_robot` show keyframes, sites and hinge ranges in degrees; the
preview stands in `home` or `stand`."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests._extras import needs_mcp, needs_sim

# A one-triangle mesh is enough for MuJoCo to compile a mesh geom.
OBJ = (
    "v 0 0 0\nv 0.05 0 0\nv 0 0.05 0\nv 0 0 0.05\nf 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n"
)
SCENE = """<mujoco model="scene">
  <include file="robot.xml"/>
  <worldbody><geom type="plane" size="1 1 0.1"/></worldbody>
</mujoco>"""
ROBOT = """<mujoco model="robot">
  <compiler meshdir="assets" angle="radian"/>
  <asset><mesh name="leg" file="leg.obj"/></asset>
  <worldbody>
    <body name="trunk" pos="0 0 0.3">
      <freejoint/>
      <geom type="box" size="0.05 0.05 0.05"/>
      <site name="imu" pos="0.01 0 0.02"/>
      <body name="thigh">
        <joint name="knee" type="hinge" range="-0.2094 0.3229"/>
        <geom type="mesh" mesh="leg"/>
        <site name="foot" pos="0 0 -0.1"/>
      </body>
    </body>
  </worldbody>
  <keyframe>
    <key name="zero" qpos="0 0 0.3 1 0 0 0 0"/>
    <key name="stand" qpos="0 0 0.3 1 0 0 0 0.2"/>
    <key name="crouch" qpos="0 0 0.3 1 0 0 0 0.5"/>
  </keyframe>
</mujoco>"""


def _robot_folder(root: Path) -> Path:
    """The issue's folder: the model, its include and meshes, and what is
    not the robot (earlier variants, test output, an unused mesh folder)."""
    folder = root / "src"
    (folder / "assets").mkdir(parents=True)
    (folder / "assets_v1").mkdir()
    (folder / "scene.xml").write_text(SCENE)
    (folder / "robot.xml").write_text(ROBOT)
    (folder / "assets" / "leg.obj").write_text(OBJ)
    (folder / "assets_v1" / "leg_old.obj").write_text(OBJ)
    (folder / "robot_v1.xml").write_text(ROBOT)
    (folder / "analysis.json").write_text("{}")
    (folder / "LICENSE").write_text("MIT")
    (folder / "limits.json").write_text("{}")  # trainnr's own: kept (docs/88)
    return folder


@needs_sim
class TheBundle(unittest.TestCase):
    def test_only_what_the_model_reaches_is_copied(self) -> None:
        from trainnr.robot.onboarding import onboard_mjcf  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            source = _robot_folder(Path(tmp))
            out = onboard_mjcf(source / "scene.xml", "duck", Path(tmp) / "duck", {})
            self.assertEqual(
                out["files"]["copied"],
                ["LICENSE", "assets/leg.obj", "limits.json", "robot.xml", "scene.xml"],
            )
            self.assertEqual(out["files"]["left_beside"], 3)
            bundle = Path(tmp) / "duck"
            self.assertFalse((bundle / "assets_v1").exists())
            self.assertFalse((bundle / "analysis.json").exists())
            # an unrelated file beside the model never moves the stamp
            (source / "analysis.json").write_text('{"changed": true}')
            again = onboard_mjcf(source / "scene.xml", "duck", Path(tmp) / "again", {})
            self.assertEqual(again["stamp"], out["stamp"])

    def test_a_file_outside_the_folder_is_refused_by_name(self) -> None:
        from trainnr.robot.onboarding import onboard_mjcf  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            source = _robot_folder(Path(tmp))
            (Path(tmp) / "shared.obj").write_text(OBJ)
            robot = (source / "robot.xml").read_text()
            (source / "robot.xml").write_text(
                robot.replace('file="leg.obj"', 'file="../../shared.obj"')
            )
            with self.assertRaisesRegex(ValueError, "outside its folder"):
                onboard_mjcf(source / "scene.xml", "duck", Path(tmp) / "duck", {})


@needs_sim
class TheRevisions(unittest.TestCase):
    """#119: the robot again under its own name is a revision, the one it
    replaces kept and out of the lists; what changed is said in model terms."""

    def test_the_same_name_again_adds_a_revision_and_keeps_the_last(self) -> None:
        from trainnr.project import create_project, index_project  # noqa: PLC0415
        from trainnr.project.details import robot_sections  # noqa: PLC0415
        from trainnr.robot.onboarding import onboard  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            project = create_project(Path(tmp) / "p", "p")
            source = _robot_folder(Path(tmp))
            first = onboard(
                source / "scene.xml", "duck", project.robots / "duck", revise=True
            )
            same = onboard(
                source / "scene.xml", "duck", project.robots / "duck", revise=True
            )
            self.assertEqual(same["stamp"], first["stamp"])
            self.assertIn("unchanged", same["revision"])
            robot = (source / "robot.xml").read_text()
            (source / "robot.xml").write_text(
                robot.replace('range="-0.2094 0.3229"', 'range="-0.2094 0.3500"')
            )
            second = onboard(
                source / "scene.xml", "duck", project.robots / "duck", revise=True
            )
            self.assertNotEqual(second["stamp"], first["stamp"])
            self.assertEqual(second["replaces"], first["stamp"])
            [change] = [c for c in second["changes"] if c.startswith("joint knee")]
            self.assertIn("range (rad)", change)
            self.assertIn("18.5°", change)
            self.assertIn("20.1°", change)
            # a run built on the first revision says so
            run = project.root / "runs" / "walked"
            run.mkdir(parents=True)
            (run / "identity.json").write_text(
                json.dumps({"robot": first["stamp"], "task": "x"})
            )
            index = index_project(project)
            robots = [a for a in index.artifacts if a.kind == "robot"]
            self.assertEqual([a.stamp for a in robots], [second["stamp"]])
            self.assertEqual(robots[0].summary["revisions"], 2)
            [walked] = [a for a in index.artifacts if a.kind == "run"]
            self.assertIn(second["stamp"], walked.summary["superseded robot"])
            page = {
                s["title"]: s
                for s in robot_sections(project.robots / "duck", second["stamp"])
            }
            rows = dict(page["Revisions"]["rows"])
            self.assertEqual(rows["replaces"], first["stamp"])
            self.assertIn("knee", rows["change 1"])


@needs_sim
class ThePage(unittest.TestCase):
    def test_keyframes_sites_and_degrees(self) -> None:
        import mujoco  # noqa: PLC0415

        from trainnr.project.details import robot_sections  # noqa: PLC0415
        from trainnr.project.previews import preview_keyframe  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            source = _robot_folder(Path(tmp))
            model = mujoco.MjModel.from_xml_path(str(source / "scene.xml"))
            self.assertEqual(preview_keyframe(model), 1)  # `stand`, not `zero`
            sections = {s["title"]: s for s in robot_sections(source, "duck@x")}
        joints = sections["Joints"]["rows"]
        knee = next(r for r in joints if r[0] == "knee")
        self.assertIn("(-12.0°, 18.5°)", knee[2])
        keyframes = sections["Keyframes"]
        self.assertEqual(keyframes["columns"], ["joint", "zero", "stand", "crouch"])
        [row] = keyframes["rows"]
        self.assertEqual(row[:3], ["knee", "0.0°", "11.5°"])
        self.assertIn("outside the range", row[3])
        self.assertIn("'stand'", keyframes["note"])
        sites = {r[0]: r[1] for r in sections["Sites"]["rows"]}
        self.assertEqual(sites, {"imu": "trunk", "foot": "thigh"})


@needs_sim
class TheViewer(unittest.TestCase):
    def test_every_keyframe_is_a_step_ending_on_the_preview_pose(self) -> None:
        try:
            import rerun as rr  # noqa: PLC0415
        except ImportError:  # pragma: no cover - the viz extra is absent
            self.skipTest("rerun not installed")
        from trainnr.project import create_project, index_project  # noqa: PLC0415
        from trainnr.project.present import _present_robot  # noqa: PLC0415
        from trainnr.robot.onboarding import onboard_mjcf  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            project = create_project(Path(tmp) / "p", "p")
            source = _robot_folder(Path(tmp))
            onboard_mjcf(source / "scene.xml", "duck", project.robots / "duck", {})
            [robot] = [a for a in index_project(project).artifacts if a.kind == "robot"]
            stream = rr.RecordingStream("trainnr-test")
            stream.memory_recording()
            steps: list[int] = []
            names: list[str] = []
            real_log = stream.log

            def log(entity: str, value: object, **kw: object) -> None:
                if entity.endswith("/keyframe"):
                    names.append(str(value.text.as_arrow_array()[0]))  # type: ignore[attr-defined]
                real_log(entity, value, **kw)

            with (
                mock.patch.object(
                    stream, "set_time", lambda _t, sequence: steps.append(sequence)
                ),
                mock.patch.object(stream, "log", log),
            ):
                shown = _present_robot(project, robot, stream)
        self.assertEqual(shown["view"], "3D")
        self.assertEqual(steps, [0, 1, 2])
        self.assertEqual(names, ["zero", "crouch", "stand"])  # the preview pose last


@needs_mcp
@needs_sim
class TheTool(unittest.TestCase):
    def test_describe_robot_returns_the_page_and_says_no_fits(self) -> None:
        from trainnr import mcp_server  # noqa: PLC0415
        from trainnr.project import create_project  # noqa: PLC0415
        from trainnr.project.locate import PROJECT_ENV  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            project = create_project(Path(tmp) / "p", "p")
            source = _robot_folder(Path(tmp))
            with mock.patch.dict(os.environ, {PROJECT_ENV: str(project.root)}):
                onboarded = mcp_server.onboard_robot(str(source / "scene.xml"), "duck")
                self.assertEqual(onboarded["status"], "done")
                detail = mcp_server.describe_robot("duck")
        self.assertEqual(detail["Asset"]["keyframes"], 3)
        self.assertIn("knee", [j["joint"] for j in detail["Joints"]])
        self.assertEqual({s["site"] for s in detail["Sites"]}, {"imu", "foot"})
        self.assertIn("none yet", detail["fits"])


if __name__ == "__main__":
    unittest.main()
