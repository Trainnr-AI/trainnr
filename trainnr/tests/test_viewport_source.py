"""A deployment in the Studio's own MuJoCo viewport
(`deploy.viewport_source`): the scene grammar and its refusals, the
twist the human drives with (decoded, clipped to the trained ranges),
the poses a gate and a pre-flight save (`deploy.poses`) and their replay
into a model, and the scene list the drawer offers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from tests._extras import needs_sim
from tests.test_course import _raw, _Walker
from trainnr.deploy.gate import POSES_KEY, gate
from trainnr.deploy.manifest import MANIFEST_FILE, Manifest
from trainnr.deploy.poses import PoseFile, PoseTrack, trial_segment
from trainnr.deploy.viewport_source import (
    ATTRIBUTION,
    DEPLOY_PREFIX,
    GATE,
    LIVE,
    MODES,
    PREFLIGHT,
    TwistInput,
    clip_twist,
    parse_scene,
    replay_forever,
    scene_text,
    scenes_of,
)

PLANE = {"file": "scene.xml"}  # a plane deployment: held twists, no course


def _manifest() -> Manifest:
    return Manifest(root=Path("/nowhere"), raw=_raw(scene=PLANE))


class TheSceneGrammar(unittest.TestCase):
    def test_each_mode_round_trips(self) -> None:
        for text in (
            "deploy:go2-c2",
            "deploy:go2-c2:gate:dds:3",
            "deploy:go2-c2:preflight:4",
            "deploy:go2-c2:attribution:armature:2",
            "deploy:go2-c2:attribution:fit:fit",
        ):
            self.assertEqual(parse_scene(text).text(), text)
        self.assertEqual(parse_scene("deploy:go2-c2").mode, LIVE)
        self.assertEqual(scene_text("a", GATE, "mujoco", 1), "deploy:a:gate:mujoco:1")
        self.assertEqual(scene_text("a", PREFLIGHT, 0), "deploy:a:preflight:0")
        self.assertEqual(set(MODES), {LIVE, GATE, PREFLIGHT, ATTRIBUTION})

    def test_a_malformed_scene_is_refused_by_name(self) -> None:
        for text, word in (
            ("go2-c2", DEPLOY_PREFIX),
            ("deploy:", "names its deployment"),
            ("deploy:a:fly", "no viewport mode 'fly'"),
            ("deploy:a:gate:dds", "takes 2"),
            ("deploy:a:preflight:1:2", "takes 1"),
        ):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, word):
                parse_scene(text)


class TheTwist(unittest.TestCase):
    def test_the_mailbox_decodes_world_zero_and_hands_back(self) -> None:
        twist = TwistInput("twist", 3)
        twist.on_command("twist", 0, 0.6)  # world 0, forward
        twist.on_command("twist", 2, -0.3)  # world 0, turn
        twist.on_command("twist", 4, 9.0)  # world 1: not this scene's robot
        twist.on_command("speed", 0, 5.0)  # another word: not a twist
        np.testing.assert_allclose(twist.value, [0.6, 0.0, -0.3], atol=1e-6)
        twist.on_command("twist", -1, 0.0)  # handed back
        np.testing.assert_array_equal(twist.value, [0.0, 0.0, 0.0])

    def test_the_command_stays_inside_the_trained_ranges(self) -> None:
        clipped = clip_twist(_manifest(), np.array([3.0, -2.0, 0.9]))
        np.testing.assert_allclose(clipped, [1.0, -1.0, 0.5])
        inside = clip_twist(_manifest(), np.array([0.3, 0.2, -0.1]))
        np.testing.assert_allclose(inside, [0.3, 0.2, -0.1], atol=1e-6)


class ThePoseFile(unittest.TestCase):
    def _track(self) -> PoseTrack:
        track = PoseTrack(0.02, ("a", "b"))
        for name, frames in (("first", 3), ("second", 2)):
            track.begin(name)
            for i in range(frames):
                track.add(
                    (np.array([i, 0, 0.3]), np.array([1, 0, 0, 0]), np.array([i, -i]))
                )
        return track

    def test_a_track_round_trips_by_segment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._track().save(Path(tmp) / ".viewer" / "x-poses.npz")
            poses = PoseFile.load(path)
        self.assertEqual(poses.segments, ("first", "second"))
        self.assertEqual(poses.joint_names, ("a", "b"))
        self.assertEqual(poses.dt, 0.02)
        position, quat, joints = poses.frames(1)
        self.assertEqual(len(position), 2)
        np.testing.assert_allclose(joints[1], [1, -1])
        np.testing.assert_allclose(quat[0], [1, 0, 0, 0])
        with self.assertRaisesRegex(IndexError, "no segment 2"):
            poses.frames(2)

    def test_what_cannot_be_replayed_is_refused_by_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "call begin"):
            PoseTrack(0.02, ("a",)).add((np.zeros(3), np.zeros(4), np.zeros(1)))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "nothing to replay"):
                PoseTrack(0.02, ("a",)).save(Path(tmp) / "p.npz")
            with self.assertRaisesRegex(FileNotFoundError, "run it again"):
                PoseFile.load(Path(tmp) / "absent-poses.npz")
            bad = Path(tmp) / "bad.npz"
            np.savez(bad, segments=np.array(["x"]))
            with self.assertRaisesRegex(ValueError, "not a pose file"):
                PoseFile.load(bad)


def _plane_deployment(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / MANIFEST_FILE).write_text(json.dumps(_raw(scene=PLANE)))
    (folder / "policy.onnx").write_bytes(b"\x00")
    (folder / "scene.xml").write_text("<mujoco/>")
    return folder


def _walker(manifest: Manifest, assets_dir: Any = None) -> _Walker:
    return _Walker(manifest.control.step_dt)


class TheGateKeepsItsPoses(unittest.TestCase):
    """Every gate saves each trial's poses, whatever drives the policy: a
    trial run in another process's simulator can then be replayed."""

    def test_each_trial_is_a_segment_of_the_saved_poses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = _plane_deployment(Path(tmp))
            record = gate(folder, assets_dir=None, trials=2, seed=1, open=_walker)
            poses = PoseFile.load(folder / record[POSES_KEY])
        self.assertEqual(record[POSES_KEY], ".viewer/gate-mujoco-poses.npz")
        self.assertEqual(poses.segments, (trial_segment(0), trial_segment(1)))
        per_trial = _manifest().control.episode_ticks
        self.assertEqual(len(poses.frames(0)[0]), per_trial)
        self.assertEqual(poses.joint_names, ("j",))

    def test_the_drawer_lists_what_it_can_show_and_nothing_it_would_refuse(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = _plane_deployment(Path(tmp) / "dep")
            gate(folder, assets_dir=None, trials=2, seed=1, open=_walker)
            gate(folder, assets_dir=None, runtime="dds", trials=2, seed=1, open=_walker)
            listed = [scene for _label, scene in scenes_of(folder)]
            self.assertEqual(listed[0], "deploy:dep")
            self.assertIn("deploy:dep:gate:mujoco:1", listed)  # re-run in ours
            self.assertIn("deploy:dep:gate:dds:1", listed)  # replayed: poses saved
            (folder / ".viewer" / "gate-dds-poses.npz").unlink()
            listed = [scene for _label, scene in scenes_of(folder)]
            self.assertNotIn("deploy:dep:gate:dds:0", listed)  # no poses: not offered
            self.assertIn("deploy:dep:gate:mujoco:0", listed)


# One robot with a floating base and one hinge, the manifest's "j".
TINY = """<mujoco>
  <worldbody>
    <body name="base" pos="0 0 0.3"><freejoint/><geom size="0.1"/>
      <body name="leg"><joint name="j" axis="0 1 0"/>
        <geom type="capsule" size="0.03" fromto="0 0 0 0 0 -0.2"/></body>
    </body>
  </worldbody>
  <actuator><position joint="j"/></actuator>
  <keyframe><key qpos="0 0 0.3 1 0 0 0 0"/></keyframe>
</mujoco>"""


class _Stop(Exception):  # noqa: N818 - the loop ends, not an error
    pass


class _Loop:
    """The physics pump, stood in for: records what each tick published."""

    def __init__(self, ticks: int) -> None:
        self.on_command = None
        self.left = ticks
        self.seen: list[tuple[np.ndarray, float]] = []

    def tick(self, data: Any, pace_seconds: float) -> None:
        self.seen.append((data.qpos.copy(), float(data.time)))
        self.left -= 1
        if not self.left:
            raise _Stop


@needs_sim
class TheReplay(unittest.TestCase):
    def test_a_segment_is_set_into_the_model_frame_by_frame(self) -> None:
        import mujoco  # noqa: PLC0415

        model = mujoco.MjModel.from_xml_string(TINY)
        track = PoseTrack(0.02, ("j",))
        track.begin("stop · zeroed")
        for i in range(4):
            track.add(
                (
                    np.array([0.1 * i, 0, 0.3]),
                    np.array([1, 0, 0, 0]),
                    np.array([0.2 * i]),
                )
            )
        with tempfile.TemporaryDirectory() as tmp:
            poses = PoseFile.load(track.save(Path(tmp) / "p-poses.npz"))
        loop = _Loop(ticks=4)
        with self.assertRaises(_Stop):
            replay_forever(model, loop, _manifest(), poses, 0)
        for i, (qpos, time) in enumerate(loop.seen):
            self.assertAlmostEqual(qpos[0], 0.1 * i, places=5)  # base x
            self.assertAlmostEqual(qpos[7], 0.2 * i, places=5)  # the hinge
            self.assertAlmostEqual(time, 0.02 * i)  # the recording's own clock


if __name__ == "__main__":
    unittest.main()


class TheDressing(unittest.TestCase):
    """The Studio's dressing is visual only: a scene exported for a
    runtime (no lights, a bare plane) gains a sky, a key light and the
    checker floor, and steps bit for bit like the bare scene (2026-09-24:
    the deploy viewport showed a bare plane under a black sky)."""

    BARE = """<mujoco><worldbody>
      <geom name="floor" type="plane" size="0 0 0.01"/>
      <body pos="0 0 0.5"><freejoint/>
        <geom type="box" size="0.1 0.2 0.05" mass="1"/></body>
    </worldbody></mujoco>"""

    def test_the_dressed_scene_steps_exactly_like_the_bare_one(self) -> None:
        import mujoco  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        from trainnr.tasks.scene import dress  # noqa: PLC0415

        bare = mujoco.MjModel.from_xml_string(self.BARE)
        spec = mujoco.MjSpec.from_string(self.BARE)
        dress(spec)
        dressed = spec.compile()
        self.assertEqual(bare.nlight, 0)
        self.assertEqual(dressed.nlight, 1)
        self.assertIn(mujoco.mjtTexture.mjTEXTURE_SKYBOX, list(dressed.tex_type))
        self.assertGreaterEqual(int(dressed.geom_matid[0]), 0)
        runs = []
        for model in (bare, dressed):
            data = mujoco.MjData(model)
            data.qvel[:3] = [0.3, -0.2, 0.0]
            for _ in range(500):
                mujoco.mj_step(model, data)
            runs.append(data.qpos.copy())
        np.testing.assert_array_equal(runs[0], runs[1])

    def test_a_scene_with_its_own_light_and_floor_material_keeps_them(self) -> None:
        import mujoco  # noqa: PLC0415

        from trainnr.tasks.scene import dress  # noqa: PLC0415

        xml = self.BARE.replace(
            "<worldbody>",
            '<asset><material name="m" rgba="1 0 0 1"/></asset>'
            '<worldbody><light pos="0 0 3"/>',
        ).replace('size="0 0 0.01"/>', 'size="0 0 0.01" material="m"/>')
        spec = mujoco.MjSpec.from_string(xml)
        dress(spec)
        model = spec.compile()
        self.assertEqual(model.nlight, 1)
        self.assertEqual(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, model.geom_matid[0]),
            "m",
        )
