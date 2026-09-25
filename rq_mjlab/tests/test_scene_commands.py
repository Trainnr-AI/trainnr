"""A captured scene's command schedule starts slow (rq_mjlab.go2_walk)."""

from __future__ import annotations

import unittest


class TheSceneCommands(unittest.TestCase):
    def test_slow_first_then_the_plane_first_stage(self) -> None:
        from rq_mjlab.go2_walk import scene_command_stages  # noqa: PLC0415

        stages = scene_command_stages(24)
        self.assertEqual([s["step"] for s in stages], [0, 12000, 24000])
        self.assertEqual(stages[0]["lin_vel_x"], (-0.5, 0.5))
        self.assertEqual(stages[0]["lin_vel_y"], (-0.3, 0.3))
        # the last stage is the plane's first: the same envelope a plane
        # walker is certified at, so the two are judged alike
        self.assertEqual(stages[-1]["lin_vel_x"], (-1.0, 1.0))
        self.assertEqual(stages[-1]["lin_vel_y"], (-1.0, 1.0))
        self.assertEqual(stages[-1]["ang_vel_z"], (-0.5, 0.5))

    def test_a_play_config_keeps_its_own_commands(self) -> None:
        """Play has no curriculum; the scene schedule left it alone (a
        KeyError on the first scene play, 2026-09-25)."""
        from types import SimpleNamespace  # noqa: PLC0415

        from rq_mjlab.go2_walk import gentle_scene_commands  # noqa: PLC0415

        ranges = SimpleNamespace(
            lin_vel_x=(-2.0, 2.0), lin_vel_y=(-1, 1), ang_vel_z=(-1, 1)
        )
        cfg = SimpleNamespace(
            curriculum={}, commands={"twist": SimpleNamespace(ranges=ranges)}
        )
        gentle_scene_commands(cfg)
        self.assertEqual(ranges.lin_vel_x, (-2.0, 2.0))

    def test_the_verdict_pins_the_stage_a_checkpoint_reached(self) -> None:
        from rq_mjlab.envelope import stage_reached  # noqa: PLC0415
        from rq_mjlab.go2_walk import scene_command_stages  # noqa: PLC0415

        stages = scene_command_stages(24)
        self.assertEqual(stage_reached(stages, (299 + 1) * 24), 0)
        self.assertEqual(stage_reached(stages, (2999 + 1) * 24), 2)


class TheScansSeeTheScene(unittest.TestCase):
    """The scene's ground sits in its own group, and the height scans look
    there: in the collision group they saw nothing (a blind walker, and a
    crash without a camera, 2026-09-25)."""

    def test_the_scans_look_in_the_ground_group(self) -> None:
        from types import SimpleNamespace  # noqa: PLC0415

        from rq_mjlab.scene_stage import (  # noqa: PLC0415
            TERRAIN_SCAN_GROUP,
            scans_see_the_scene,
        )

        sensors = (
            SimpleNamespace(name="terrain_scan", include_geom_groups=(0,)),
            SimpleNamespace(name="foot_height_scan", include_geom_groups=(0,)),
            SimpleNamespace(name="feet_ground_contact"),
        )
        scans_see_the_scene(sensors)
        self.assertEqual(sensors[0].include_geom_groups, (TERRAIN_SCAN_GROUP,))
        self.assertEqual(sensors[1].include_geom_groups, (TERRAIN_SCAN_GROUP,))
        with self.assertRaisesRegex(ValueError, "foot_height_scan"):
            scans_see_the_scene(sensors[:1])

    def test_the_ground_group_is_apart_from_pictures_and_robot(self) -> None:
        from rq_pipeline.scenes.cameras import VISUAL_GROUPS  # noqa: PLC0415
        from rq_pipeline.scenes.record import COLLISION_GROUP  # noqa: PLC0415

        from rq_mjlab.scene_stage import TERRAIN_SCAN_GROUP  # noqa: PLC0415

        self.assertNotIn(
            TERRAIN_SCAN_GROUP, VISUAL_GROUPS
        )  # the splat stays the picture
        self.assertNotEqual(TERRAIN_SCAN_GROUP, COLLISION_GROUP)  # not the robot's legs


if __name__ == "__main__":
    unittest.main()
