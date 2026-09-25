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

        from rq_mjlab.envelope import COMMAND_TERM  # noqa: PLC0415
        from rq_mjlab.go2_walk import gentle_scene_commands  # noqa: PLC0415

        ranges = SimpleNamespace(
            lin_vel_x=(-2.0, 2.0), lin_vel_y=(-1, 1), ang_vel_z=(-1, 1)
        )
        cfg = SimpleNamespace(
            curriculum={}, commands={COMMAND_TERM: SimpleNamespace(ranges=ranges)}
        )
        gentle_scene_commands(cfg)
        self.assertEqual(ranges.lin_vel_x, (-2.0, 2.0))

    def test_a_training_config_takes_the_scene_schedule(self) -> None:
        from types import SimpleNamespace  # noqa: PLC0415

        from rq_mjlab.envelope import (  # noqa: PLC0415
            COMMAND_TERM,
            CURRICULUM_TERM,
            STAGES_KEY,
        )
        from rq_mjlab.go2_walk import (  # noqa: PLC0415
            gentle_scene_commands,
            go2_agent,
            scene_command_stages,
        )

        ranges = SimpleNamespace(
            lin_vel_x=(-1, 1), lin_vel_y=(-1, 1), ang_vel_z=(-1, 1)
        )
        term = SimpleNamespace(params={STAGES_KEY: "the plane's"})
        cfg = SimpleNamespace(
            curriculum={CURRICULUM_TERM: term},
            commands={COMMAND_TERM: SimpleNamespace(ranges=ranges)},
        )
        gentle_scene_commands(cfg)
        stages = scene_command_stages(go2_agent(1).num_steps_per_env)
        self.assertEqual(term.params[STAGES_KEY], stages)
        self.assertEqual(ranges.lin_vel_x, stages[0]["lin_vel_x"])
        self.assertEqual(ranges.lin_vel_y, stages[0]["lin_vel_y"])

    def test_the_smoke_agent_keys_the_stages_alike(self) -> None:
        """The schedule converts iterations with the Go2 recipe's steps
        per iteration; a smoke run must count the same, or its stages
        would open at other iterations."""
        from rq_mjlab.go2_walk import go2_agent  # noqa: PLC0415
        from rq_mjlab.walk_train import smoke_agent  # noqa: PLC0415

        self.assertEqual(
            smoke_agent(1).num_steps_per_env, go2_agent(1).num_steps_per_env
        )

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

        from rq_mjlab.walks import TERRAIN_SCAN_GROUP  # noqa: PLC0415

        self.assertNotIn(
            TERRAIN_SCAN_GROUP, VISUAL_GROUPS
        )  # the splat stays the picture
        self.assertNotEqual(TERRAIN_SCAN_GROUP, COLLISION_GROUP)  # not the robot's legs


class ThePlayShowsTheGround(unittest.TestCase):
    """Viewers draw groups 0-2 by default; a scene's ground (group 4) was
    invisible in the play window and the Studio (2026-09-25)."""

    def test_both_viewers_turn_the_ground_group_on(self) -> None:
        from types import SimpleNamespace  # noqa: PLC0415

        from rq_mjlab.walk_play import showing_the_ground  # noqa: PLC0415
        from rq_mjlab.walks import TERRAIN_SCAN_GROUP  # noqa: PLC0415

        synced = []

        class Browser:
            def setup(self) -> None:
                self._scene = SimpleNamespace(
                    geom_groups_visible=[True, True, True, False, False, False],
                    _sync_visibilities=lambda: synced.append(True),
                )

        class Window:
            def setup(self) -> None:
                self.viewer = SimpleNamespace(
                    opt=SimpleNamespace(geomgroup=[1, 1, 1, 0, 0, 0])
                )

        browser = showing_the_ground(Browser)()
        browser.setup()
        self.assertTrue(browser._scene.geom_groups_visible[TERRAIN_SCAN_GROUP])
        self.assertEqual(synced, [True])
        window = showing_the_ground(Window)()
        window.setup()
        self.assertEqual(window.viewer.opt.geomgroup[TERRAIN_SCAN_GROUP], 1)
        self.assertEqual(
            window.viewer.opt.geomgroup[3], 0
        )  # robot colliders stay hidden


if __name__ == "__main__":
    unittest.main()
