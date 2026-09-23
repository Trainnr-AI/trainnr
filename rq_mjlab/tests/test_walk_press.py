"""The rollout->dataset writer's pure parts (D2): the failure row, the
state names off an observation manager, the checkpoint stamp, the
export descriptor - everything a batch says about itself, pinned
without a GPU."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, ClassVar

from rq_pipeline.collect.demo_export import ExportSpec
from rq_pipeline.collect.kitting_export import DemoLayout

from rq_mjlab.walk_press import (
    ACTION_SEMANTICS,
    CAMERA_KEY,
    checkpoint_stamp,
    failure_row,
    state_names,
)
from rq_mjlab.walk_verdict import EpisodeOutcome, WorldEpisode


class _Manager:
    """The two observation-manager attributes state_names reads."""

    active_terms: ClassVar = {"actor": ["base_ang_vel", "joint_pos", "command"]}
    group_obs_term_dim: ClassVar = {"actor": [(3,), (14,), (3,)]}


class StateNames(unittest.TestCase):
    def test_one_name_per_component_in_term_order(self) -> None:
        names = state_names(_Manager())
        self.assertEqual(len(names), 3 + 14 + 3)
        self.assertEqual(
            names[:3], ["base_ang_vel[0]", "base_ang_vel[1]", "base_ang_vel[2]"]
        )
        self.assertEqual(names[3], "joint_pos[0]")
        self.assertEqual(names[-1], "command[2]")


class FailureRows(unittest.TestCase):
    def test_a_discard_keeps_its_command_and_why(self) -> None:
        fell = WorldEpisode(
            EpisodeOutcome(steps=412, fell=True, mean_err=0.05, mean_cmd=0.3),
            command=[0.3, 0.0, 0.1],
        )
        row = failure_row(77, 4, fell)
        self.assertEqual((row["batch_seed"], row["world"]), (77, 4))
        self.assertTrue(row["fell"])
        self.assertEqual(row["command"], [0.3, 0.0, 0.1])
        # JSON-clean: the file is one row per line, nothing numpy.
        json.dumps(row)


class TheExpertStamp(unittest.TestCase):
    def test_the_checkpoint_bytes_are_the_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "model_10.pt"
            a.write_bytes(b"weights-a")
            first = checkpoint_stamp(a)
            self.assertTrue(first.startswith("model_10@"))
            a.write_bytes(b"weights-b")
            self.assertNotEqual(first, checkpoint_stamp(a))


class TheExportDescriptor(unittest.TestCase):
    def test_round_trips_and_names_its_file(self) -> None:
        spec = ExportSpec(
            state_width=2,
            state_names=["a", "b"],
            cameras=[CAMERA_KEY],
            instruction="walk",
            bundle="microduck",
            action_semantics=ACTION_SEMANTICS,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = spec.write_to(Path(tmp))
            self.assertEqual(path.name, DemoLayout.EXPORT_FILE)
            self.assertEqual(ExportSpec.read_from(Path(tmp)), spec)

    def test_a_batch_without_one_is_refused_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as ctx:
                ExportSpec.read_from(Path(tmp))
            self.assertIn(DemoLayout.EXPORT_FILE, str(ctx.exception))


class DaggerRelabel(unittest.TestCase):
    """DAgger's one move, pinned without a GPU: the student's episode
    keeps its judgment, command and states; only the actions become the
    teacher's labels — and a shape mismatch is refused."""

    def test_the_teacher_labels_the_students_states(self) -> None:
        import numpy as np  # noqa: PLC0415

        from rq_mjlab.walk_press import relabel  # noqa: PLC0415

        obs = np.arange(12, dtype=np.float32).reshape(3, 4)
        student_actions = np.zeros((3, 2), dtype=np.float32)
        episode = WorldEpisode(
            EpisodeOutcome(steps=3, fell=False, mean_err=0.1, mean_cmd=0.3),
            command=[0.3, 0.0, 0.0],
            observations=obs,
            actions=student_actions,
            qpos=np.ones((3, 5), dtype=np.float32),
        )
        labeled = relabel(episode, lambda o: o[:, :2] * 10)
        np.testing.assert_array_equal(labeled.actions, obs[:, :2] * 10)
        np.testing.assert_array_equal(labeled.observations, obs)  # the student's states
        self.assertIs(labeled.outcome, episode.outcome)  # the student's judgment
        self.assertEqual(labeled.command, episode.command)
        with self.assertRaisesRegex(ValueError, "do not match"):
            relabel(episode, lambda o: o)  # (3, 4) labels for (3, 2) actions
        with self.assertRaisesRegex(ValueError, "captured"):
            relabel(WorldEpisode(episode.outcome, episode.command), lambda o: o)

    def test_the_labeler_hands_the_actor_its_observation_group(self) -> None:
        """An rsl-rl actor indexes a TensorDict by observation group; a
        bare tensor raises IndexError inside the model (the crash that
        killed DAgger round 1's first launch, 2026-09-04)."""
        import numpy as np  # noqa: PLC0415

        from rq_mjlab.walk_press import teacher_labeler  # noqa: PLC0415
        from rq_mjlab.walk_verdict import ACTOR_OBS_GROUP  # noqa: PLC0415

        seen: dict[str, Any] = {}

        def actor(obs: Any) -> Any:  # what rsl-rl's MLPModel.get_latent does
            seen["batch"] = tuple(obs.batch_size)
            return obs[ACTOR_OBS_GROUP][:, :2] * 10

        label = teacher_labeler(actor, "cpu")
        obs = np.arange(12, dtype=np.float32).reshape(3, 4)
        np.testing.assert_array_equal(label(obs), obs[:, :2] * 10)
        self.assertEqual(seen["batch"], (3,))


class TheSceneProvenance(unittest.TestCase):
    """A scene batch names where its pictures and its ground came from
    (docs/78 E3): the task stamp carries the built identity, the visual
    basis the scene's version and gap, the physics basis the floor as
    the scene declares it."""

    def test_the_source_is_the_walk_and_its_identity(self) -> None:
        from rq_mjlab.walk_press import walk_source  # noqa: PLC0415

        a = walk_source("go2", {"robot": "go2@1", "scene": "fake@2"})
        b = walk_source("go2", {"robot": "go2@1"})
        self.assertTrue(a.startswith("go2-walk@"))
        self.assertNotEqual(a, b)  # the scene is part of the identity

    def test_the_scene_visuals_name_the_stamp_the_gap_and_the_floor(self) -> None:
        from rq_mjlab.scene_stage import scene_stamp  # noqa: PLC0415
        from rq_mjlab.walk_press import scene_visuals  # noqa: PLC0415
        from tests.test_scene_stage import _scene  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            scene = _scene(Path(tmp))
            visual, physics = scene_visuals(scene)
            self.assertIn(scene_stamp(scene), visual)
            self.assertIn("no visual draws", visual)
            self.assertIn("gap chamfer None m", visual)  # the fake's gap is unmeasured
            self.assertIn("floor friction [1.25, 0.3, 0.3] declared ±0.2", physics)


class TheEpisodeCap(unittest.TestCase):
    """The press caps every rollout at the training episode: the play
    env's own episodes never end (the Go2's run 1e9 s), and a walker
    that never falls would roll out forever."""

    def test_the_cap_is_the_training_episode_at_the_control_rate(self) -> None:
        from rq_pipeline.tasks.walks import DEFAULT_EPISODE_S  # noqa: PLC0415

        from rq_mjlab.walk_press import episode_ticks  # noqa: PLC0415

        self.assertEqual(episode_ticks(DEFAULT_EPISODE_S, 0.02), 1000)
        self.assertEqual(episode_ticks(2.0, 0.02), 100)


class TheEpisodesFrames(unittest.TestCase):
    """On a scene the frames are the pictures the episode's own camera
    took; off one, the chase camera's replay; a missing camera refuses."""

    def test_the_episodes_own_pictures_every_n_ticks(self) -> None:
        import numpy as np  # noqa: PLC0415

        from rq_mjlab.walk_press import frames_of  # noqa: PLC0415

        # the rollout snapped one picture every 2 ticks: three for five ticks
        pictures = np.arange(3)[:, None, None, None] * np.ones((3, 2, 2, 3), np.uint8)
        episode = WorldEpisode(
            EpisodeOutcome(steps=5, fell=False, mean_err=0.1, mean_cmd=0.3),
            command=[0.3, 0.0, 0.0],
            frames={"head": pictures},
        )
        frames = frames_of(episode, None, "head", 2)
        self.assertEqual([tick for tick, _ in frames], [0, 2, 4])
        self.assertEqual(int(frames[1][1][0, 0, 0]), 1)
        with self.assertRaisesRegex(ValueError, "no frames from camera 'course'"):
            frames_of(episode, None, "course", 1)

    def test_the_chase_camera_replays_when_given(self) -> None:
        from rq_mjlab.walk_press import frames_of  # noqa: PLC0415

        class _Chase:
            def frames(self, qpos: Any, every: int) -> list[tuple[int, Any]]:
                return [(0, "chase")]

        episode = WorldEpisode(
            EpisodeOutcome(steps=1, fell=False, mean_err=0.1, mean_cmd=0.3),
            command=[0.3, 0.0, 0.0],
        )
        self.assertEqual(frames_of(episode, _Chase(), "chase", 1), [(0, "chase")])  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
