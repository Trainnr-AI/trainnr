"""The demo generator as a function: its accounting, its files, its
numbering — on a short protocol where the referee cannot pass (the
give-up path) and on one kept episode (the keep path)."""

import importlib.util
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

MUJOCO_PRESENT = importlib.util.find_spec("mujoco") is not None
SHORT_STEPS = (
    400  # the parts never reach the slots in 0.8 s: every attempt is a discard
)
SPARSE_FRAMES = 100  # one frame per 100 control ticks: 14 frames per kept episode


@unittest.skipUnless(MUJOCO_PRESENT, "sim extra not installed (uv sync --extra sim)")
class TheGenerator(unittest.TestCase):
    def test_gives_up_after_max_attempts_and_writes_nothing(self) -> None:
        from rq_pipeline.collect.kitting_demos import generate_demos  # noqa: PLC0415
        from rq_pipeline.tasks.aloha2 import KITTING_SPEC  # noqa: PLC0415

        said: list[str] = []
        with TemporaryDirectory() as tmp:
            batch = generate_demos(
                Path(tmp) / "demos",
                episodes=1,
                seed=1,
                frame_every=SPARSE_FRAMES,
                max_attempts=1,
                spec=replace(KITTING_SPEC, steps=SHORT_STEPS),
                say=said.append,
            )
            self.assertEqual(
                (batch.kept, batch.attempts, batch.complete), (0, 1, False)
            )
            self.assertEqual(list((Path(tmp) / "demos").iterdir()), [])
        self.assertTrue(said[0].startswith("expert kitting-expert@"))
        self.assertIn("gave up after 1 attempts", said[-1])

    def test_a_kept_episode_is_numbered_from_first_episode_with_its_stamp(self) -> None:
        from rq_pipeline.collect.kitting_demos import generate_demos  # noqa: PLC0415
        from rq_pipeline.collect.kitting_export import (  # noqa: PLC0415
            Manifest,
            episode_dirs,
        )

        with TemporaryDirectory() as tmp:
            batch = generate_demos(
                Path(tmp) / "demos",
                episodes=1,
                seed=20260828,
                dr_span=0.0,  # nominal dynamics: the expert's 12/12 band
                frame_every=SPARSE_FRAMES,
                first_episode=7,
                say=lambda _text: None,
            )
            self.assertTrue(batch.complete, batch)
            self.assertEqual(batch.last_episode, 7)
            [episode] = episode_dirs(Path(tmp) / "demos")
            self.assertEqual(episode.name, "episode_0007")
            manifest = Manifest.read_from(episode)
            self.assertEqual(manifest.expert, batch.expert)
            self.assertEqual(manifest.frame_every_control_ticks, SPARSE_FRAMES)
            self.assertEqual(set(manifest.draws), {"right", "left"})


if __name__ == "__main__":
    unittest.main()
