"""The task-agnostic press loop, with a fake attempt callable — the
accounting, the numbering, the ceiling, and the go-forward sidecar —
no simulator anywhere near it.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rq_pipeline.collect.press import EpisodeManifest, PressResult, press
from tests._extras import needs_numpy


def scripted_attempts(*outcomes: bool):
    """An attempt_fn that succeeds/fails per the given script, with a
    tiny trajectory and a named dynamics draw per attempt."""
    remaining = list(outcomes)

    def attempt_fn(rng, *, frame_every):
        succeeded = remaining.pop(0)
        return PressResult(
            succeeded=succeeded,
            dynamics={"friction": float(rng.uniform(0.9, 1.1))},
            draws={"goal": 3},
            retries=[],
            states=[[0.0], [1.0]],
            sensors=[[0.0], [0.0]],
            actions=[[0.5], [0.5]],
            frames=[],
            note="" if succeeded else "referee",
        )

    return attempt_fn


@needs_numpy
class TheLoop(unittest.TestCase):
    def run_press(self, tmp: str, attempt_fn, **kwargs):
        return press(
            Path(tmp) / "demos",
            task="demo-task@abc123",
            expert="demo-expert@def456",
            instrument="mujoco-3.11.0+test",
            control_hz=50,
            attempt_fn=attempt_fn,
            seed=7,
            say=lambda _text: None,
            **kwargs,
        )

    def test_n_episodes_means_n_successful_episodes(self) -> None:
        with TemporaryDirectory() as tmp:
            batch = self.run_press(
                tmp, scripted_attempts(False, True, False, True), episodes=2
            )
            self.assertEqual((batch.kept, batch.attempts), (2, 4))
            self.assertTrue(batch.complete)

    def test_the_ceiling_stops_an_impossible_regime_loudly(self) -> None:
        said: list[str] = []
        with TemporaryDirectory() as tmp:
            batch = press(
                Path(tmp) / "demos",
                task="t@1",
                expert="e@1",
                instrument="i",
                control_hz=50,
                attempt_fn=scripted_attempts(False, False, False),
                episodes=5,
                seed=7,
                max_attempts=3,
                say=said.append,
            )
            self.assertEqual((batch.kept, batch.attempts), (0, 3))
            self.assertFalse(batch.complete)
            self.assertEqual(list((Path(tmp) / "demos").iterdir()), [])
        self.assertIn("gave up after 3 attempts", said[-1])

    def test_the_sidecar_answers_the_arcs_unanswerable_questions(self) -> None:
        # Which task, which engine, which expert, which dynamics — the
        # per-episode record no dataset in the studied ecosystems
        # carries (docs/e2e-research/60 §5).
        with TemporaryDirectory() as tmp:
            self.run_press(tmp, scripted_attempts(True), episodes=1, first_episode=4)
            episode = Path(tmp) / "demos" / "episode_0004"
            manifest = EpisodeManifest.read_from(episode)
            self.assertEqual(manifest.task, "demo-task@abc123")
            self.assertEqual(manifest.expert, "demo-expert@def456")
            self.assertEqual(manifest.instrument, "mujoco-3.11.0+test")
            self.assertIn("friction", manifest.dynamics)
            self.assertEqual(manifest.draws, {"goal": 3})
            self.assertEqual(manifest.verdict, "success (task referee)")


if __name__ == "__main__":
    unittest.main()
