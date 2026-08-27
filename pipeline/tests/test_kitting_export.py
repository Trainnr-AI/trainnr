"""Kitting demos → LeRobot dataset: the round trip on a synthetic batch.

The demos-directory contract (trajectory.npz + frames/<tick>.jpg +
manifest) is checked without LeRobot; the conversion itself needs the
`train` extra and runs in the train venv.
"""

import json
import tempfile
import unittest
from pathlib import Path

from tests._extras import needs_numpy, needs_train

STEPS = 60  # physics steps; the generator writes sensors per physics step
CONTROL_TICKS = 6  # and actions per control tick (10 physics steps each)
FRAME_TICKS = (0, 10, 20, 30, 40, 50)  # frame_every=1 at 500 Hz physics / 50 Hz control
EXPERT = "kitting-expert@0123456789ab"


def synthetic_batch(
    root: Path, episodes: int = 2, frame_every: int = 1, experts: tuple = (EXPERT,)
) -> Path:
    import numpy as np  # noqa: PLC0415

    from rq_pipeline.collect.kitting_export import (  # noqa: PLC0415
        Manifest,
        write_episode,
    )

    rng = np.random.default_rng(0)
    demos = root / "demos"
    for index in range(episodes):
        write_episode(
            demos,
            index,
            states=rng.normal(size=(STEPS, 40)),
            sensors=rng.normal(size=(STEPS, 34)),
            actions=rng.normal(size=(CONTROL_TICKS, 14)),
            frames=[
                (tick, rng.integers(0, 255, (48, 64, 3), dtype=np.uint8))
                for tick in FRAME_TICKS
            ],
            manifest=Manifest(
                seed=0,
                attempt=index + 1,
                draws={},
                dr_span=0.0,
                damping_scale=1.0,
                gain_scale=1.0,
                retries=[],
                control_hz=50,
                frame_every_control_ticks=frame_every,
                expert=experts[index % len(experts)],
            ),
        )
    return demos


class DemosDirectoryContract(unittest.TestCase):
    def test_empty_batch_is_refused(self) -> None:
        from rq_pipeline.collect.kitting_export import episode_dirs  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            episode_dirs(Path(tmp))

    @needs_numpy
    def test_episodes_are_found_in_order(self) -> None:
        from rq_pipeline.collect.kitting_export import episode_dirs  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            demos = synthetic_batch(Path(tmp), episodes=3)
            names = [p.name for p in episode_dirs(demos)]
        self.assertEqual(names, ["episode_0000", "episode_0001", "episode_0002"])

    @needs_numpy
    def test_a_manifest_carries_the_expert_and_an_old_one_reads(self) -> None:
        from rq_pipeline.collect.kitting_export import (  # noqa: PLC0415
            UNSTAMPED_EXPERT,
            DemoLayout,
            Manifest,
        )

        with tempfile.TemporaryDirectory() as tmp:
            demos = synthetic_batch(Path(tmp), episodes=1)
            episode = demos / "episode_0000"
            self.assertEqual(Manifest.read_from(episode).expert, EXPERT)
            # A batch written before the stamp existed: the field is absent.
            raw = json.loads(
                (episode / DemoLayout.MANIFEST_FILE).read_text(encoding="utf-8")
            )
            del raw["expert"]
            (episode / DemoLayout.MANIFEST_FILE).write_text(
                json.dumps(raw), encoding="utf-8"
            )
            self.assertEqual(Manifest.read_from(episode).expert, UNSTAMPED_EXPERT)


@needs_train
class RoundTrip(unittest.TestCase):
    def test_export_then_reload(self) -> None:
        import numpy as np  # noqa: PLC0415
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

        from rq_pipeline.collect.kitting_export import (  # noqa: PLC0415
            PROVENANCE_FILE,
            STATE_WIDTH,
            export_kitting_demos,
        )

        with tempfile.TemporaryDirectory() as tmp:
            demos = synthetic_batch(Path(tmp), episodes=2)
            root = Path(tmp) / "dataset"
            export_kitting_demos(demos, root, repo_id="test/kitting", use_videos=False)
            provenance = json.loads(
                (root / PROVENANCE_FILE).read_text(encoding="utf-8")
            )
            self.assertEqual(provenance["fps"], 50)
            self.assertEqual(provenance["episodes"], 2)
            self.assertIn("@", provenance["bundle"])
            self.assertEqual(provenance["expert"], EXPERT)
            dataset = LeRobotDataset("test/kitting", root=root)
            self.assertEqual(len(dataset), 2 * len(FRAME_TICKS))
            sample = dataset[0]
            self.assertEqual(tuple(sample["observation.state"].shape), (STATE_WIDTH,))
            self.assertEqual(tuple(sample["action"].shape), (STATE_WIDTH,))
            self.assertEqual(
                tuple(sample["observation.images.top"].shape)[-2:], (48, 64)
            )
            # The state is the sensor block at the frame's physics tick and
            # the action the control tick that contains it, verbatim.
            trajectory = np.load(demos / "episode_0000" / "trajectory.npz")
            np.testing.assert_allclose(
                sample["observation.state"].numpy(),
                trajectory["sensors"][0, :STATE_WIDTH],
                rtol=1e-6,
            )
            last = dataset[len(FRAME_TICKS) - 1]  # frame at physics tick 50
            np.testing.assert_allclose(
                last["action"].numpy(), trajectory["actions"][5], rtol=1e-6
            )

    def test_two_experts_in_one_batch_are_refused(self) -> None:
        from rq_pipeline.collect.kitting_export import (  # noqa: PLC0415
            export_kitting_demos,
        )

        with tempfile.TemporaryDirectory() as tmp:
            demos = synthetic_batch(
                Path(tmp), episodes=2, experts=(EXPERT, "kitting-expert@ba9876543210")
            )
            with self.assertRaisesRegex(ValueError, "disagree on the expert"):
                export_kitting_demos(
                    demos, Path(tmp) / "dataset", repo_id="test/mixed", use_videos=False
                )


if __name__ == "__main__":
    unittest.main()
