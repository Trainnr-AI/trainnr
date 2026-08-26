"""Kitting demos → LeRobot dataset: the round trip on a synthetic batch.

The demos-directory contract (trajectory.npz + frames/<tick>.jpg +
manifest) is checked without LeRobot; the conversion itself needs the
`train` extra and runs in the train venv.
"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

LEROBOT_PRESENT = importlib.util.find_spec("lerobot") is not None
NUMPY_PRESENT = importlib.util.find_spec("numpy") is not None

STEPS = 60  # physics steps; the generator writes sensors per physics step
CONTROL_TICKS = 6  # and actions per control tick (10 physics steps each)
FRAME_TICKS = (0, 10, 20, 30, 40, 50)  # frame_every=1 at 500 Hz physics / 50 Hz control


def synthetic_batch(root: Path, episodes: int = 2, frame_every: int = 1) -> Path:
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
            ),
        )
    return demos


class DemosDirectoryContract(unittest.TestCase):
    def test_empty_batch_is_refused(self) -> None:
        from rq_pipeline.collect.kitting_export import episode_dirs  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            episode_dirs(Path(tmp))

    @unittest.skipUnless(NUMPY_PRESENT, "sim extra not installed")
    def test_episodes_are_found_in_order(self) -> None:
        from rq_pipeline.collect.kitting_export import episode_dirs  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            demos = synthetic_batch(Path(tmp), episodes=3)
            names = [p.name for p in episode_dirs(demos)]
        self.assertEqual(names, ["episode_0000", "episode_0001", "episode_0002"])


@unittest.skipUnless(LEROBOT_PRESENT, "train extra not installed (use .venv-train)")
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
            provenance = json.loads((root / PROVENANCE_FILE).read_text())
            self.assertEqual(provenance["fps"], 50)
            self.assertEqual(provenance["episodes"], 2)
            self.assertIn("@", provenance["bundle"])
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


if __name__ == "__main__":
    unittest.main()
