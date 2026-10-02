"""The generic demos→LeRobot converter: task-shaped features, the
frame/state/action alignment inherited from the kitting converter, and
the NEW refusal — one dynamics basis per dataset (58 §8)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._extras import needs_train

CONTROL_HZ = 50
FRAME_EVERY = 5
STATE_WIDTH = 2


class _Camera:
    key = "front"
    camera_name = "front"
    height = 8
    width = 8


class _Task:
    state_width = STATE_WIDTH
    cameras = (_Camera(),)
    instruction = "hold it up"


def _batch(tmp: Path, *, bases: tuple[str, ...]) -> Path:
    import numpy as np  # noqa: PLC0415

    from trainnr.collect.kitting_export import write_episode  # noqa: PLC0415
    from trainnr.collect.press import EpisodeManifest  # noqa: PLC0415

    demos = tmp / "demos"
    steps, ticks = 20, 4
    for index, basis in enumerate(bases):
        manifest = EpisodeManifest(
            seed=1,
            attempt=index + 1,
            task="hold@test",
            expert="expert@test",
            instrument="cpu",
            dynamics={"damping": 1.0 + index},
            draws={"trial": index},
            retries=[],
            control_hz=CONTROL_HZ,
            frame_every_control_ticks=FRAME_EVERY,
            dynamics_basis=basis,
        )
        rng = np.random.default_rng(index)
        write_episode(
            demos,
            index,
            states=rng.random((steps, 5)),
            sensors=np.arange(steps * 3, dtype=float).reshape(steps, 3),
            actions=rng.random((ticks, STATE_WIDTH)),
            frames=[
                (tick, (rng.random((8, 8, 3)) * 255).astype(np.uint8))
                for tick in (0, 5, 10, 15)
            ],
            manifest=manifest,
        )
    return demos


@needs_train
class ExportDemos(unittest.TestCase):
    def test_round_trip_with_task_shaped_features(self) -> None:
        import numpy as np  # noqa: PLC0415
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

        from trainnr.collect.demo_export import export_demos  # noqa: PLC0415
        from trainnr.collect.provenance import PROVENANCE_FILE  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            demos = _batch(tmp, bases=("declared span", "declared span"))
            task = _Task()
            task.bundle_dir = tmp / "demos"  # any real dir: the stamp's input
            root = tmp / "dataset"
            export_demos(demos, root, task=task, repo_id="test/hold", use_videos=False)
            provenance = json.loads((root / PROVENANCE_FILE).read_text())
            self.assertEqual(provenance["fps"], CONTROL_HZ // FRAME_EVERY)
            self.assertEqual(provenance["episodes"], 2)
            self.assertIn("@", provenance["bundle"])
            # The batch by NAME (the union check's key) and by STAMP (lineage).
            self.assertEqual(provenance["source"], demos.name)
            self.assertTrue(provenance["source_stamp"].startswith(f"{demos.name}@"))
            self.assertIn(str(STATE_WIDTH), provenance["state_semantics"])
            dataset = LeRobotDataset("test/hold", root=root)
            self.assertEqual(len(dataset), 8)  # 2 episodes x 4 frames
            sample = dataset[1]  # frame at physics tick 5
            self.assertEqual(tuple(sample["observation.state"].shape), (STATE_WIDTH,))
            self.assertEqual(
                tuple(sample["observation.images.front"].shape)[-2:], (8, 8)
            )
            trajectory = np.load(demos / "episode_0000" / "trajectory.npz")
            np.testing.assert_allclose(
                sample["observation.state"].numpy(),
                trajectory["sensors"][5, :STATE_WIDTH],
                rtol=1e-6,
            )
            np.testing.assert_allclose(
                sample["action"].numpy(), trajectory["actions"][1], rtol=1e-6
            )

    def test_constant_dims_get_unit_std(self) -> None:
        import json  # noqa: PLC0415

        from trainnr.collect.demo_export import (  # noqa: PLC0415
            export_demos,
            guard_constant_dims,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            demos = _batch(tmp, bases=("declared span", "declared span"))
            task = _Task()
            task.bundle_dir = tmp / "demos"
            root = tmp / "dataset"
            export_demos(demos, root, task=task, repo_id="test/hold", use_videos=False)
            stats = json.loads((root / "meta" / "stats.json").read_text())

            def leaves(node):
                if isinstance(node, list):
                    for item in node:
                        yield from leaves(item)
                else:
                    yield node

            for feature in stats.values():
                for value in leaves(feature.get("std", [])):
                    self.assertGreater(abs(value), 1e-7)
            # Idempotent: a second pass finds nothing left to patch.
            self.assertEqual(guard_constant_dims(root), [])

    def test_two_bases_in_one_batch_are_refused(self) -> None:
        from trainnr.collect.demo_export import export_demos  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            demos = _batch(tmp, bases=("declared span", "identified-interval"))
            task = _Task()
            task.bundle_dir = tmp / "demos"
            with self.assertRaisesRegex(ValueError, "dynamics basis"):
                export_demos(
                    demos,
                    tmp / "dataset",
                    task=task,
                    repo_id="test/mixed",
                    use_videos=False,
                )


if __name__ == "__main__":
    unittest.main()
