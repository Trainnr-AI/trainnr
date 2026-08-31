"""The multiplication core on the one-hinge task: a seed episode
multiplied into keepers under fresh dynamics, the Mimic accounting
(nearest-neighbour selection, the abort), and the manifests naming
their source."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._extras import needs_mjx, needs_sim
from tests.test_press_batch import STEPS, TARGET, XML, _task


def _seed_episode(tmp: Path):
    """One kept episode from the reference instrument, on disk and back."""
    import numpy as np  # noqa: PLC0415

    from rq_pipeline.collect.kitting_export import write_episode  # noqa: PLC0415
    from rq_pipeline.collect.press import EpisodeManifest  # noqa: PLC0415
    from rq_pipeline.collect.press_batch import (  # noqa: PLC0415
        Seed,
        cpu_execute,
        expand_controls,
    )
    from rq_pipeline.physics.mujoco_backend import MuJoCoBackend  # noqa: PLC0415

    task = _task()
    backend = MuJoCoBackend()
    backend.load_spec(task.spec)
    home = backend.default_initial_state()
    actions = np.full((STEPS // 10, 1), TARGET)
    controls = expand_controls(actions, control_interval=10, steps=STEPS)
    ok, states, sensors = cpu_execute(task, home, controls)
    assert ok
    manifest = EpisodeManifest(
        seed=1,
        attempt=1,
        task="hinge@test",
        expert="hinge-expert@test",
        instrument="cpu",
        dynamics={"damping": 1.0},
        draws={"angle": 0.0},
        retries=[],
        control_hz=50,
        frame_every_control_ticks=1,
    )
    write_episode(
        tmp,
        0,
        states=states,
        sensors=sensors,
        actions=actions,
        frames=[],
        manifest=manifest,
    )
    return task, home, Seed.read(tmp / "episode_0000")


@needs_sim
class NearestSeed(unittest.TestCase):
    def test_picks_the_closest_by_the_callers_key(self) -> None:
        from rq_pipeline.collect.multiply import nearest_seed  # noqa: PLC0415
        from rq_pipeline.collect.press_batch import Seed  # noqa: PLC0415

        seeds = [
            Seed(
                states=None,
                sensors=None,
                actions=None,
                manifest={"draws": {"part": [x, 0.0]}},
            )
            for x in (0.0, 0.5, 1.0)
        ]
        picked = nearest_seed(seeds, [0.6, 0.0], lambda s: s.manifest["draws"]["part"])
        self.assertEqual(picked, 1)


@needs_sim
class KittingAdapter(unittest.TestCase):
    def test_variants_stay_in_the_measured_basin(self) -> None:
        import numpy as np  # noqa: PLC0415

        from rq_pipeline.collect.kitting_demos import (  # noqa: PLC0415
            SPAWN_NOISE,
            kitting_variant_fn,
        )
        from rq_pipeline.collect.press_batch import Seed  # noqa: PLC0415
        from rq_pipeline.tasks.aloha2 import (  # noqa: PLC0415
            PART_ORDER,
            PART_STATE_SLICE,
        )

        seeds = [
            Seed(
                states=None,
                sensors=None,
                actions=None,
                manifest={"draws": {arm: [x, -x] for arm in PART_ORDER}},
            )
            for x in (0.10, 0.20)
        ]
        size = max(PART_STATE_SLICE[arm].stop for arm in PART_ORDER) + 4
        variant_fn = kitting_variant_fn(np.zeros(size))
        rng = np.random.default_rng(2)
        for _ in range(20):
            variant = variant_fn(rng, seeds)
            for arm in PART_ORDER:
                part = PART_STATE_SLICE[arm]
                drawn = np.array(variant.draws[arm])
                # written into the initial state verbatim
                np.testing.assert_allclose(
                    [
                        variant.initial_state[part.start],
                        variant.initial_state[part.start + 1],
                    ],
                    drawn,
                )
                # within the measured basin of at least one seed
                gaps = [np.abs(drawn - s.manifest["draws"][arm]).max() for s in seeds]
                self.assertLessEqual(min(gaps), SPAWN_NOISE)
            self.assertIn(variant.seed_index, (0, 1))


@needs_mjx
class Multiplication(unittest.TestCase):
    def test_keeps_stamps_and_aborts(self) -> None:
        import mujoco  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        from rq_pipeline.collect.multiply import (  # noqa: PLC0415
            MultiplyPlan,
            Variant,
            multiply,
        )
        from rq_pipeline.collect.press import EpisodeManifest  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            task, home, seed_episode = _seed_episode(tmp / "seeds")

            def variant_fn(rng, seeds):
                initial = seeds[0].states[0].copy()
                jitter = float(rng.uniform(-0.02, 0.02))
                initial[1] = home[1] + jitter
                return Variant(
                    seed_index=0, initial_state=initial, draws={"angle_jitter": jitter}
                )

            def dynamics_fn(rng):
                scale = float(1.0 + rng.uniform(-0.1, 0.1))
                spec = mujoco.MjSpec.from_string(XML)
                spec.joints[0].damping = spec.joints[0].damping * scale
                return spec, {"damping": scale}, "test span ±0.1"

            plan = MultiplyPlan(
                variant_fn=variant_fn,
                dynamics_fn=dynamics_fn,
                episodes=2,
                worlds=4,
                action_noise=0.002,
                engine={"impl": "jax"},
            )
            out = tmp / "multiplied"
            result = multiply(
                out,
                task=task,
                seeds=[seed_episode],
                plan=plan,
                seed=3,
                instrument="test",
                control_hz=50,
                say=lambda _line: None,
            )
            self.assertTrue(result.complete)
            self.assertEqual(result.kept, 2)
            self.assertGreaterEqual(result.device_kept, result.kept)
            manifest = EpisodeManifest.read_from(out / "episode_0000")
            self.assertTrue(manifest.expert.startswith("multiplied:hinge-expert"))
            self.assertEqual(manifest.draws["multiplied_from"], 0)
            self.assertIn("angle_jitter", manifest.draws)
            self.assertEqual(manifest.dynamics_basis, "test span ±0.1")
            self.assertEqual(manifest.verdict, "success (task referee, CPU reference)")
            with np.load(out / "episode_0000" / "trajectory.npz") as data:
                self.assertEqual(data["states"].shape[0], STEPS)
                self.assertEqual(data["actions"].shape[0], STEPS // 10)

            # The abort: hopeless dynamics exhaust max_rounds, keep nothing
            # (damping x100 glues the joint — the servo cannot reach the
            # target inside the episode, so every world fails the referee).
            def glue(rng):
                del rng
                spec = mujoco.MjSpec.from_string(XML)
                spec.joints[0].damping = spec.joints[0].damping * 100.0
                return spec, {"damping": 100.0}, "glue"

            starved = multiply(
                tmp / "starved",
                task=task,
                seeds=[seed_episode],
                plan=MultiplyPlan(
                    variant_fn=variant_fn,
                    dynamics_fn=glue,
                    episodes=1,
                    worlds=2,
                    max_rounds=2,
                    engine={"impl": "jax"},
                ),
                seed=3,
                instrument="test",
                control_hz=50,
                say=lambda _line: None,
            )
            self.assertFalse(starved.complete)
            self.assertEqual(starved.rounds, 2)


if __name__ == "__main__":
    unittest.main()
