"""Visual DR in the press (docs/66 §4) and the press's feed seam
(a note): draws through the REAL appliers on a compiled model,
the refusals at the adapter door, the manifest and datasheet carrying
the draws, the multi-camera episode layout, and the loop reporting
every attempt to its feed."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._extras import needs_sim
from tests.test_press_batch import HOLD_STEPS, STEPS, TARGET, TOLERANCE, _task

DR = {"damping": (0.95, 1.05), "gain": (0.95, 1.05)}
DIFFUSE_LO, DIFFUSE_HI = 0.5, 1.5  # the diffuse-scale range every test draws
OFFSET = 0.01  # the camera-offset half-width, metres
XYZ = 3

# The one-hinge task with a light and two cameras: what visual DR
# needs a model to have. Same servo and referee as test_press_batch.
LIT_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <light name="sun" pos="0 0 2" diffuse="0.8 0.8 0.8"/>
    <camera name="cam_a" pos="0.5 0 0.3"/>
    <camera name="cam_b" pos="0 0.5 0.3"/>
    <body>
      <joint name="j" axis="0 1 0" damping="0.08"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.2" size="0.01" mass="0.1"/>
    </body>
  </worldbody>
  <actuator><position name="servo" joint="j" kp="8" ctrlrange="-1.5 1.5"/></actuator>
  <sensor><jointpos joint="j"/></sensor>
</mujoco>
"""


def _lit_task():
    """The `_task` shape over LIT_XML."""
    import mujoco  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    from trainnr.protocol import EpisodeProtocol  # noqa: PLC0415

    def _success(states: object, sensors: object) -> bool:
        del sensors
        tail = np.asarray(states)[-HOLD_STEPS:, 1]
        return bool(np.all(np.abs(tail - TARGET) < TOLERANCE))

    class _LitTask:
        spec = mujoco.MjSpec.from_string(LIT_XML)
        protocol = EpisodeProtocol(
            trials=1,
            steps=STEPS,
            control_interval=10,
            perturb=lambda _trial, home: home,
            success=_success,
        )

    return _LitTask


def _policy(step: int, sensordata: object) -> list[float]:
    del step, sensordata
    return [TARGET]


@needs_sim
class DrawRandom(unittest.TestCase):
    def test_uniform_scalar_stays_in_range_and_is_seeded(self) -> None:
        import numpy as np  # noqa: PLC0415

        from trainnr.evaluate.variations import (  # noqa: PLC0415
            Uniform,
            Variation,
            draw_random,
        )

        knob = Variation(
            "lights", "diffuse_scale", Uniform((DIFFUSE_LO,), (DIFFUSE_HI,))
        )
        values = [
            draw_random(knob, np.random.default_rng(7)),
            draw_random(knob, np.random.default_rng(7)),
            draw_random(knob, np.random.default_rng(8)),
        ]
        self.assertEqual(values[0], values[1])  # the seed's stream, twice
        self.assertNotEqual(values[0], values[2])
        for value in values:
            self.assertTrue(DIFFUSE_LO <= value <= DIFFUSE_HI)

    def test_uniform_box_draws_per_component(self) -> None:
        import numpy as np  # noqa: PLC0415

        from trainnr.evaluate.variations import (  # noqa: PLC0415
            Uniform,
            Variation,
            draw_random,
        )

        knob = Variation(
            "cam_a", "offset_m", Uniform((-OFFSET,) * XYZ, (OFFSET,) * XYZ)
        )
        value = draw_random(knob, np.random.default_rng(1))
        self.assertEqual(len(value), XYZ)
        for component in value:
            self.assertTrue(-OFFSET <= component <= OFFSET)

    def test_choice_picks_a_label(self) -> None:
        import numpy as np  # noqa: PLC0415

        from trainnr.evaluate.variations import (  # noqa: PLC0415
            Choice,
            Variation,
            draw_random,
        )

        knob = Variation("lights", "diffuse_scale", Choice(("a", "b")))
        self.assertIn(draw_random(knob, np.random.default_rng(3)), ("a", "b"))


@needs_sim
class ApplyVisuals(unittest.TestCase):
    def test_draws_land_on_the_compiled_model(self) -> None:
        """Degenerate ranges pin exact values: the light halves, the
        camera moves — through the SAME appliers evaluation uses."""
        import mujoco  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        from trainnr.collect.scripted_demos import _apply_visuals  # noqa: PLC0415
        from trainnr.evaluate.variations import Uniform, Variation  # noqa: PLC0415

        model = mujoco.MjSpec.from_string(LIT_XML).compile()
        nominal_diffuse = model.light_diffuse.copy()
        nominal_cam = model.cam_pos.copy()
        drawn = _apply_visuals(
            model,
            [
                Variation(
                    "lights", "diffuse_scale", Uniform((DIFFUSE_LO,), (DIFFUSE_LO,))
                ),
                Variation(
                    "cam_a", "offset_m", Uniform((OFFSET,) * XYZ, (OFFSET,) * XYZ)
                ),
            ],
            np.random.default_rng(0),
        )
        np.testing.assert_allclose(model.light_diffuse, nominal_diffuse * DIFFUSE_LO)
        np.testing.assert_allclose(model.cam_pos[0], nominal_cam[0] + OFFSET)
        np.testing.assert_allclose(model.cam_pos[1], nominal_cam[1])  # cam_b: untouched
        self.assertEqual(drawn["lights.diffuse_scale"], DIFFUSE_LO)
        self.assertEqual(drawn["cam_a.offset_m"], [OFFSET] * XYZ)

    def test_a_lightless_scene_refuses_the_lights_knob(self) -> None:
        """The so101 scenes have nlight == 0 (measured 2026-09-02): the
        lights knob there would be a silent no-op — refused, naming the
        headlight knob that actually works."""
        import mujoco  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        from trainnr.collect.scripted_demos import _apply_visuals  # noqa: PLC0415
        from trainnr.evaluate.variations import Uniform, Variation  # noqa: PLC0415

        model = mujoco.MjSpec.from_string(
            LIT_XML.replace('<light name="sun" pos="0 0 2" diffuse="0.8 0.8 0.8"/>', "")
        ).compile()
        self.assertEqual(model.nlight, 0)
        with self.assertRaisesRegex(ValueError, "headlight"):
            _apply_visuals(
                model,
                [
                    Variation(
                        "lights",
                        "diffuse_scale",
                        Uniform((DIFFUSE_LO,), (DIFFUSE_LO,)),
                    )
                ],
                np.random.default_rng(0),
            )

    def test_the_headlight_knob_scales_the_builtin_light(self) -> None:
        import mujoco  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        from trainnr.collect.scripted_demos import _apply_visuals  # noqa: PLC0415
        from trainnr.evaluate.variations import Uniform, Variation  # noqa: PLC0415

        model = mujoco.MjSpec.from_string(LIT_XML).compile()
        nominal = model.vis.headlight.diffuse.copy()
        _apply_visuals(
            model,
            [
                Variation(
                    "headlight",
                    "diffuse_scale",
                    Uniform((DIFFUSE_LO,), (DIFFUSE_LO,)),
                )
            ],
            np.random.default_rng(0),
        )
        np.testing.assert_allclose(model.vis.headlight.diffuse, nominal * DIFFUSE_LO)

    def test_a_missing_camera_is_refused_by_name(self) -> None:
        import mujoco  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        from trainnr.collect.scripted_demos import _apply_visuals  # noqa: PLC0415
        from trainnr.evaluate.variations import Uniform, Variation  # noqa: PLC0415

        model = mujoco.MjSpec.from_string(LIT_XML).compile()
        with self.assertRaisesRegex(ValueError, "ghost"):
            _apply_visuals(
                model,
                [Variation("ghost", "offset_m", Uniform((0.0,) * XYZ, (0.0,) * XYZ))],
                np.random.default_rng(0),
            )


@needs_sim
class TheAdapterDoor(unittest.TestCase):
    def test_a_dynamics_knob_is_refused_as_a_visual(self) -> None:
        from trainnr.collect.scripted_demos import (  # noqa: PLC0415
            generate_scripted_demos,
        )
        from trainnr.evaluate.variations import Uniform, Variation  # noqa: PLC0415

        with (
            tempfile.TemporaryDirectory() as tmp,
            self.assertRaisesRegex(ValueError, "visual DR"),
        ):
            generate_scripted_demos(
                Path(tmp),
                task_factory=_task,
                policy=_policy,
                expert="e",
                dr=DR,
                basis="b",
                episodes=1,
                seed=1,
                frame_every=0,
                visuals=[Variation("joints", "damping_scale", Uniform((0.9,), (1.1,)))],
                visual_basis="v",
            )

    def test_visuals_without_a_basis_are_refused(self) -> None:
        from trainnr.collect.scripted_demos import (  # noqa: PLC0415
            generate_scripted_demos,
        )
        from trainnr.evaluate.variations import Uniform, Variation  # noqa: PLC0415

        with (
            tempfile.TemporaryDirectory() as tmp,
            self.assertRaisesRegex(ValueError, "visual_basis"),
        ):
            generate_scripted_demos(
                Path(tmp),
                task_factory=_task,
                policy=_policy,
                expert="e",
                dr=DR,
                basis="b",
                episodes=1,
                seed=1,
                frame_every=0,
                visuals=[
                    Variation(
                        "lights", "diffuse_scale", Uniform((DIFFUSE_LO,), (DIFFUSE_HI,))
                    )
                ],
            )


@needs_sim
class VisualDrawsRideTheBatch(unittest.TestCase):
    def test_manifest_and_datasheet_carry_the_draws(self) -> None:
        from trainnr.collect.datasheet import render, summarize  # noqa: PLC0415
        from trainnr.collect.press import EpisodeManifest  # noqa: PLC0415
        from trainnr.collect.scripted_demos import (  # noqa: PLC0415
            generate_scripted_demos,
        )
        from trainnr.evaluate.variations import Uniform, Variation  # noqa: PLC0415

        visuals = [
            Variation("lights", "diffuse_scale", Uniform((DIFFUSE_LO,), (DIFFUSE_HI,))),
            Variation("cam_a", "offset_m", Uniform((-OFFSET,) * XYZ, (OFFSET,) * XYZ)),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            batch = generate_scripted_demos(
                Path(tmp),
                task_factory=_lit_task,
                policy=_policy,
                expert="hold-up@test",
                dr=DR,
                basis="test-declared span",
                episodes=2,
                seed=9,
                frame_every=0,
                visuals=visuals,
                visual_basis="test-declared visual span (docs/66 §4)",
                say=lambda _line: None,
            )
            self.assertTrue(batch.complete)
            manifests = [
                EpisodeManifest.read_from(Path(tmp) / f"episode_{i:04d}")
                for i in range(2)
            ]
            summary = summarize(Path(tmp))
        for manifest in manifests:
            self.assertTrue(
                DIFFUSE_LO <= manifest.visuals["lights.diffuse_scale"] <= DIFFUSE_HI
            )
            self.assertEqual(len(manifest.visuals["cam_a.offset_m"]), XYZ)
            self.assertEqual(
                manifest.visual_basis, "test-declared visual span (docs/66 §4)"
            )
        # Two episodes, two draws — the randomiser randomises visuals too.
        self.assertNotEqual(
            manifests[0].visuals["lights.diffuse_scale"],
            manifests[1].visuals["lights.diffuse_scale"],
        )
        self.assertIn("lights.diffuse_scale", summary.visuals)
        self.assertIn("cam_a.offset_m[2]", summary.visuals)
        self.assertEqual(
            summary.visual_bases, ("test-declared visual span (docs/66 §4)",)
        )
        spread = summary.visuals["lights.diffuse_scale"]
        self.assertTrue(DIFFUSE_LO <= spread.low <= spread.high <= DIFFUSE_HI)
        self.assertIn("## Visual draws", render(summary))


@needs_sim
class MultiCameraLayout(unittest.TestCase):
    def test_camera_frames_land_in_per_camera_subdirs(self) -> None:
        import numpy as np  # noqa: PLC0415

        from trainnr.collect.demo_export import episode_camera_keys  # noqa: PLC0415
        from trainnr.collect.kitting_export import (  # noqa: PLC0415
            DemoLayout,
            write_episode,
        )
        from trainnr.collect.press import EpisodeManifest  # noqa: PLC0415

        image = np.zeros((4, 4, 3), dtype=np.uint8)
        manifest = EpisodeManifest(
            seed=1,
            attempt=1,
            task="t@0",
            expert="e@0",
            instrument="i",
            dynamics={},
            draws={},
            retries=[],
            control_hz=50,
            frame_every_control_ticks=1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            episode = write_episode(
                Path(tmp),
                0,
                states=[[0.0]],
                sensors=[[0.0]],
                actions=[[0.0]],
                frames=[],
                camera_frames={
                    "cam_a": [(0, image), (10, image)],
                    "cam_b": [(0, image), (10, image)],
                },
                manifest=manifest,
            )
            self.assertEqual(episode_camera_keys(episode), ("cam_a", "cam_b"))
            for camera in ("cam_a", "cam_b"):
                frames = sorted(
                    (episode / DemoLayout.FRAMES_DIR / camera).glob("*.jpg")
                )
                self.assertEqual([f.stem for f in frames], ["000000", "000010"])
            # The flat layout reads as no per-camera subdirs.
            with self.assertRaisesRegex(ValueError, "not both"):
                write_episode(
                    Path(tmp),
                    1,
                    states=[[0.0]],
                    sensors=[[0.0]],
                    actions=[[0.0]],
                    frames=[(0, image)],
                    camera_frames={"cam_a": [(0, image)]},
                    manifest=manifest,
                )


class _FakeFeed:
    def __init__(self) -> None:
        self.attempts: list[tuple[int, int, int, bool]] = []
        self.batches: list = []

    def attempt(self, attempts, kept, wanted, result) -> None:
        self.attempts.append((attempts, kept, wanted, result.succeeded))

    def done(self, batch) -> None:
        self.batches.append(batch)


@needs_sim
class TheFeedSeam(unittest.TestCase):
    def test_the_loop_reports_every_attempt_and_the_batch(self) -> None:
        from trainnr.collect.scripted_demos import (  # noqa: PLC0415
            generate_scripted_demos,
        )

        feed = _FakeFeed()
        with tempfile.TemporaryDirectory() as tmp:
            batch = generate_scripted_demos(
                Path(tmp),
                task_factory=_task,
                policy=_policy,
                expert="e",
                dr=DR,
                basis="b",
                episodes=2,
                seed=3,
                frame_every=0,
                feed=feed,
                say=lambda _line: None,
            )
        self.assertEqual(len(feed.attempts), batch.attempts)
        self.assertEqual(feed.attempts[-1][1], batch.kept)
        self.assertEqual([b.kept for b in feed.batches], [batch.kept])


if __name__ == "__main__":
    unittest.main()
