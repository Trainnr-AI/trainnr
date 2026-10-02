"""The batched open-loop verifier on a task small enough to be fast: a
one-hinge 'hold up' referee, judged states-only like kitting's. Pins:
`expand_controls`' hold semantics; device filter vs CPU reference
agreeing on an easy pass and an easy fail; a seed round-trip through
`DemoLayout`."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._extras import needs_mjx, needs_sim

HOLD_STEPS = 40
STEPS = 200
TARGET = 0.5
TOLERANCE = 0.2

XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body>
      <joint name="j" axis="0 1 0" damping="0.08"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.2" size="0.01" mass="0.1"/>
    </body>
  </worldbody>
  <actuator><position name="servo" joint="j" kp="8" ctrlrange="-1.5 1.5"/></actuator>
  <sensor><jointpos joint="j"/></sensor>
</mujoco>
"""


def _task():
    """A minimal Task-shaped object: spec + a states-only protocol."""
    import mujoco  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    from trainnr.protocol import EpisodeProtocol  # noqa: PLC0415

    def _success(states: object, sensors: object) -> bool:
        del sensors  # states-only, kitting's shape
        tail = np.asarray(states)[-HOLD_STEPS:, 1]
        return bool(np.all(np.abs(tail - TARGET) < TOLERANCE))

    class _Task:
        spec = mujoco.MjSpec.from_string(XML)
        protocol = EpisodeProtocol(
            trials=1,
            steps=STEPS,
            control_interval=10,
            perturb=lambda _trial, home: home,
            success=_success,
        )

    return _Task


@needs_sim
class ExpandControls(unittest.TestCase):
    def test_hold_and_pad_semantics(self) -> None:
        import numpy as np  # noqa: PLC0415

        from trainnr.collect.press_batch import expand_controls  # noqa: PLC0415

        actions = np.array([[1.0], [2.0]])
        out = expand_controls(actions, control_interval=3, steps=10)
        np.testing.assert_array_equal(out[:, 0], [1, 1, 1, 2, 2, 2, 2, 2, 2, 2])
        out = expand_controls(actions, control_interval=3, steps=4)
        np.testing.assert_array_equal(out[:, 0], [1, 1, 1, 2])


@needs_sim
class SeedRoundTrip(unittest.TestCase):
    def test_read_back_what_the_press_writes(self) -> None:
        import numpy as np  # noqa: PLC0415

        from trainnr.collect.kitting_export import write_episode  # noqa: PLC0415
        from trainnr.collect.press import EpisodeManifest  # noqa: PLC0415
        from trainnr.collect.press_batch import Seed  # noqa: PLC0415

        manifest = EpisodeManifest(
            seed=7,
            attempt=1,
            task="t@0",
            expert="e@0",
            instrument="i",
            dynamics={"damping": 1.0},
            draws={},
            retries=[],
            control_hz=50,
            frame_every_control_ticks=1,
            dynamics_basis="test",
        )
        with tempfile.TemporaryDirectory() as tmp:
            write_episode(
                Path(tmp),
                0,
                states=np.zeros((5, 3)),
                sensors=np.ones((5, 2)),
                actions=np.full((2, 1), 0.5),
                frames=[],
                manifest=manifest,
            )
            seed = Seed.read(Path(tmp) / "episode_0000")
        self.assertEqual(seed.states.shape, (5, 3))
        self.assertEqual(float(seed.actions[0, 0]), 0.5)
        self.assertEqual(seed.manifest["dynamics_basis"], "test")


@needs_mjx
class TheFilterAndTheReference(unittest.TestCase):
    def test_device_filter_and_cpu_reference_agree_on_pass_and_fail(self) -> None:
        import numpy as np  # noqa: PLC0415

        from trainnr.collect.press_batch import (  # noqa: PLC0415
            batched_success,
            cpu_execute,
            expand_controls,
        )
        from trainnr.physics.mujoco_backend import MuJoCoBackend  # noqa: PLC0415

        task = _task()
        backend = MuJoCoBackend()
        backend.load_spec(task.spec)
        home = backend.default_initial_state()
        # World 0 commands the target; world 1 commands the opposite.
        ticks = STEPS // 10
        good = expand_controls(
            np.full((ticks, 1), TARGET), control_interval=10, steps=STEPS
        )
        bad = expand_controls(
            np.full((ticks, 1), -TARGET), control_interval=10, steps=STEPS
        )
        initials = np.stack([home, home])
        controls = np.stack([good, bad])
        successes, states = batched_success(
            task, initials, controls, engine={"impl": "jax"}
        )
        self.assertEqual(states.shape[0], 2)
        np.testing.assert_array_equal(successes, [True, False])
        # The keepers' instrument: the CPU agrees on both.
        ok, _states, sensors = cpu_execute(task, home, good)
        self.assertTrue(ok)
        self.assertEqual(sensors.shape, (STEPS, 1))
        ok, _states, _sensors = cpu_execute(task, home, bad)
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
