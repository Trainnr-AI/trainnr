"""The legged fit's core, proved on a true model before any log: the
synthetic Go2 chirp (docs/e2e-research/26's rule) with the base fixed
and with the base shaken in the air, the tick-aligned derivatives, the
bound flag, the refusals."""

from __future__ import annotations

import unittest

import numpy as np

from tests._extras import needs_sim
from tests._fixtures import go2_model_or_skip

# The recovery the study measured on 2026-09-24 (36/36 fixed, 33/36
# shaken within 3 %); the test asks for a little less so a numpy or
# MuJoCo patch release does not turn the proof red.
WITHIN = 0.05
MIN_WITHIN_FIXED = 34
MIN_WITHIN_SHAKEN = 30


def _recover(posture: str) -> tuple[int, int, list[str]]:
    from rq_pipeline.robot.quadruped_synth import (  # noqa: PLC0415
        prepared_spec,
        simulate,
    )
    from rq_pipeline.robot.torque_balance import (  # noqa: PLC0415
        JointSamples,
        fit_terms,
        tick_aligned,
    )

    xml = go2_model_or_skip()
    recording, truth = simulate(xml, posture=posture)
    spec, joints = prepared_spec(xml, posture)
    model = spec.compile()
    position = recording.channels["joint.position"]
    times, q, v, a = tick_aligned(position.times, position.values)
    base = {}
    if "base.pose" in recording.channels:
        base = {
            "base_pose": recording.channels["base.pose"].values[:-1],
            "base_velocity": recording.channels["base.twist"].values[:-1],
            "base_acceleration": recording.channels["base.acceleration"].values[:-1],
        }
    samples = JointSamples(
        times=times,
        joints=joints,
        position=q,
        velocity=v,
        acceleration=a,
        torque=recording.channels["joint.effort"].values[:-1],
        **base,
    )
    balance = fit_terms(model, samples)
    within, misses = 0, []
    for p in balance.result.parameters:
        joint, term = p.name.split(".")
        true = getattr(truth[joint], term)
        if abs(p.estimate - true) <= WITHIN * true:
            within += 1
        else:
            misses.append(f"{p.name} {p.estimate:.4g} vs {true:.4g}")
    return within, len(balance.result.parameters), misses


@needs_sim
class SyntheticRecovery(unittest.TestCase):
    def test_the_fixed_base_chirp_recovers_every_joint(self) -> None:
        within, total, misses = _recover("fixed")
        self.assertEqual(total, 36)
        self.assertGreaterEqual(within, MIN_WITHIN_FIXED, misses)

    def test_the_shaken_base_chirp_recovers_through_the_base_state(self) -> None:
        within, total, misses = _recover("shaken")
        self.assertEqual(total, 36)
        self.assertGreaterEqual(within, MIN_WITHIN_SHAKEN, misses)


class TickAligned(unittest.TestCase):
    def test_one_sample_fewer_and_the_forward_difference(self) -> None:
        from rq_pipeline.robot.torque_balance import (  # noqa: PLC0415
            Smoothing,
            tick_aligned,
        )

        times = np.arange(0, 2.0, 0.004)
        position = np.column_stack([np.sin(2 * np.pi * 1.0 * times), 0.5 * times**2])
        t, q, v, a = tick_aligned(times, position, smoothing=Smoothing(window_s=0.02))
        self.assertEqual(len(t), len(times) - 1)
        self.assertEqual(q.shape, (len(times) - 1, 2))
        # The quadratic's acceleration is 1 everywhere, the sine's velocity
        # amplitude is 2*pi; both within the derivative's few-percent bias.
        self.assertAlmostEqual(float(np.median(a[10:-10, 1])), 1.0, delta=0.05)
        self.assertAlmostEqual(
            float(np.max(np.abs(v[10:-10, 0]))), 2 * np.pi, delta=0.2
        )

    def test_refuses_a_window_too_short_for_a_cubic(self) -> None:
        from rq_pipeline.robot.torque_balance import (  # noqa: PLC0415
            Smoothing,
            tick_aligned,
        )

        times = np.arange(0, 1.0, 0.1)
        with self.assertRaisesRegex(ValueError, "too few for a cubic"):
            tick_aligned(times, np.zeros((10, 1)), smoothing=Smoothing(window_s=0.2))


@needs_sim
class Verdicts(unittest.TestCase):
    def test_an_estimate_on_its_bound_is_flagged_and_never_pinned(self) -> None:
        """A hinge with no friction at all under a chirp: the Coulomb term
        lands on its lower bound, and a bound hit is not a measurement."""
        import mujoco  # noqa: PLC0415

        from rq_pipeline.robot.torque_balance import (  # noqa: PLC0415
            JointSamples,
            fit_terms,
            tick_aligned,
        )

        xml = """
        <mujoco><option timestep="0.001" integrator="RK4"/>
        <worldbody><body pos="0 0 1">
        <joint name="j" axis="0 1 0" damping="0.2" armature="0.01"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.02" mass="1"/>
        </body></worldbody>
        <actuator><motor name="m" joint="j" gear="1"/></actuator></mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        n, hz = 3000, 250
        times = np.arange(n) / hz
        q, v, tau = np.zeros((n, 1)), np.zeros((n, 1)), np.zeros((n, 1))
        for i in range(n):
            target = 0.6 * np.sin(
                2 * np.pi * (0.3 + 2.0 * times[i] / times[-1]) * times[i]
            )
            data.ctrl[0] = 30.0 * (target - data.qpos[0]) - 0.5 * data.qvel[0]
            mujoco.mj_forward(model, data)
            q[i], v[i], tau[i] = data.qpos[0], data.qvel[0], data.actuator_force[0]
            for _ in range(4):
                mujoco.mj_step(model, data)
        t, qm, vm, am = tick_aligned(times, q)
        samples = JointSamples(t, ("j",), qm, vm, am, tau[:-1])
        balance = fit_terms(model, samples)
        by_name = {p.name: p for p in balance.result.parameters}
        friction = by_name["j.frictionloss"]
        self.assertTrue(friction.at_bound)
        self.assertFalse(friction.pinned)
        self.assertIn("AT BOUND", balance.result.summary())
        self.assertAlmostEqual(by_name["j.damping"].estimate, 0.2, delta=0.02)
        self.assertAlmostEqual(by_name["j.armature"].estimate, 0.01, delta=0.002)

    def test_refuses_too_few_usable_samples(self) -> None:
        import mujoco  # noqa: PLC0415

        from rq_pipeline.robot.torque_balance import (  # noqa: PLC0415
            JointSamples,
            fit_terms,
        )

        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body><joint name="j" axis="0 1 0"/>'
            '<geom type="sphere" size="0.05" mass="1"/></body></worldbody></mujoco>'
        )
        n = 20
        samples = JointSamples(
            np.arange(n) / 100.0,
            ("j",),
            np.zeros((n, 1)),
            np.full((n, 1), 2.0),
            np.zeros((n, 1)),
            np.zeros((n, 1)),
        )
        with self.assertRaisesRegex(ValueError, "fewer than two bootstrap blocks"):
            fit_terms(model, samples)
