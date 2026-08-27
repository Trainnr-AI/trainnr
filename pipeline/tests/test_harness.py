"""Gate A end to end in simulation: policies → scores → join → certificate.

A gravity-loaded pendulum must be swung to a target angle. Four scripted
policies of visibly different competence are scored under the identical
paired-trial protocol; their sim ranking must come out in competence
order, and the joined certificate must behave the way the statistics
promise at n=4: honest FAIL on the interval gate, small exact p.
"""

import unittest

from tests._extras import needs_sim

PENDULUM = """
<mujoco>
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <worldbody>
    <body>
      <joint name="arm" type="hinge" axis="0 1 0" damping="0.02"/>
      <geom type="capsule" fromto="0 0 0  0.3 0 0" size="0.02" mass="0.2"/>
    </body>
  </worldbody>
  <actuator><motor joint="arm" gear="1"/></actuator>
  <sensor><jointpos joint="arm"/><jointvel joint="arm"/></sensor>
</mujoco>
"""

INERT_SCENE = """
<mujoco>
  <worldbody><geom type="plane" size="1 1 0.1"/></worldbody>
</mujoco>
"""

TARGET_ANGLE = 0.8
TOLERANCE = 0.15
# Gravity torque scale at horizontal: mass x g x lever of the capsule's
# centre of mass. Used by the compensating policies below.
_MGL = 0.2 * 9.81 * 0.15
# FULLPHYSICS state layout is [time, qpos..., qvel..., act...]; index 1 is
# the single hinge's angle.
_QPOS_INDEX = 1


def _pd_controller(kp: float, kd: float, gravity_comp: float = 1.0):
    """PD over the two sensors (angle, velocity) plus gravity feedforward.

    Gains are chosen for DISCRETE-loop stability: a velocity gain held
    for T=50 ms on inertia I=0.006 multiplies velocity by (1 - kd*T/I)
    per control tick, so kd must stay below I/T = 0.12. This suite
    learned that twice: its first run showed pure P at kp=3 diverging
    to 1e6 rad, and the fourth review (R7) found its original kd=0.3
    design was only stable BECAUSE of the one-tick sensor-lag bug —
    fixing the lag exposed the genuinely unstable controller. Sampled
    dynamics are exactly the fidelity the harness exists to capture.
    """
    from math import cos  # noqa: PLC0415

    def act(step, sensordata):
        angle, velocity = sensordata[0], sensordata[1]
        return [
            kp * (TARGET_ANGLE - angle)
            - kd * velocity
            - gravity_comp * _MGL * cos(angle)
        ]

    return act


def _perturb(trial, home):
    initial = home.copy()
    initial[_QPOS_INDEX] = -0.3 + 0.15 * trial
    return initial


def _settled_near_target(states, sensors):
    tail = sensors[-50:, 0]
    return bool(abs(tail.mean() - TARGET_ANGLE) < TOLERANCE)


@needs_sim
class GateAInSimulation(unittest.TestCase):
    def _scores(self):
        from rq_pipeline.evaluate.harness import (  # noqa: PLC0415
            EpisodeProtocol,
            SimPolicy,
            evaluate_policies,
        )
        from rq_pipeline.physics.mujoco_backend import MuJoCoBackend  # noqa: PLC0415

        backend = MuJoCoBackend()
        backend.load_mjcf_string(PENDULUM)
        # Competence order by proportional gain: with identical gravity
        # feedforward, weaker kp converges slower, so fewer of the paired
        # starts settle within the episode. Probed with aligned
        # observations 2026-08-23: 6/2/1/0 successes out of 6.
        policies = [
            SimPolicy("tuned", _pd_controller(1.0, 0.08)),
            SimPolicy("soft", _pd_controller(0.13, 0.10)),
            SimPolicy("sluggish", _pd_controller(0.10, 0.10)),
            SimPolicy("reversed", _pd_controller(-0.8, 0.08, gravity_comp=0.0)),
        ]
        protocol = EpisodeProtocol(
            trials=6,
            steps=150,
            control_interval=5,
            perturb=_perturb,
            success=_settled_near_target,
        )
        return evaluate_policies(
            backend, policies, protocol, source="pendulum-test@000000000000"
        )

    def test_ranking_matches_competence_and_certificate_is_honest(self) -> None:
        from rq_pipeline.evaluate.certificate import certify  # noqa: PLC0415
        from rq_pipeline.evaluate.harness import join_with_real  # noqa: PLC0415

        scores = self._scores()
        by_name = {score.name: score.score for score in scores}
        self.assertEqual(by_name["tuned"], 1.0)
        self.assertEqual(by_name["reversed"], 0.0)
        self.assertGreater(by_name["tuned"], by_name["soft"])
        self.assertGreater(by_name["soft"], by_name["sluggish"])
        self.assertGreater(by_name["sluggish"], by_name["reversed"])

        real = {
            "tuned": (44, 50),
            "soft": (31, 50),
            "sluggish": (18, 50),
            "reversed": (4, 50),
        }
        certificate = certify(
            robot_bundle="pendulum-test@000000000000",
            scene_bundle="bench@000000000000",
            outcomes=join_with_real(scores, real),
            gate_threshold=0.5,
            instrument="mujoco-cpu",
        )
        # Four policies in perfect agreement: the interval gate must still
        # FAIL (n=4 cannot certify strength), while the exact permutation
        # p says the agreement itself is unlikely to be luck (1/24).
        self.assertFalse(certificate.gate_passed)
        self.assertAlmostEqual(certificate.exact_p_value, 1.0 / 24.0)
        self.assertGreater(certificate.top_pick, 0.5)
        # The sim trial count reaches the certificate (docs/32 §6).
        self.assertEqual({r.sim_trials for r in certificate.policies}, {6})

    def test_records_are_written_as_trials_finish_and_fold_back(self) -> None:
        import tempfile  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        from rq_pipeline.evaluate.harness import (  # noqa: PLC0415
            EpisodeProtocol,
            SimPolicy,
            evaluate_policies,
        )
        from rq_pipeline.evaluate.records import fold, read_records  # noqa: PLC0415
        from rq_pipeline.physics.mujoco_backend import MuJoCoBackend  # noqa: PLC0415

        backend = MuJoCoBackend()
        backend.load_mjcf_string(PENDULUM)
        protocol = EpisodeProtocol(
            trials=3,
            steps=20,
            control_interval=5,
            perturb=_perturb,
            success=_settled_near_target,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.jsonl"
            scores = evaluate_policies(
                backend,
                [SimPolicy("limp", lambda _step, _sense: [0.0])],
                protocol,
                source="pendulum-test@000000000000",
                record_to=path,
            )
            records = read_records(path)
        self.assertEqual([r.trial for r in records], [0, 1, 2])
        self.assertEqual(records[0].steps, 20)
        self.assertTrue(records[0].instrument.startswith("mujoco-"))
        self.assertEqual(records[0].protocol["control_interval"], 5)
        self.assertEqual(fold(records), scores)

    def test_dead_model_is_refused_before_any_episode(self) -> None:
        from rq_pipeline.evaluate.harness import (  # noqa: PLC0415
            EpisodeProtocol,
            SimPolicy,
            evaluate_policies,
        )
        from rq_pipeline.physics.mujoco_backend import MuJoCoBackend  # noqa: PLC0415
        from rq_pipeline.robot.model_checks import DeadModelError  # noqa: PLC0415

        backend = MuJoCoBackend()
        backend.load_mjcf_string(INERT_SCENE)
        protocol = EpisodeProtocol(
            trials=1,
            steps=10,
            control_interval=1,
            perturb=lambda _trial, home: home,
            success=lambda _states, _sensors: True,
        )
        with self.assertRaises(DeadModelError):
            evaluate_policies(
                backend,
                [SimPolicy("any", lambda _step, _sense: [])],
                protocol,
                source="inert@000000000000",
            )

    def test_unstamped_source_refused_before_any_episode(self) -> None:
        from rq_pipeline.evaluate.harness import (  # noqa: PLC0415
            EpisodeProtocol,
            SimPolicy,
            evaluate_policies,
        )
        from rq_pipeline.physics.mujoco_backend import MuJoCoBackend  # noqa: PLC0415

        backend = MuJoCoBackend()
        backend.load_mjcf_string(PENDULUM)
        protocol = EpisodeProtocol(
            trials=1,
            steps=10,
            control_interval=1,
            perturb=lambda _trial, home: home,
            success=lambda _states, _sensors: True,
        )
        with self.assertRaises(ValueError):
            evaluate_policies(
                backend,
                [SimPolicy("any", lambda _step, _sense: [0.0])],
                protocol,
                source="pendulum-unstamped",
            )

    def test_mismatched_policy_sets_are_refused(self) -> None:
        from rq_pipeline.evaluate.harness import (  # noqa: PLC0415
            SimScore,
            join_with_real,
        )

        scores = [SimScore("a", 5, 10), SimScore("b", 3, 10)]
        with self.assertRaises(ValueError):
            join_with_real(scores, {"a": (5, 10)})
        with self.assertRaises(ValueError):
            join_with_real(scores, {"a": (5, 10), "b": (3, 10), "c": (1, 10)})


if __name__ == "__main__":
    unittest.main()
