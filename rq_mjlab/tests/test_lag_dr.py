"""The command lag drawn in training (rq_mjlab.lag_dr)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace


def _cfg(*, identified: int = 0):
    from mjlab.actuator import BuiltinPositionActuatorCfg  # noqa: PLC0415

    actuators = tuple(
        BuiltinPositionActuatorCfg(
            target_names_expr=(name,),
            stiffness=20.0,
            damping=0.5,
            effort_limit=23.7,
            delay_min_lag=identified,
            delay_max_lag=identified,
        )
        for name in (".*hip_.*", ".*calf_.*")
    )
    return SimpleNamespace(
        sim=SimpleNamespace(mujoco=SimpleNamespace(timestep=0.005)),
        scene=SimpleNamespace(
            entities={
                "robot": SimpleNamespace(
                    articulation=SimpleNamespace(actuators=actuators)
                )
            }
        ),
    )


class TheLagDraw(unittest.TestCase):
    def test_milliseconds_to_physics_steps_round_up(self) -> None:
        from rq_mjlab.lag_dr import lag_steps  # noqa: PLC0415

        self.assertEqual(lag_steps(40, 0.005), 8)  # two 20 ms control ticks
        self.assertEqual(lag_steps(41, 0.005), 9)  # a part step still covered
        self.assertEqual(lag_steps(0, 0.005), 0)
        with self.assertRaisesRegex(ValueError, "not negative"):
            lag_steps(-1, 0.005)

    def test_every_actuator_draws_the_lag_redrawn_every_second(self) -> None:
        from rq_mjlab.lag_dr import with_command_lag  # noqa: PLC0415

        cfg = _cfg()
        identity = with_command_lag(cfg, {"dr_basis": "gains ±0.1"}, 40)
        for actuator in cfg.scene.entities["robot"].articulation.actuators:
            self.assertEqual((actuator.delay_min_lag, actuator.delay_max_lag), (0, 8))
            self.assertEqual(actuator.delay_update_period, 200)  # 1 s of 5 ms steps
            self.assertTrue(actuator.delay_per_env_phase)
        self.assertIn(
            "command lag 0-40 ms (0-8 physics steps of 5 ms)", identity["lag_dr"]
        )
        self.assertTrue(identity["dr_basis"].startswith("gains ±0.1; command lag"))

    def test_on_top_of_the_identified_delay(self) -> None:
        from rq_mjlab.lag_dr import with_command_lag  # noqa: PLC0415

        cfg = _cfg(identified=2)
        with_command_lag(cfg, {"dr_basis": "x"}, 40)
        actuator = cfg.scene.entities["robot"].articulation.actuators[0]
        self.assertEqual((actuator.delay_min_lag, actuator.delay_max_lag), (2, 10))

    def test_zero_changes_nothing(self) -> None:
        from rq_mjlab.lag_dr import with_command_lag  # noqa: PLC0415

        cfg = _cfg()
        before = cfg.scene.entities["robot"].articulation.actuators
        identity = {"dr_basis": "x"}
        self.assertIs(with_command_lag(cfg, identity, 0), identity)
        self.assertIs(cfg.scene.entities["robot"].articulation.actuators, before)


if __name__ == "__main__":
    unittest.main()
