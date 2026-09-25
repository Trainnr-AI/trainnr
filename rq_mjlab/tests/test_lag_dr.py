"""The command lag drawn in training (rq_mjlab.lag_dr)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from rq_mjlab.walks import ROBOT_ENTITY, Identity


def _articulation(*, identified: int = 0):
    from mjlab.actuator import BuiltinPositionActuatorCfg  # noqa: PLC0415
    from mjlab.entity import EntityArticulationInfoCfg  # noqa: PLC0415

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
    return EntityArticulationInfoCfg(actuators=actuators)


def _cfg(*, identified: int = 0, articulation=None):
    return SimpleNamespace(
        sim=SimpleNamespace(mujoco=SimpleNamespace(timestep=0.005)),
        scene=SimpleNamespace(
            entities={
                ROBOT_ENTITY: SimpleNamespace(
                    articulation=articulation or _articulation(identified=identified)
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
        identity = with_command_lag(cfg, {Identity.DR_BASIS: "gains ±0.1"}, 40)
        for actuator in cfg.scene.entities[ROBOT_ENTITY].articulation.actuators:
            self.assertEqual((actuator.delay_min_lag, actuator.delay_max_lag), (0, 8))
            self.assertEqual(actuator.delay_update_period, 200)  # 1 s of 5 ms steps
            self.assertTrue(actuator.delay_per_env_phase)
        self.assertIn(
            "command lag 0-40 ms (0-8 physics steps of 5 ms)", identity[Identity.LAG_DR]
        )
        self.assertTrue(
            identity[Identity.DR_BASIS].startswith("gains ±0.1; command lag")
        )

    def test_on_top_of_the_identified_delay(self) -> None:
        from rq_mjlab.lag_dr import with_command_lag  # noqa: PLC0415

        cfg = _cfg(identified=2)
        with_command_lag(cfg, {Identity.DR_BASIS: "x"}, 40)
        actuator = cfg.scene.entities[ROBOT_ENTITY].articulation.actuators[0]
        self.assertEqual((actuator.delay_min_lag, actuator.delay_max_lag), (2, 10))

    def test_a_shared_articulation_is_never_edited(self) -> None:
        """Two configs sharing one articulation (the Go2's used to): the
        lag on the first must not reach the second (review 2026-09-26)."""
        from rq_mjlab.lag_dr import lag_steps, with_command_lag  # noqa: PLC0415

        shared = _articulation()
        first, second = _cfg(articulation=shared), _cfg(articulation=shared)
        with_command_lag(first, {Identity.DR_BASIS: "x"}, 40)
        self.assertTrue(all(a.delay_max_lag == 0 for a in shared.actuators))
        self.assertTrue(
            all(
                a.delay_max_lag == 0
                for a in second.scene.entities[ROBOT_ENTITY].articulation.actuators
            )
        )
        lagged = first.scene.entities[ROBOT_ENTITY].articulation.actuators
        steps = lag_steps(40, 0.005)
        self.assertTrue(all(a.delay_max_lag == steps for a in lagged))

    def test_zero_changes_nothing(self) -> None:
        from rq_mjlab.lag_dr import with_command_lag  # noqa: PLC0415

        cfg = _cfg()
        before = cfg.scene.entities[ROBOT_ENTITY].articulation.actuators
        identity = {Identity.DR_BASIS: "x"}
        self.assertIs(with_command_lag(cfg, identity, 0), identity)
        self.assertIs(cfg.scene.entities[ROBOT_ENTITY].articulation.actuators, before)


if __name__ == "__main__":
    unittest.main()
