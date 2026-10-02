"""One law, one spelling: the torch kernel's friction budget against the
pipeline's numpy transcription, across the sign/tie grid where the two
copies disagreed until 2026-09-01's review (the numpy side lacked BAM's
sign gate and paid the tie). The reference both must match is
bam/model.py 172-205, read verbatim (docs/e2e-research/57 §2).

The kernel's budget excludes viscous (it rides `dof_damping`); the
pipeline's includes it — so the comparison adds `friction_viscous·|qd|`
to the kernel side, stating the known, deliberate split once.
"""

from __future__ import annotations

import itertools
import unittest

import torch
from trainnr.robot.friction_budget import (
    FrictionParams,
    friction_torque_budget,
)

from trainnr_mjlab.kernel import LawParams, friction_budget

VISCOUS = 0.03  # pipeline-side only; the kernel delegates it to dof_damping
# M5/M6: the DIRECTIONAL load family.
DIRECTIONAL = {
    "friction_base": 0.05,
    "friction_stribeck": 0.02,
    "dtheta_stribeck": 0.37,
    "alpha": 1.8,
    "load_friction_motor": 0.04,
    "load_friction_external": 0.24,
    "load_friction_motor_stribeck": 0.01,
    "load_friction_external_stribeck": 0.14,
    "load_friction_motor_quad": 0.007,
    "load_friction_external_quad": 0.003,
}

# M3/M4: the UNDIRECTED load family — the third of the store the torch
# kernel could not express at all until 2026-09-01's second review.
UNDIRECTED = {
    "friction_base": 0.05,
    "friction_stribeck": 0.02,
    "dtheta_stribeck": 0.37,
    "alpha": 1.8,
    "load_friction_base": 0.06,
    "load_friction_stribeck": 0.09,
}

# The grid: both signs, dominance both ways, the tie, zero, and a
# spread of velocities including rest (Stribeck envelope = 1).
TORQUES = (-5.0, -1.0, 0.0, 1.0, 5.0)
VELOCITIES = (0.0, 0.1, 2.0)


class OneLawOneSpelling(unittest.TestCase):
    def _assert_parity(self, params: dict) -> None:
        law = LawParams(kt=1.0, R=1.0, **params)
        numpy_params = FrictionParams(friction_viscous=VISCOUS, **params)
        for tau_m, tau_e, qd in itertools.product(TORQUES, TORQUES, VELOCITIES):
            with self.subTest(tau_m=tau_m, tau_e=tau_e, qd=qd):
                kernel = float(
                    friction_budget(
                        law,
                        torch.tensor([[tau_m]], dtype=torch.float64),
                        torch.tensor([[tau_e]], dtype=torch.float64),
                        torch.tensor([[qd]], dtype=torch.float64),
                    )
                ) + VISCOUS * abs(qd)
                reference = float(
                    friction_torque_budget(
                        numpy_params, theta_dot=qd, tau_m=tau_m, tau_e=tau_e
                    )
                )
                self.assertAlmostEqual(kernel, reference, places=9)

    def test_the_directional_family_m5_m6(self) -> None:
        self._assert_parity(DIRECTIONAL)

    def test_the_undirected_family_m3_m4(self) -> None:
        # 16 of the 48 committed bundles are m3/m4 — this family is a
        # third of the store, and the kernel had no field for it.
        self._assert_parity(UNDIRECTED)

    def test_scalable_and_rig_params_are_disjoint_across_trees(self) -> None:
        # The pipeline says "never jitter these"; SCALABLE says "draw
        # these per world". One parameter sat in both lists once
        # (error_gain_ratio — second review, 2026-09-01); this pin keeps
        # the two trees' physics claims from contradicting again.
        from trainnr.robot.actuator_bundle import (  # noqa: PLC0415
            RIG_AND_FIRMWARE_PARAMS,
        )

        from trainnr_mjlab.actuator import BamActuator  # noqa: PLC0415

        overlap = set(BamActuator.SCALABLE) & RIG_AND_FIRMWARE_PARAMS
        self.assertEqual(overlap, set(), f"both drawable and forbidden: {overlap}")

    def test_every_committed_bundle_builds_a_law(self) -> None:
        # The sweep the first pass lacked: a roster that refuses real
        # fits is an outage, not a safeguard (review 2026-09-01).
        import json  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        from trainnr_mjlab.actuator import _CONSUMED_KEYS  # noqa: PLC0415

        store = sorted(
            (Path(__file__).resolve().parents[2] / "robots" / "actuator-bundles").glob(
                "*.bundle.json"
            )
        )
        self.assertTrue(store, "no committed bundles to sweep")
        for path in store:
            with self.subTest(bundle=path.name):
                params = json.loads(path.read_text())["params"]
                unknown = sorted(set(params) - set(_CONSUMED_KEYS))
                self.assertEqual(unknown, [], f"{path.name} carries {unknown}")


if __name__ == "__main__":
    unittest.main()
