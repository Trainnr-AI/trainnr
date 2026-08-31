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
from rq_pipeline.robot.friction_budget import (
    FrictionParams,
    friction_torque_budget,
)

from rq_mjlab.kernel import LawParams, friction_budget

VISCOUS = 0.03  # pipeline-side only; the kernel delegates it to dof_damping
PARAMS = {
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

# The grid: both signs, dominance both ways, the tie, zero, and a
# spread of velocities including rest (Stribeck envelope = 1).
TORQUES = (-5.0, -1.0, 0.0, 1.0, 5.0)
VELOCITIES = (0.0, 0.1, 2.0)


class OneLawOneSpelling(unittest.TestCase):
    def test_torch_kernel_matches_numpy_reference_on_the_grid(self) -> None:
        law = LawParams(kt=1.0, R=1.0, **PARAMS)
        numpy_params = FrictionParams(friction_viscous=VISCOUS, **PARAMS)
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


if __name__ == "__main__":
    unittest.main()
