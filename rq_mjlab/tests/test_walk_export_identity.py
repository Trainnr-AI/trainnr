"""The deployment manifest names the randomization its policy TRAINED
under, not the export env's own (the review of 2026-09-24: every Go2
`deploy.json` said "none: ... no actuator DR" while the run's identity
said "declared ±0.1 scale on kp, kd, armature")."""

from __future__ import annotations

import unittest

from rq_pipeline.deploy.manifest import Key

from rq_mjlab.walk_export import UNRECORDED, manifest_identity

BUILT = {
    Key.ROBOT: "go2@5003bf617b5f",
    Key.ACTUATOR: "unitree-go2-declared-pd@3b68245f8d2a",
    Key.DR_BASIS: "none: the reference's declared PD gains exactly (no actuator DR)",
    Key.TASK: "go2-walk@built",
}
TRAINED = {
    Key.ROBOT: "go2@5003bf617b5f",
    Key.ACTUATOR: "unitree-go2-declared-pd@3b68245f8d2a",
    Key.DR_BASIS: "declared ±0.1 scale on kp, kd, armature around the reference's "
    "declared constants",
    Key.SEED: None,
    Key.TASK: "go2-walk@0e7e123a7de7",
}


class TheManifestNamesTheTrainedRandomization(unittest.TestCase):
    def test_the_runs_basis_seed_and_task_win(self) -> None:
        out = manifest_identity(BUILT, TRAINED)
        self.assertEqual(out[Key.DR_BASIS], TRAINED[Key.DR_BASIS])
        self.assertEqual(out[Key.TASK], TRAINED[Key.TASK])
        self.assertIn(Key.SEED, out)
        self.assertEqual(out[Key.ROBOT], BUILT[Key.ROBOT])

    def test_a_run_without_the_record_says_unrecorded_never_none(self) -> None:
        out = manifest_identity(BUILT, {Key.TASK: "go2-walk@x"})
        self.assertEqual(out[Key.DR_BASIS], UNRECORDED)
        self.assertEqual(out[Key.TASK], "go2-walk@x")

    def test_a_run_without_a_task_keeps_the_built_one(self) -> None:
        out = manifest_identity(BUILT, {Key.DR_BASIS: TRAINED[Key.DR_BASIS]})
        self.assertEqual(out[Key.TASK], BUILT[Key.TASK])


if __name__ == "__main__":
    unittest.main()
