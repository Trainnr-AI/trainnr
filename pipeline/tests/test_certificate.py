"""End-to-end: bundles in, signable certificate out."""

import json
import tempfile
import unittest
from pathlib import Path

from rq_pipeline.bundles.hashing import stamp
from rq_pipeline.evaluate.certificate import PolicyOutcome, certify

# Twelve policies with strong-but-imperfect sim/real agreement — the
# realistic shape of a passing Gate A run at the sample size the defect
# register says the gate actually needs. Sim successes out of 100 trials.
SIM_SUCCESSES = [15, 22, 28, 35, 41, 48, 55, 61, 68, 74, 81, 88]
SIM_TRIALS = 100
REAL_SUCCESSES = [8, 12, 13, 19, 20, 26, 27, 24, 33, 38, 40, 44]
OUTCOMES = [
    PolicyOutcome(f"policy-{index:02d}", sim, SIM_TRIALS, successes, 50)
    for index, (sim, successes) in enumerate(
        zip(SIM_SUCCESSES, REAL_SUCCESSES, strict=True)
    )
]
GATE_THRESHOLD = 0.5


def _stamped_bundle(root: Path, name: str) -> str:
    root.mkdir()
    (root / "model.xml").write_text(f"<mujoco model='{name}'/>")
    return stamp(name, root)


class CertifyEndToEnd(unittest.TestCase):
    def test_full_run_produces_auditable_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            robot = _stamped_bundle(Path(directory) / "robot", "toy-arm")
            scene = _stamped_bundle(Path(directory) / "scene", "toy-lab")
            certificate = certify(
                robot_bundle=robot,
                scene_bundle=scene,
                outcomes=OUTCOMES,
                gate_threshold=GATE_THRESHOLD,
                physics_backend="mujoco-3.11.0",
            )
        self.assertTrue(certificate.gate_passed)
        self.assertEqual(certificate.physics_backend, "mujoco-3.11.0")
        self.assertEqual(certificate.policy_count, 12)
        self.assertGreaterEqual(certificate.rank_lower, GATE_THRESHOLD)
        # Twelve policies exceed the 8-policy enumeration limit, so no
        # exact p is claimed — Fisher-z carries the claim alone.
        self.assertIsNone(certificate.exact_p_value)
        self.assertGreater(certificate.top_pick, 0.5)
        # Every policy carries its own real-side interval AND its sim n —
        # the count, never just the rate (docs/32 §6).
        for result in certificate.policies:
            self.assertLess(result.real_lower, result.real_upper)
            self.assertEqual(result.sim_trials, SIM_TRIALS)
        # The artifact is machine-readable and self-describing.
        parsed = json.loads(certificate.to_json())
        self.assertEqual(parsed["robot_bundle"], robot)
        self.assertEqual(len(parsed["policies"]), 12)
        self.assertEqual(parsed["policies"][0]["sim_successes"], SIM_SUCCESSES[0])
        self.assertIn("PASS", certificate.summary())

    def test_counts_are_validated_on_construction(self) -> None:
        with self.assertRaises(ValueError):
            PolicyOutcome("p", 5, 4, 1, 10)  # more sim successes than trials
        with self.assertRaises(ValueError):
            PolicyOutcome("p", 1, 4, 1, 0)  # no real trials

    def test_five_policies_cannot_buy_a_pass_with_agreement(self) -> None:
        # The C1 lesson, now enforced by the artifact itself: near-perfect
        # observed agreement over five policies fails a 0.5 lower-bound
        # gate, because n = 5 cannot support the claim.
        with tempfile.TemporaryDirectory() as directory:
            robot = _stamped_bundle(Path(directory) / "robot", "toy-arm")
            scene = _stamped_bundle(Path(directory) / "scene", "toy-lab")
            certificate = certify(
                robot_bundle=robot,
                scene_bundle=scene,
                outcomes=OUTCOMES[:5],
                gate_threshold=GATE_THRESHOLD,
            )
        self.assertFalse(certificate.gate_passed)
        self.assertIn("FAIL", certificate.summary())
        # The R2 fix: at n <= 8 the certificate carries the exact
        # permutation p alongside the interval, and it says something the
        # FAIL verdict does not — the agreement itself is real (small p),
        # the sample is just too small to certify its strength.
        self.assertIsNotNone(certificate.exact_p_value)
        self.assertLess(certificate.exact_p_value, 0.05)
        self.assertIn("exact p=", certificate.summary())

    def test_unstamped_bundle_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            certify(
                robot_bundle="toy-arm",  # no @hash
                scene_bundle="toy-lab@abcdefabcdef",
                outcomes=OUTCOMES,
                gate_threshold=GATE_THRESHOLD,
            )
        self.assertIn("name@hash", str(caught.exception))

    def test_bad_gate_threshold_refused(self) -> None:
        with self.assertRaises(ValueError):
            certify(
                robot_bundle="a@000000000000",
                scene_bundle="b@000000000000",
                outcomes=OUTCOMES,
                gate_threshold=1.0,
            )


if __name__ == "__main__":
    unittest.main()
