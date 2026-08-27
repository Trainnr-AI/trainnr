"""The recreated Paper 0 producer must reproduce its own committed fits.

The 2026-08-25 product review found the flagship records were written by
an uncommitted scratchpad script. drivetrain_fit.py recreates it; this
test pins the recreation to the committed estimates so the repo can
always reproduce its own deliverable. Tolerance is optimizer-noise wide
(different initial guesses, same optimum), not physics wide.
"""

import json
import unittest
from pathlib import Path

from tests._extras import needs_sim

REPO = Path(__file__).resolve().parents[2]
BUNDLE = REPO / "robots" / "rig-drivetrain"
RECORDING = REPO / "recordings" / "sweep-2026-08-24-b.wire"
COMMITTED = BUNDLE / "fits" / "sweep-2026-08-24-b@1f1e7cd6f0bf.json"


@needs_sim
class ReproduceCommittedFit(unittest.TestCase):
    def test_sweep_b_estimates_match_the_committed_record(self) -> None:
        from rq_pipeline.robot.drivetrain_fit import fit_drivetrain  # noqa: PLC0415

        result, path = fit_drivetrain(BUNDLE, RECORDING, write=False)
        self.assertIsNone(path)
        committed = {
            parameter["name"]: parameter
            for parameter in json.loads(COMMITTED.read_text(encoding="utf-8"))[
                "parameters"
            ]
        }
        for parameter in result.parameters:
            if parameter.name == "scale_ref_damping":
                # The anchor: fixed by hand, unbounded interval by
                # construction — nothing to reproduce.
                self.assertFalse(parameter.pinned)
                continue
            expected = committed[parameter.name]["estimate"]
            relative = abs(parameter.estimate - expected) / abs(expected)
            self.assertLess(relative, 0.01, f"{parameter.name}: {relative:.4f}")

    def test_committed_records_are_strict_json_with_full_provenance(self) -> None:
        for record_path in sorted((BUNDLE / "fits").glob("sweep-*.json")):
            raw = json.loads(
                record_path.read_text(encoding="utf-8"), parse_constant=self._refuse
            )
            for key in ("profile", "model", "code", "units", "pinned_criterion"):
                self.assertIn(key, raw, f"{record_path.name} missing {key}")
                self.assertIsNotNone(raw[key], f"{record_path.name} {key} is null")
            for name in ("profile", "model"):
                self.assertIn("@", raw[name], f"{record_path.name} {name} unstamped")

    @staticmethod
    def _refuse(token: str) -> None:
        raise AssertionError(f"non-RFC-8259 token in a fit record: {token}")


if __name__ == "__main__":
    unittest.main()
