"""The bundle profile: one source of truth per robot, loudly validated."""

import json
import tempfile
import unittest
from pathlib import Path

from rq_pipeline.bundles.profile import PROFILE_FILE, RobotProfile, load_profile

REPO_ROOT = Path(__file__).resolve().parents[2]
RIG_BUNDLE = REPO_ROOT / "robots" / "rig-drivetrain"


def _valid_raw() -> dict:
    return {
        "name": "test-bot",
        "ticks_per_revolution": 960.0,
        "camera_fps": 4,
        "servo_pulse_floor_us": 1100.0,
        "servo_pulse_ceiling_us": 1900.0,
        "duty_ceiling_percent": 100.0,
    }


class RigBundleProfile(unittest.TestCase):
    def test_the_committed_rig_profile_loads(self) -> None:
        profile = load_profile(RIG_BUNDLE)
        self.assertEqual(profile.name, "rig-drivetrain")
        self.assertEqual(profile.model_file, "model.xml")
        self.assertTrue((RIG_BUNDLE / profile.model_file).exists())
        # The unverified number must SAY it is unverified, in the file
        # itself — the whole point of provenance-per-value.
        self.assertIn("UNVERIFIED", profile.provenance["ticks_per_revolution"])


class Validation(unittest.TestCase):
    def _load(self, raw: dict) -> RobotProfile:
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / PROFILE_FILE).write_text(json.dumps(raw))
            return load_profile(Path(directory))

    def test_valid_profile_loads(self) -> None:
        self.assertEqual(self._load(_valid_raw()).camera_fps, 4)

    def test_unknown_field_is_an_error_not_a_default(self) -> None:
        raw = _valid_raw()
        raw["ticks_per_revoluton"] = raw.pop("ticks_per_revolution")  # typo
        with self.assertRaises(ValueError):
            self._load(raw)

    def test_nonpositive_values_refused(self) -> None:
        for field_name in (
            "ticks_per_revolution",
            "camera_fps",
            "duty_ceiling_percent",
        ):
            raw = _valid_raw()
            raw[field_name] = 0
            with self.assertRaises(ValueError):
                self._load(raw)

    def test_empty_servo_band_refused(self) -> None:
        raw = _valid_raw()
        raw["servo_pulse_floor_us"] = 1900.0
        raw["servo_pulse_ceiling_us"] = 1100.0
        with self.assertRaises(ValueError):
            self._load(raw)

    def test_provenance_for_unknown_field_refused(self) -> None:
        raw = _valid_raw()
        raw["provenance"] = {"gear_ratio": "not a profile field"}
        with self.assertRaises(ValueError):
            self._load(raw)


if __name__ == "__main__":
    unittest.main()
