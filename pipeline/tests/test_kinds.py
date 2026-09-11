"""The kind rules as one table, the project's JSON doors, the checkout
resolver and the twist-envelope wording (review 2026-09-12)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rq_pipeline.envs.lerobot_train_log import CHECKPOINT_WEIGHTS
from rq_pipeline.evaluate.commands import describe_twist
from rq_pipeline.paths import CHECKOUT_ENV, CHECKOUT_MARKER, checkout
from rq_pipeline.project.files import read_json, write_json
from rq_pipeline.project.index import UNRECORDED, interval_of, ratio_of
from rq_pipeline.project.kinds import (
    FILE_RULES,
    FITS_DIR,
    MARKERS,
    RULES,
    Kind,
    UnknownKindError,
    detect,
)


class TheRules(unittest.TestCase):
    def test_every_kind_but_finding_has_a_row_and_the_markers_follow(self) -> None:
        self.assertEqual(set(RULES) | {Kind.FINDING}, set(Kind))
        self.assertEqual(
            set(MARKERS), {k for k, r in RULES.items() if not r[0].fallback}
        )
        self.assertEqual(
            {r.kind for r in FILE_RULES}, {Kind.POLICY, Kind.FINDING, Kind.FIT}
        )

    def test_single_files_detect_by_the_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / FITS_DIR).mkdir()
            (root / "findings").mkdir()
            for path, kind in (
                (root / "model_7999.pt", Kind.POLICY),
                (root / CHECKPOINT_WEIGHTS, Kind.POLICY),
                (root / FITS_DIR / "a.json", Kind.FIT),
                (root / "findings" / "f.json", Kind.FINDING),
            ):
                path.write_text("{}")
                self.assertIs(detect(path), kind, path.name)
            (root / "notes.txt").write_text("")
            with self.assertRaises(UnknownKindError):
                detect(root / "notes.txt")

    def test_a_fallback_marker_yields_to_a_named_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "rl"
            root.mkdir()
            (root / "identity.json").write_text("{}")
            (root / "model_1.pt").write_bytes(b"\x00")
            self.assertIs(detect(root), Kind.RUN)


class TheJsonDoors(unittest.TestCase):
    def test_missing_ok_reads_empty_and_a_write_is_atomic_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "deep" / "r.json"
            self.assertEqual(read_json(path, missing_ok=True), {})
            with self.assertRaises(FileNotFoundError):
                read_json(path)
            write_json(path, {"word": "ünïcode"})
            self.assertEqual(read_json(path)["word"], "ünïcode")
            self.assertEqual(sorted(p.name for p in path.parent.iterdir()), ["r.json"])


class TheReaders(unittest.TestCase):
    def test_interval_and_ratio_never_invent(self) -> None:
        self.assertEqual(interval_of({"ci95": [0.1, 0.9]}), (0.1, 0.9))
        self.assertEqual(interval_of({"ci": [0.0, 1.0]}), (0.0, 1.0))
        self.assertIsNone(interval_of({"ci95": [0.5]}))
        self.assertIsNone(interval_of({}))
        self.assertEqual(ratio_of({"successes": 3, "trials": 4}), "3 / 4")
        self.assertEqual(ratio_of({"trials": 4}), UNRECORDED)
        self.assertEqual(ratio_of({"a": 1, "n": 2}, "a", "n"), "1 / 2")


class TheCheckout(unittest.TestCase):
    def test_the_variable_wins_and_the_marker_is_found_otherwise(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.dict(os.environ, {CHECKOUT_ENV: tmp}),
        ):
            self.assertEqual(checkout(), Path(tmp).resolve())
        with mock.patch.dict(os.environ, {CHECKOUT_ENV: ""}):
            self.assertTrue((checkout() / CHECKOUT_MARKER).is_file())


class TheTwist(unittest.TestCase):
    def test_the_envelope_in_words(self) -> None:
        self.assertEqual(
            describe_twist({"lin_vel_x": [-1.5, 2.0], "ang_vel_z": [-0.7, 0.7]}),
            "forward -1.5 to 2 m/s, turn ±0.7 rad/s",
        )
        self.assertEqual(
            describe_twist({"lin_vel_x": [0, 1], "lin_vel_y": [-0.5, 0.5]}),
            "forward 0 to 1 m/s, sideways -0.5 to 0.5 m/s",
        )
        self.assertIsNone(describe_twist({"other": [0, 1]}))
        self.assertIsNone(describe_twist(None))
