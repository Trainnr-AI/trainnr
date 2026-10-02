"""The stills tool's pure parts: which checkpoints it picks, and the
sheet it lays them on. The rollout and the render need the GPU venv
and are exercised by running the tool on a run."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from trainnr_mjlab.walk_stills import (
    FELL,
    SHEET_FILE,
    SURVIVED_OFF_COMMAND,
    SURVIVED_TRACKING,
    checkpoints_every,
    outcome_colour,
    write_sheet,
)


class CheckpointSelection(unittest.TestCase):
    def test_every_hundredth_and_the_last(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        for n in (0, 50, 100, 150, 200, 250, 1234):
            (tmp / f"model_{n}.pt").write_bytes(b"w")
        picked = [n for n, _ in checkpoints_every(tmp, 100)]
        self.assertEqual(picked, [0, 100, 200, 1234])

    def test_no_checkpoints(self) -> None:
        self.assertEqual(checkpoints_every(Path(tempfile.mkdtemp()), 100), [])


class TheSheet(unittest.TestCase):
    def test_colours_by_outcome(self) -> None:
        self.assertEqual(outcome_colour({"fell": True, "tracked": False}), FELL)
        self.assertEqual(
            outcome_colour({"fell": False, "tracked": True}), SURVIVED_TRACKING
        )
        self.assertEqual(
            outcome_colour({"fell": False, "tracked": False}), SURVIVED_OFF_COMMAND
        )

    def test_every_still_lands_on_the_page(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        rows = []
        outcomes = {0: (False, False), 100: (True, False), 200: (False, True)}
        for n, (fell, tracked) in outcomes.items():
            Image.new("RGB", (48, 36), (n, 0, 0)).save(tmp / f"model_{n}.png")
            rows.append(
                {
                    "iteration": n,
                    "file": f"model_{n}.png",
                    "fell": fell,
                    "tracked": tracked,
                }
            )
        out = write_sheet(rows, tmp)
        self.assertEqual(out, tmp / SHEET_FILE)
        page = Image.open(out)
        self.assertEqual(page.width, 3 * 240)
        (tmp / "stills.json").write_text(json.dumps(rows))
