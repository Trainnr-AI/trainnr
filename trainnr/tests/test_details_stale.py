"""A detail is a view: it is rewritten when the artifact changes under
it (a review landing beside a task), and a task shows one Acceptance."""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from trainnr.project import create_project, index_project, write_index
from trainnr.project.details import details_path
from trainnr.project.previews import _first_sentence


class StaleDetails(unittest.TestCase):
    def test_a_task_detail_is_rewritten_when_acceptance_lands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = create_project(Path(tmp) / "p", "p")
            task = project.root / "tasks" / "t"
            task.mkdir(parents=True)
            (task / "task.json").write_text(
                json.dumps(
                    {
                        "task_id": "acme/none",
                        "stamp": "none@0123456789ab",
                        "kind": "declared",
                    }
                )
            )
            write_index(project, index_project(project))
            stamp = index_project(project).artifacts[0].stamp
            first = json.loads(details_path(project, stamp).read_text())
            titles = [s["title"] for s in first["sections"]]
            self.assertEqual(titles.count("Acceptance"), 1)
            self.assertEqual(first["sections"][1]["rows"][0], ["verdict", "unreviewed"])
            # The review lands later; the detail file is older than the folder now.
            later = time.time() + 5
            (task / "acceptance.json").write_text(
                json.dumps(
                    {
                        "accepted": True,
                        "expert_successes": 4,
                        "trials": 4,
                        "floor_successes": 0,
                        "funnel": {"expert": [4, 4]},
                        "milestones": ["moved", "placed"],
                    }
                )
            )
            os.utime(task / "acceptance.json", (later, later))
            write_index(project, index_project(project))
            second = json.loads(details_path(project, stamp).read_text())
            acceptance = [s for s in second["sections"] if s["title"] == "Acceptance"]
            self.assertEqual(len(acceptance), 1)
            rows = dict((k, v) for k, v in acceptance[0]["rows"])
            self.assertEqual(rows["verdict"], "accepted")
            self.assertEqual(rows["funnel · expert"], "moved 4/4 · placed 4/4")

    def test_first_sentence_stops_at_a_sentence_not_a_decimal(self) -> None:
        self.assertEqual(
            _first_sentence("Rate 0.53 held. Then not."), "Rate 0.53 held."
        )
        self.assertEqual(_first_sentence("No stop here"), "No stop here")


if __name__ == "__main__":
    unittest.main()
