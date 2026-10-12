"""The release's notarization step (tools/notarize.py): it reads
notarytool's JSON reply and knows which files Apple takes as they are and
which can carry a stapled ticket. The submission itself runs only on the
release's macOS runner."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "notarize", REPO / "tools" / "notarize.py"
)
assert _SPEC is not None and _SPEC.loader is not None
notarize = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(notarize)


class Notarize(unittest.TestCase):
    def test_the_reply_gives_the_status_and_the_submission(self) -> None:
        reply = '{"id": "2efe2717-52ef-43a5-96dc-0797e4ca1041", "status": "Accepted"}'
        self.assertEqual(
            notarize.outcome(reply),
            ("Accepted", "2efe2717-52ef-43a5-96dc-0797e4ca1041"),
        )
        self.assertEqual(notarize.outcome('{"status": "Invalid"}'), ("Invalid", ""))

    def test_what_is_submitted_as_it_is_and_what_is_stapled(self) -> None:
        self.assertIn(".dmg", notarize.SUBMITTABLE)
        self.assertNotIn("", notarize.SUBMITTABLE, "a bare binary goes in a zip")
        self.assertIn(".dmg", notarize.STAPLEABLE)
        self.assertNotIn(".zip", notarize.STAPLEABLE, "a zip cannot hold a ticket")


if __name__ == "__main__":
    unittest.main()
