"""A union keeps its story: the sidecars ride along verbatim, the
counts add up, and disagreeing sources are refused by name."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rq_pipeline.collect.dataset_ops import check_union
from rq_pipeline.collect.provenance import PROVENANCE_FILE


def _dataset(
    root: Path,
    *,
    fps: int = 10,
    expert: str = "t@1",
    basis: str = "b",
    episodes: int = 2,
) -> Path:
    root.mkdir(parents=True)
    (root / PROVENANCE_FILE).write_text(
        json.dumps(
            {
                "bundle": "microduck@1",
                "source": root.name,
                "expert": expert,
                "episodes": episodes,
                "frames": [3] * episodes,
                "fps": fps,
                "manifests": [{"dynamics_basis": basis, "visual_basis": ""}] * episodes,
            }
        )
    )
    return root


class TheUnionCheck(unittest.TestCase):
    def test_agreeing_sources_pass_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            a = _dataset(Path(tmp) / "a")
            b = _dataset(Path(tmp) / "b", episodes=5)
            sidecars = check_union([a, b])
        self.assertEqual([s["source"] for s in sidecars], ["a", "b"])
        self.assertEqual(sum(s["episodes"] for s in sidecars), 7)

    def test_disagreements_are_refused_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            a = _dataset(Path(tmp) / "a")
            with self.assertRaisesRegex(ValueError, "fps"):
                check_union([a, _dataset(Path(tmp) / "b", fps=50)])
            with self.assertRaisesRegex(ValueError, "expert"):
                check_union([a, _dataset(Path(tmp) / "c", expert="other@2")])
            with self.assertRaisesRegex(ValueError, "dynamics_basis"):
                check_union([a, _dataset(Path(tmp) / "d", basis="another")])
            with self.assertRaisesRegex(ValueError, "at least two"):
                check_union([a])
            with self.assertRaisesRegex(FileNotFoundError, "sidecar"):
                check_union([a, Path(tmp) / "missing"])


if __name__ == "__main__":
    unittest.main()
