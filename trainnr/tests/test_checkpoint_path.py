"""`evaluate_walk` hands its job the checkpoint's resolved path, never the
bare `run/model_N.pt` form: the judge runs in another directory and the
bare form died there on FileNotFoundError after the existence check had
passed (stranger test, 2026-10-03)."""

import tempfile
import unittest
from pathlib import Path

from trainnr.mcp_server import RUNS_FOLDER, checkpoint_missing, checkpoint_path


class CheckpointPath(unittest.TestCase):
    def test_bare_and_runs_forms_name_the_same_absolute_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / RUNS_FOLDER / "c0"
            run.mkdir(parents=True)
            (run / "model_149.pt").write_bytes(b"")
            expected = run / "model_149.pt"
            for spoken in ("c0/model_149.pt", f"{RUNS_FOLDER}/c0/model_149.pt"):
                with self.subTest(spoken=spoken):
                    resolved = checkpoint_path(spoken, root)
                    self.assertEqual(resolved, expected)
                    self.assertTrue(resolved.is_absolute())
                    self.assertIsNone(checkpoint_missing(spoken, root))

    def test_absolute_path_is_kept_and_no_root_means_as_given(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "model_1.pt"
            file.write_bytes(b"")
            self.assertEqual(checkpoint_path(str(file), Path(tmp) / "elsewhere"), file)
        self.assertEqual(checkpoint_path("c0/model_1.pt", None), Path("c0/model_1.pt"))

    def test_missing_checkpoint_names_the_run_folders_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / RUNS_FOLDER / "c0"
            run.mkdir(parents=True)
            (run / "model_0.pt").write_bytes(b"")
            why = checkpoint_missing("c0/model_150.pt", root)
            self.assertIsNotNone(why)
            self.assertIn("model_0.pt", why or "")

    def test_an_experiment_names_its_newest_checkpoint(self) -> None:
        """An agent passed the experiment's name, the way every other
        argument names an artifact, and was refused twice (2026-10-04)."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / RUNS_FOLDER / "c0"
            run.mkdir(parents=True)
            for n in (9, 149, 50):  # numeric order, not text order
                (run / f"model_{n}.pt").write_bytes(b"")
            for spoken in ("c0", f"{RUNS_FOLDER}/c0", str(run)):
                with self.subTest(spoken=spoken):
                    self.assertEqual(
                        checkpoint_path(spoken, root), run / "model_149.pt"
                    )
                    self.assertIsNone(checkpoint_missing(spoken, root))

    def test_an_experiment_without_checkpoints_is_refused_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / RUNS_FOLDER / "c0").mkdir(parents=True)
            self.assertIn("no checkpoints", checkpoint_missing("c0", root) or "")

    def test_a_checkpoint_named_without_its_suffix_is_found(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / RUNS_FOLDER / "c0"
            run.mkdir(parents=True)
            (run / "model_149.pt").write_bytes(b"")
            self.assertEqual(
                checkpoint_path("c0/model_149", root), run / "model_149.pt"
            )
            self.assertIsNone(checkpoint_missing("c0/model_149", root))


if __name__ == "__main__":
    unittest.main()
