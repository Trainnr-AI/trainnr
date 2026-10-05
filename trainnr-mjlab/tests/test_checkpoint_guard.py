"""A checkpoint that carries code is refused before rsl_rl unpickles it
(security review, 2026-10-04); a real rsl_rl checkpoint loads."""

import os
import pickle
import tempfile
import unittest
from pathlib import Path

from trainnr_mjlab.checkpoint_guard import require_tensor_only


class _Runs:
    def __reduce__(self) -> tuple:
        return (os.system, ("touch PWNED",))


class TheGuard(unittest.TestCase):
    def test_tensors_and_plain_data_pass(self) -> None:
        import torch  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model_1.pt"
            torch.save(
                {
                    "actor_state_dict": {"mlp.0.weight": torch.zeros(4, 3)},
                    "iter": 1,
                    "infos": {"note": "plain"},
                },
                path,
            )
            require_tensor_only(path)

    def test_a_pickle_that_runs_code_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model_1.pt"
            path.write_bytes(pickle.dumps({"actor_state_dict": _Runs()}))
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                with self.assertRaisesRegex(ValueError, "could run code"):
                    require_tensor_only(path)
                self.assertFalse((Path(tmp) / "PWNED").exists())
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
