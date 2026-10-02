"""The gsplat installer refuses by name: a Mac, a missing interpreter, a
CUDA this tool has no wheels for, an environment with neither uv nor pip;
and its wheel pins are the four CUDA 13 pieces at torch's minor."""

from __future__ import annotations

import runpy
import subprocess
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tools" / "install-gsplat.py"


def load() -> dict:
    return runpy.run_path(str(SCRIPT), run_name="installer_under_test")


class TheInstaller(unittest.TestCase):
    def test_the_wheels_are_the_four_cuda_13_pieces(self) -> None:
        tool = load()
        self.assertEqual(
            tool["COMPILER_WHEELS"]["13"],
            ("nvidia-cuda-nvcc", "nvidia-nvvm", "nvidia-cuda-crt", "nvidia-cuda-cccl"),
        )
        self.assertEqual(tool["GSPLAT_VERSION"], "1.5.3")

    def test_a_mac_is_refused_naming_brush(self) -> None:
        tool = load()
        with (
            mock.patch("platform.system", return_value="Darwin"),
            mock.patch("sys.argv", ["install-gsplat.py"]),
            self.assertRaises(SystemExit) as caught,
        ):
            tool["main"]()
        self.assertIn("Brush", str(caught.exception))

    def test_no_torch_and_no_installer_are_refused_by_name(self) -> None:
        tool = load()
        boom = subprocess.CalledProcessError(1, ["python"])
        with (
            mock.patch("subprocess.run", side_effect=boom),
            self.assertRaises(SystemExit) as caught,
        ):
            tool["torch_cuda"](Path("/venv/bin/python"))
        self.assertIn("cannot import torch", str(caught.exception))
        no_pip = subprocess.CompletedProcess(["python"], 1)
        with (
            mock.patch("shutil.which", return_value=None),
            mock.patch("subprocess.run", return_value=no_pip),
            self.assertRaises(SystemExit) as caught,
        ):
            tool["installer"](Path("/venv/bin/python"))
        self.assertIn("ensurepip", str(caught.exception))

    def test_another_cuda_is_refused_naming_this_tools_limit(self) -> None:
        tool = load()
        self.assertIn("CUDA 13", tool["OTHER_CUDA"])
        self.assertIsNone(tool["COMPILER_WHEELS"].get("12"))


if __name__ == "__main__":
    unittest.main()
