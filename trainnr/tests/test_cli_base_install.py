"""`trainnr version` and the CLI's other light commands run on a plain
`pip install trainnr`, with no simulation extra: importing the CLI loads
neither MuJoCo nor the job layer that pulls it in. v0.2.0-rc.1's CLI did,
at the top of the module, and release-verify's clean install failed."""

from __future__ import annotations

import subprocess
import sys
import unittest


class ABaseInstall(unittest.TestCase):
    def test_importing_the_cli_loads_no_simulator(self) -> None:
        probe = (
            "import sys, trainnr.cli\n"
            "heavy = [m for m in ('mujoco', 'trainnr.mcp_jobs', 'trainnr.project')"
            " if m in sys.modules]\n"
            "print(','.join(heavy))\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", probe], check=True, capture_output=True, text=True
        ).stdout.strip()
        self.assertEqual(out, "", f"the CLI loaded {out} on import")


if __name__ == "__main__":
    unittest.main()
