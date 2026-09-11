"""The Studio's cross-language constants stay mirrored: values that
exist in both the Python tools and the Rust shell are read from BOTH
sources as text and compared, so a one-sided edit becomes a red build
instead of a silently broken stream (the test_firmware_mirror idiom).
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RENDER_STREAM = (REPO / "tools" / "studio-render-stream.py").read_text(encoding="utf-8")
VIEWPORT_RS = (REPO / "crates" / "studio-shell" / "src" / "viewport.rs").read_text(
    encoding="utf-8"
)
MAIN_RS = (REPO / "crates" / "studio-shell" / "src" / "main.rs").read_text(
    encoding="utf-8"
)
VIZ = (REPO / "pipeline" / "rq_pipeline" / "viz.py").read_text(encoding="utf-8")


def constant(source: str, pattern: str) -> str:
    """The one capture group of `pattern`, which must match exactly once."""
    matches = re.findall(pattern, source, flags=re.MULTILINE)
    if len(matches) != 1:
        raise AssertionError(f"{pattern!r} matched {len(matches)} times")
    return matches[0]


class RenderSideCap(unittest.TestCase):
    def test_python_and_rust_agree_on_the_render_cap(self) -> None:
        # tools/studio-render-stream.py declares itself a mirror of
        # crates/studio-shell/src/viewport.rs; nothing enforced it
        # until 2026-09-01.
        py = constant(RENDER_STREAM, r"^MAX_RENDER_SIDE = (\d+)")
        rs = constant(VIEWPORT_RS, r"const MAX_RENDER_SIDE: u32 = (\d+);")
        self.assertEqual(py, rs)


class StudioPort(unittest.TestCase):
    def test_the_ingest_port_is_the_one_the_shell_binds(self) -> None:
        # rq_pipeline.viz.STUDIO_ADDRESS is the one Python home; the
        # Rust shell binds the same port (documented mirror).
        address = constant(VIZ, r'STUDIO_ADDRESS = "rerun\+http://127\.0\.0\.1:(\d+)/')
        bound = constant(MAIN_RS, r'const GRPC_BIND: &str = "0\.0\.0\.0:(\d+)"')
        self.assertEqual(address, bound)


if __name__ == "__main__":
    unittest.main()
