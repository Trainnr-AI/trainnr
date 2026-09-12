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


class WireTags(unittest.TestCase):
    def test_the_pan_tag_and_its_payload_agree(self) -> None:
        # The pan message (WASD/QE, 2026-09-12): the tag number and the
        # three f32 seconds on both sides of the pipe.
        py_tag = constant(RENDER_STREAM, r"^TAG_PAN = (\d+)")
        rs_tag = constant(VIEWPORT_RS, r"const TAG_PAN: u8 = (\d+);")
        self.assertEqual(py_tag, rs_tag)
        py_bytes = int(constant(RENDER_STREAM, r"^    TAG_PAN: (\d+),"))
        rs_bytes = int(constant(VIEWPORT_RS, r"fn encode_pan\(.*?\) -> \[u8; (\d+)\]"))
        self.assertEqual(py_bytes + 1, rs_bytes)  # the tag byte leads

    def test_the_group_tag_agrees(self) -> None:
        # The group-mask message (u8 kind, u8 group, u8 on): one tag,
        # four bytes on the wire, the kinds named by the stream's report.
        py_tag = constant(RENDER_STREAM, r"^TAG_GROUP = (\d+)")
        rs_tag = constant(VIEWPORT_RS, r"const TAG_GROUP: u8 = (\d+);")
        self.assertEqual(py_tag, rs_tag)
        self.assertEqual(constant(RENDER_STREAM, r"^    TAG_GROUP: (\d+),"), "3")
        self.assertIn("&[TAG_GROUP, kind, group, u8::from(on)]", VIEWPORT_RS)


class StudioPort(unittest.TestCase):
    def test_the_ingest_port_is_the_one_the_shell_binds(self) -> None:
        # rq_pipeline.viz.STUDIO_ADDRESS is the one Python home; the
        # Rust shell binds the same port (documented mirror).
        address = constant(VIZ, r'STUDIO_ADDRESS = "rerun\+http://127\.0\.0\.1:(\d+)/')
        bound = constant(MAIN_RS, r'const GRPC_BIND: &str = "0\.0\.0\.0:(\d+)"')
        self.assertEqual(address, bound)


if __name__ == "__main__":
    unittest.main()
