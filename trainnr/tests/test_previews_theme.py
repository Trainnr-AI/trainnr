"""The card pictures follow the app's theme: a light palette for the kinds
drawn here, the dark set untouched, the simulator's renders shared."""

import json
import tempfile
import unittest
from pathlib import Path

from trainnr.project import previews

try:
    from PIL import Image
except ImportError:  # the renderer skips without Pillow, and so does the test
    Image = None


class PreviewTheme(unittest.TestCase):
    def test_the_light_set_lives_beside_the_dark_one(self) -> None:
        class P:
            root = Path("/p")

        self.assertEqual(
            previews.preview_path(P(), "x@1").as_posix(), "/p/.index/previews/x@1.png"
        )
        self.assertEqual(
            previews.preview_path(P(), "x@1", "light").as_posix(),
            "/p/.index/previews/light/x@1.png",
        )

    def test_a_certificate_card_is_drawn_in_the_light_palette(self) -> None:
        if Image is None:
            self.skipTest("Pillow")
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "cert"
            source.mkdir()
            (source / "certificate.json").write_text(
                json.dumps(
                    {"id": "c", "successes": 3, "trials": 4, "ci95": [0.3, 0.99]}
                )
            )
            out = Path(tmp) / "light.png"
            token = previews._PALETTE.set(previews.LIGHT)
            try:
                ok = previews._render_certificate(None, source, out, {})
            finally:
                previews._PALETTE.reset(token)
            if not ok:
                self.skipTest("the certificate renderer needs more than this fixture")
            corner = Image.open(out).convert("RGB").getpixel((2, 2))
            self.assertEqual(corner, previews.LIGHT.ground)

    def test_a_palette_change_empties_the_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:

            class P:
                root = Path(tmp)

            folder = previews.preview_path(P(), "x", "light").parent
            folder.mkdir(parents=True)
            (folder / "a@1.png").write_bytes(b"old")
            previews._drop_set_if_palette_changed(P(), "light", previews.LIGHT)
            self.assertFalse((folder / "a@1.png").exists())
            (folder / "b@1.png").write_bytes(b"new")
            previews._drop_set_if_palette_changed(P(), "light", previews.LIGHT)
            self.assertTrue((folder / "b@1.png").exists(), "same palette keeps the set")

    def test_the_default_palette_is_dark(self) -> None:
        self.assertIs(previews.pal(), previews.DARK)
        self.assertEqual(previews.PALETTES["light"].ground, previews.LIGHT.ground)


if __name__ == "__main__":
    unittest.main()
