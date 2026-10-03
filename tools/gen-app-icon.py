"""Generate the Studio's app icon — the committed asset's provenance.

    cd trainnr && uv run --extra sim python ../tools/gen-app-icon.py

Writes `crates/trainnr-studio/assets/icon-256.rgba`: 256x256 raw RGBA the
shell embeds via `include_bytes!` (raw, so no PNG decoder joins the
dependency tree for one image). Rounded dark tile, the info-blue accent
dot from the brand bar, a lowercase "tr" in Inter Medium (the face the
Studio's interface uses, shipped in re_ui under the SIL Open Font
Licence), falling back to a common sans where re_ui is not in the cargo
registry. Also writes the README's logo, `docs/figures/readme/logo.png`.
Re-run this and commit the files together if the mark ever changes.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SIZE = 256
RADIUS = 56
BACKGROUND = (23, 26, 31, 255)  # near re_ui's dark surface
ACCENT = (90, 158, 224, 255)  # the brand bar's info blue
INK = (236, 239, 244, 255)

image = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
draw = ImageDraw.Draw(image)
draw.rounded_rectangle([0, 0, SIZE - 1, SIZE - 1], RADIUS, fill=BACKGROUND)

# First bold-ish sans available on this machine — macOS, then common
# Linux paths; the mark is committed, so the generator only needs to run
# where someone is redesigning it, but it shouldn't be macOS-only.
MARK = "tr"
INTER = sorted(Path.home().glob(".cargo/registry/src/*/re_ui-*/data/Inter-Medium.otf"))
FONTS = (
    *(str(f) for f in reversed(INTER)),
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
)
for candidate in FONTS:
    try:
        font = ImageFont.truetype(candidate, 128)
        break
    except OSError:
        continue
else:
    raise SystemExit(f"no usable font found; tried {FONTS}")
box = draw.textbbox((0, 0), MARK, font=font)
x = (SIZE - (box[2] - box[0])) / 2 - box[0]
y = (SIZE - (box[3] - box[1])) / 2 - box[1] - 10
draw.text((x, y), MARK, font=font, fill=INK)
draw.ellipse([SIZE - 78, 52, SIZE - 54, 76], fill=ACCENT)

out = Path(__file__).resolve().parent.parent / "crates/trainnr-studio/assets"
out.mkdir(parents=True, exist_ok=True)
(out / "icon-256.rgba").write_bytes(image.tobytes())
image.save(out / "icon-preview.png")  # for humans; the app embeds the .rgba
readme = Path(__file__).resolve().parent.parent / "docs/figures/readme"
readme.mkdir(parents=True, exist_ok=True)
image.resize((128, 128), Image.LANCZOS).save(readme / "logo.png", optimize=True)
print(
    f"wrote {out / 'icon-256.rgba'} ({SIZE}x{SIZE} RGBA), "
    "a PNG preview and the README logo"
)
