#!/usr/bin/env bash
# Build firmware for the PHYSICAL Pico 2 W (RP2350, Cortex-M33) and emit a
# .uf2 you can drag onto the board.
#
#   tools/build-pico2.sh pico-blink
#   tools/build-pico2.sh pico-robot usb      # extra features, comma-separated
#
# Flash it with picotool — hold BOOTSEL, plug in USB, then:
#
#   picotool load -x firmware/<crate>/<crate>-pico2.uf2
#
# Prefer that over dragging the .uf2 onto the mounted drive: the
# mass-storage path stalled at "preparing to copy" on one of our boards
# while picotool flashed the same file first try (docs/07, 2026-08-07).
set -euo pipefail
CRATE="${1:-pico-blink}"
# Everything after the crate name is added to the feature list, so a
# transport or debug flag does not need its own build script.
EXTRA="${2:-}"
FEATURES="pico2${EXTRA:+,$EXTRA}"
TARGET=thumbv8m.main-none-eabihf
cd "$(dirname "$0")/../firmware/$CRATE"

cargo build --release --no-default-features --features "$FEATURES" --target "$TARGET"

BIN="target/$TARGET/release/$CRATE"
if command -v picotool >/dev/null; then
  picotool uf2 convert "$BIN" -t elf "$CRATE-pico2.uf2" --family rp2350-arm-s
  echo "wrote firmware/$CRATE/$CRATE-pico2.uf2"
else
  echo
  echo "Built: firmware/$CRATE/$BIN"
  echo
  echo "To make a .uf2 you need picotool (elf2uf2-rs is RP2040-only):"
  echo "    brew install picotool"
  echo "then re-run this script."
fi
