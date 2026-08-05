#!/usr/bin/env bash
# Build firmware for the PHYSICAL Pico 2 W (RP2350, Cortex-M33) and emit a
# .uf2 you can drag onto the board.
#
#   tools/build-pico2.sh pico-blink
#
# Flash it: hold BOOTSEL, plug in USB, drop the .uf2 on the drive that
# appears. No Debug Probe or soldering needed.
set -euo pipefail
CRATE="${1:-pico-blink}"
TARGET=thumbv8m.main-none-eabihf
cd "$(dirname "$0")/../firmware/$CRATE"

cargo build --release --no-default-features --features pico2 --target "$TARGET"

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
