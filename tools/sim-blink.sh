#!/bin/bash
# One command: build firmware -> UF2 -> run in the local rp2040js emulator,
# watching the LED pins. Usage: tools/sim-blink.sh [sim-seconds]
set -e
cd "$(dirname "$0")/.."
source "$HOME/.cargo/env" 2>/dev/null || true

# Ensure the emulator is set up (clone + patch + harnesses).
if [ ! -d tools/rp2040js/node_modules ]; then
  echo "Emulator not set up yet — running tools/setup-emulator.sh"
  tools/setup-emulator.sh
fi
# Keep harnesses in sync with tools/harness/ (the version we track in git).
cp tools/harness/*.ts tools/rp2040js/demo/

(cd firmware/pico-blink \
  && cargo build --release \
  && elf2uf2-rs target/thumbv6m-none-eabi/release/pico-blink pico-blink.uf2)
cd tools/rp2040js
npx tsx demo/watch-blink.ts ../../firmware/pico-blink/pico-blink.uf2 "${1:-3}" 2>&1 \
  | grep -v "Unimplemented\|SEV"
