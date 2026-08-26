#!/usr/bin/env bash
# H3 loop: build pico-encoder -> UF2 -> emulate with a virtual quadrature
# encoder that ramps up in speed. Usage: tools/sim-encoder.sh
set -e
cd "$(dirname "$0")/.."
source "$HOME/.cargo/env" 2>/dev/null || true
source tools/_firmware.sh

# Ensure the emulator is set up (clone + patch + harnesses).
if [ ! -d tools/rp2040js/node_modules ]; then
  echo "Emulator not set up yet — running tools/setup-emulator.sh"
  tools/setup-emulator.sh
fi
# Keep harnesses in sync with tools/harness/ (the version we track in git).
cp tools/harness/*.ts tools/rp2040js/demo/

build_uf2 pico-encoder thumbv6m-none-eabi
cd tools/rp2040js
npx tsx demo/watch-encoder.ts ../../firmware/pico-encoder/pico-encoder.uf2 2>&1 \
  | grep -v "Unimplemented\|SEV"
