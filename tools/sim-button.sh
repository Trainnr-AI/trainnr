#!/usr/bin/env bash
# H1 loop: build pico-button -> UF2 -> emulate with scheduled button
# presses and duty-cycle measurement. Usage: tools/sim-button.sh
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

build_uf2 pico-button thumbv6m-none-eabi
cd tools/rp2040js
npx tsx demo/watch-button.ts ../../firmware/pico-button/pico-button.uf2 2>&1 \
  | grep -v "Unimplemented\|SEV"
