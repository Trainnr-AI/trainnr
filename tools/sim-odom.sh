#!/usr/bin/env bash
# LEVEL 1 PROOF: build pico-odom (firmware using sim-core's Odometry) ->
# UF2 -> emulate with two virtual wheel encoders, comparing the on-chip
# pose against ground truth. Usage: tools/sim-odom.sh
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

build_uf2 pico-odom thumbv6m-none-eabi
cd tools/rp2040js
npx tsx demo/watch-odom.ts ../../firmware/pico-odom/pico-odom.uf2 2>&1 \
  | grep -v "Unimplemented\|SEV"
