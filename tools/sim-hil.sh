#!/bin/bash
# LEVEL 3 — hardware-in-the-loop.
#
# The firmware (on an emulated RP2040) is the robot's brain; hil-host is its
# body and the physical world. They talk over the emulated UART every
# control tick. Watch it in the Rerun window.
#
# Usage: tools/sim-hil.sh
set -e
cd "$(dirname "$0")/.."
source "$HOME/.cargo/env" 2>/dev/null || true

# Ensure the emulator is set up (clone + patch + harnesses).
if [ ! -d tools/rp2040js/node_modules ]; then
  echo "Emulator not set up yet — running tools/setup-emulator.sh"
  tools/setup-emulator.sh
fi
cp tools/harness/*.ts tools/rp2040js/demo/

(cd firmware/pico-robot \
  && cargo build --release \
  && elf2uf2-rs target/thumbv6m-none-eabi/release/pico-robot pico-robot.uf2)

cargo run -q -p hil-host -- firmware/pico-robot/pico-robot.uf2
