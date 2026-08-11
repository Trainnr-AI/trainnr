#!/usr/bin/env bash
# Build pico-robot for BOTH targets, from one command.
#
#   tools/build-robot.sh
#
# Why this exists: the two images are built by different paths — the
# RP2040 one by cargo + elf2uf2-rs, the RP2350 one by build-pico2.sh — and
# on 2026-08-07 a firmware change was rebuilt for one and not the other.
# The stale image flashed cleanly, ran, and failed to answer a new message,
# which reads exactly like a protocol bug. It cost two hardware runs.
#
# Both, or neither.
set -euo pipefail
cd "$(dirname "$0")/.."
source "$HOME/.cargo/env" 2>/dev/null || true

echo "[1/2] RP2040 (emulator, UART)"
# ⚠️ ASK cargo where the binary is. This was `target/…` relative to the
# crate, which broke silently when `firmware/` became one workspace on
# 2026-08-11 — cargo moved its output to `firmware/target/` and this kept
# pointing at the abandoned per-crate directory. `build-pico2.sh` had the
# identical bug and produced five byte-identical images from a stale elf.
(cd firmware/pico-robot \
  && cargo build --release --quiet \
  && TARGET_DIR=$(cargo metadata --format-version 1 --no-deps --offline 2>/dev/null \
       | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])') \
  && elf2uf2-rs "${TARGET_DIR}/thumbv6m-none-eabi/release/pico-robot" pico-robot.uf2)
echo "      firmware/pico-robot/pico-robot.uf2"

echo "[2/2] RP2350 (real board, USB)"
tools/build-pico2.sh pico-robot usb >/dev/null
echo "      firmware/pico-robot/pico-robot-pico2-usb.uf2"

# Same source, same minute. If these disagree, something did not rebuild.
for f in firmware/pico-robot/pico-robot.uf2 firmware/pico-robot/pico-robot-pico2-usb.uf2; do
  printf "      %-46s %s\n" "$f" "$(date -r "$f" '+%H:%M:%S')"
done
