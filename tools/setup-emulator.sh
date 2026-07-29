#!/bin/bash
# One-time setup for the local RP2040 emulator.
#
# Clones wokwi/rp2040js (MIT — the same engine behind Wokwi's web Pico
# simulator), applies our SEV fix, and installs our harnesses into it.
# The clone is deliberately NOT committed to this repo: it's upstream
# code. Everything of ours lives in tools/harness/ and tools/patches/.
#
# Usage: tools/setup-emulator.sh
set -e
cd "$(dirname "$0")/.."
ROOT="$PWD"

if ! command -v node >/dev/null; then
  echo "Node.js is required (brew install node)." >&2
  exit 1
fi

if [ ! -d tools/rp2040js ]; then
  echo "==> cloning wokwi/rp2040js"
  git clone --depth 1 https://github.com/wokwi/rp2040js tools/rp2040js
fi

cd "$ROOT/tools/rp2040js"

echo "==> installing npm dependencies"
npm install --silent

# The SEV fix: upstream implements the ARM SEV instruction as a log-only
# no-op, so embassy's WFE/SEV executor wake protocol deadlocks. C and
# MicroPython never hit this (they wake via interrupts). See
# docs/08-hardware-sim.md. TODO: upstream this as a PR.
if grep -q "eventRegistered = true;" src/cortex-m0-core.ts && \
   grep -q "embassy executor's wake protocol" src/cortex-m0-core.ts; then
  echo "==> SEV fix already applied"
else
  echo "==> applying SEV fix"
  git apply "$ROOT/tools/patches/rp2040js-sev-fix.patch"
fi

echo "==> installing harnesses"
cp "$ROOT"/tools/harness/*.ts demo/

echo
echo "Ready. Try:  tools/sim-blink.sh"
