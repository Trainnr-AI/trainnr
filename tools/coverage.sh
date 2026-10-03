#!/usr/bin/env bash
# Coverage + lines-of-code report for the workspace.
#
#   tools/coverage.sh            summary by group
#   tools/coverage.sh --html     also open a browsable HTML report
#
# Firmware crates are absent by design: they are `no_std`, build for a
# different target, and cannot run host tests. What covers them is
# `pico-selftest` (same maths, real silicon, diffed against the host) and
# the HIL rig.
set -euo pipefail
cd "$(dirname "$0")/.."
source "$HOME/.cargo/env" 2>/dev/null || true

command -v cargo-llvm-cov >/dev/null || {
  echo "cargo-llvm-cov not installed:  cargo install cargo-llvm-cov"; exit 1; }

if [ "${1:-}" = "--html" ]; then
  (cd crates/trainnr-studio && cargo llvm-cov --html)
  echo "open crates/trainnr-studio/target/llvm-cov/html/index.html"
  exit 0
fi

(cd crates/trainnr-studio && cargo llvm-cov --summary-only 2>/dev/null) | tail -n +2
echo
python3 tools/loc-report.py | tail -25
