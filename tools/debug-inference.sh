#!/usr/bin/env bash
# Run a command with BOTH logging gates open, so ONNX Runtime execution
# provider diagnostics are actually visible.
#
# Why two: ONNX Runtime filters in C++ (ORT_LOG, default `error`) before
# anything reaches Rust's tracing (RUST_LOG). Setting only one gives
# silence that looks like "nothing wrong". See docs/11-perception-stack.md.
#
#   tools/debug-inference.sh cargo run --release -p vision --bin bench
#
# The line worth grepping for is VerifyEachNodeIsAssignedToAnEp — it says
# exactly which provider took which nodes.
set -e
cd "$(dirname "$0")/.."
export ORT_LOG="${ORT_LOG:-verbose}"
export RUST_LOG="${RUST_LOG:-ort=trace}"
echo "ORT_LOG=$ORT_LOG  RUST_LOG=$RUST_LOG" >&2
exec "$@"
