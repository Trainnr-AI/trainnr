#!/usr/bin/env bash
# Prove the whole system, end to end, in one command.
#
#   tools/verify.sh                  everything that needs no hardware
#   tools/verify.sh --serial <port>  ALSO run against a real Pico
#
# Ordered cheapest-first, so a broken build fails in seconds rather than
# after the two-minute emulator run.
set -uo pipefail
cd "$(dirname "$0")/.."
source "$HOME/.cargo/env" 2>/dev/null || true

SERIAL=""
[ "${1:-}" = "--serial" ] && SERIAL="${2:-}"

pass=0; fail=0
step() {                      # step "name" "command"
  printf "%-46s" "$1"
  if out=$(eval "$2" 2>&1); then
    echo "ok"; pass=$((pass+1))
  else
    echo "FAIL"; echo "$out" | tail -15; fail=$((fail+1))
  fi
}

echo "=== robotiq end-to-end verification ==="
step "formatting"            "cargo fmt --all --check"
step "clippy (all targets)"  "! cargo clippy -q --workspace --all-targets 2>&1 | grep -qE '^error'"
step "tests"                 "cargo test -q --workspace"
step "docs describe real code" "python3 tools/check-docs.py"
step "unsafe forbidden everywhere" "python3 tools/check-unsafe-gates.py"
step "simulator solves the U-trap" \
     "cargo run -q -p sim-run | grep -q 'Waypoints reached: 1/1'"

# Firmware. pico-led and pico-selftest are RP2350-only by design.
for c in pico-blink pico-button pico-encoder pico-imu pico-odom pico-robot; do
  step "firmware $c (RP2040)" "(cd firmware/$c && cargo build -q --release --target thumbv6m-none-eabi)"
done
for c in $(ls firmware | grep pico-); do
  step "firmware $c (RP2350)" "tools/build-pico2.sh $c"
done

# The two committed fixtures: a real hardware session, and a synthetic
# perception one. Both fail on a behaviour change, neither needs hardware.
step "replay: RP2350 wire recording" \
     "cargo run -q -p hil-host -- --replay recordings/rp2350-utrap.wire | grep -q 'matched the recording exactly'"
step "replay: perception fixture" \
     "cargo run -q -p vision --bin chase -- --replay recordings/chase-sweep.perc | grep -q 'every command matched'"

# Slowest, and needs npx + the emulator checkout.
step "HIL on the emulator (RP2040)" \
     "tools/build-robot.sh && cargo run -q -p hil-host | grep -q 'waypoints:   1/1'"

if [ -n "$SERIAL" ]; then
  step "HIL on real silicon ($SERIAL)" \
       "cargo run -q -p hil-host -- --serial $SERIAL | grep -q 'waypoints:   1/1'"
else
  echo "HIL on real silicon                            skipped (pass --serial <port>)"
fi

echo
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ] || exit 1
