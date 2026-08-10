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

# --fast skips the two slow/blocked steps and runs everything else.
#
#   the emulator      ~9 minutes since the measured robot speed landed,
#                     because it simulates a 3.86x slower machine
#   the wire replay   STALE pending a hardware re-record (see below)
#
# Everything else — every build, every test, both firmware architectures,
# the perception replay — still runs. Use it while iterating; run the
# whole thing before pushing.
CI=""
[ "${1:-}" = "--fast" ] && CI=1

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
  step "firmware $c (RP2350)" "SKIP_UF2=1 tools/build-pico2.sh $c"
done
# The loop above builds each crate's DEFAULT transport, which for
# pico-encoder and pico-odom is the UART one the emulator speaks. Their USB
# builds are what run on a real board, and nothing else here compiles them.
# (pico-robot's USB build is covered, but only transitively — the HIL step
# below runs tools/build-robot.sh, which builds it.)
step "firmware pico-encoder (RP2350, USB)" "SKIP_UF2=1 tools/build-pico2.sh pico-encoder usb"
step "firmware pico-odom (RP2350, USB)" "SKIP_UF2=1 tools/build-pico2.sh pico-odom usb"
# `teleop` replaces the calibration sweep with a host command channel, so
# it compiles a different half of the file — the watchdog, the signed duty
# path and the H-bridge failsafe. Nothing above reaches any of it.
step "firmware pico-odom (RP2350, teleop)" "SKIP_UF2=1 tools/build-pico2.sh pico-odom teleop"
# The radio transport, and the tee that sends every line down BOTH wires.
# Neither is reachable from any build above: `wifi` brings in the CYW43
# driver, the IP stack and a third `Report` impl, and `usb,wifi` compiles
# the `Tee` combinator that nothing else instantiates.
#
# Built WITHOUT credentials on purpose. `WIFI_SSID` unset is a supported
# state — the firmware blinks a distinct pattern and never tries to join —
# precisely so that this step can exist on a machine that has none. A
# `compile_error!` there would have made the radio build the one variant
# never checked.
step "firmware pico-odom (RP2350, wifi)" "SKIP_UF2=1 tools/build-pico2.sh pico-odom wifi"
step "firmware pico-odom (RP2350, usb+wifi)" "SKIP_UF2=1 tools/build-pico2.sh pico-odom usb,wifi"

# The two committed fixtures: a real hardware session, and a synthetic
# perception one. Both fail on a behaviour change, neither needs hardware.
# ⚠️ STALE since 2026-08-10, and deliberately still run.
#
# `RobotSpec::REAL_BOT` gained its measured values that day, so `hil-host`
# now commands the chip differently and this replay diverges — correctly.
# The recording is a capture of a REAL RP2350 over USB, so re-making it
# needs the board:
#
#   picotool load -x firmware/pico-robot/pico-robot-pico2-usb.uf2   # after BOOTSEL
#   cargo run -p hil-host -- --serial /dev/cu.usbmodem11 \
#       --record recordings/rp2350-utrap.wire
#
# Left FAILING rather than skipped, because a skip is a thing people stop
# reading. A red step with this comment above it is a re-record that has
# not happened yet; a green one would be a regression that nobody noticed.
# ⚠️ Excluded from --ci ONLY because it is currently stale. A CI that is
# red from its first commit teaches everyone to ignore red, which is worse
# than a skip. Locally it still runs and still fails, so it is not hidden.
# **Delete the `[ -z "$CI" ] &&` guard the moment it is re-recorded.**
[ -z "$CI" ] && \
step "replay: RP2350 wire recording  [STALE - re-record on hardware]" \
     "cargo run -q -p hil-host -- --replay recordings/rp2350-utrap.wire | grep -q 'matched the recording exactly'"
# The teleop page is compiled into the binary with `include_str!`, so a
# missing or renamed file is a build error rather than a 404 discovered by
# someone standing in a room holding a phone over a robot. This checks the
# thing that file is FOR: that the joystick handler survives to the
# `touch-action` rule iOS needs, without which the page silently does
# nothing on the only device it exists to run on.
step "teleop page still has its touch handlers" \
     "grep -q 'touch-action: none' crates/teleop-web/src/index.html && \
      grep -q touchcancel crates/teleop-web/src/index.html"
step "replay: perception fixture" \
     "cargo run -q -p vision --bin chase -- --replay recordings/chase-sweep.perc | grep -q 'every command matched'"

# Slowest, and needs npx + the emulator checkout.
[ -z "$CI" ] && \
step "HIL on the emulator (RP2040)" \
     "tools/build-robot.sh && cargo run -q -p hil-host | grep -q 'waypoints:   1/1'"

if [ -n "$SERIAL" ]; then
  step "HIL on real silicon ($SERIAL)" \
       "cargo run -q -p hil-host -- --serial $SERIAL | grep -q 'waypoints:   1/1'"
  # The failures a SUCCESSFUL mission never exercises: a corrupt command,
  # and a host that dies and reconnects. Both were broken on hardware and
  # invisible from the laptop.
  step "chip conformance (corrupt cmd, reconnect)" \
       "cargo run -q -p hil-host --example chip_probe -- $SERIAL"
else
  echo "HIL on real silicon                            skipped (pass --serial <port>)"
fi

echo
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ] || exit 1
