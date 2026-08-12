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

# --fast skips ONE step and runs everything else.
#
#   the emulator      ~9 minutes since the measured robot speed landed,
#                     because it simulates a 3.86x slower machine
#
# It used to skip the wire replay too, which was stale from 2026-08-10 to
# 2026-08-11. Re-recorded, so it runs everywhere again.
#
# ⚠️ `--fast` is a real gap, not just a slower/faster choice: the emulator
# is the ONLY step where firmware EXECUTES rather than merely compiling.
# Eight crates were refactored behind it on 2026-08-11 and the resulting
# images were never run. Run the whole thing before pushing.
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

# The two crates that must compile for the chip as well as the laptop.
# `cargo test` proves neither: it builds the std shape only, so an
# accidental `Vec`, `String` or `std::` in either would pass every test
# here and fail the moment the firmware tried to link it — the crate's
# whole reason for existing (a guard that outlives the host) lost to a
# convenience import nobody noticed.
for target in thumbv6m-none-eabi thumbv8m.main-none-eabihf; do
  step "sim-core is still no_std ($target)" \
       "cargo build -q -p sim-core --no-default-features --target $target"
  step "arm is still no_std ($target)" \
       "cargo build -q -p arm --no-default-features --target $target"
  step "n20-joint is still no_std ($target)" \
       "cargo build -q -p n20-joint --no-default-features --target $target"
done

# Firmware. pico-led and pico-selftest are RP2350-only by design.
for c in pico-arm pico-blink pico-button pico-encoder pico-imu pico-odom pico-robot; do
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
# The arm joint's only telemetry path. Nothing else compiles its `link`
# module or the `J` messages it emits.
step "firmware pico-arm (RP2350, USB)" "SKIP_UF2=1 tools/build-pico2.sh pico-arm usb"
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
#
# Re-recorded 2026-08-11 against board #1 (chipid 0x12ea158439ef5cea)
# running `pico-robot` over USB, after `RobotSpec::REAL_BOT` gained its
# measured values and made the previous capture diverge by 2,264 steps.
# 1/1 waypoints, 0 wall bumps, 0.394 m drift, worst compute 319 us.
#
# ⚠️ It is a capture of REAL silicon, so re-making it needs the board:
#
#   picotool load -x firmware/pico-robot/pico-robot-pico2-usb.uf2   # after BOOTSEL
#   cargo run -p hil-host -- --serial /dev/cu.usbmodem11 \
#       --record recordings/rp2350-utrap.wire
#
# No longer excluded from --fast: it passes, so it guards every run
# rather than only the ones somebody remembered to make slow.
step "replay: RP2350 wire recording" \
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
# A real two-joint session on real motors, 2026-08-12. Guards the whole
# chain in one line: the firmware's `J` encoder, the shared parser, and
# the phases the guard actually produced — including the `held` that only
# happens when a commander goes quiet.
#
# ⚠️ Re-making it needs the board:
#
#   tools/build-pico2.sh pico-arm usb   # then copy the .uf2 after BOOTSEL
#   cargo run -p hil-host --example joint_viz -- /dev/cu.usbmodem11 \
#       --record recordings/bench-two-joint.wire
step "replay: two-joint bench recording" \
     "cargo run -q -p hil-host --example joint -- --replay recordings/bench-two-joint.wire \
      | grep -q '0 unparsable'"
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
