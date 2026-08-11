#!/usr/bin/env bash
# Build firmware for the PHYSICAL Pico 2 W (RP2350, Cortex-M33) and emit a
# .uf2 you can drag onto the board.
#
#   tools/build-pico2.sh pico-blink
#   tools/build-pico2.sh pico-robot usb      # extra features, comma-separated
#
# Flash it with picotool — hold BOOTSEL, plug in USB, then:
#
#   picotool load -x firmware/<crate>/<crate>-pico2.uf2
#
# Prefer that over dragging the .uf2 onto the mounted drive: the
# mass-storage path stalled at "preparing to copy" on one of our boards
# while picotool flashed the same file first try (docs/07, 2026-08-07).
set -euo pipefail
CRATE="${1:-pico-blink}"
# Everything after the crate name is added to the feature list, so a
# transport or debug flag does not need its own build script.
EXTRA="${2:-}"
FEATURES="pico2${EXTRA:+,$EXTRA}"
TARGET=thumbv8m.main-none-eabihf
# ⚠️ The .uf2 is named after its FEATURES, not just its crate.
#
# It used to be `<crate>-pico2.uf2` for every build, which meant
# `tools/verify.sh` — which builds `wifi` and `usb,wifi` back to back with
# no credentials, because that is the point of it — silently overwrote a
# credentialed binary that was about to be flashed. On 2026-08-10 that
# cost a flash cycle and looked exactly like the radio failing to join:
# the LED reported no credentials because the binary genuinely had none.
#
# Same shape as every other bug this repo has found: two things writing
# one name, and nothing comparing them.
OUT="$CRATE-pico2${EXTRA:+-${EXTRA//,/-}}.uf2"
cd "$(dirname "$0")/../firmware/$CRATE"

cargo build --release --no-default-features --features "$FEATURES" --target "$TARGET"

# ⚠️ ASK cargo where it put the binary. Do not assume.
#
# This was hardcoded to `target/…` relative to the crate directory, which
# was right until `firmware/` became one workspace on 2026-08-11 — cargo
# then started building into `firmware/target/` while this kept reading
# the abandoned `firmware/<crate>/target/`. Every .uf2 after that point
# was a conversion of a STALE elf from before the migration, so all five
# feature variants came out byte-identical and several hours of hardware
# debugging compared the same binary against itself.
TARGET_DIR=$(cargo metadata --format-version 1 --no-deps --offline 2>/dev/null \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])' 2>/dev/null)
BIN="${TARGET_DIR:-target}/$TARGET/release/$CRATE"

if [ ! -f "$BIN" ]; then
  echo "no binary at $BIN — cargo metadata said target_directory=${TARGET_DIR:-<unset>}" >&2
  exit 1
fi

# Check the CONTENT matches the features asked for, not the timestamp.
#
# An mtime check was the obvious guard and it is wrong: cargo hardlinks a
# cached artifact and keeps its original timestamp, so a legitimate
# rebuild looks stale. This asks a question only the right binary can
# answer — a `wifi` build embeds 231 KB of CYW43 firmware and a build
# without it does not, so their sizes cannot be close.
SIZE=$(wc -c < "$BIN")
case "$FEATURES" in
  *wifi*) [ "$SIZE" -gt 2500000 ] || { echo "⚠️  $BIN is ${SIZE}B — too small to contain the radio firmware, so this is not a wifi build" >&2; exit 1; } ;;
  *)      [ "$SIZE" -lt 2500000 ] || { echo "⚠️  $BIN is ${SIZE}B — large enough to contain the radio firmware, but wifi was not requested" >&2; exit 1; } ;;
esac
# `SKIP_UF2=1` builds and stops. `tools/verify.sh` uses it because the
# gate only needs the COMPILE to succeed — writing a .uf2 there produced a
# flashable artifact nobody asked for, and on 2026-08-10 that artifact
# overwrote a credentialed binary between building it and flashing it.
# The feature suffix above separates `usb` from `wifi`; it cannot separate
# "built with WIFI_SSID" from "built without", because those are the same
# features. Not writing the file at all is what closes that.
if [ -n "${SKIP_UF2:-}" ]; then
  echo "built (no .uf2): firmware/$CRATE $FEATURES"
elif command -v picotool >/dev/null; then
  picotool uf2 convert "$BIN" -t elf "$OUT" --family rp2350-arm-s
  echo "wrote firmware/$CRATE/$OUT"
else
  echo
  echo "Built: firmware/$CRATE/$BIN"
  echo
  echo "To make a .uf2 you need picotool (elf2uf2-rs is RP2040-only):"
  echo "    brew install picotool"
  echo "then re-run this script."
fi
