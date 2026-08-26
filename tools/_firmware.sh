# Sourced by the sim-*.sh scripts: one place that asks cargo where a
# firmware binary lands. `firmware/` is one workspace, so its output is in
# `firmware/target/`, never `firmware/<crate>/target/` — the bug
# build-robot.sh and build-pico2.sh fixed on 2026-08-11 and the six sim
# scripts still carried until 2026-08-27.
#
#   build_uf2 <crate> [target-triple]   -> firmware/<crate>/<crate>.uf2

build_uf2() {
  local crate="$1" triple="${2:-thumbv6m-none-eabi}"
  (cd "firmware/$crate" \
    && cargo build --release --quiet \
    && target_dir=$(cargo metadata --format-version 1 --no-deps --offline 2>/dev/null \
         | "${PYTHON:-python3}" -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])') \
    && elf2uf2-rs "${target_dir}/${triple}/release/${crate}" "${crate}.uf2")
}
