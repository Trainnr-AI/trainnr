//! Linker setup — see `firmware/build-support`, which holds the one
//! copy of the memory layouts and the chip-selection logic.

fn main() {
    firmware_build_support::configure();
}
