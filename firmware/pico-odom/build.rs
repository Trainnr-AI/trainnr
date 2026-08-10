//! Linker setup — see `firmware/build-support`, which holds the one
//! copy of the memory layouts and the chip-selection logic.

fn main() {
    firmware_build_support::configure();

    // The `wifi` transport bakes credentials into the binary with
    // `option_env!`. Cargo cannot see that on its own, so without these a
    // changed password reuses the cached binary and the board keeps
    // failing to join with nothing in the build output to say why.
    println!("cargo:rerun-if-env-changed=WIFI_SSID");
    println!("cargo:rerun-if-env-changed=WIFI_PASSWORD");
}
