//! Linker setup for every firmware crate in this repo.
//!
//! Runs on the HOST at build time. Each crate's `build.rs` is one line:
//!
//! Gated like everything else even though **nothing here reaches the
//! chip** — it emits linker flags on the laptop. A build script that
//! could run arbitrary unsafe at compile time is still worth forbidding,
//! and an exception here would be one more thing to remember.
//!
//! ```ignore
//! fn main() { firmware_build_support::configure(); }
//! ```
//!
//! # Why this crate exists
//!
//! The RP2040 and RP2350 need different memory layouts and different
//! linker arguments, and every firmware crate needs the same logic to
//! choose between them. That logic and both `memory-*.x` files were
//! copied into eight crates — 32 files with 4 distinct contents. One
//! copy now.
//!
//! `.unwrap()` is deliberate throughout: a build script SHOULD fail the
//! build loudly rather than limp on with a broken linker configuration.

#![forbid(unsafe_code)]

use std::env;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

/// The RP2040 layout: reserves 256 bytes for the BOOT2 second-stage
/// bootloader, because the chip has no internal flash and must be taught
/// how to read the external chip before anything else can run.
pub const MEMORY_RP2040: &[u8] = include_bytes!("../memory-rp2040.x");

/// The RP2350 layout: no BOOT2 (the ROM bootloader is real), but an
/// `.start_block` section for the `IMAGE_DEF` the ROM looks for.
pub const MEMORY_RP2350: &[u8] = include_bytes!("../memory-rp2350.x");

/// Pick the layout from the crate's features and hand it to the linker.
///
/// Panics with a readable message if both chips or neither are selected —
/// otherwise the failure lands deep in the linker with nothing to read.
pub fn configure() {
    let rp2040 = env::var_os("CARGO_FEATURE_RP2040").is_some();
    let rp2350 = env::var_os("CARGO_FEATURE_PICO2").is_some();

    assert!(
        !(rp2040 && rp2350),
        "\n\n  Pick ONE chip.\n  \
           RP2040 (emulator):  cargo build --release\n  \
           RP2350 (Pico 2 W):  cargo build --release --no-default-features \\\n  \
                                 --features pico2 --target thumbv8m.main-none-eabihf\n\n  \
           (`--features pico2` alone still leaves the default `rp2040` on.)\n"
    );
    assert!(
        rp2040 || rp2350,
        "\n\n  No chip selected. Use the default (RP2040) or --features pico2.\n"
    );

    let layout = if rp2350 { MEMORY_RP2350 } else { MEMORY_RP2040 };

    // Always written as `memory.x` — cortex-m-rt's link.x includes that
    // name specifically.
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    File::create(out.join("memory.x"))
        .unwrap()
        .write_all(layout)
        .unwrap();
    println!("cargo:rustc-link-search={}", out.display());
    println!("cargo:rerun-if-changed=build.rs");

    println!("cargo:rustc-link-arg-bins=--nmagic");
    println!("cargo:rustc-link-arg-bins=-Tlink.x");
    // link-rp.x carries the RP2040's BOOT2 section. embassy-rp only emits
    // it under its `rp2040` feature, so asking for it on RP2350 fails the
    // link with a missing-file error.
    if !rp2350 {
        println!("cargo:rustc-link-arg-bins=-Tlink-rp.x");
    }
}
