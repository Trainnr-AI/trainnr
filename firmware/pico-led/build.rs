//! Build script: hands the linker the right memory layout for whichever
//! chip we are targeting. Runs on the HOST at compile time (this one file
//! is normal std Rust, not firmware).
//!
//! Two chips, two layouts:
//!   default        RP2040 (Pico 1)  — what the emulator runs
//!   --features pico2   RP2350 (Pico 2 W) — the physical board
//!
//! `.unwrap()` is deliberate here: a build script SHOULD fail the build
//! loudly rather than limp on with a broken linker configuration.

use std::env;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

fn main() {
    let rp2040 = env::var_os("CARGO_FEATURE_RP2040").is_some();
    let rp2350 = env::var_os("CARGO_FEATURE_PICO2").is_some();

    // Both at once would hand embassy-rp two mutually exclusive chip
    // features and fail deep inside the linker with nothing useful to
    // read. Fail here instead, where the message can say what to do.
    assert!(
        !(rp2040 && rp2350),
        "\n\n  Pick ONE chip.\n\
           RP2040 (emulator):  cargo build --release\n\
           RP2350 (Pico 2 W):  cargo build --release --no-default-features \\\n\
                                 --features pico2 --target thumbv8m.main-none-eabihf\n\n\
           (`--features pico2` alone still leaves the default `rp2040` on.)\n"
    );
    assert!(
        rp2040 || rp2350,
        "\n\n  No chip selected. Use the default (RP2040) or --features pico2.\n"
    );

    let (layout, source) = if rp2350 {
        (
            &include_bytes!("memory-rp2350.x")[..],
            "memory-rp2350.x",
        )
    } else {
        (
            &include_bytes!("memory-rp2040.x")[..],
            "memory-rp2040.x",
        )
    };

    // Put the chosen layout where the linker looks, always under the name
    // `memory.x` — cortex-m-rt's link.x includes that name specifically.
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    File::create(out.join("memory.x"))
        .unwrap()
        .write_all(layout)
        .unwrap();
    println!("cargo:rustc-link-search={}", out.display());
    println!("cargo:rerun-if-changed={source}");
    println!("cargo:rerun-if-changed=build.rs");

    println!("cargo:rustc-link-arg-bins=--nmagic");
    println!("cargo:rustc-link-arg-bins=-Tlink.x");
    // link-rp.x carries the RP2040's BOOT2 section. embassy-rp only emits
    // it when its `rp2040` feature is on, so asking for it on RP2350
    // fails the link with a missing-file error.
    if !rp2350 {
        println!("cargo:rustc-link-arg-bins=-Tlink-rp.x");
    }
}
