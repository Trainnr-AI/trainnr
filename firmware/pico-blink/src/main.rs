//! H0 — blink. The "hello world" that proves the entire toolchain:
//! Rust → cross-compiled ELF → simulated (or real) silicon.
//!
//! Two LEDs blink at DIFFERENT rates from two independent async tasks —
//! embassy's whole pitch in one screen: concurrency on a chip with no OS,
//! no threads, no heap.
//!
//! `#![no_std]`: no standard library — no println!, no Vec, no files.
//! There is no operating system underneath; this program IS the computer.
//! `#![no_main]`: no ordinary main — the `#[embassy_executor::main]` macro
//! generates the real entry point that the chip's reset vector jumps to.

#![no_std]
#![no_main]
// No unsafe anywhere in this firmware. `forbid`, not `deny`: it cannot be
// switched off locally with an `#[allow]`. Embassy's HAL already wraps the
// peripheral access that would otherwise need it — if that ever stops being
// true, the argument belongs in a commit that changes this line.
#![forbid(unsafe_code)]

use embassy_executor::Spawner;
use embassy_rp::gpio::{Level, Output};
use embassy_time::Timer;
// A panic on a chip with no OS has nowhere to report to (yet — real logging
// arrives with the Debug Probe). panic-halt parks the CPU in a loop.
use panic_halt as _;

/// A separate task: blinks whatever LED it's handed, forever.
/// `'static` says the pin must live for the program's whole life — on
/// firmware, that's most things (there's no "shutdown").
///
/// `pool_size = 2`: embassy pre-allocates task memory at COMPILE time (no
/// heap!), one slot per pool entry — and the default is 1. We spawn this
/// task twice, so we need two slots. Without this, the second spawn's
/// unwrap() panics before any LED ever toggles — a bug the rp2040js
/// emulator caught on first boot (it would have failed identically on the
/// real board).
#[embassy_executor::task(pool_size = 2)]
async fn blink(mut led: Output<'static>, interval_ms: u64) {
    loop {
        led.set_high();
        Timer::after_millis(interval_ms).await;
        led.set_low();
        Timer::after_millis(interval_ms).await;
    }
}

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    // Take ownership of the chip's peripherals. This is Rust's ownership
    // system doing real hardware work: each pin can be claimed exactly
    // once — two drivers fighting over one pin is a COMPILE error.
    let p = embassy_rp::init(Default::default());

    // Pin 25 = the Pico's onboard LED. Pin 15 = external LED on the
    // breadboard (see diagram.json for the wiring).
    let onboard = Output::new(p.PIN_25, Level::Low);
    let external = Output::new(p.PIN_16, Level::Low);

    // Two tasks, two rhythms, one core, no OS. Calling a #[task] fn
    // creates a SpawnToken (Err if the task pool is exhausted — embassy
    // 0.10 API); spawning the token starts it.
    spawner.spawn(blink(onboard, 50).unwrap());
    spawner.spawn(blink(external, 800).unwrap());

    // main is itself a task; with nothing left to do, let it end.
}
