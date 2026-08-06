//! H0-W — blink the onboard LED on a **Pico 2 W**.
//!
//! `pico-blink` drives `PIN_25` and, on a W board, nothing happens. That is
//! not a bug in it: on the wireless Picos **GP25 is the radio's chip-select
//! line**, and the LED is wired to the CYW43439's own GPIO 0. There is no
//! way to reach it with a plain `Output` — you have to boot the radio.
//!
//! So this is the smallest program that lights that LED, and it is
//! dramatically more work than `pico-blink`:
//!
//! ```text
//!   ┌─ RP2350 ────────────┐         ┌─ CYW43439 (the radio) ─┐
//!   │                     │  a fake │                        │
//!   │  PIO0 ──────────────┼─ SPI ──▶│  ┌──────────────┐      │
//!   │  (programmable I/O) │  bus    │  │ 231 KB of    │      │
//!   │                     │         │  │ firmware we  │      │
//!   │  GP23 ── power ────▶│         │  │ upload every │      │
//!   │  GP24 ── data ─────▶│         │  │ boot         │      │
//!   │  GP25 ── chip sel ─▶│         │  └──────────────┘      │
//!   │  GP29 ── clock ────▶│         │        │               │
//!   └─────────────────────┘         │        ▼  GPIO 0 ──▶ 💡│
//!                                   └────────────────────────┘
//! ```
//!
//! Three things make it awkward, all worth knowing:
//!
//! 1. **The radio has no flash of its own.** Its 231 KB of firmware lives
//!    in *our* flash and is uploaded across the bus on every single boot.
//!    That is why this binary is far larger than `pico-blink`'s 6 KB.
//! 2. **The bus is not real SPI.** It is a half-duplex variant where the
//!    data line reverses direction mid-transaction, which no hardware SPI
//!    peripheral can do. `cyw43-pio` implements it on **PIO** — a small
//!    programmable state-machine block in the RP2350 built for exactly
//!    this "invent your own protocol" problem.
//! 3. **It needs a background task.** `cyw43::Runner` must keep running to
//!    service the chip, so this is the first firmware here that genuinely
//!    *needs* concurrency rather than using it for tidiness.
//!
//! ```sh
//! tools/build-pico2.sh pico-led
//! picotool load firmware/pico-led/pico-led-pico2.uf2
//! ```

#![no_std]
#![no_main]

use cyw43::{Aligned, A4};
use cyw43_pio::{PioSpi, RM2_CLOCK_DIVIDER};
use embassy_executor::Spawner;
use embassy_rp::bind_interrupts;
use embassy_rp::gpio::{Level, Output};
use embassy_rp::dma;
use embassy_rp::peripherals::{DMA_CH0, PIO0};
use embassy_rp::pio::{InterruptHandler, Pio};
use embassy_time::Timer;
use panic_halt as _;
use static_cell::StaticCell;

// The PIO block raises an interrupt when a transfer finishes; this hands
// that interrupt to embassy's handler so the driver can `.await` on it
// instead of spinning.
// embassy-rp 0.10 wants BOTH bound: PIO drives the fake SPI bus, and DMA
// moves the 231 KB of firmware across it without the CPU copying byte by
// byte. All eight DMA channels share one interrupt, DMA_IRQ_0.
bind_interrupts!(struct Irqs {
    PIO0_IRQ_0 => InterruptHandler<PIO0>;
    DMA_IRQ_0 => dma::InterruptHandler<DMA_CH0>;
});

/// The radio's firmware, copied into 4-byte-aligned statics.
///
/// `include_bytes!` gives no alignment guarantee, but the driver moves
/// these by DMA and needs 4-byte alignment — hence `Aligned`, a
/// `#[repr(align(4))]` newtype. Getting it wrong would be a misaligned DMA
/// read at runtime, not a compile error, so the type does the work.
static FW: Aligned<A4, [u8; 231_077]> = Aligned(*cyw43_firmware::CYW43_43439A0);
/// Board-specific radio settings (MAC, board type, antenna trim). Not in
/// the published `cyw43-firmware` crate, so vendored — 742 bytes.
static NVRAM: Aligned<A4, [u8; 742]> =
    Aligned(*include_bytes!("../cyw43-firmware/nvram_rp2040.bin"));

/// Services the radio. Must run forever, or the chip stops answering.
#[embassy_executor::task]
async fn cyw43_task(
    runner: cyw43::Runner<'static, cyw43::SpiBus<Output<'static>, PioSpi<'static, PIO0, 0>>>,
) -> ! {
    runner.run().await
}

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    let p = embassy_rp::init(Default::default());

    // The four lines to the radio. These pin numbers are fixed by the
    // board's wiring, not chosen by us.
    let pwr = Output::new(p.PIN_23, Level::Low); // radio power enable
    let cs = Output::new(p.PIN_25, Level::High); // chip select — NOT an LED
    let mut pio = Pio::new(p.PIO0, Irqs);
    let spi = PioSpi::new(
        &mut pio.common,
        pio.sm0,
        RM2_CLOCK_DIVIDER,
        pio.irq0,
        cs,
        p.PIN_24, // data, bidirectional
        p.PIN_29, // clock
        dma::Channel::new(p.DMA_CH0, Irqs),
    );

    // `'static` state the driver borrows for its whole life. StaticCell
    // hands out a `&'static mut` exactly once — the no-allocator way to
    // get a long-lived buffer without writing `unsafe`.
    static STATE: StaticCell<cyw43::State> = StaticCell::new();
    let state = STATE.init(cyw43::State::new());

    // Boots the radio and uploads its firmware across the PIO bus.
    // Nothing works until this finishes.
    let (_net_device, mut control, runner) = cyw43::new(state, pwr, spi, &FW, &NVRAM).await;
    // A #[task] fn returns Result<SpawnToken, SpawnError> — the Err is
    // "the task's fixed memory pool is full". Pool size is 1 and we spawn
    // once, so it cannot fire; but halting quietly would leave the LED
    // dark, which is indistinguishable from the bug this program exists
    // to fix. So: fail loudly.
    match cyw43_task(runner) {
        Ok(token) => spawner.spawn(token),
        Err(_) => panic!("cyw43 task pool exhausted"),
    }

    // The country/regulatory table. Radios have to be told where they are.
    // Passed as a plain slice — unlike the firmware and nvram it is not
    // DMA'd, so it needs no alignment wrapper.
    control.init(cyw43_firmware::CYW43_43439A0_CLM).await;

    // GPIO 0 *of the radio*, not of the RP2350. This is the LED.
    loop {
        control.gpio_set(0, true).await;
        Timer::after_millis(500).await;
        control.gpio_set(0, false).await;
        Timer::after_millis(500).await;
    }
}
