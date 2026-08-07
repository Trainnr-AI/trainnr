//! H3b — reading a wheel encoder on (emulated) silicon.
//!
//! Two GPIO pins carry the A/B quadrature signals. A tight loop samples
//! them and feeds YOUR decoder; every 250 ms the firmware reports its
//! position, speed and error count over UART.
//!
//! The demo has a sting in the tail: the harness spins the virtual wheel
//! faster and faster. Because this firmware POLLS at a fixed rate, there
//! is a speed above which it starts missing transitions — and you will
//! watch the error counter come alive. That is not a bug in your decoder;
//! it is the fundamental limit of polling, and the reason real motor
//! control uses hardware decoders (RP2040 PIO, STM32 timer encoder mode)
//! or pin interrupts.

#![no_std]
#![no_main]
// No unsafe anywhere in this firmware. `forbid`, not `deny`: it cannot be
// switched off locally with an `#[allow]`. Embassy's HAL already wraps the
// peripheral access that would otherwise need it — if that ever stops being
// true, the argument belongs in a commit that changes this line.
#![forbid(unsafe_code)]

use core::fmt::Write as _;
use embassy_executor::Spawner;
use embassy_rp::gpio::{Input, Level, Output, Pull};
// ⚠️ RP2350-E9 (see docs/09): our physical board is stepping A2, where an
// input with an internal PULL-DOWN can latch high instead of reading a
// clean low. GP16/GP17 below use exactly that configuration. It may
// never show — a push-pull encoder drives the line itself — but if ticks
// stick or counts only ever rise, add an external pull-down <=4.7k before
// suspecting this code. The emulated RP2040 is unaffected.
use embassy_rp::uart::{Blocking, Config as UartConfig, Uart, UartTx};
use embassy_time::{Instant, Timer};
use panic_halt as _;
use quad_encoder::QuadratureDecoder;

/// How often the sampling loop looks at the pins. 100 µs = 10 kHz.
/// Sampling theory: we can only track transitions slower than half this.
const POLL_US: u64 = 100;

#[embassy_executor::task]
async fn heartbeat(mut led: Output<'static>) {
    loop {
        led.toggle();
        Timer::after_millis(500).await;
    }
}

/// Samples the encoder pins forever, reporting periodically.
#[embassy_executor::task]
async fn encoder_task(
    a: Input<'static>,
    b: Input<'static>,
    mut uart: UartTx<'static, Blocking>,
) {
    let mut dec = QuadratureDecoder::new(a.is_high(), b.is_high());
    let mut line: heapless::String<160> = heapless::String::new();
    let mut last_report = Instant::now();
    let mut last_count = 0i32;

    loop {
        // ---- sample ----
        dec.update(a.is_high(), b.is_high());

        // ---- report every 250 ms ----
        let now = Instant::now();
        if now.duration_since(last_report).as_millis() >= 250 {
            let delta = dec.count - last_count;
            // ticks per second over the reporting window
            let ticks_per_s = delta * 4; // 250 ms window -> x4
            let dir = if delta > 0 {
                "fwd"
            } else if delta < 0 {
                "rev"
            } else {
                "---"
            };
            let _ = write!(
                line,
                "count={:>7}  {}  {:>7} ticks/s  errors={}\r\n",
                dec.count, dir, ticks_per_s, dec.errors
            );
            let _ = uart.blocking_write(line.as_bytes());
            line.clear();
            last_report = now;
            last_count = dec.count;
        }

        Timer::after_micros(POLL_US).await;
    }
}

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    let p = embassy_rp::init(Default::default());

    let uart = Uart::new_blocking(p.UART0, p.PIN_0, p.PIN_1, UartConfig::default());
    let (tx, _rx) = uart.split();

    // Encoder A on GP16, B on GP17. Pull::Down so an unconnected pin
    // reads low instead of floating randomly (real wiring matters here).
    let a = Input::new(p.PIN_16, Pull::Down);
    let b = Input::new(p.PIN_17, Pull::Down);

    spawner.spawn(heartbeat(Output::new(p.PIN_25, Level::Low)).unwrap());
    spawner.spawn(encoder_task(a, b, tx).unwrap());
}
