//! The parts every firmware in this directory was writing out again.
//!
//! Nothing here is clever. It exists because four crates had four
//! byte-identical copies of the USB bring-up and had **already started to
//! disagree** — two wrote `max_packet_size_0 = MAX_PACKET as u8` and two
//! hardcoded `64`. They happened to be the same number. That is the shape
//! of every bug this project has spent a day on: two facts held in two
//! places, and nothing comparing them.

#![no_std]
// No unsafe anywhere in this crate. `forbid`, not `deny`: it cannot be
// switched off locally with an `#[allow]`.
#![forbid(unsafe_code)]

use embassy_rp::gpio::Output;
use embassy_time::{Instant, Timer};

pub mod motor;
#[cfg(feature = "usb")]
pub mod usb;

/// Milliseconds since boot.
///
/// A named function rather than `Instant::now().as_millis()` at each call
/// site, because "what does the chip think the time is" is one decision:
/// it is monotonic, it starts at boot, and it never comes from a host.
/// Anything doing timeout arithmetic — the command watchdog, the stall
/// detector — has to agree about that or the comparisons are meaningless.
pub fn now_ms() -> u64 {
    Instant::now().as_millis()
}

/// Toggles an LED forever, at `period_ms` on and `period_ms` off.
///
/// The sign of life for a board with no other output. ⚠️ **On a Pico 2 W
/// this lights nothing**: GP25 is the CYW43 radio's chip-select there, not
/// an LED, and reaching the actual LED means booting the radio — see
/// `firmware/pico-led`. Do not read "no blink" as "dead board"; on a W the
/// sign of life is the USB port appearing.
#[embassy_executor::task]
pub async fn heartbeat(mut led: Output<'static>, period_ms: u64) {
    loop {
        led.toggle();
        Timer::after_millis(period_ms).await;
    }
}

/// Somewhere a status line can be sent, whatever the wire is.
///
/// # The contract, which is load-bearing
///
/// **Reports are best-effort. Ticks are not.**
///
/// A sampling loop's product is an accurate count; the status line is
/// disposable commentary on it. A transport that stalls — a USB host that
/// stopped draining, a terminal nobody opened, a radio whose transmit
/// buffer is full — must never hold up sampling, because every microsecond
/// spent blocked here is a microsecond of missed transitions, and a missed
/// transition corrupts the very number the loop exists to measure.
///
/// How long "quietly" takes is the implementation's business: the UART
/// cannot block, USB carries a deadline, and the radio is polled exactly
/// once and dropped if it is not ready.
///
/// ⚠️ **This has been violated once, and it took the board down.** On
/// 2026-08-10 a UDP implementation awaited `send_to`, which returns
/// `Pending` on a full transmit buffer. The report loop wedged — and the
/// USB half wedged with it, because the two sends were joined and a join
/// finishes only when both halves do. If an implementation of this trait
/// can ever return `Pending` indefinitely, it is wrong.
///
/// `async_fn_in_trait` is allowed rather than boxed: the warning is about
/// callers being unable to require `Send`, and nothing here ever needs it.
/// Every implementation runs on one thread-local embassy executor, and a
/// `Box<dyn Future>` would want an allocator this target does not have.
#[allow(async_fn_in_trait)]
pub trait Report {
    async fn send(&mut self, bytes: &[u8]);
}
