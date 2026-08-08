//! H3b — reading a wheel encoder, on emulated or real silicon.
//!
//! Two GPIO pins carry the A/B quadrature signals. A tight loop samples
//! them and feeds YOUR decoder; every 250 ms the firmware reports its
//! position, speed and error count.
//!
//! The demo has a sting in the tail: the harness spins the virtual wheel
//! faster and faster. Because this firmware POLLS at a fixed rate, there
//! is a speed above which it starts missing transitions — and you will
//! watch the error counter come alive. That is not a bug in your decoder;
//! it is the fundamental limit of polling, and the reason real motor
//! control uses hardware decoders (RP2040 PIO, STM32 timer encoder mode)
//! or pin interrupts.
//!
//! # Two transports, one sampling loop
//!
//! ```text
//!   default          UART on GP0/GP1   → the rp2040js emulator
//!   --features usb   USB CDC serial    → a REAL Pico on a USB cable
//! ```
//!
//! On a real board GP0/GP1 are bare header pins wired to nothing, so the
//! UART build runs perfectly and reports into the void. The USB build is
//! what you flash to measure a physical motor.
//!
//! [`sample_forever`] is written **once** and takes a [`Report`]; only
//! the plumbing differs. Same discipline as `pico-robot`'s `Link`: when a
//! second copy is about to appear, give the shared thing a home instead.
//!
//! # Measuring `ticks_per_revolution` with this
//!
//! `crates/sim-core/src/spec.rs` is emphatic that this number must be
//! measured rather than computed from `PPR × 4 × gear_ratio`, because
//! advertised gear ratios are approximations. Wire the encoder only —
//! VCC to 3V3, GND to GND, C1/C2 to GP16/GP17, **motor leads left
//! disconnected** — turn the output shaft exactly ten revolutions by
//! hand, and divide the reported `count` by ten.
//!
//! Power the encoder from **3V3, not 5V**: its C1/C2 outputs swing to
//! whatever VCC is, and 5 V logic into an RP2350 input is out of spec.

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
use embassy_rp::peripherals::{PIN_16, PIN_17, PIN_25};
use embassy_rp::Peri;
use embassy_time::{Instant, Timer};
use panic_halt as _;
use quad_encoder::QuadratureDecoder;

/// How often the sampling loop looks at the pins. 100 µs = 10 kHz.
/// Sampling theory: we can only track transitions slower than half this.
const POLL_US: u64 = 100;

/// How often a status line goes out. Nothing is derived from this by
/// hand — the rate below divides by the *measured* elapsed time, so
/// changing this number cannot silently make the reported speed wrong.
const REPORT_MS: u64 = 250;

/// Where a status line goes. The sampling loop does not care.
trait Report {
    /// Send bytes, or give up quietly. **How long "quietly" takes is the
    /// implementation's business** — only the USB transport can stall, so
    /// only it carries a deadline.
    ///
    /// **Dropping output is deliberate, and the priority is the point.**
    /// This loop's product is an accurate tick count; the status line is
    /// disposable commentary on it. A transport that stalls — a USB host
    /// that stops draining, a terminal nobody opened — must never hold up
    /// sampling, because every microsecond spent blocked here is a
    /// microsecond of missed transitions, and a missed transition
    /// corrupts the very number we are here to measure.
    ///
    /// So: reports are best-effort, ticks are not.
    async fn send(&mut self, bytes: &[u8]);
}

#[embassy_executor::task]
async fn heartbeat(mut led: Output<'static>) {
    loop {
        led.toggle();
        Timer::after_millis(500).await;
    }
}

/// Samples the encoder pins forever, reporting periodically.
///
/// Deliberately a plain `async fn` rather than an `#[embassy_executor::task]`:
/// tasks cannot be generic, and being generic over [`Report`] is exactly
/// what keeps this loop from being written twice.
async fn sample_forever(a: Input<'static>, b: Input<'static>, out: &mut impl Report) -> ! {
    let mut dec = QuadratureDecoder::new(a.is_high(), b.is_high());
    let mut line: heapless::String<160> = heapless::String::new();
    let mut last_report = Instant::now();
    let mut last_count = 0i32;

    loop {
        // ---- sample ----
        dec.update(a.is_high(), b.is_high());

        // ---- report every REPORT_MS ----
        let now = Instant::now();
        let elapsed_ms = now.duration_since(last_report).as_millis();
        if elapsed_ms >= REPORT_MS {
            let delta = dec.count - last_count;
            // Divide by what the clock actually says, not by REPORT_MS.
            // The loop can only ever overshoot its deadline — a report
            // that took 260 ms and one that took 250 ms are different
            // rates, and assuming the nominal interval would quietly
            // report the wrong speed. `elapsed_ms >= REPORT_MS >= 1`, so
            // this cannot divide by zero.
            let ticks_per_s = i64::from(delta) * 1000 / elapsed_ms as i64;
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
            out.send(line.as_bytes()).await;
            line.clear();
            last_report = now;
            last_count = dec.count;
        }

        Timer::after_micros(POLL_US).await;
    }
}

/// The pins and the LED, which are the same whichever transport is built.
///
/// Takes the three peripherals by value rather than the whole
/// `Peripherals` struct, so the caller can still hand the transport its
/// own UART or USB block out of the same `p`.
fn shared_setup(
    spawner: Spawner,
    pin_a: Peri<'static, PIN_16>,
    pin_b: Peri<'static, PIN_17>,
    pin_led: Peri<'static, PIN_25>,
) -> (Input<'static>, Input<'static>) {
    // ⚠️ RP2350-E9 (see docs/09): our physical board is stepping A2, where
    // an input with an internal PULL-DOWN can latch high instead of
    // reading a clean low. These two pins use exactly that configuration.
    //
    // `Pull::Down` is right for a PUSH-PULL encoder, which drives the line
    // itself. If the motor's Hall outputs turn out to be open-drain the
    // line is only ever pulled low, and this reads a constant — the
    // symptom is a count that never moves. In that case switch both to
    // `Pull::Up`, which also sidesteps E9 entirely, since the erratum is
    // specifically about pull-downs.
    //
    // If ticks stick or counts only ever rise with a push-pull encoder,
    // add an external pull-down <=4.7k before suspecting this code. The
    // emulated RP2040 is unaffected.
    let a = Input::new(pin_a, Pull::Down);
    let b = Input::new(pin_b, Pull::Down);

    // ⚠️ On a **Pico 2 W this lights nothing**: GP25 is the CYW43 radio's
    // chip-select there, not an LED — see `firmware/pico-led`, which boots
    // the radio precisely because that is the only way to reach it.
    // Driving it is harmless, and it is a real heartbeat on a non-W board
    // and on the emulated RP2040. But do not read "no blink" as "dead
    // board": on a W, the sign of life is the USB port appearing.
    spawner.spawn(heartbeat(Output::new(pin_led, Level::Low)).unwrap());

    (a, b)
}

// ---------------------------------------------------------------------
// Transport A — UART on GP0/GP1. What the emulator speaks.
// ---------------------------------------------------------------------
#[cfg(not(feature = "usb"))]
mod transport {
    use super::*;
    use embassy_rp::uart::{Blocking, Config as UartConfig, Uart, UartTx};

    struct UartReport(UartTx<'static, Blocking>);

    impl Report for UartReport {
        /// **Never actually blocks for long.** `blocking_write` pushes
        /// into the hardware FIFO at 115200 baud with nothing on the other
        /// end to apply back-pressure, so the timeout the trait permits is
        /// not needed here. The emulator is the only consumer.
        async fn send(&mut self, bytes: &[u8]) {
            let _ = self.0.blocking_write(bytes);
        }
    }

    pub async fn run(p: embassy_rp::Peripherals, spawner: Spawner) -> ! {
        let (a, b) = shared_setup(spawner, p.PIN_16, p.PIN_17, p.PIN_25);

        let uart = Uart::new_blocking(p.UART0, p.PIN_0, p.PIN_1, UartConfig::default());
        let (tx, _rx) = uart.split();

        sample_forever(a, b, &mut UartReport(tx)).await
    }
}

// ---------------------------------------------------------------------
// Transport B — USB CDC serial. What a real Pico on a cable speaks.
// ---------------------------------------------------------------------
#[cfg(feature = "usb")]
mod transport {
    use super::*;
    use embassy_futures::join::join;
    use embassy_rp::bind_interrupts;
    use embassy_rp::peripherals::USB;
    use embassy_rp::usb::{Driver, InterruptHandler};
    use embassy_time::{with_timeout, Duration};
    use embassy_usb::class::cdc_acm::{CdcAcmClass, State};
    use embassy_usb::driver::Driver as UsbDriver;
    use embassy_usb::{Builder, Config};
    use static_cell::StaticCell;

    bind_interrupts!(struct Irqs {
        USBCTRL_IRQ => InterruptHandler<USB>;
    });

    /// The endpoint's max packet size. A longer write must be split or
    /// the transfer is rejected, and our status line is up to 160 bytes.
    const MAX_PACKET: usize = 64;

    /// Longest a status line may spend trying to reach the host before it
    /// is abandoned. Sized against the 1 ms USB full-speed frame: a
    /// healthy write of three packets finishes in well under this, so the
    /// timeout only ever fires on a host that has genuinely stopped
    /// reading. At 10 kHz sampling, hitting it costs ~50 samples.
    const REPORT_TIMEOUT_MS: u64 = 5;

    struct UsbReport<'d, D: UsbDriver<'d>>(CdcAcmClass<'d, D>);

    impl<'d, D: UsbDriver<'d>> Report for UsbReport<'d, D> {
        /// Skipped entirely when no terminal has opened the port, and
        /// abandoned after [`REPORT_TIMEOUT_MS`] if one has but has
        /// stopped reading.
        ///
        /// `dtr()` — "data terminal ready" — is the CDC line the host
        /// raises when something opens `/dev/cu.usbmodem…`. Checking it
        /// first means the common case of a board powered from a charger
        /// costs nothing at all, rather than one timeout per report.
        ///
        /// # On cancelling the write
        ///
        /// `pico-robot` justifies cancelling its USB *read* by having
        /// checked embassy-rp's source: the endpoint read registers a
        /// waker and tests a bit, with every side effect after the await.
        /// **We have not made that argument for `write_packet`**, so a
        /// cancelled write may in principle leave a partial or repeated
        /// packet — a garbled status line.
        ///
        /// Taking that risk is still right, and the asymmetry is the
        /// reason: a mangled line costs a glance, while blocking the
        /// sampler costs ticks, and ticks are the measurement. If a
        /// garbled line is ever seen in practice, the fix is a channel and
        /// a separate reporting task — not a longer timeout.
        async fn send(&mut self, bytes: &[u8]) {
            if !self.0.dtr() {
                return;
            }
            let write_all = async {
                for chunk in bytes.chunks(MAX_PACKET) {
                    if self.0.write_packet(chunk).await.is_err() {
                        return;
                    }
                }
            };
            let _ = with_timeout(Duration::from_millis(REPORT_TIMEOUT_MS), write_all).await;
        }
    }

    pub async fn run(p: embassy_rp::Peripherals, spawner: Spawner) -> ! {
        let (a, b) = shared_setup(spawner, p.PIN_16, p.PIN_17, p.PIN_25);

        let driver = Driver::new(p.USB, Irqs);
        // 0x2e8a is Raspberry Pi's vendor ID. The product ID differs from
        // `pico-robot`'s 0x000a so both can be plugged in at once and
        // still be told apart in `ioreg`/`lsusb`.
        let mut config = Config::new(0x2e8a, 0x000b);
        config.manufacturer = Some("robotiq");
        config.product = Some("pico-encoder");
        config.serial_number = Some("1");
        config.max_power = 100;
        config.max_packet_size_0 = MAX_PACKET as u8;

        static CONFIG_DESC: StaticCell<[u8; 256]> = StaticCell::new();
        static BOS_DESC: StaticCell<[u8; 256]> = StaticCell::new();
        static CONTROL_BUF: StaticCell<[u8; 64]> = StaticCell::new();
        static STATE: StaticCell<State> = StaticCell::new();

        let state = STATE.init(State::new());
        let mut builder = Builder::new(
            driver,
            config,
            CONFIG_DESC.init([0; 256]),
            BOS_DESC.init([0; 256]),
            &mut [],
            CONTROL_BUF.init([0; 64]),
        );
        let class = CdcAcmClass::new(&mut builder, state, MAX_PACKET as u16);
        let mut usb = builder.build();
        let mut out = UsbReport(class);

        // The USB stack and the sampler must both run; `join` polls them
        // on one stack.
        //
        // Note what is NOT here: `pico-robot` waits for `wait_connection()`
        // before starting its control loop, because a robot with no host
        // has nothing to do. **Sampling starts immediately instead.** The
        // decoder's whole job is to miss nothing, and gating it on a
        // terminal being open would mean any ticks between power-up and
        // your first `screen` are silently gone. `send` already declines
        // to write while `dtr()` is low, so nothing is wasted.
        join(usb.run(), sample_forever(a, b, &mut out)).await;

        // `join` over a `!` future never returns, but the compiler wants a
        // value for the `-> !` signature.
        loop {
            Timer::after_secs(1).await;
        }
    }
}

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    let p = embassy_rp::init(Default::default());
    transport::run(p, spawner).await
}
