//! LEVEL 1 PROOF — the laptop robot's brain, running on the chip.
//!
//! Every line of dead reckoning here comes from `sim-core`: the SAME
//! `Odometry`, `Pose` and `DiffDrive` that drive the Rerun simulator,
//! compiled with `--no-default-features` for bare metal. Nothing was
//! rewritten or ported. The two islands are now one codebase:
//!
//!   quad-encoder  (H3)  ->  sim-core::Odometry  (Stage 0 exercise 3)
//!   pin transitions      ->  believed pose, computed on emulated ARM
//!
//! Two encoders (left GP16/17, right GP18/19) feed the same tick-to-pose
//! math the simulator uses, and the firmware reports x/y/heading.
//!
//! # Two transports, one odometry loop
//!
//! ```text
//!   default          UART on GP0/GP1   → the rp2040js emulator
//!   --features usb   USB CDC serial    → a REAL Pico on a USB cable
//! ```
//!
//! On a real board GP0/GP1 are bare header pins wired to nothing, so the
//! UART build runs perfectly and reports into the void.
//!
//! # What this can and cannot prove on real hardware
//!
//! The geometry it reads from [`RobotSpec::REAL_BOT`] is **still
//! placeholder** — see that constant's docs. In particular its
//! `ticks_per_revolution` is `1024.0` where the bench says **4290**, so
//! reported distances are roughly 4.2x too large.
//!
//! **The shape is still right, and that is what this firmware
//! demonstrates.** Turn both wheels the same way and the pose travels in
//! a straight line; turn them opposite and it spins on the spot; turn one
//! and it arcs. Those follow from the kinematics, not from the scale, so
//! `DiffDrive` and `Odometry` can be validated against real sensors
//! before a single dimension has been measured.

#![no_std]
#![no_main]
// No unsafe anywhere in this firmware. `forbid`, not `deny`: it cannot be
// switched off locally with an `#[allow]`. Embassy's HAL already wraps the
// peripheral access that would otherwise need it — if that ever stops being
// true, the argument belongs in a commit that changes this line.
#![forbid(unsafe_code)]

use core::fmt::Write as _;
use core::sync::atomic::{AtomicU16, Ordering};
use embassy_executor::Spawner;
use embassy_rp::gpio::{Input, Level, Output, Pull};
use embassy_rp::peripherals::PWM_SLICE3;
use embassy_rp::peripherals::{PIN_16, PIN_17, PIN_18, PIN_19, PIN_25, PIN_6, PIN_7, PIN_8, PIN_9};
use embassy_rp::pwm::{Config as PwmConfig, Pwm};
use embassy_rp::Peri;
use embassy_time::{Instant, Timer};
use panic_halt as _;
use quad_encoder::QuadratureDecoder;
use sim_core::{Odometry, Pose, RobotSpec};

/// Encoder sampling period. 100 µs = 10 kHz (see math-09 on aliasing).
const POLL_US: u64 = 100;

/// How often a pose line goes out.
const REPORT_MS: u64 = 100;

/// PWM counter wrap. 5000 counts at ~150 MHz gives roughly **30 kHz**,
/// deliberately above the audible band — a motor driven at 2 kHz whines,
/// and the winding heats more on the switching edges.
const PWM_TOP: u16 = 5000;

/// The duty sweep, as percentages held for [`STEP_SECS`] each.
///
/// It **ends at zero and parks**, rather than looping. A bench motor on
/// four thin encoder wires should not run unattended, and a firmware whose
/// natural end state is "stopped" cannot be left running by accident.
const SWEEP: [u16; 6] = [0, 25, 50, 75, 100, 0];
const STEP_SECS: u64 = 3;

/// The duty the sweep is currently commanding, so the report line can say
/// so. Written by one task and read by another, which is the whole reason
/// it is an atomic rather than a `static mut`.
///
/// `Relaxed` is sufficient: it orders nothing else, and a report that
/// catches the value one tick early is a cosmetic mislabel, not a wrong
/// measurement. Note that RP2040 (thumbv6m) has no atomic compare-and-swap
/// — plain load and store like this are fine, which is why the emulator
/// build still compiles.
static DUTY_PERCENT: AtomicU16 = AtomicU16::new(0);

/// Where a status line goes. The odometry loop does not care.
///
/// ⚠️ **This is the second copy of this transport**, the first being
/// `firmware/pico-encoder`. The repo's own rule is to give a shared thing
/// a home when the second copy appears, and it is not being followed here
/// deliberately: `embassy-rp`'s chip selection is a *feature* set by the
/// top-level binary, so a shared library would have to forward those
/// features through its own flags. That is a real piece of work, not a
/// tidy-up, and doing it badly at the moment this firmware first meets
/// real hardware is the worse trade.
///
/// **Trigger to extract: a third copy** — `pico-imu` will want one.
trait Report {
    /// Send bytes, or give up quietly. Only the USB transport can stall,
    /// so only it carries a deadline.
    ///
    /// Reports are best-effort and ticks are not: every microsecond spent
    /// blocked here is a microsecond of missed transitions, and a missed
    /// transition corrupts the pose this loop exists to compute.
    async fn send(&mut self, bytes: &[u8]);
}

#[embassy_executor::task]
async fn heartbeat(mut led: Output<'static>) {
    loop {
        led.toggle();
        Timer::after_millis(500).await;
    }
}

/// Drives motor A through [`SWEEP`], then stops and parks forever.
///
/// This is the rig `crates/sim-core/src/spec.rs` step 4 asks for:
/// `max_wheel_speed` measured *"on your battery at the voltage the robot
/// actually runs at, not the datasheet's nominal 6 V"*. Read the encoder
/// rate at the 100% step — under real load, on real cells, with whatever
/// sag they have.
///
/// # The TB6612's three controls
///
/// ```text
///   STBY   low = outputs off entirely. Default state, and the reason a
///          correctly wired board looks dead: nothing moves until this
///          is driven high.
///   AIN1/2 direction. HIGH/LOW is forward, LOW/HIGH reverse,
///          LOW/LOW coasts, HIGH/HIGH brakes.
///   PWMA   speed, as a duty cycle.
/// ```
///
/// # Why it parks rather than loops
///
/// The motor sits loose on a desk attached to four thin encoder wires
/// that have already come adrift once tonight. A sweep that ends leaves
/// the bench safe if nobody is watching; one that repeats does not. The
/// last entry in [`SWEEP`] is `0` and `STBY` drops after it — belt and
/// braces, because a zero duty with the driver still enabled is a stopped
/// motor that can still be commanded, and this one should not be.
#[embassy_executor::task]
async fn drive_sweep(
    mut pwm: Pwm<'static>,
    mut ain1: Output<'static>,
    mut ain2: Output<'static>,
    mut stby: Output<'static>,
) {
    // Forward. Direction is fixed for this sweep — one variable at a time,
    // and reverse is a sign, not a separate experiment.
    ain1.set_high();
    ain2.set_low();

    let mut cfg = PwmConfig::default();
    cfg.top = PWM_TOP;
    cfg.compare_a = 0;
    pwm.set_config(&cfg);

    // Enable only after the duty is known-zero, so the first thing the
    // driver ever sees is "stopped" rather than whatever the register
    // happened to hold.
    stby.set_high();

    for percent in SWEEP {
        cfg.compare_a = PWM_TOP / 100 * percent;
        pwm.set_config(&cfg);
        DUTY_PERCENT.store(percent, Ordering::Relaxed);
        Timer::after_secs(STEP_SECS).await;
    }

    cfg.compare_a = 0;
    pwm.set_config(&cfg);
    DUTY_PERCENT.store(0, Ordering::Relaxed);
    stby.set_low();

    // Park. The odometry task keeps reporting, so the final counts stay
    // readable after the motor has stopped.
    loop {
        Timer::after_secs(60).await;
    }
}

/// Samples both encoders forever, integrating `sim-core`'s odometry.
///
/// Deliberately a plain `async fn` rather than an `#[embassy_executor::task]`:
/// tasks cannot be generic, and being generic over [`Report`] is what
/// keeps this loop from being written twice.
async fn odometry_forever(
    la: Input<'static>,
    lb: Input<'static>,
    ra: Input<'static>,
    rb: Input<'static>,
    out: &mut impl Report,
) -> ! {
    let mut left = QuadratureDecoder::new(la.is_high(), lb.is_high());
    let mut right = QuadratureDecoder::new(ra.is_high(), rb.is_high());

    // sim-core's odometry, unmodified, on a microcontroller — and reading
    // the ONE robot spec rather than a private copy of its numbers. This
    // used to hold its own WHEEL_RADIUS / TRACK_WIDTH / TICKS_PER_REV
    // constants, which is exactly the drift docs/13 set out to remove: the
    // simulator could be retuned and this would quietly disagree.
    let spec = RobotSpec::REAL_BOT;
    let mut odom = Odometry {
        model: spec.drive(),
        ticks_per_revolution: spec.ticks_per_revolution,
        pose: Pose::ORIGIN,
    };

    let mut line: heapless::String<192> = heapless::String::new();
    let mut last_report = Instant::now();
    let mut last_l = 0i32;
    let mut last_r = 0i32;

    loop {
        // ---- sample both encoders (H3's decoder) ----
        left.update(la.is_high(), lb.is_high());
        right.update(ra.is_high(), rb.is_high());

        let now = Instant::now();
        if now.duration_since(last_report).as_millis() >= REPORT_MS {
            // ---- feed the tick deltas to Stage 0's odometry ----
            let dl = left.count - last_l;
            let dr = right.count - last_r;
            last_l = left.count;
            last_r = right.count;
            odom.update(dl as i64, dr as i64);

            let p = odom.pose;
            let _ = write!(
                line,
                "pose x={:+.3} y={:+.3} th={:+.3}  ticks L={} R={}  err={}  duty={}%\r\n",
                p.x,
                p.y,
                p.heading,
                left.count,
                right.count,
                left.errors + right.errors,
                DUTY_PERCENT.load(Ordering::Relaxed)
            );
            out.send(line.as_bytes()).await;
            line.clear();
            last_report = now;
        }

        Timer::after_micros(POLL_US).await;
    }
}

/// The TB6612 side, bundled so `shared_setup`'s signature stays readable.
///
/// GP6 is PWM slice 3 channel A — on RP2040/RP2350 a GPIO's slice is
/// `n / 2` and its channel is A for even `n`, which is why the pin and the
/// slice cannot be chosen independently.
struct MotorPins {
    slice: Peri<'static, PWM_SLICE3>,
    pwm: Peri<'static, PIN_6>,
    ain1: Peri<'static, PIN_7>,
    ain2: Peri<'static, PIN_8>,
    stby: Peri<'static, PIN_9>,
}

/// The four encoder pins and the LED, identical whichever transport is built.
fn shared_setup(
    spawner: Spawner,
    la: Peri<'static, PIN_16>,
    lb: Peri<'static, PIN_17>,
    ra: Peri<'static, PIN_18>,
    rb: Peri<'static, PIN_19>,
    pin_led: Peri<'static, PIN_25>,
    motor: MotorPins,
) -> (
    Input<'static>,
    Input<'static>,
    Input<'static>,
    Input<'static>,
) {
    // ⚠️ `Pull::Up` is a MEASURED result, not a preference. Do not "tidy"
    // it back to `Pull::Down`.
    //
    // RP2350-E9 (docs/09): on stepping A2 — which our board is — an input
    // configured with an internal PULL-DOWN can latch high instead of
    // reading a clean low. Measured on `pico-encoder`, same silicon, same
    // motor, turning the encoder magnet by hand:
    //
    //     Pull::Down   44 counts,  15 errors    (~25% of steps lost)
    //     Pull::Up     53 counts,   0 errors
    //
    // Those 15 were an *undercount*, silently — which here would be a
    // quietly wrong pose rather than an obviously broken one.
    let la = Input::new(la, Pull::Up);
    let lb = Input::new(lb, Pull::Up);
    let ra = Input::new(ra, Pull::Up);
    let rb = Input::new(rb, Pull::Up);

    // ⚠️ On a **Pico 2 W this lights nothing**: GP25 is the CYW43 radio's
    // chip-select there, not an LED — see `firmware/pico-led`, which boots
    // the radio precisely because that is the only way to reach it. Do not
    // read "no blink" as "dead board"; on a W the sign of life is the USB
    // port appearing.
    spawner.spawn(heartbeat(Output::new(pin_led, Level::Low)).unwrap());

    // The driver starts DISABLED and at zero duty. Every output below is
    // created in its safe state before `drive_sweep` enables anything.
    spawner.spawn(
        drive_sweep(
            Pwm::new_output_a(motor.slice, motor.pwm, PwmConfig::default()),
            Output::new(motor.ain1, Level::Low),
            Output::new(motor.ain2, Level::Low),
            Output::new(motor.stby, Level::Low),
        )
        .unwrap(),
    );

    (la, lb, ra, rb)
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
        /// end to apply back-pressure. The emulator is the only consumer.
        async fn send(&mut self, bytes: &[u8]) {
            let _ = self.0.blocking_write(bytes);
        }
    }

    pub async fn run(p: embassy_rp::Peripherals, spawner: Spawner) -> ! {
        let (la, lb, ra, rb) = shared_setup(
            spawner,
            p.PIN_16,
            p.PIN_17,
            p.PIN_18,
            p.PIN_19,
            p.PIN_25,
            MotorPins {
                slice: p.PWM_SLICE3,
                pwm: p.PIN_6,
                ain1: p.PIN_7,
                ain2: p.PIN_8,
                stby: p.PIN_9,
            },
        );

        let uart = Uart::new_blocking(p.UART0, p.PIN_0, p.PIN_1, UartConfig::default());
        let (tx, _rx) = uart.split();

        odometry_forever(la, lb, ra, rb, &mut UartReport(tx)).await
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

    /// The endpoint's max packet size. A longer write must be split or the
    /// transfer is rejected, and our pose line is up to 192 bytes.
    const MAX_PACKET: usize = 64;

    /// Longest a status line may spend trying to reach the host before it
    /// is abandoned. Sized against the 1 ms USB full-speed frame: a
    /// healthy write finishes well inside this, so the timeout only fires
    /// on a host that has genuinely stopped reading.
    const REPORT_TIMEOUT_MS: u64 = 5;

    struct UsbReport<'d, D: UsbDriver<'d>>(CdcAcmClass<'d, D>);

    impl<'d, D: UsbDriver<'d>> Report for UsbReport<'d, D> {
        /// Skipped entirely when no terminal has opened the port, and
        /// abandoned after [`REPORT_TIMEOUT_MS`] if one has but has
        /// stopped reading. `dtr()` is the CDC line a host raises when
        /// something opens `/dev/cu.usbmodem…`, so a board on a charger
        /// costs nothing rather than one timeout per report.
        ///
        /// Cancelling the write is not proven safe the way `pico-robot`
        /// proved its USB *read* — a cancelled write may garble a line.
        /// That trade is right here for the same reason as in
        /// `pico-encoder`: a mangled line costs a glance, blocking the
        /// sampler costs ticks, and ticks are the pose.
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
        let (la, lb, ra, rb) = shared_setup(
            spawner,
            p.PIN_16,
            p.PIN_17,
            p.PIN_18,
            p.PIN_19,
            p.PIN_25,
            MotorPins {
                slice: p.PWM_SLICE3,
                pwm: p.PIN_6,
                ain1: p.PIN_7,
                ain2: p.PIN_8,
                stby: p.PIN_9,
            },
        );

        let driver = Driver::new(p.USB, Irqs);
        // 0x2e8a is Raspberry Pi's vendor ID. Product IDs differ per
        // firmware — pico-robot 0x000a, pico-encoder 0x000b — so several
        // boards can be plugged in and still told apart.
        let mut config = Config::new(0x2e8a, 0x000c);
        config.manufacturer = Some("robotiq");
        config.product = Some("pico-odom");
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

        // Sampling starts immediately rather than waiting for a host: the
        // decoders' job is to miss nothing, and gating them on a terminal
        // being open would silently lose every tick between power-up and
        // the first `screen`. `send` already declines to write while
        // `dtr()` is low.
        join(usb.run(), odometry_forever(la, lb, ra, rb, &mut out)).await;

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
