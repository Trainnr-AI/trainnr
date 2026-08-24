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
//! # Three builds, one odometry loop
//!
//! ```text
//!   default             UART on GP0/GP1  → the rp2040js emulator
//!   --features usb      USB CDC serial   → a REAL Pico on a USB cable
//!   --features teleop   USB + commands   → the host drives the motors
//! ```
//!
//! On a real board GP0/GP1 are bare header pins wired to nothing, so the
//! UART build runs perfectly and reports into the void.
//!
//! The first two run a **calibration sweep**: a fixed 0→100% duty ramp
//! that measured `max_wheel_speed`, `motor_tau` and the deadband. It is
//! kept, not superseded — `wheel_radius` still needs it once wheels exist.
//!
//! `teleop` replaces the sweep with a **command channel**. The host sends
//! `T v w` twists in `hil-protocol`'s vocabulary; the chip converts them
//! through the same `DiffDrive::inverse` the simulator uses and drives the
//! H-bridge — and stops it when the host goes quiet. See
//! [`follow_host_forever`].
//!
//! ```text
//!   camera ─▶ host control law ─▶ T v w ─▶ chip ─▶ TB6612 ─▶ motors
//!                                            │
//!                                            └── CommandWatchdog:
//!                                                silence ⇒ coast, STBY low
//! ```
//!
//! # Telemetry over the radio, and how to read the LED
//!
//! Built with `wifi`, the board reports over UDP as well as (or instead
//! of) the cable. It has one output when untethered, so that LED encodes
//! the whole state machine:
//!
//! ```text
//!   ··   ··   ··       double flash ~1/s   DELIVERING — datagrams leaving
//!   ▬▬  ▬▬  ▬▬         even blink ~1/s     joined, but nothing getting out
//!   ▪▪▪▪▪▪▪▪▪▪▪▪       fast blink ~4/s     associating, not joined yet
//!   ·          ·       one blip / 2 s      no WIFI_SSID compiled in
//! ```
//!
//! **The healthy state blinks rather than staying lit on purpose**: a
//! solid LED cannot prove the firmware is still running, so a board that
//! panicked with the light on would look exactly like one working
//! perfectly. A heartbeat is only produced by code still executing.
//!
//! It means datagrams are *leaving the chip*, not that anyone receives
//! them — a broadcast socket cannot know whether a host is listening.
//! `odom_view --udp` measures delivery, against the chip's own sequence
//! numbers.
//!
//! ```sh
//! # host the network (default) — no router needed, laptop must join it
//! WIFI_SSID=pico2w WIFI_PASSWORD=8-to-63-chars \
//!     tools/build-pico2.sh pico-odom usb,wifi
//!
//! # or join an existing one — laptop keeps its internet
//! WIFI_MODE=station WIFI_SSID=... WIFI_PASSWORD=... \
//!     tools/build-pico2.sh pico-odom usb,wifi
//!
//! cargo run -p hil-host --example odom_view -- /dev/cu.usbmodem11 --udp
//! ```
//!
//! Measured on 2026-08-11, board #2 joining a domestic 2.4 GHz AP, 60 s:
//! **2992 reports on the cable with no loss, 2876 over the air — 3.88%
//! lost, median 66 ms behind the cable, p99 128 ms.** Good enough to watch
//! a robot; not good enough to steer one.
//!
//! # What this can and cannot prove on real hardware
//!
//! The geometry it reads from [`RobotSpec::REAL_BOT`] carries the
//! bench-measured `ticks_per_revolution` (4290) and max wheel speed;
//! wheel radius and track are still ruler-grade — see that constant's
//! docs for which numbers have real provenance and which are estimates.
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

// ⚠️ This is a DESIGN boundary, not a missing feature. `teleop` puts the
// host inside the 50 Hz control loop, and doing that over a lossy radio is
// a decision with its own failure modes: `CommandWatchdog` turns silence
// into a stop, so a dropped datagram burst becomes a robot that halts
// mid-manoeuvre. That may well be the right trade — but it needs a bench
// session and a measured jitter number, not a feature flag that happens to
// compile. Telemetry is one-directional and safe to ship first.
#[cfg(all(feature = "wifi", feature = "teleop"))]
compile_error!(
    "`wifi` is telemetry-only. Commands over UDP need the watchdog timeout \
     re-derived from measured radio jitter first — see docs/e2e-research/28."
);
#[cfg(all(feature = "wifi", not(feature = "pico2")))]
compile_error!("`wifi` needs the CYW43 radio, which only the Pico 2 W here has");

use core::sync::atomic::{AtomicBool, AtomicU16, AtomicU32, Ordering};
use embassy_executor::Spawner;
use embassy_rp::gpio::{Input, Level, Output, Pull};
use embassy_rp::peripherals::{
    PIN_10, PIN_11, PIN_12, PIN_2, PIN_26, PIN_27, PIN_3, PIN_6, PIN_7, PIN_8, PIN_9,
};
#[cfg(feature = "wifi")]
use embassy_rp::peripherals::{PIN_23, PIN_24, PIN_25, PIN_29};
use embassy_rp::peripherals::{PWM_SLICE3, PWM_SLICE5};
use embassy_rp::pwm::{Config as PwmConfig, Pwm};
use embassy_rp::Peri;
use embassy_time::{Instant, Timer};
#[cfg(any(feature = "teleop", feature = "chase", feature = "fetch"))]
use embassy_time::Duration;
use firmware_support::motor::{Channel, Encoders, Motors, PWM_TOP};
use firmware_support::Report;
use panic_halt as _;
use quad_encoder::QuadratureDecoder;
use sim_core::{Odometry, Pose, RobotSpec};
#[cfg(any(feature = "teleop", feature = "chase", feature = "fetch"))]
use sim_core::{BodyTwist, DUTY_FULL};
#[cfg(feature = "teleop")]
use sim_core::CommandWatchdog;

/// Encoder sampling period. 100 µs = 10 kHz (see math-09 on aliasing).
const POLL_US: u64 = 100;

/// How often a pose line goes out — and how often odometry integrates.
///
/// **20 ms, not 100 ms, and the reason is measurement.** The argument for
/// dropping it was that a first-order motor lag with tau ~0.15 s would
/// give only one or two samples per transient at 100 ms — nowhere near
/// enough to fit a curve to.
///
/// ⚠️ **That 0.15 s was the placeholder, and the sweep this rate enabled
/// went on to measure tau at ~0.03–0.05 s** — see `sim_core::Motor::tau`.
/// So the conclusion held and the reasoning did not: 20 ms does not give
/// ~8 samples inside the first tau, it gives **one or two**, which is
/// exactly why the answer came back as "30 ms or 50 ms" and not as a
/// fitted curve. The measurement is trustworthy because eight independent
/// transitions clustered, not because any one of them was well resolved.
///
/// **To resolve tau properly, this must drop to ~5 ms**, which the note
/// below says the UART build cannot carry. That is the real cost of the
/// shared constant, and it is not being paid today: ±20 ms on a 40 ms
/// time constant is already enough to show the simulator was 4x out.
///
/// It also makes the odometry itself better: `Odometry::update` integrates
/// an arc per call, and five times more calls is five times finer.
///
/// Not lower than this. At 10 ms the UART build would need ~10 KB/s, which
/// is the whole 115200-baud budget, and the USB write deadline
/// (`REPORT_TIMEOUT_MS`, 5 ms) becomes a quarter of the reporting period
/// rather than a twentieth — and a blocked write is missed encoder ticks.
const REPORT_MS: u64 = 20;

/// PWM counter wrap. 5000 counts at ~150 MHz gives roughly **30 kHz**,
/// The duty sweep, as percentages held for [`STEP_SECS`] each.
///
/// It **ends at zero and parks**, rather than looping. A bench motor on
/// four thin encoder wires should not run unattended, and a firmware whose
/// natural end state is "stopped" cannot be left running by accident.
#[cfg(not(any(feature = "teleop", feature = "chase", feature = "fetch")))]
const SWEEP: [u16; 6] = [0, 25, 50, 75, 100, 0];
#[cfg(not(any(feature = "teleop", feature = "chase", feature = "fetch")))]
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

/// Set once a host has actually opened the serial port.
///
/// [`drive_sweep`] waits for this before it enables anything, so **the
/// motor cannot move unless someone is watching.** Plugging the board into
/// a charger, or into a laptop with no terminal open, leaves it inert.
///
/// That started as a way to make the run catchable — an 18-second sweep
/// that begins at power-up is over before you can start reading it — but
/// the safety property is the better reason to keep it. A bench motor that
/// spins the instant it receives power is one loose battery lead away from
/// a surprise.
static HOST_WATCHING: AtomicBool = AtomicBool::new(false);

/// Total distance the encoders have seen, published by the odometry loop
/// so [`drive_sweep`] can check whether the robot did what it was told.
///
/// **Plain `store`, never `fetch_add`.** RP2040 is thumbv6m, which has no
/// atomic read-modify-write — the emulator build would not compile. The
/// odometry task already holds the running totals, so it publishes them
/// rather than accumulating here.
static TOTAL_TICKS: AtomicU32 = AtomicU32::new(0);

/// Set when the sweep commanded motion and the encoders disagreed.
///
/// # Why this exists
///
/// On 2026-08-09 the battery pack's switch was off. The firmware
/// commanded a full 0→100% sweep, the encoders read zero throughout, and
/// **nothing anywhere said so** — the run looked exactly like a successful
/// one until a human noticed the motors were silent.
///
/// That is the same failure this repo keeps finding in other clothes: two
/// facts held in the same program that nobody compared. The fix is to
/// compare them.
///
/// It is also a *safety* fix rather than a diagnostic one. A motor that is
/// commanded and not turning is either disconnected — harmless — or
/// **stalled**, drawing near its ~500 mA stall current and heating. The
/// firmware cannot tell which, so it stops driving and says why.
static STALLED: AtomicBool = AtomicBool::new(false);

/// Stop the motors if the host goes quiet for this long.
///
/// The same 200 ms `pico-robot` uses, and the same reasoning: ten missed
/// ticks at 50 Hz, and about 9 cm of travel at top speed — well beyond any
/// real scheduling hiccup, well short of a table edge.
///
/// ⚠️ The 9 cm assumed `max_wheel_speed = 30.0`. The bench measured
/// **7.77 rad/s**, so this robot covers roughly **2 cm** in a lost 200 ms.
/// The number is left alone: it was chosen as a bound on *jitter*, which
/// has not changed, and it is now four times more conservative in
/// distance than it was designed to be.
#[cfg(feature = "teleop")]
const COMMAND_TIMEOUT_MS: u64 = 200;

/// How long a read waits before coming up for air to check the watchdog.
/// Shorter than [`COMMAND_TIMEOUT_MS`], so staleness is acted on within
/// one poll of becoming true.
#[cfg(any(feature = "teleop", feature = "chase", feature = "fetch"))]
const POLL: Duration = Duration::from_millis(50);



/// Duty above which a commanded motor **must** be turning, in the
/// `±DUTY_FULL` units the wire carries.
///
/// 150 of 1000 is 15%, clear of the measured ~4.3% deadband with margin —
/// below that, "not moving" is legitimate physics rather than a fault.
#[cfg(any(feature = "teleop", feature = "chase", feature = "fetch"))]
const STALL_DUTY_FLOOR: i32 = 150;

/// Consecutive polls of commanded-but-not-moving before cutting the
/// motors. At [`POLL`] = 50 ms, four polls is 200 ms — long enough that a
/// motor still overcoming its own inertia is never mistaken for a stall,
/// short enough that a genuinely locked rotor is not held at full duty.
#[cfg(any(feature = "teleop", feature = "chase", feature = "fetch"))]
const STALL_POLLS: u32 = 4;

/// Duty above which the motor **must** move, or something is wrong.
///
/// The measured deadband is ~4.6% (docs/07, 2026-08-09), so 15% is clear
/// of it with margin — below that, "not moving" is legitimate physics
/// rather than a fault.
#[cfg(not(any(feature = "teleop", feature = "chase", feature = "fetch")))]
const MUST_MOVE_ABOVE: u16 = 15;

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
#[cfg(not(any(feature = "teleop", feature = "chase", feature = "fetch")))]
#[embassy_executor::task]
async fn drive_sweep(mut motors: Motors) {
    // Nothing moves until a host is listening. See `HOST_WATCHING`.
    while !HOST_WATCHING.load(Ordering::Relaxed) {
        Timer::after_millis(100).await;
    }

    let mut cfg = PwmConfig::default();
    cfg.top = PWM_TOP;
    motors.left.arm(&mut cfg);
    motors.right.arm(&mut cfg);

    // Enable only after BOTH channels read zero, so the first thing the
    // driver ever sees is "stopped" rather than whatever the registers
    // happened to hold. `STBY` gates all four switches of both bridges.
    motors.standby.set_high();

    for percent in SWEEP {
        motors.left.set_duty(&mut cfg, percent);
        motors.right.set_duty(&mut cfg, percent);
        DUTY_PERCENT.store(percent, Ordering::Relaxed);

        // Commanded versus achieved. The two numbers were always both in
        // this program; nothing compared them until a dead battery pack
        // produced a textbook-looking run with zero motion in it.
        let before = TOTAL_TICKS.load(Ordering::Relaxed);
        Timer::after_secs(STEP_SECS).await;
        let moved = TOTAL_TICKS.load(Ordering::Relaxed) != before;

        if percent > MUST_MOVE_ABOVE && !moved {
            STALLED.store(true, Ordering::Relaxed);
            break; // fall through to the shutdown below
        }
    }

    motors.left.set_duty(&mut cfg, 0);
    motors.right.set_duty(&mut cfg, 0);
    DUTY_PERCENT.store(0, Ordering::Relaxed);
    motors.standby.set_low();

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
async fn odometry_forever(pins: Encoders, out: &mut impl Report) -> ! {
    let Encoders {
        left_a: la,
        left_b: lb,
        right_a: ra,
        right_b: rb,
    } = pins;
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

    // Whatever the radio wants said, sent down the same transports as the
    // pose. Drained HERE rather than by a task of its own because this is
    // the one place that already owns `out` — and because a diagnostic
    // channel with its own transport is a second thing that can be broken
    // while the first looks fine.
    // ⚠️ Unconditional now. It used to be a no-op without `wifi`, which
    // meant a note queued by anything else vanished silently — and the
    // camera queues one at boot.
    //
    // ⚠️ And gated on `HOST_WATCHING`, which is not decoration.
    //
    // `try_receive` REMOVES the note from the queue; `Report::send` then
    // silently declines while `dtr()` is low. Ungated, those two compose
    // into a shredder: every note queued before a terminal opens is
    // consumed and dropped, and the camera queues its identity at boot —
    // ~34 seconds before a human gets to `screen` on a good day. Measured
    // exactly that way on 2026-08-14: telemetry at `n=1698`, camera line
    // nowhere, because it had been drained into a closed port at `n=0`.
    //
    // Waiting costs nothing. `NOTES` is a drop queue, so a board nobody
    // ever attaches to fills eight slots and discards the rest, which is
    // the same outcome as before minus the pretence of having reported.
    macro_rules! drain_notes {
        ($out:expr) => {
            if HOST_WATCHING.load(Ordering::Relaxed) {
                while let Ok(note) = crate::diag::NOTES.try_receive() {
                    $out.send(&note).await;
                }
            }
        };
    }

    let mut line: heapless::String<192> = heapless::String::new();
    // Counts reports, not loop iterations, so a gap on the host side means
    // exactly one thing: a line this loop emitted did not arrive. Never
    // reset — a sequence that goes backwards is the chip having rebooted,
    // which is worth being able to see.
    let mut seq = 0u64;
    let mut last_report = Instant::now();
    let mut last_l = 0i32;
    let mut total_motion: u32 = 0;
    let mut last_r = 0i32;

    loop {
        // ---- sample both encoders (H3's decoder) ----
        left.update(la.is_high(), lb.is_high());
        right.update(ra.is_high(), rb.is_high());

        let now = Instant::now();
        if now.duration_since(last_report).as_millis() >= REPORT_MS {
            // ---- feed the tick deltas to Stage 0's odometry ----
            //
            // The sign is applied ONCE, here, where counts leave the
            // decoder — so the odometry, the reported totals and anything
            // the host derives from them cannot disagree about which way
            // a wheel turned. See `LEFT_ENCODER_SIGN`.
            // Two independent facts: `LEFT_ENCODER_SIGN` is the mirrored
            // assembly, `DRIVETRAIN_SIGN` is the whole pair facing
            // backwards on the chassis. Both are documented where they are
            // defined; neither is a preference.
            let left_count = left.count
                * firmware_support::motor::LEFT_ENCODER_SIGN
                * firmware_support::motor::DRIVETRAIN_SIGN;
            let right_count = right.count * firmware_support::motor::DRIVETRAIN_SIGN;
            let dl = left_count - last_l;
            let dr = right_count - last_r;
            last_l = left_count;
            last_r = right_count;
            odom.update(dl as i64, dr as i64);

            // Publish for the stall check: a MONOTONE odometer of
            // motion. The old sum of |cumulative| counts could hold
            // constant during a spin after forward travel (|left|
            // falling while |right| rises cancel exactly), reading as
            // "not moving" while the robot turned — a phantom stall
            // found in review 2026-08-24, never on the floor.
            total_motion =
                total_motion.wrapping_add(dl.unsigned_abs() + dr.unsigned_abs());
            TOTAL_TICKS.store(total_motion, Ordering::Relaxed);

            let p = odom.pose;
            // ONE definition of this line, shared with both hosts.
            //
            // It used to be a `write!` here and two independent parsers
            // over there. When the error counter was split per wheel, one
            // of those parsers kept looking for the old key, dropped every
            // line, and drew an empty viewer while the gate stayed green.
            // `hil_protocol::Status` round-trips, so that cannot recur.
            seq += 1;
            let _ = hil_protocol::Status {
                seq,
                x: p.x,
                y: p.y,
                heading: p.heading,
                ticks_left: i64::from(left_count),
                ticks_right: i64::from(right_count),
                errors_left: left.errors as u64,
                errors_right: right.errors as u64,
                duty_percent: u64::from(DUTY_PERCENT.load(Ordering::Relaxed)),
                stalled: STALLED.load(Ordering::Relaxed),
            }
            .write_into(&mut line);
            out.send(line.as_bytes()).await;
            line.clear();
            // Restate the camera every ~10 s at 50 Hz. A boot-time note
            // is drained the first time ANY host touches the port, so
            // without this the line is unobservable in practice.
            // ⚠️ The interval and the rate divisor are the same fact, so
            // they come from one constant. 50 Hz reporting x 10 s.
            #[cfg(feature = "camera")]
            if seq.is_multiple_of((1000 / REPORT_MS) * u64::from(camera::ANNOUNCE_SECONDS)) {
                camera::announce();
            }
            drain_notes!(out);
            last_report = now;
        }

        Timer::after_micros(POLL_US).await;
    }
}

/// The shared back half of every motor commander: stall guard, standby
/// gating, the mounting sign, and the duty telemetry.
///
/// `teleop` and `chase` differ only in **who computes the twist** — a
/// host on a cable, or the camera. Everything that touches hardware is
/// here, once. `firmware/support/motor.rs` already warned that direction
/// pins and deadband notes "diverge silently once there are two of them",
/// and chase's first draft proved it by copying forty-five lines of this
/// verbatim.
#[cfg(any(feature = "teleop", feature = "chase", feature = "fetch"))]
struct Drivetrain {
    motors: Motors,
    cfg: PwmConfig,
    enabled: bool,
    ticks_at_last_check: u32,
    stalled_polls: u32,
    /// Polls left in the post-stall cooldown; motors are held at zero
    /// while it runs down. See the retry note on [`Self::apply`].
    stall_cooldown: u32,
}

/// How long a stall cuts the motors before ONE retry is allowed:
/// 40 polls × 50 ms = 2 s. ⚠️ Chosen so a genuinely blocked motor sees
/// at most 200 ms of stall current every 2 s — a 10% duty cycle on a
/// brief ~500 mA draw, thermally trivial — while a transient snag (a
/// carpet edge, a cable, one wheel late to break away) costs 2 s
/// instead of the rest of the run.
#[cfg(any(feature = "teleop", feature = "chase", feature = "fetch"))]
const STALL_COOLDOWN_POLLS: u32 = 40;

#[cfg(any(feature = "teleop", feature = "chase", feature = "fetch"))]
impl Drivetrain {
    /// Take the motors, stopped.
    fn new(mut motors: Motors) -> Self {
        let mut cfg = PwmConfig::default();
        cfg.top = PWM_TOP;
        motors.left.set_signed(&mut cfg, 0);
        motors.right.set_signed(&mut cfg, 0);
        Drivetrain {
            motors,
            cfg,
            enabled: false,
            ticks_at_last_check: TOTAL_TICKS.load(Ordering::Relaxed),
            stalled_polls: 0,
            stall_cooldown: 0,
        }
    }

    /// A body twist to per-wheel duty, in the ±`DUTY_FULL` wire units.
    fn duty_for(spec: &RobotSpec, forward_speed: f64, turn_rate: f64) -> (i32, i32) {
        let wheels = spec.fit_wheels(spec.drive().inverse(BodyTwist {
            forward_speed,
            turn_rate,
        }));
        (spec.duty(wheels.left), spec.duty(wheels.right))
    }

    /// Drive at `(left, right)` duty — through the stall guard.
    ///
    /// # Commanded, but not moving
    ///
    /// The sweep has had this check since a dead battery produced a
    /// textbook-looking run with zero motion in it, and `teleop` once
    /// shipped without it — a host could hold a stalled motor at full
    /// duty indefinitely, drawing near its ~500 mA stall current, and
    /// nothing would say so. The chip STOPS rather than escaping:
    /// reversing is recovery, recovery needs to know what is behind the
    /// robot, and only the commander knows that. `STALLED` appears in
    /// the report line for the commander to act on.
    ///
    /// # ⚠️ Cooldown-and-retry, added the day the robot met the floor
    ///
    /// The original latch held until the wheels moved — which, with the
    /// duty cut to zero, meant until reboot. That philosophy assumed a
    /// host with hands. The camera-commander has none, and on the ground
    /// a 200 ms friction hiccup (one wheel late to break away in a turn)
    /// bricked three autonomous runs in one afternoon. Now a stall cuts
    /// the motors for [`STALL_COOLDOWN_POLLS`] (2 s), then clears for
    /// ONE retry. A truly blocked motor re-stalls 200 ms into each retry
    /// — thermally nothing — and the `STALLED` flag pulses in the report
    /// line so a watching host still sees the truth.
    ///
    /// # Two layers of stop
    ///
    /// Zero duty is a *software* stop — the bridge still obeys the next
    /// write. `STBY` low is a *hardware* stop, disabling all four
    /// switches of both bridges. It is touched only on a transition,
    /// because reporting it every 50 ms would bury the one event that
    /// matters in a thousand that do not.
    fn apply(&mut self, mut left: i32, mut right: i32) {
        let ticks_now = TOTAL_TICKS.load(Ordering::Relaxed);
        let commanded_hard =
            left.unsigned_abs().max(right.unsigned_abs()) > STALL_DUTY_FLOOR.unsigned_abs();
        if commanded_hard && ticks_now == self.ticks_at_last_check {
            self.stalled_polls += 1;
            if self.stalled_polls >= STALL_POLLS && !STALLED.load(Ordering::Relaxed) {
                STALLED.store(true, Ordering::Relaxed);
                self.stall_cooldown = STALL_COOLDOWN_POLLS;
            }
        } else {
            self.stalled_polls = 0;
            if ticks_now != self.ticks_at_last_check {
                STALLED.store(false, Ordering::Relaxed);
                self.stall_cooldown = 0;
            }
        }
        self.ticks_at_last_check = ticks_now;
        if STALLED.load(Ordering::Relaxed) {
            left = 0;
            right = 0;
            // The cooldown runs down with the motors safe at zero; when
            // it expires the latch opens for one retry.
            self.stall_cooldown = self.stall_cooldown.saturating_sub(1);
            if self.stall_cooldown == 0 {
                STALLED.store(false, Ordering::Relaxed);
                self.stalled_polls = 0;
            }
        }

        let should_run = left != 0 || right != 0;
        if should_run != self.enabled {
            self.motors.standby.set_level(Level::from(should_run));
            self.enabled = should_run;
            // Clear only on the way UP: clearing on the disable
            // transition un-latched a stall in the same call that set
            // it, blipping STALLED=false at the host once per latch.
            if should_run {
                STALLED.store(false, Ordering::Relaxed);
            }
        }
        // The same mounting fact as everywhere else, applied on the way
        // out — see `firmware_support::motor::DRIVETRAIN_SIGN`.
        let facing = firmware_support::motor::DRIVETRAIN_SIGN;
        self.motors.left.set_signed(&mut self.cfg, left * facing);
        self.motors.right.set_signed(&mut self.cfg, right * facing);

        // A magnitude percentage — all the status line has room to say.
        // The sign is visible in the encoder counts either way.
        DUTY_PERCENT.store(
            (left.unsigned_abs().max(right.unsigned_abs()) * 100 / DUTY_FULL.unsigned_abs()) as u16,
            Ordering::Relaxed,
        );
    }
}

/// Drive the motors from `T v w` twists sent by the host, and **stop them
/// when the host goes quiet**.
///
/// This is the failsafe `firmware/pico-robot` describes and could not
/// perform: its own comment reads *"⚠️ AT H4 THIS IS WHERE THE H-BRIDGE
/// GETS WRITTEN TO ZERO"*, because on that firmware the host owned the
/// motors and the chip owned nothing physical. Here the chip owns the
/// H-bridge, so here the marker gets filled in.
///
/// ```text
///   host                              chip
///   ────                              ────
///   T 0.20 0.00   ──── USB CDC ────▶  DiffDrive::inverse   twist → wheels
///                                     RobotSpec::fit_wheels  clamp to real
///                                     RobotSpec::duty        rad/s → ±1000
///                                     CommandWatchdog::gate  ← the failsafe
///                                     TB6612 AIN/BIN + PWM
///   (silence)     ─────────────────▶  zero duty, STBY low, wheels coast
/// ```
///
/// # Two layers of stop, on purpose
///
/// Zeroing the duty is a *software* stop: the bridge is still enabled and
/// would obey the next value written to it. Dropping `STBY` is a
/// *hardware* stop — it disables all four switches of both bridges at
/// once, and no PWM register can undo it. A failsafe that shares its
/// failure modes with the thing it is guarding is not one, so both fire.
///
/// # Why the watchdog is not optional here
///
/// It has existed in `sim-core` since Stage 0 and has never guarded
/// anything that could move. A host that crashes mid-command leaves the
/// last twist latched in the H-bridge, and an H-bridge holds its output
/// indefinitely — the robot does not coast to a stop, it drives into the
/// wall at whatever it was last told. [`COMMAND_TIMEOUT_MS`] is the whole
/// distance between those two outcomes.
#[cfg(feature = "teleop")]
async fn follow_host_forever(
    rx: &mut impl CommandSource,
    motors: Motors,
    spec: RobotSpec,
) -> ! {
    use hil_protocol::{LineReader, Message};

    let mut drivetrain = Drivetrain::new(motors);
    let mut watchdog = CommandWatchdog::new(COMMAND_TIMEOUT_MS);
    let mut reader: LineReader<64> = LineReader::new();
    let mut rx_bytes = [0u8; 64];
    // What the host last asked for, before the watchdog has its say.
    let mut wanted = (0i32, 0i32);

    loop {
        // Bounded, so staleness is noticed within one poll of becoming
        // true rather than whenever the next byte happens to arrive. A
        // watchdog that only runs when the thing it watches is alive is
        // not a watchdog.
        let n = rx.recv(&mut rx_bytes, POLL).await;
        for &byte in &rx_bytes[..n] {
            let Some(line) = reader.push(byte) else {
                continue;
            };
            // Anything that is not a twist is ignored rather than
            // rejected — including this firmware's OWN status lines, so
            // looping the port back on itself cannot command the motors.
            if let Ok(Message::Twist { v, w }) = Message::parse(line) {
                wanted = Drivetrain::duty_for(&spec, v, w);
                // Fed on a VALID twist only. A host dribbling malformed
                // bytes is a host that has lost its mind, and must not
                // count as one that is still in control.
                watchdog.feed(firmware_support::now_ms());
            }
        }

        let (left, right) = watchdog.gate(firmware_support::now_ms(), wanted);
        drivetrain.apply(left, right);
    }
}

/// The read half of the host link, mirroring [`Report`] for the other
/// direction.
///
/// Separate from `Report` because the two halves have different owners
/// once `CdcAcmClass` is split, and because the deadline belongs here:
/// `pico-robot` measured that wrapping a blocking read in `with_timeout`
/// from outside cost the emulator 95% of its throughput.
#[cfg(feature = "teleop")]
trait CommandSource {
    /// Fill `buf` with whatever has arrived, waiting at most `poll`.
    /// Returns how many bytes, possibly 0.
    async fn recv(&mut self, buf: &mut [u8], poll: Duration) -> usize;
}

/// The TB6612 side, bundled so `shared_setup`'s signature stays readable.
///
/// GP6 is PWM slice 3 channel A — on RP2040/RP2350 a GPIO's slice is
/// `n / 2` and its channel is A for even `n`, which is why the pin and the
/// slice cannot be chosen independently.
struct MotorPins {
    slice_a: Peri<'static, PWM_SLICE3>,
    pwma: Peri<'static, PIN_6>,
    ain1: Peri<'static, PIN_7>,
    ain2: Peri<'static, PIN_8>,
    stby: Peri<'static, PIN_9>,
    slice_b: Peri<'static, PWM_SLICE5>,
    pwmb: Peri<'static, PIN_10>,
    bin1: Peri<'static, PIN_11>,
    bin2: Peri<'static, PIN_12>,
}

// ---------------------------------------------------------------------
// Notes — prose the chip wants the host to read, on the wire the poses
// already use.
// ---------------------------------------------------------------------
//
// # Why this is not inside `wifi_link` any more
//
// It was, and it was right to be: the radio invented it on 2026-08-10
// because an untethered board's only output was **one LED**, and "is it
// blinking fast or slow" was load-bearing evidence.
//
// The camera is the second subsystem that needs exactly the same thing —
// say one sentence, never block the loop that is counting encoder ticks,
// and drop it on the floor if nobody is listening. Two users is when a
// thing stops being the radio's private business.
//
// ⚠️ The drop-if-full rule is the whole design. Diagnostics must never be
// able to block the thing they diagnose, which on this board means the
// 10 kHz sampler: a note that waits for a reader costs encoder ticks, and
// ticks are the measurement.
mod diag {
    use embassy_sync::blocking_mutex::raw::CriticalSectionRawMutex;
    use embassy_sync::channel::Channel;

    /// One line waiting to go out.
    pub type Line = heapless::Vec<u8, 192>;

    /// Prose queued for the host, drained by the report loop and sent
    /// down whatever transports exist.
    ///
    /// Lines start with `#` so a host can tell prose from a pose without
    /// parsing it, and so `Status::parse` rejects them — which it does
    /// anyway, having no keys to find.
    pub static NOTES: Channel<CriticalSectionRawMutex, Line, 8> = Channel::new();

    /// Queue a note, WAITING for room.
    ///
    /// ⚠️ For bulk output only — a thumbnail is thirty lines and the
    /// queue is eight deep, so `try_send` would drop five sixths of an
    /// image and produce a picture with holes in it. Anything on a
    /// control path must use [`note`] instead: this one can block, and
    /// blocking is exactly what diagnostics are forbidden to do.
    #[cfg(any(feature = "camera", feature = "arm"))]
    pub async fn note_blocking(text: &str) {
        let mut line = Line::new();
        if line.extend_from_slice(text.as_bytes()).is_ok()
            && line.extend_from_slice(b"\r\n").is_ok()
        {
            NOTES.send(line).await;
        }
    }

    /// Queue a note, dropping it if nobody is draining.
    #[cfg(any(feature = "camera", feature = "arm"))]
    pub fn note(text: &str) {
        let mut line = Line::new();
        if line.extend_from_slice(text.as_bytes()).is_ok()
            && line.extend_from_slice(b"\r\n").is_ok()
        {
            let _ = NOTES.try_send(line);
        }
    }
}

// ---------------------------------------------------------------------
// The camera, on the same board as the motors.
// ---------------------------------------------------------------------
//
// # ⚠️ Identified ONCE, at boot, and never touched again
//
// A blocking bus transaction inside `odometry_forever` would stall the
// 10 kHz sampler for however long it took — a lesson this repo has paid
// for twice: 50 Hz encoder polling counted 404 ticks where ~17,000 were
// expected, and the servo task's blocking writes were measured raising
// decode errors. The bus is therefore built in ASYNC mode (the camera's
// boot-time calls use its blocking face, which is harmless before the
// loops start), and nothing touches it from the sampler's executor
// without an `await`.
//
// The camera's identity is a boot-time fact — it cannot change while the
// robot drives — so it is read once, announced as a note, and the bus is
// then left alone. Anything that wants live camera data later needs DMA or
// PIO, not a blocking read wedged into the control loop.
//
// # What it does not conflict with
//
// `pico-odom` drives motors from PWM slices 3 (GP6) and 5 (GP10); XCLK is
// slice 2 (GP21). I2C0 and GP4/GP5 are unused by the motor firmware. So
// the two subsystems share nothing but the report stream — which is the
// point of running them together.
#[cfg(all(feature = "chase", feature = "teleop"))]
compile_error!(
    "`chase` and `teleop` are two commanders of one drivetrain — the camera \
     and the host would both write duty, and whichever wrote last would win \
     silently. Two things that must agree, with nothing comparing them: \
     refused at compile time instead."
);

#[cfg(all(feature = "camera", feature = "wifi"))]
compile_error!(
    "`camera` and `wifi` both claim PIO0 and DMA_CH0 — the radio uses them for \
     its SPI, the camera for parallel capture. Pick one. Refused here rather \
     than discovered as a silently corrupt frame or a radio that will not join."
);

// The shared I2C bus interrupt: async mode hands the bus wait back to
// the executor instead of blinding it — see `servo.rs` for the measured
// reason. Bound here because the bus is a board resource built here.
#[cfg(feature = "camera")]
embassy_rp::bind_interrupts!(struct I2cIrqs {
    I2C0_IRQ => embassy_rp::i2c::InterruptHandler<embassy_rp::peripherals::I2C0>;
});

/// The fetch errand as a state machine over the chase primitives.
///
/// SEEK is turn-and-glance (never turning while looking — the camera
/// is too slow to track a stiction-rate turn). ARRIVE latches on three
/// consecutive stationary glances of a big centred blob — one glimpse is
/// a hand waving past the lens, half a second of one is the prop. SPIN
/// and BACK are open-loop timed (constants trimmed on the floor, like
/// every drive constant in this file). PICK hands the body to the servo
/// task through `servo::PICK_GO`/`PICK_DONE`, wheels frozen. CARRY
/// creeps forward with the prize and parks. Single-shot: replug to run
/// the errand again — a fetch robot that immediately re-hunts with a
/// full claw would chase its own cargo.
#[cfg(feature = "fetch")]
async fn fetch_forever(motors: Motors, spec: RobotSpec) -> ! {
    use core::sync::atomic::Ordering;

    /// ⚠️ RAISED from the chase's 0.06/0.9 on 2026-08-24, attempt 1:
    /// the fetch rig carries the arm, PCA and loom the chase never did,
    /// and the old turn duty (~29%) stalled against the new breakaway —
    /// guard latched at 175 ticks. Same promotion path as the chase's
    /// own constants (raised the day THAT robot left the stand).
    const CREEP_M_PER_S: f64 = 0.10;
    const TURN_RAD_PER_S: f64 = 1.4;
    const CENTRE_DEADBAND: f32 = 0.20;
    /// Blob at least this big = the prop fills the near field = parked.
    const AREA_ARRIVED: u32 = 1800;
    /// Timed 180° at TURN_RAD_PER_S ≈ π/1.4 s; trimmed on the floor.
    const SPIN_MS: u64 = 2250;
    /// Reverse leg that lays the prop into the arm's pocket.
    const BACK_MS: u64 = 1400;
    const CARRY_MS: u64 = 1500;

    while !HOST_WATCHING.load(Ordering::Relaxed) {
        Timer::after_millis(50).await;
    }
    crate::diag::note("# fetch SEEKING");
    servo::SALUTE_GO.store(true, Ordering::Relaxed);

    let mut drivetrain = Drivetrain::new(motors);
    // ---- SEEK (turn-and-glance) ----
    // Attempt 4's lesson, straight off the wire: the camera delivers
    // ~4 fps and the stiction-breaking turn rate swings ~20 deg per
    // frame, so CONTINUOUS turning orbits the prop forever (+-33k
    // ticks of pure spin). The car never turns while looking now: a
    // short burst, a full stop, a fresh frame, then decide again. Only
    // the straight creep - which the camera CAN track - runs
    // continuously, and arrival needs three stationary glances.
    const TURN_BURST_MS: u64 = 150;
    const GLANCE_MS: u64 = 700;
    const CREEP_LEG_MS: u64 = 1500;
    const ARRIVE_GLANCES: u32 = 3;

    let mut arrive_streak: u32 = 0;
    let mut search_left = true;
    let mut last_frame_seen = camera::frame_total();
    let mut last_frame_at = firmware_support::now_ms();
    'seek: loop {
        {
            let (left, right) = Drivetrain::duty_for(&spec, 0.0, 0.0);
            drivetrain.apply(left, right);
        }
        let glance_from = camera::frame_total();
        let glance_start = firmware_support::now_ms();
        while camera::frame_total() == glance_from
            && firmware_support::now_ms().saturating_sub(glance_start) < GLANCE_MS
        {
            Timer::after(POLL).await;
        }
        let fresh = camera::frame_total() != glance_from;
        if fresh {
            last_frame_seen = camera::frame_total();
            last_frame_at = firmware_support::now_ms();
        } else if firmware_support::now_ms().saturating_sub(last_frame_at) > 5000 {
            // The chase parks on a dead image; the fetch must too — a
            // wedged capture reads as "no blob" and an unguarded seek
            // pirouettes on it forever (review 2026-08-24).
            let _ = last_frame_seen;
            let (left, right) = Drivetrain::duty_for(&spec, 0.0, 0.0);
            drivetrain.apply(left, right);
            loop {
                crate::diag::note("# fetch camera DEAD — parked, replug to retry");
                Timer::after_millis(5000).await;
            }
        }
        // A glance that timed out without a fresh frame decides on the
        // PRE-turn image — treat it as no blob instead.
        let burst = match camera::blob_error() {
            Some((x, _y, area)) if fresh => {
                let centred = x.abs() <= CENTRE_DEADBAND;
                if centred && area >= AREA_ARRIVED {
                    arrive_streak += 1;
                    if arrive_streak >= ARRIVE_GLANCES {
                        break 'seek;
                    }
                    None
                } else if !centred {
                    arrive_streak = 0;
                    search_left = x < 0.0;
                    Some(if x < 0.0 { TURN_RAD_PER_S } else { -TURN_RAD_PER_S })
                } else {
                    // Centred but far: creep straight while it stays so,
                    // re-glancing at least every CREEP_LEG_MS.
                    arrive_streak = 0;
                    {
                        let (left, right) = Drivetrain::duty_for(&spec, CREEP_M_PER_S, 0.0);
                        drivetrain.apply(left, right);
                    }
                    let creep_from = firmware_support::now_ms();
                    loop {
                        {
                            let (left, right) =
                                Drivetrain::duty_for(&spec, CREEP_M_PER_S, 0.0);
                            drivetrain.apply(left, right);
                        }
                        Timer::after(POLL).await;
                        let leg_done = firmware_support::now_ms()
                            .saturating_sub(creep_from)
                            > CREEP_LEG_MS;
                        let keep = match camera::blob_error() {
                            Some((cx, _cy, carea)) => {
                                cx.abs() <= CENTRE_DEADBAND && carea < AREA_ARRIVED
                            }
                            None => false,
                        };
                        if leg_done || !keep {
                            break;
                        }
                    }
                    None
                }
            }
            _ => {
                arrive_streak = 0;
                Some(if search_left { TURN_RAD_PER_S } else { -TURN_RAD_PER_S })
            }
        };
        if let Some(w) = burst {
            {
                let (left, right) = Drivetrain::duty_for(&spec, 0.0, w);
                drivetrain.apply(left, right);
            }
            Timer::after_millis(TURN_BURST_MS).await;
            {
                let (left, right) = Drivetrain::duty_for(&spec, 0.0, 0.0);
                drivetrain.apply(left, right);
            }
        }
    }
    {
        let (left, right) = Drivetrain::duty_for(&spec, 0.0, 0.0);
        drivetrain.apply(left, right);
    }
    crate::diag::note("# fetch ARRIVED, spinning");
    Timer::after_millis(300).await;

    // ---- SPIN 180° (timed) ----
    // Every timed leg RE-APPLIES its duty each POLL: the stall guard
    // only advances inside apply(), so a single apply + sleep held a
    // blocked wheel at full duty for the whole leg with the guard
    // asleep (review 2026-08-24).
    let leg = firmware_support::now_ms();
    while firmware_support::now_ms().saturating_sub(leg) < SPIN_MS {
        let (left, right) = Drivetrain::duty_for(&spec, 0.0, TURN_RAD_PER_S);
        drivetrain.apply(left, right);
        Timer::after(POLL).await;
    }
    {
        let (left, right) = Drivetrain::duty_for(&spec, 0.0, 0.0);
        drivetrain.apply(left, right);
    }
    crate::diag::note("# fetch SPUN, backing up");
    Timer::after_millis(300).await;

    // ---- BACK onto the prop ----
    let leg = firmware_support::now_ms();
    while firmware_support::now_ms().saturating_sub(leg) < BACK_MS {
        let (left, right) = Drivetrain::duty_for(&spec, -CREEP_M_PER_S, 0.0);
        drivetrain.apply(left, right);
        Timer::after(POLL).await;
    }
    {
        let (left, right) = Drivetrain::duty_for(&spec, 0.0, 0.0);
        drivetrain.apply(left, right);
    }
    crate::diag::note("# fetch PARKED, arm's turn");

    // ---- PICK (the arm's show; wheels frozen) ----
    servo::PICK_DONE.store(false, Ordering::Relaxed);
    servo::PICK_GO.store(true, Ordering::Relaxed);
    // Bounded: a dead arm task (bus error) can never set PICK_DONE,
    // and an unbounded wait froze the errand forever at "arm's turn".
    let pick_started = firmware_support::now_ms();
    while !servo::PICK_DONE.load(Ordering::Relaxed) {
        if firmware_support::now_ms().saturating_sub(pick_started) > 60_000 {
            crate::diag::note("# fetch ABANDONED — arm never reported done");
            break;
        }
        Timer::after_millis(100).await;
    }
    crate::diag::note("# fetch CARRYING");

    // ---- CARRY and park ----
    let leg = firmware_support::now_ms();
    while firmware_support::now_ms().saturating_sub(leg) < CARRY_MS {
        let (left, right) = Drivetrain::duty_for(&spec, CREEP_M_PER_S, 0.0);
        drivetrain.apply(left, right);
        Timer::after(POLL).await;
    }
    {
        let (left, right) = Drivetrain::duty_for(&spec, 0.0, 0.0);
        drivetrain.apply(left, right);
    }
    loop {
        crate::diag::note("# fetch DONE — errand complete, replug to rerun");
        Timer::after_millis(5000).await;
    }
}

#[cfg(feature = "camera")]
mod camera;
#[cfg(feature = "arm")]
mod servo;

/// Drive the motors from the camera's blob, and **stop when the image
/// stops**.
///
/// This is the loop the whole rig has been building toward: pixels in,
/// wheel duty out, nothing but the chip in between.
///
/// ```text
///   OV7670 ─▶ PIO ─▶ DMA ─▶ blob::find ─▶ (x, area)
///                                            │ sign only
///                                            ▼
///   blob left of centre  ─▶ turn left     too small ─▶ creep forward
///   blob right of centre ─▶ turn right    too big   ─▶ back away
///                                            │
///                              DiffDrive::inverse ─▶ duty ─▶ TB6612
/// ```
///
/// # Sign only, and tiny
///
/// `crates/blob`'s own doc says it: *visual servoing never asks where
/// anything is* — only which side of centre. So this controller has no
/// gains to tune: inside the deadband it is still, outside it nudges at
/// one fixed creep speed. A proportional law can come later, measured;
/// a sign law cannot be wrong by a factor of anything.
///
/// # The failsafe is freshness, not a host
///
/// `teleop`'s watchdog stops the wheels when the HOST goes silent. Here
/// the commander is the camera, so the equivalent event is **the frame
/// counter not advancing** — lens covered, capture wedged, sensor
/// unplugged. [`FRESH_MS`] without a new frame reads as "no image", and
/// no image means stop: [`camera::blob_error`]'s `None` and a stale
/// counter land in the same branch, because "looked and found nothing"
/// and "stopped looking" must both park the robot.
#[cfg(feature = "chase")]
async fn chase_forever(motors: Motors, spec: RobotSpec) -> ! {
    /// Forward/backward creep, metres per second. "Slightly" made a number.
    const CREEP_M_PER_S: f64 = 0.06;
    /// Turn nudge, radians per second.
    ///
    /// ⚠️ MEASURED UP from 0.5 on 2026-08-15, the day the robot left the
    /// stand: 0.5 rad/s is a 16% wheel duty, which spins free wheels and
    /// does not move a chassis on the ground — the guard then reads
    /// commanded-but-motionless (161 is just over its 150 floor) and
    /// latches, leaving a robot that "flinched once and died" while the
    /// servos dance on. 0.9 rad/s commands ~29%, above the observed
    /// breakaway; the 25% retreat creep was seen moving the chassis, so
    /// forward/backward stays as it is.
    const TURN_RAD_PER_S: f64 = 0.9;
    /// How far off-centre (−1..+1) the blob may sit before turning.
    const CENTRE_DEADBAND: f32 = 0.20;
    /// Blob smaller than this (pixels) → it is far → creep forward.
    ///
    /// ⚠️ MEASURED 2026-08-15, replacing guesses that drove the robot
    /// backwards around the room: a live session's blob ran 431–1,889
    /// pixels (mean 932), straddling the old NEAR=1500 — so "too close,
    /// retreat" fired on the ordinary ambient patch, and retreating from
    /// a room's brightness does not shrink it. The band now BRACKETS the
    /// observed range: approach below it, hold inside it, retreat only
    /// when something genuinely fills the view. Scene-tuned bench
    /// numbers — an area→distance calibration replaces them when a lens
    /// model exists.
    const AREA_FAR: u32 = 400;
    /// Blob bigger than this → too close → back away.
    const AREA_NEAR: u32 = 2500;
    /// No new frame for this long → the image has stopped → so do we.
    const FRESH_MS: u64 = 1000;

    // Nothing moves until a host opens the port — the same invariant the
    // sweep and teleop keep. A robot that starts hunting on power-up,
    // with nobody watching and no way to stop it, is the surprise
    // `HOST_WATCHING` exists to prevent.
    while !HOST_WATCHING.load(Ordering::Relaxed) {
        Timer::after_millis(50).await;
    }

    let mut drivetrain = Drivetrain::new(motors);
    let mut last_frame_total = camera::frame_total();
    let mut last_frame_at = firmware_support::now_ms();

    loop {
        Timer::after(POLL).await;
        let now = firmware_support::now_ms();

        // ---- is the image alive? ----
        let frames = camera::frame_total();
        if frames != last_frame_total {
            last_frame_total = frames;
            last_frame_at = now;
        }
        let image_alive = now.saturating_sub(last_frame_at) < FRESH_MS;

        // ---- the whole control law ----
        let (v, w) = match camera::blob_error() {
            Some((x, _y, area)) if image_alive => {
                let v = if area < AREA_FAR {
                    CREEP_M_PER_S
                } else if area > AREA_NEAR {
                    -CREEP_M_PER_S
                } else {
                    0.0
                };
                // x positive = blob right of centre = turn right, and
                // positive `w` is a LEFT turn — so the sign flips here,
                // once, where the image meets the body frame.
                let w = if x > CENTRE_DEADBAND {
                    -TURN_RAD_PER_S
                } else if x < -CENTRE_DEADBAND {
                    TURN_RAD_PER_S
                } else {
                    0.0
                };
                (v, w)
            }
            // "Found nothing" and "stopped looking" both stop the robot.
            _ => (0.0, 0.0),
        };

        let (left, right) = Drivetrain::duty_for(&spec, v, w);
        drivetrain.apply(left, right);
    }
}

/// The four encoder pins and the motor driver, identical whichever
/// transport is built.
///
/// # Why the LED is not set up here
///
/// It used to be, taking `PIN_25` alongside the encoder pins. That stopped
/// working the moment a third transport existed: **on a Pico 2 W, GP25 is
/// the radio's chip-select**, so the `wifi` build has to hand that pin to
/// `PioSpi` and blink the radio's own GPIO 0 instead. A function that
/// claimed the pin unconditionally would have made the wifi transport
/// impossible to write without a flag saying "don't do the thing you were
/// named for".
///
/// So the heartbeat belongs to the transport, which is the thing that
/// knows what a sign of life looks like on its board.
fn shared_setup(
    la: Peri<'static, PIN_2>,
    lb: Peri<'static, PIN_3>,
    ra: Peri<'static, PIN_26>,
    rb: Peri<'static, PIN_27>,
    motor: MotorPins,
) -> (Encoders, Motors) {
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

    // The driver is built DISABLED and at zero duty, and handed back
    // rather than committed to a job. Who drives it is a build-time
    // choice — the calibration sweep, or the host — and every output here
    // is created in its safe state either way.
    let motors = Motors {
        left: Channel {
            pwm: Pwm::new_output_a(motor.slice_a, motor.pwma, PwmConfig::default()),
            in1: Output::new(motor.ain1, Level::Low),
            in2: Output::new(motor.ain2, Level::Low),
        },
        right: Channel {
            pwm: Pwm::new_output_a(motor.slice_b, motor.pwmb, PwmConfig::default()),
            in1: Output::new(motor.bin1, Level::Low),
            in2: Output::new(motor.bin2, Level::Low),
        },
        standby: Output::new(motor.stby, Level::Low),
    };

    let encoders = Encoders {
        left_a: la,
        left_b: lb,
        right_a: ra,
        right_b: rb,
    };
    (encoders, motors)
}

// ---------------------------------------------------------------------
// Transport A — UART on GP0/GP1. What the emulator speaks.
// ---------------------------------------------------------------------
#[cfg(not(any(feature = "usb", feature = "wifi")))]
mod transport {
    use super::*;
    use embassy_rp::uart::{Blocking, Config as UartConfig, Uart, UartTx};

    struct UartReport(UartTx<'static, Blocking>);

    impl Report for UartReport {
        /// **Never actually blocks for long.** `blocking_write` pushes
        /// into the hardware FIFO at 115200 baud with nothing on the other
        /// end to apply back-pressure. The emulator is the only consumer.
        async fn send(&mut self, bytes: &[u8]) {
            // The emulator is always listening, so the gate opens at once.
            HOST_WATCHING.store(true, Ordering::Relaxed);
            let _ = self.0.blocking_write(bytes);
        }
    }

    pub async fn run(p: embassy_rp::Peripherals, spawner: Spawner) -> ! {
        let (encoders, motors) = shared_setup(
            p.PIN_2,
            p.PIN_3,
            p.PIN_26,
            p.PIN_27,
            MotorPins {
                slice_a: p.PWM_SLICE3,
                pwma: p.PIN_6,
                ain1: p.PIN_7,
                ain2: p.PIN_8,
                stby: p.PIN_9,
                slice_b: p.PWM_SLICE5,
                pwmb: p.PIN_10,
                bin1: p.PIN_11,
                bin2: p.PIN_12,
            },
        );
        spawner.spawn(firmware_support::heartbeat(Output::new(p.PIN_25, Level::Low), 500).unwrap());
        // `teleop` implies `usb`, so this transport is never built with it
        // — the sweep is the only thing that can own the motors here.
        spawner.spawn(drive_sweep(motors).unwrap());

        let uart = Uart::new_blocking(p.UART0, p.PIN_0, p.PIN_1, UartConfig::default());
        let (tx, _rx) = uart.split();

        odometry_forever(encoders, &mut UartReport(tx)).await
    }
}

// ---------------------------------------------------------------------
// Transport B — USB CDC serial. What a real Pico on a cable speaks, and —
// when built with `wifi` too — one half of the transport comparison.
// ---------------------------------------------------------------------
#[cfg(feature = "usb")]
mod transport {
    use super::*;
    #[cfg(not(any(feature = "teleop", feature = "chase", feature = "fetch")))]
    use embassy_futures::join::join;
    use embassy_rp::bind_interrupts;
    use embassy_rp::peripherals::USB;
    use embassy_rp::usb::{Driver, InterruptHandler};
    use embassy_time::{with_timeout, Duration};
    #[cfg(any(feature = "teleop", feature = "chase", feature = "fetch"))]
    use embassy_futures::join::join3;
    #[cfg(feature = "teleop")]
    use embassy_usb::class::cdc_acm::Receiver;
    use embassy_usb::class::cdc_acm::Sender;
    use embassy_usb::driver::Driver as UsbDriver;

    bind_interrupts!(struct Irqs {
        USBCTRL_IRQ => InterruptHandler<USB>;
    });


    /// Longest a status line may spend trying to reach the host before it
    /// is abandoned. Sized against the 1 ms USB full-speed frame: a
    /// healthy write finishes well inside this, so the timeout only fires
    /// on a host that has genuinely stopped reading.
    const REPORT_TIMEOUT_MS: u64 = 5;

    struct UsbReport<'d, D: UsbDriver<'d>>(Sender<'d, D>);

    /// The read half. Its deadline lives inside `recv` rather than being
    /// wrapped around it by the caller — see [`CommandSource`].
    ///
    /// A read that times out is **not** an error and must not zero the
    /// command by itself: silence for 50 ms is normal, silence for
    /// [`COMMAND_TIMEOUT_MS`] is not, and only the watchdog is allowed to
    /// tell those apart.
    #[cfg(feature = "teleop")]
    struct UsbCommands<'d, D: UsbDriver<'d>>(Receiver<'d, D>);

    #[cfg(feature = "teleop")]
    impl<'d, D: UsbDriver<'d>> CommandSource for UsbCommands<'d, D> {
        async fn recv(&mut self, buf: &mut [u8], poll: Duration) -> usize {
            // Cancelling this read IS proven safe: `pico-robot` justifies
            // it from embassy-rp's source — the endpoint read registers a
            // waker and tests a bit, with every side effect after the
            // await. That argument is why the timeout can sit here at all.
            match with_timeout(poll, self.0.read_packet(buf)).await {
                Ok(Ok(n)) => n,
                // Timed out, or the host vanished mid-packet. Both are
                // "no bytes", and the watchdog decides what that means.
                _ => 0,
            }
        }
    }

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
            HOST_WATCHING.store(true, Ordering::Relaxed);
            let write_all = async {
                for chunk in bytes.chunks(firmware_support::usb::MAX_PACKET) {
                    if self.0.write_packet(chunk).await.is_err() {
                        return;
                    }
                }
            };
            let _ = with_timeout(Duration::from_millis(REPORT_TIMEOUT_MS), write_all).await;
        }
    }

    pub async fn run(p: embassy_rp::Peripherals, spawner: Spawner) -> ! {
        let (encoders, motors) = shared_setup(
            p.PIN_2,
            p.PIN_3,
            p.PIN_26,
            p.PIN_27,
            MotorPins {
                slice_a: p.PWM_SLICE3,
                pwma: p.PIN_6,
                ain1: p.PIN_7,
                ain2: p.PIN_8,
                stby: p.PIN_9,
                slice_b: p.PWM_SLICE5,
                pwmb: p.PIN_10,
                bin1: p.PIN_11,
                bin2: p.PIN_12,
            },
        );
        // ⚠️ On a **Pico 2 W this lights nothing**: GP25 is the CYW43
        // radio's chip-select there, not an LED — see `firmware/pico-led`,
        // which boots the radio precisely because that is the only way to
        // reach it. Do not read "no blink" as "dead board"; on a W the
        // sign of life is the USB port appearing.
        //
        // With `wifi` the pin goes to the radio instead, and the radio's
        // own LED takes over as the sign of life.
        #[cfg(not(feature = "wifi"))]
        spawner.spawn(firmware_support::heartbeat(Output::new(p.PIN_25, Level::Low), 500).unwrap());

        let driver = Driver::new(p.USB, Irqs);
        // 0x2e8a is Raspberry Pi's vendor ID. Product IDs differ per
        // firmware — pico-robot 0x000a, pico-encoder 0x000b — so several
        // boards can be plugged in and still told apart.
        let (mut usb, class) = firmware_support::usb::cdc(driver, "pico-odom", 0x000c);
        let (tx, _rx) = class.split();

        // ⚠️ Bound to a name, not `_`. `_camera_clock` keeps XCLK running
        // for the life of the program; `let _ = ...` would drop the PWM
        // immediately and the camera would go deaf the moment it was
        // identified — working once, then never again, which is the most
        // expensive kind of working.
        //
        // Done before the loops start because it is a blocking bus read.
        // The note it queues is drained by the report loop, so it reaches
        // the host whenever the host turns up, rather than being written
        // into a port nobody has opened yet.
        // One bus, three tenants: the camera's SCCB, the PCA9685, and
        // whatever joins GP4/GP5 next. Built HERE because a bus is a
        // board resource — each module borrows it and hands it back.
        // ⚠️ ASYNC mode, deliberately, even though the camera only ever
        // uses it blocking at boot: async-mode hardware still serves the
        // blocking trait, and the servo task's writes become await
        // points the 10 kHz sampler can preempt instead of ~0.5 ms
        // blind spots per transaction.
        #[cfg(feature = "camera")]
        let shared_bus = embassy_rp::i2c::I2c::new_async(
            p.I2C0,
            p.PIN_5,
            p.PIN_4,
            I2cIrqs,
            embassy_rp::i2c::Config::default(),
        );
        #[cfg(feature = "camera")]
        let (_camera_clock, _shared_bus) = camera::start(spawner, shared_bus, camera::CameraPins {
            xclk_slice: p.PWM_SLICE2,
            xclk: p.PIN_21,
            vsync: p.PIN_1,
            href: p.PIN_22,
            pclk: p.PIN_0,
            pio: p.PIO0,
            dma: p.DMA_CH0,
            data: (
                p.PIN_13,
                p.PIN_14,
                p.PIN_15,
                p.PIN_16,
                p.PIN_17,
                p.PIN_18,
                p.PIN_19,
                p.PIN_20,
            ),
        })
        .await;
        #[cfg(feature = "arm")]
        {
            let bus = servo::probe(_shared_bus).await;
            // The sweep task owns the bus from here; see `servo::sweep`.
            spawner.spawn(servo::run(bus).unwrap());
        }

        // One report stream, two wires. Both carry the same `Status::seq`,
        // which is what turns "the radio feels laggy" into a number.
        #[cfg(feature = "wifi")]
        let mut out = Tee(
            UsbReport(tx),
            wifi_link::start(
                spawner,
                wifi_link::RadioPins {
                    pwr: p.PIN_23,
                    cs: p.PIN_25,
                    dio: p.PIN_24,
                    clk: p.PIN_29,
                    pio: p.PIO0,
                    dma: p.DMA_CH0,
                },
            ),
        );
        #[cfg(not(feature = "wifi"))]
        let mut out = UsbReport(tx);

        // Sampling starts immediately rather than waiting for a host: the
        // decoders' job is to miss nothing, and gating them on a terminal
        // being open would silently lose every tick between power-up and
        // the first `screen`. `send` already declines to write while
        // `dtr()` is low.
        #[cfg(not(any(feature = "teleop", feature = "chase", feature = "fetch")))]
        {
            spawner.spawn(drive_sweep(motors).unwrap());
            join(usb.run(), odometry_forever(encoders, &mut out)).await;
        }

        // The whole errand: seek the prop by camera, park, spin, back up,
        // hand the body to the arm, carry the prize. Same three-futures
        // shape as chase, one state machine deeper.
        #[cfg(feature = "fetch")]
        {
            join3(
                usb.run(),
                odometry_forever(encoders, &mut out),
                fetch_forever(motors, RobotSpec::REAL_BOT),
            )
            .await;
        }

        // The camera commands the wheels; the host only watches. Same
        // three-futures shape as teleop, different commander.
        #[cfg(feature = "chase")]
        {
            join3(
                usb.run(),
                odometry_forever(encoders, &mut out),
                chase_forever(motors, RobotSpec::REAL_BOT),
            )
            .await;
        }

        // Three futures on one stack: the USB stack, the sampler, and the
        // command loop. `follow_host_forever` is a plain `async fn` for
        // the same reason `odometry_forever` is — an embassy task cannot
        // be generic, and both are generic over their transport.
        #[cfg(feature = "teleop")]
        {
            let mut commands = UsbCommands(_rx);
            join3(
                usb.run(),
                odometry_forever(encoders, &mut out),
                follow_host_forever(&mut commands, motors, RobotSpec::REAL_BOT),
            )
            .await;
        }

        // `join` over a `!` future never returns, but the compiler wants a
        // value for the `-> !` signature.
        loop {
            Timer::after_secs(1).await;
        }
    }
}

/// Sends every line down two transports.
///
/// Built for one question — **does the radio deliver what the cable
/// delivers?** — and it can only answer it because both halves receive the
/// *same* bytes from the *same* loop iteration, carrying the same
/// [`Status::seq`]. Any difference in what arrives is therefore the
/// transport's and not the robot's, which is not something two separate
/// runs could ever establish.
///
/// The two sends are joined rather than sequenced. USB's write carries a
/// 5 ms deadline and UDP's cannot block at all; running them in order
/// would put the cable's worst case in front of the radio's every single
/// report, and then measure the delay it had just caused.
#[cfg(all(feature = "usb", feature = "wifi"))]
struct Tee<A, B>(A, B);

#[cfg(all(feature = "usb", feature = "wifi"))]
impl<A: Report, B: Report> Report for Tee<A, B> {
    async fn send(&mut self, bytes: &[u8]) {
        embassy_futures::join::join(self.0.send(bytes), self.1.send(bytes)).await;
    }
}

// ---------------------------------------------------------------------
// The radio link — UDP over CYW43. A `Report`, not a transport: the USB
// build tees into it, and the radio-only build wraps it in Transport C.
// ---------------------------------------------------------------------
//
// # Why this is a `Report` impl and not a new firmware
//
// The whole transport is ~40 lines of actual logic. Everything else here
// is bringing the radio up, and `firmware/pico-led` already proved that
// sequence on this exact board — 6 power cycles, 6 successes
// (docs/e2e-research/28). The odometry loop, the pose integrator and the
// wire format are untouched, because `Report` was already the seam.
//
// # Why UDP, and why broadcast
//
// `Report`'s contract says reports are best-effort and ticks are not: a
// late report is worse than a missing one, because every microsecond
// blocked here is a missed encoder transition. UDP *is* that contract —
// fire the datagram, never wait for an acknowledgement. TCP would add
// retransmission delay fighting a design that has already chosen to drop.
//
// Broadcast (255.255.255.255) rather than a configured host address, so
// there is nothing to set up on either side: the board shouts, and any
// machine on the LAN that binds the port hears it. That is right for a
// bench and wrong for a customer site — it does not cross subnets and it
// authenticates nobody. docs/e2e-research/28 is explicit that the radio
// stays a bench side-channel; this is that side-channel.
//
// # Why a radio-only board never starts the motor sweep
//
// It falls out of the existing gate rather than needing a rule. The sweep
// waits on `HOST_WATCHING`, which only a transport that can *tell* sets —
// USB raises it when a host opens the port, UART when the emulator
// attaches. **A broadcast socket cannot know whether anyone is
// listening**, so `UdpReport::send` never raises it, and a radio-only
// board therefore parks at the top of `drive_sweep` forever.
//
// That is the correct behaviour and it is worth stating why: an untethered
// robot beginning a blind motor sweep on power-up, with no cable attached
// to stop it, is precisely the surprise `HOST_WATCHING` was added to
// prevent. Tee'd with USB the sweep runs as it always did, because then
// there genuinely is a host on a cable.
#[cfg(feature = "wifi")]
mod wifi_link {
    use super::*;
    use core::fmt::Write as _;
    use core::net::Ipv4Addr;
    use core::task::Poll;
    use cyw43::{Aligned, A4};
    use cyw43_pio::{PioSpi, RM2_CLOCK_DIVIDER};
    use embassy_futures::poll_once;
    use embassy_net::udp::{PacketMetadata, UdpSocket};
    use embassy_net::{IpEndpoint, StackResources};
    use embassy_sync::blocking_mutex::raw::CriticalSectionRawMutex;
    use embassy_sync::channel::Channel;
    use embassy_time::{with_timeout, Duration};
    use embassy_rp::bind_interrupts;
    use embassy_rp::clocks::RoscRng;
    use embassy_rp::dma;
    use embassy_rp::peripherals::{DMA_CH0, PIO0};
    use embassy_rp::pio::{InterruptHandler, Pio};
    use static_cell::StaticCell;

    bind_interrupts!(struct Irqs {
        PIO0_IRQ_0 => InterruptHandler<PIO0>;
        DMA_IRQ_0 => dma::InterruptHandler<DMA_CH0>;
    });

    /// Where status lines are broadcast, and where `odom_view --udp`
    /// listens. One number, hardcoded on both sides on purpose: a port
    /// that can disagree is one more pair of facts nobody compares.
    const TELEMETRY_PORT: u16 = 9870;

    /// Credentials, baked in at build time:
    ///
    /// ```sh
    /// WIFI_SSID='bench' WIFI_PASSWORD='...' tools/build-pico2.sh pico-odom wifi
    /// ```
    ///
    /// **Empty is a supported state, not a build failure.** `verify.sh`
    /// has to be able to compile this transport on a machine with no
    /// credentials — that is the whole point of building every firmware
    /// variant — and a `compile_error!` here would mean the wifi build was
    /// the one thing never checked. So an unset SSID compiles, and the
    /// board says so at runtime by blinking [`BLINK_NO_CREDENTIALS`]
    /// rather than pretending to join.
    ///
    /// `build.rs` marks both as rebuild triggers, so changing the password
    /// rebuilds rather than silently reusing a binary with the old one.
    const SSID: &str = match option_env!("WIFI_SSID") {
        Some(s) => s,
        None => "",
    };
    const PASSWORD: &str = match option_env!("WIFI_PASSWORD") {
        Some(s) => s,
        None => "",
    };

    /// The radio's firmware and this board's settings — the same bytes
    /// `pico-led` uploads, from the same shared directory. See that crate
    /// for why 231 KB of it lives in our flash.
    static FW: Aligned<A4, [u8; 231_077]> = Aligned(*cyw43_firmware::CYW43_43439A0);
    static NVRAM: Aligned<A4, [u8; 742]> =
        Aligned(*include_bytes!("../../cyw43-firmware/nvram_rp2040.bin"));

    /// LED cadence, in milliseconds on and off, for each thing that can be
    /// true. The LED is the **only** output an untethered board has, so it
    /// has to distinguish the failures rather than just proving power:
    ///
    /// ```text
    ///   long gap, short flash   no credentials — rebuild with WIFI_SSID
    ///   fast even blink         joining, or joined but no DHCP lease yet
    ///   solid on                address held, datagrams going out
    /// ```
    const BLINK_NO_CREDENTIALS: (u64, u64) = (60, 1940);
    const BLINK_JOINING: (u64, u64) = (120, 120);
    /// Half of the delivering heartbeat: two quick flashes, then a pause.
    const BLINK_DELIVERING: (u64, u64) = (70, 90);
    const DELIVERING_PAUSE_MS: u64 = 700;

    /// Longest a single association attempt may take before it is
    /// abandoned and retried. Generous — real APs can take several seconds
    /// — but finite, which `cyw43::Control::join` is not.
    const JOIN_TIMEOUT: Duration = Duration::from_secs(15);

    /// Shortest passphrase WPA2-PSK permits. IEEE 802.11i fixes the range
    /// at 8–63 ASCII characters, so anything shorter **cannot** be a WPA2
    /// key, whatever it was called when it was written down.
    const SHORTEST_WPA2_KEY: usize = 8;
    const LONGEST_WPA2_KEY: usize = 63;

    /// Whether the robot **hosts** the network or joins one.
    ///
    /// Hosting is the default, and it is the right default for a machine
    /// that drives around: it needs no router, no site credentials and no
    /// permission from anyone's IT department, and it works identically in
    /// a lab, a car park and a customer's warehouse. Joining an existing
    /// network is `WIFI_MODE=station`, and is what you want when several
    /// robots must be watched from one laptop.
    ///
    /// ⚠️ Hosting means the laptop leaves whatever network it was on.
    const HOSTING: bool = match option_env!("WIFI_MODE") {
        Some(m) => matches!(m.as_bytes(), b"ap"),
        None => true,
    };

    /// 2.4 GHz channel to host on. Six is the middle of the three
    /// non-overlapping channels (1, 6, 11) and the conventional default.
    const AP_CHANNEL: u8 = 6;

    /// The robot's address when it is hosting. Clients get no DHCP — see
    /// the note where the AP starts.
    const AP_ADDRESS: Ipv4Addr = Ipv4Addr::new(192, 168, 4, 1);

    /// How to authenticate, decided from the passphrase itself.
    ///
    /// A passphrase too short to be a WPA2 key almost certainly means the
    /// network is open and the string is something else — a hotspot name,
    /// a note to self. Sending it as a PMK anyway is what this build did
    /// on 2026-08-10, and the radio simply never answered.
    fn join_options() -> cyw43::JoinOptions<'static> {
        if PASSWORD.len() < SHORTEST_WPA2_KEY {
            cyw43::JoinOptions::new_open()
        } else {
            cyw43::JoinOptions::new(PASSWORD.as_bytes())
        }
    }
    /// Joined, addressed, and **nothing is getting out**. Its own pattern
    /// because that state used to be invisible: the link was up, so the
    /// LED went solid, while every datagram was being dropped. A board
    /// that looks identical whether it is delivering or silently failing
    /// is the thing that cost this session an hour.
    const BLINK_LINKED_BUT_MUTE: (u64, u64) = (500, 500);

    /// True once DHCP has given us an address. Read by the LED task to go
    /// solid, and by [`UdpReport`] to skip the send entirely — a datagram
    /// with no source address is an error return we would only throw away.
    static LINK_UP: AtomicBool = AtomicBool::new(false);

    /// Set by [`UdpReport::send`] when a datagram actually left, cleared
    /// by the LED task each time it looks. Distinguishes "linked and
    /// delivering" from "linked and dropping every packet", which are the
    /// same thing to `LINK_UP` and were the same thing to the LED.
    static SENT_RECENTLY: AtomicBool = AtomicBool::new(false);

    /// Services the radio. Must run forever, or the chip stops answering.
    #[embassy_executor::task]
    async fn cyw43_task(
        runner: cyw43::Runner<'static, cyw43::SpiBus<Output<'static>, PioSpi<'static, PIO0, 0>>>,
    ) -> ! {
        runner.run().await
    }

    /// Services the TCP/IP stack: ARP, DHCP renewal, and moving frames
    /// between smoltcp and the radio.
    #[embassy_executor::task]
    async fn net_task(
        mut runner: embassy_net::Runner<'static, cyw43::NetDriver<'static>>,
    ) -> ! {
        runner.run().await
    }

    /// Owns `control` for the life of the program: joins the network, then
    /// blinks what happened.
    ///
    /// Joining lives here rather than in `run` because `control` cannot be
    /// shared, and a link that drops has to be able to rejoin — which means
    /// whoever blinks the LED must also be whoever can call `join`.
    #[embassy_executor::task]
    async fn link_task(mut control: cyw43::Control<'static>) -> ! {
        control.init(cyw43_firmware::CYW43_43439A0_CLM).await;

        // ⚠️ Default power management is PM2, which parks the radio for up
        // to 200 ms between beacons. That is a good trade for a sensor
        // that wakes once a minute and a bad one for a 50 Hz telemetry
        // stream — it would show up as periodic 200 ms gaps in the trail
        // and look exactly like a firmware stall. Costs battery; say so
        // in the log rather than discovering it as jitter.
        control
            .set_power_management(cyw43::PowerManagementMode::None)
            .await;

        if SSID.is_empty() {
            loop {
                blink(&mut control, BLINK_NO_CREDENTIALS).await;
            }
        }

        if HOSTING {
            // ⚠️ `start_ap_wpa2` PANICS on a passphrase outside 8..=63 —
            // it is an assert in the driver, not an error return. A panic
            // here halts the executor, which on a W board means a dark LED
            // and no USB: the exact unfalsifiable state that cost this
            // session hours. So the length is checked HERE, before the
            // driver gets a chance to be right about it in the worst
            // possible way.
            if (SHORTEST_WPA2_KEY..=LONGEST_WPA2_KEY).contains(&PASSWORD.len()) {
                control.start_ap_wpa2(SSID, PASSWORD, AP_CHANNEL).await;
                note("# hosting WPA2 network");
            } else {
                control.start_ap_open(SSID, AP_CHANNEL).await;
                note("# hosting OPEN network — set an 8..=63 character WIFI_PASSWORD for WPA2");
            }
            announce_ap();
            // No DHCP server: a client that joins will self-assign a
            // link-local address rather than being handed one. That is
            // enough for broadcast telemetry, which is addressed to
            // 255.255.255.255 and delivered at layer 2 regardless of
            // whose subnet anyone thinks they are on. It is NOT enough to
            // reach this board by address, which is what a command path
            // would need — that is when a DHCP server earns its keep.
            let mut ticks = 0u32;
            loop {
                // ⚠️ Re-announced, not said once.
                //
                // `UsbReport` drops anything written while DTR is low —
                // correctly, since nothing is listening — so a banner
                // emitted at boot is gone by the time someone opens the
                // port. The station path got away with this only because
                // its join loop happened to repeat every 15 seconds. A
                // diagnostic you have to have been present for is not a
                // diagnostic.
                if ticks % 20 == 0 {
                    announce_ap();
                }
                ticks += 1;
                show_link_health(&mut control).await;
            }
        }

        loop {
            // ⚠️ Blink BEFORE attempting, not only after failing.
            //
            // `join` does not return until the radio answers, and on
            // 2026-08-10 it did not answer — so `link_task` sat inside it
            // with the LED never having been touched. Dark is also what a
            // panicked board looks like, and what a board stuck in
            // firmware upload looks like. Three very different faults, one
            // indistinguishable symptom, on the only output an untethered
            // board has. Blinking first makes "trying" visible.
            for _ in 0..8 {
                blink(&mut control, BLINK_JOINING).await;
            }

            // What the radio can actually SEE, said out loud. A failed
            // join has several causes that look identical from outside —
            // wrong name, out of range, wrong band, wrong security — and
            // a scan separates them in one flash instead of four guesses.
            // The CYW43439 is **2.4 GHz only**, so an access point missing
            // from this list is very often one that exists perfectly well
            // on 5 GHz.
            let mut seen = 0u32;
            let mut scanner = control.scan(cyw43::ScanOptions::default()).await;
            while let Some(bss) = scanner.next().await {
                let len = (bss.ssid_len as usize).min(bss.ssid.len());
                let Ok(name) = core::str::from_utf8(&bss.ssid[..len]) else {
                    continue;
                };
                if name.is_empty() {
                    continue;
                }
                seen += 1;
                let mut text: heapless::String<160> = heapless::String::new();
                let _ = write!(
                    text,
                    "# saw ssid={:?} rssi={} chanspec={:#06x}{}",
                    name,
                    bss.rssi,
                    bss.chanspec,
                    if name == SSID { "  <-- OURS" } else { "" }
                );
                note(&text);
            }
            drop(scanner);
            let mut summary: heapless::String<160> = heapless::String::new();
            let _ = write!(
                summary,
                "# scan done: {} networks visible; joining {:?} as {}",
                seen,
                SSID,
                if PASSWORD.len() < SHORTEST_WPA2_KEY {
                    "open"
                } else {
                    "wpa2"
                }
            );
            note(&summary);

            // ⚠️ Bounded. `join` awaits a radio event with no timeout of
            // its own, so a radio that never answers parks this task
            // forever — and with it any chance of retrying, rejoining
            // after driving out of range, or saying anything on the LED.
            let attempt = with_timeout(JOIN_TIMEOUT, control.join(SSID, join_options())).await;
            if !matches!(attempt, Ok(Ok(()))) {
                note("# join failed or timed out");
                continue;
            }
            note("# joined; waiting for a DHCP lease");

            // Associated. DHCP may still be outstanding, so the LED stays
            // in the joining pattern until `LINK_UP` says otherwise, and
            // goes solid once an address is held.
            while !LINK_UP.load(Ordering::Relaxed) {
                blink(&mut control, BLINK_JOINING).await;
            }
            while LINK_UP.load(Ordering::Relaxed) {
                show_link_health(&mut control).await;
            }
            control.gpio_set(0, false).await;
        }
    }

    /// One on/off cycle of the radio's own GPIO 0 — the LED on a Pico 2 W.
    async fn blink(control: &mut cyw43::Control<'static>, (on, off): (u64, u64)) {
        control.gpio_set(0, true).await;
        Timer::after_millis(on).await;
        control.gpio_set(0, false).await;
        Timer::after_millis(off).await;
    }

    /// Shows, on the one output an untethered board has, whether telemetry
    /// is actually going out. Shared by both modes, because "is the radio
    /// working" has one answer and should have one pattern.
    ///
    /// ```text
    ///   ··   ··   ··      double flash, ~1/s   DELIVERING — datagrams leaving
    ///   ▬▬  ▬▬  ▬▬        even blink,  ~1/s    linked, but nothing getting out
    ///   ▪▪▪▪▪▪▪▪▪▪        fast blink,  ~4/s    associating, not yet joined
    ///   ·         ·       one blip / 2 s       no SSID compiled in
    /// ```
    ///
    /// # Why the healthy state blinks rather than staying lit
    ///
    /// Because a solid LED cannot prove the firmware is still running. A
    /// board that panicked with the light on looks exactly like a board
    /// delivering perfectly — and this session already spent hours on
    /// states that were indistinguishable from the outside. A heartbeat is
    /// only produced by code that is still executing.
    ///
    /// # What it can and cannot tell you
    ///
    /// It means **datagrams are leaving the chip**, not that anyone is
    /// receiving them. A broadcast socket has no way to know whether a
    /// host is listening. For delivery you need the other end —
    /// `odom_view --udp` reports loss against the chip's own sequence
    /// numbers, which is the measurement this cannot make alone.
    async fn show_link_health(control: &mut cyw43::Control<'static>) {
        if SENT_RECENTLY.swap(false, Ordering::Relaxed) {
            blink(control, BLINK_DELIVERING).await;
            blink(control, BLINK_DELIVERING).await;
            Timer::after_millis(DELIVERING_PAUSE_MS).await;
        } else {
            blink(control, BLINK_LINKED_BUT_MUTE).await;
        }
    }

    /// Polls the stack for a DHCP lease and publishes it to [`LINK_UP`].
    ///
    /// A task rather than a future joined into the report loop, so that
    /// the USB build can tee into the radio without also having to know
    /// that the radio needs servicing.
    #[embassy_executor::task]
    async fn link_state_task(stack: embassy_net::Stack<'static>) -> ! {
        loop {
            LINK_UP.store(stack.is_config_up(), Ordering::Relaxed);
            Timer::after_millis(200).await;
        }
    }

    /// One status line waiting to go out. Shared with the rest of the
    /// firmware — see the top-level `diag` module for why it moved.
    use crate::diag::{note, Line};

    /// Lines handed to the radio but not yet transmitted.
    ///
    /// **Deliberately tiny.** This is a drop queue, not a buffer: if the
    /// radio is slower than 50 Hz, the right thing is to lose the oldest
    /// telemetry rather than accumulate a backlog that arrives late and
    /// describes a robot that has since moved.
    static OUTBOX: Channel<CriticalSectionRawMutex, Line, 2> = Channel::new();


    /// What network the robot is hosting, and where it is.
    fn announce_ap() {
        let mut text: heapless::String<160> = heapless::String::new();
        let _ = write!(
            text,
            "# hosting ssid={:?} channel={} address={} — join it, then odom_view --udp",
            SSID, AP_CHANNEL, AP_ADDRESS
        );
        note(&text);
    }

    /// A handle, not a socket.
    ///
    /// # Why the radio is on the other side of a queue
    ///
    /// Because on 2026-08-10 it took the whole board down twice, in two
    /// different ways, and both were structural rather than bad luck:
    ///
    /// 1. `send_to` awaited on a full transmit buffer, so the report loop
    ///    stopped — **and USB stopped with it**, because `Tee` joins both
    ///    sends and a join finishes only when both halves do.
    /// 2. `cyw43::new()` hung during firmware upload, and since USB was
    ///    started *after* radio bring-up, the board enumerated nothing at
    ///    all. A dark LED and no serial port is an unfalsifiable state:
    ///    "the radio hung", "the flash is bad" and "the board is dead"
    ///    look identical.
    ///
    /// A queue fixes the class. Nothing the radio does — uploading
    /// firmware, scanning, associating, failing DHCP, blocking on a full
    /// buffer — can now reach the odometry loop or the USB stack. A radio
    /// that never comes up costs exactly one dropped datagram per report,
    /// and the cable keeps saying so.
    pub struct UdpReport;

    impl Report for UdpReport {
        /// Never awaits anything. `try_send` either takes the line or
        /// does not, which is precisely this trait's contract: *every
        /// microsecond spent blocked here is a microsecond of missed
        /// encoder transitions.*
        ///
        /// `HOST_WATCHING` is deliberately never set: see the module
        /// header. There is no host to watch on a broadcast socket, so the
        /// motor gate it guards is left shut.
        async fn send(&mut self, bytes: &[u8]) {
            let mut line = Line::new();
            if line.extend_from_slice(bytes).is_err() {
                return;
            }
            let _ = OUTBOX.try_send(line);
        }
    }

    /// The four pins the radio needs. Named rather than passed loose,
    /// because `PIN_25` arriving here instead of at the heartbeat is the
    /// whole difference between a W board that talks and one that blinks.
    pub struct RadioPins {
        pub pwr: Peri<'static, PIN_23>,
        pub cs: Peri<'static, PIN_25>,
        pub dio: Peri<'static, PIN_24>,
        pub clk: Peri<'static, PIN_29>,
        pub pio: Peri<'static, PIO0>,
        pub dma: Peri<'static, DMA_CH0>,
    }

    /// Hands back a report sink **immediately**, and does every slow or
    /// fallible thing in a task behind it.
    ///
    /// ⚠️ Deliberately not `async`. It used to await radio bring-up, which
    /// put `cyw43::new()` — a 231 KB firmware upload over a bit-banged SPI
    /// bus — in front of USB enumeration. When that upload hung, the board
    /// presented no serial port, no LED and no way to tell a hung radio
    /// from a bad flash. Nothing that can fail belongs on this path.
    pub fn start(spawner: Spawner, pins: RadioPins) -> UdpReport {
        spawner.spawn(radio_task(spawner, pins).unwrap());
        UdpReport
    }

    /// Owns the radio for the life of the program: brings it up, then
    /// drains [`OUTBOX`] onto the air.
    ///
    /// Everything in here is allowed to be slow, to fail, or to hang. That
    /// is the whole point of it being over here.
    #[embassy_executor::task]
    async fn radio_task(spawner: Spawner, pins: RadioPins) -> ! {
        // The four lines to the radio. Fixed by the board's wiring — and
        // the reason this cannot coexist with the GP25 heartbeat the
        // cable-only builds spawn.
        let pwr = Output::new(pins.pwr, Level::Low);
        let cs = Output::new(pins.cs, Level::High);
        let mut pio = Pio::new(pins.pio, Irqs);
        let spi = PioSpi::new(
            &mut pio.common,
            pio.sm0,
            RM2_CLOCK_DIVIDER,
            pio.irq0,
            cs,
            pins.dio,
            pins.clk,
            dma::Channel::new(pins.dma, Irqs),
        );

        static STATE: StaticCell<cyw43::State> = StaticCell::new();
        let (net_device, control, runner) =
            cyw43::new(STATE.init(cyw43::State::new()), pwr, spi, &FW, &NVRAM).await;
        spawner.spawn(cyw43_task(runner).unwrap());
        spawner.spawn(link_task(control).unwrap());

        // smoltcp seeds its port and sequence randomness from this. The
        // ring oscillator is the one entropy source available before the
        // network exists.
        let seed = RoscRng.next_u64();
        static RESOURCES: StaticCell<StackResources<2>> = StaticCell::new();
        // Hosting means nobody is going to hand us an address, so we pick
        // one; joining means waiting for a lease.
        let config = if HOSTING {
            embassy_net::Config::ipv4_static(embassy_net::StaticConfigV4 {
                address: embassy_net::Ipv4Cidr::new(AP_ADDRESS, 24),
                gateway: None,
                dns_servers: heapless::Vec::new(),
            })
        } else {
            embassy_net::Config::dhcpv4(Default::default())
        };
        let (stack, net_runner) = embassy_net::new(
            net_device,
            config,
            RESOURCES.init(StackResources::new()),
            seed,
        );
        spawner.spawn(net_task(net_runner).unwrap());
        spawner.spawn(link_state_task(stack).unwrap());

        // One datagram of headroom each way. Status lines are ~100 bytes
        // and sent every 20 ms; a backlog would be stale data we would
        // rather drop than deliver late.
        static RX_META: StaticCell<[PacketMetadata; 4]> = StaticCell::new();
        static RX_BUF: StaticCell<[u8; 512]> = StaticCell::new();
        static TX_META: StaticCell<[PacketMetadata; 4]> = StaticCell::new();
        static TX_BUF: StaticCell<[u8; 512]> = StaticCell::new();
        let mut socket = UdpSocket::new(
            stack,
            RX_META.init([PacketMetadata::EMPTY; 4]),
            RX_BUF.init([0; 512]),
            TX_META.init([PacketMetadata::EMPTY; 4]),
            TX_BUF.init([0; 512]),
        );
        // Bound to the same port it broadcasts to, so a host that replies
        // has somewhere to reply *to* when the command path lands.
        let _ = socket.bind(TELEMETRY_PORT);
        let broadcast = IpEndpoint::new(Ipv4Addr::BROADCAST.into(), TELEMETRY_PORT);

        loop {
            let line = OUTBOX.receive().await;
            if !LINK_UP.load(Ordering::Relaxed) {
                continue;
            }
            // ⚠️ Polled ONCE, never awaited. `send_to` returns
            // `Poll::Pending` when the transmit buffer is full and only
            // wakes when space appears — and if the link is associated but
            // frames are not actually leaving, that space never comes.
            // Awaiting it here would stall this task forever, which is
            // survivable now (the queue just fills and drops) but would
            // still hide the fault behind a silent radio rather than
            // showing it as the mute-LED pattern.
            if let Poll::Ready(Ok(())) = poll_once(socket.send_to(&line, broadcast)) {
                SENT_RECENTLY.store(true, Ordering::Relaxed);
            }
        }
    }
}

// ---------------------------------------------------------------------
// Transport C — the radio alone. An untethered board with no cable at all.
// ---------------------------------------------------------------------
#[cfg(all(feature = "wifi", not(feature = "usb")))]
mod transport {
    use super::*;

    pub async fn run(p: embassy_rp::Peripherals, spawner: Spawner) -> ! {
        let (encoders, motors) = shared_setup(
            p.PIN_2,
            p.PIN_3,
            p.PIN_26,
            p.PIN_27,
            MotorPins {
                slice_a: p.PWM_SLICE3,
                pwma: p.PIN_6,
                ain1: p.PIN_7,
                ain2: p.PIN_8,
                stby: p.PIN_9,
                slice_b: p.PWM_SLICE5,
                pwmb: p.PIN_10,
                bin1: p.PIN_11,
                bin2: p.PIN_12,
            },
        );
        // Spawned exactly as the cable builds spawn it, and it will park
        // forever on `HOST_WATCHING` — see the note above `wifi_link`.
        // Left spawned rather than `#[cfg]`'d away so that the safety
        // property is enforced by the gate that exists for it, instead of
        // by this build happening not to call the function.
        spawner.spawn(drive_sweep(motors).unwrap());

        let mut out = wifi_link::start(
            spawner,
            wifi_link::RadioPins {
                pwr: p.PIN_23,
                cs: p.PIN_25,
                dio: p.PIN_24,
                clk: p.PIN_29,
                pio: p.PIO0,
                dma: p.DMA_CH0,
            },
        );

        odometry_forever(encoders, &mut out).await
    }
}

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    let p = embassy_rp::init(Default::default());
    transport::run(p, spawner).await
}
