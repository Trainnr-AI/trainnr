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
#[cfg(any(feature = "teleop", feature = "chase"))]
use embassy_time::Duration;
use firmware_support::motor::{Channel, Encoders, Motors, PWM_TOP};
use firmware_support::Report;
use panic_halt as _;
use quad_encoder::QuadratureDecoder;
use sim_core::{Odometry, Pose, RobotSpec};
#[cfg(any(feature = "teleop", feature = "chase"))]
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
/// deliberately above the audible band — a motor driven at 2 kHz whines,
/// and the winding heats more on the switching edges.

/// The duty sweep, as percentages held for [`STEP_SECS`] each.
///
/// It **ends at zero and parks**, rather than looping. A bench motor on
/// four thin encoder wires should not run unattended, and a firmware whose
/// natural end state is "stopped" cannot be left running by accident.
#[cfg(not(any(feature = "teleop", feature = "chase")))]
const SWEEP: [u16; 6] = [0, 25, 50, 75, 100, 0];
#[cfg(not(any(feature = "teleop", feature = "chase")))]
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
#[cfg(any(feature = "teleop", feature = "chase"))]
const POLL: Duration = Duration::from_millis(50);



/// Duty above which a commanded motor **must** be turning, in the
/// `±DUTY_FULL` units the wire carries.
///
/// 150 of 1000 is 15%, clear of the measured ~4.3% deadband with margin —
/// below that, "not moving" is legitimate physics rather than a fault.
#[cfg(any(feature = "teleop", feature = "chase"))]
const STALL_DUTY_FLOOR: i32 = 150;

/// Consecutive polls of commanded-but-not-moving before cutting the
/// motors. At [`POLL`] = 50 ms, four polls is 200 ms — long enough that a
/// motor still overcoming its own inertia is never mistaken for a stall,
/// short enough that a genuinely locked rotor is not held at full duty.
#[cfg(any(feature = "teleop", feature = "chase"))]
const STALL_POLLS: u32 = 4;

/// Duty above which the motor **must** move, or something is wrong.
///
/// The measured deadband is ~4.6% (docs/07, 2026-08-09), so 15% is clear
/// of it with margin — below that, "not moving" is legitimate physics
/// rather than a fault.
#[cfg(not(any(feature = "teleop", feature = "chase")))]
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
#[cfg(not(feature = "teleop"))]
#[cfg(not(any(feature = "teleop", feature = "chase")))]
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

            // Publish for the stall check. Sum of magnitudes, so motion in
            // either direction on either wheel counts as "it moved".
            TOTAL_TICKS.store(
                left_count.unsigned_abs() + right_count.unsigned_abs(),
                Ordering::Relaxed,
            );

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
            if seq % u64::from(50 * camera::ANNOUNCE_SECONDS) == 0 {
                camera::announce();
            }
            drain_notes!(out);
            last_report = now;
        }

        Timer::after_micros(POLL_US).await;
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
    mut motors: Motors,
    spec: RobotSpec,
) -> ! {
    use hil_protocol::{LineReader, Message};

    let mut cfg = PwmConfig::default();
    cfg.top = PWM_TOP;
    motors.left.set_signed(&mut cfg, 0);
    motors.right.set_signed(&mut cfg, 0);

    let mut watchdog = CommandWatchdog::new(COMMAND_TIMEOUT_MS);
    let mut reader: LineReader<64> = LineReader::new();
    let mut rx_bytes = [0u8; 64];
    // What the host last asked for, before the watchdog has its say.
    let mut wanted = (0i32, 0i32);
    let mut enabled = false;
    let mut ticks_at_last_check = TOTAL_TICKS.load(Ordering::Relaxed);
    let mut stalled_polls: u32 = 0;

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
                let wheels = spec.fit_wheels(spec.drive().inverse(BodyTwist {
                    forward_speed: v,
                    turn_rate: w,
                }));
                wanted = (spec.duty(wheels.left), spec.duty(wheels.right));
                // Fed on a VALID twist only. A host dribbling malformed
                // bytes is a host that has lost its mind, and must not
                // count as one that is still in control.
                watchdog.feed(firmware_support::now_ms());
            }
        }

        let (mut left, mut right) = watchdog.gate(firmware_support::now_ms(), wanted);

        // ---- commanded, but not moving ----
        //
        // The sweep has had this check since a dead battery produced a
        // textbook-looking run with zero motion in it. **`teleop` shipped
        // without it**, because `STALLED` and `MUST_MOVE_ABOVE` were gated
        // out with the sweep — so the host could hold a stalled motor at
        // full duty indefinitely, drawing near its ~500 mA stall current
        // and heating, and nothing would say so.
        //
        // The chip STOPS rather than escaping. Reversing is recovery, and
        // recovery needs to know what is behind the robot — which the host
        // knows and the chip does not. Tier 0 protects the hardware; Tier
        // 2 decides where to go. The host sees `STALLED` in the report
        // line and can act on it.
        let ticks_now = TOTAL_TICKS.load(Ordering::Relaxed);
        let commanded_hard = left.unsigned_abs().max(right.unsigned_abs())
            > STALL_DUTY_FLOOR.unsigned_abs();
        if commanded_hard && ticks_now == ticks_at_last_check {
            stalled_polls += 1;
            if stalled_polls >= STALL_POLLS {
                STALLED.store(true, Ordering::Relaxed);
            }
        } else {
            stalled_polls = 0;
            // Clears itself once the wheels turn again, so a single
            // scuff does not latch the robot off for the session.
            if ticks_now != ticks_at_last_check {
                STALLED.store(false, Ordering::Relaxed);
            }
        }
        ticks_at_last_check = ticks_now;
        if STALLED.load(Ordering::Relaxed) {
            left = 0;
            right = 0;
        }

        let should_run = left != 0 || right != 0;

        // `STBY` is touched only on a transition. It is a GPIO write
        // either way, but reporting it every 50 ms would bury the one
        // event that matters in a thousand that do not.
        if should_run != enabled {
            motors.standby.set_level(Level::from(should_run));
            enabled = should_run;
            STALLED.store(false, Ordering::Relaxed);
        }
        // The same mounting fact, on the way out. Without this a forward
        // command drives the chassis backward — measured 2026-08-13.
        let facing = firmware_support::motor::DRIVETRAIN_SIGN;
        motors.left.set_signed(&mut cfg, left * facing);
        motors.right.set_signed(&mut cfg, right * facing);

        // Reported as a magnitude percentage, which is all the existing
        // status line has room to say. The sign is visible in the encoder
        // counts either way.
        DUTY_PERCENT.store(
            (left.unsigned_abs().max(right.unsigned_abs()) * 100 / DUTY_FULL.unsigned_abs()) as u16,
            Ordering::Relaxed,
        );
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
    pub async fn note_blocking(text: &str) {
        let mut line = Line::new();
        if line.extend_from_slice(text.as_bytes()).is_ok()
            && line.extend_from_slice(b"\r\n").is_ok()
        {
            NOTES.send(line).await;
        }
    }

    /// Queue a note, dropping it if nobody is draining.
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
// `embassy_rp::i2c::I2c::new_blocking` is exactly what it says. A blocking
// transaction inside `odometry_forever` would stall the 10 kHz sampler for
// however long the bus took, and this repo has already paid for that
// lesson once: polling encoders at 50 Hz counted 404 ticks where ~17,000
// were expected.
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

#[cfg(feature = "camera")]
mod camera {
    use core::fmt::Write as _;
    use core::sync::atomic::{AtomicBool, AtomicU32, Ordering};

    use embassy_executor::Spawner;
    use embassy_rp::bind_interrupts;
    use embassy_rp::gpio::{Input, Pull};
    use embassy_rp::i2c::{Config as I2cConfig, I2c};
    use embassy_rp::peripherals::{
        DMA_CH0, I2C0, PIN_0, PIN_1, PIN_13, PIN_14, PIN_15, PIN_16, PIN_17, PIN_18, PIN_19,
        PIN_20, PIN_21, PIN_22, PIN_4, PIN_5, PIO0, PWM_SLICE2,
    };
    use embassy_rp::pio::{
        Config as PioConfig, Direction, InterruptHandler, Pio, ShiftConfig, ShiftDirection,
    };
    use embassy_rp::pwm::{Config as PwmConfig, Pwm};
    use embassy_rp::Peri;
    use embassy_time::{Duration, Timer};

    bind_interrupts!(struct CameraIrqs {
        PIO0_IRQ_0 => InterruptHandler<PIO0>;
    });
    bind_interrupts!(struct DmaIrqs {
        DMA_IRQ_0 => embassy_rp::dma::InterruptHandler<DMA_CH0>;
    });

    /// XCLK divider.
    ///
    /// ⚠️ A top of 5 gives 25 MHz on an RP2350 — the value every OV7670
    /// reference design uses, and inside the sensor's 10–24 MHz window at
    /// the RP2040's 125 MHz too.
    const XCLK_TOP: u16 = 5;
    const XCLK_COMPARE: u16 = 3;

    /// How long XCLK runs before the first transaction, and between
    /// identification retries.
    const SETTLE_MS: u64 = 100;
    /// Identification attempts before giving up for good.
    const ATTEMPTS: usize = 5;

    /// One thumbnail every this many frames.
    const THUMBNAIL_EVERY: u32 = 16;

    /// How often [`announce`] runs, in seconds. Used to turn a frame
    /// delta into a rate, so the two cannot drift apart.
    pub const ANNOUNCE_SECONDS: u32 = 10;

    /// Frame geometry.
    ///
    /// # ⚠️ 140 wide, not the 160 that QQVGA implies — and it was MEASURED
    ///
    /// The sensor's own colour-bar pattern came back correct in colour and
    /// order but sheared: every row offset from the one above. That shear
    /// is a measurement. With a row stride of `S` and a true line width of
    /// `W`, our row `r` column `c` reads stream pixel `(S*r + c) mod W`,
    /// so the offset of row `r` is `S*r mod W`. Observed, with `S` = 160:
    ///
    /// ```text
    ///   row  1  offset +20 px   ->   160 mod W = 20
    ///   row  7  offset   0      ->   1120 mod W = 0     (1120 = 8 x 140)
    ///   row 14  offset   0      ->   2240 mod W = 0     (2240 = 16 x 140)
    /// ```
    ///
    /// **W = 140**, and it predicts something it was not fitted to: a bar
    /// width of 140/8 = 17.5 px, which is why the captured runs alternate
    /// between four and five thumbnail pixels instead of being a clean
    /// five. That is the check that makes this a measurement rather than a
    /// curve fit.
    ///
    /// ⚠️ **Why the sensor does this is NOT understood.** The scaling
    /// registers ask for 640/4 = 160. The likely cause is the default
    /// `HSTART`/`HSTOP` window, which is a known OV7670 quirk — but that
    /// is a hypothesis, and 140 is the number the hardware actually
    /// produced. Tuning the window to yield a rounder figure is worth
    /// doing and is not the same job as making the picture correct.
    /// ⚠️ **120, and deliberately narrower than the sensor's line.**
    ///
    /// The true line width is not known — measured attempts gave 140 and
    /// then contradicted themselves, because the earlier PIO program lost
    /// a varying number of bytes per line and the shear it produced was
    /// therefore not a clean stride mismatch to solve for.
    ///
    /// So the width is no longer *inferred*. PIO counts out exactly this
    /// many pixels per line and then waits for the next `HREF`, which
    /// means any line at least this wide produces aligned rows. 120 is
    /// comfortably inside the ~160 the bar pattern implies, and the cost
    /// is cropped field of view rather than a corrupt picture.
    ///
    /// ⚠️ Widening this is only safe once the real line width is measured
    /// — by counting `PCLK` edges between `HREF` edges, which is a
    /// question the sensor can be asked directly rather than solved for.
    pub const FRAME_WIDTH: u16 = 120;
    pub const FRAME_HEIGHT: u16 = 120;
    const FRAME_BYTES: usize = FRAME_WIDTH as usize * FRAME_HEIGHT as usize * 2;
    /// The same frame as PIO delivers it — four bytes per pushed word.
    const FRAME_WORDS: usize = FRAME_BYTES / 4;

    /// The frame buffer. 38,400 bytes of the RP2350's 520 KB.
    ///
    /// `StaticCell` rather than a `static mut` because `unsafe_code` is
    /// forbidden repo-wide and this needs no exception.
    static PICTURE: static_cell::StaticCell<[u32; FRAME_WORDS]> = static_cell::StaticCell::new();

    /// Identity, packed, so it can be restated without touching the bus.
    static CACHED: AtomicU32 = AtomicU32::new(0);
    static SEEN: AtomicBool = AtomicBool::new(false);
    /// `applied | com7 << 8 | com15 << 16` — what the sensor says its
    /// own format registers hold, after being told.
    static CONFIG_READBACK: AtomicU32 = AtomicU32::new(0);
    /// Darkest and brightest byte of the most recent frame, `low | high << 8`.
    static PIXEL_RANGE: AtomicU32 = AtomicU32::new(0);
    /// Frames captured since boot. A rate, and a liveness signal.
    static FRAMES: AtomicU32 = AtomicU32::new(0);
    /// `centroid_x << 16 | centroid_y`, valid only when [`BLOB_AREA`] is
    /// non-zero.
    static BLOB_POS: AtomicU32 = AtomicU32::new(0);
    /// Matching pixel count. **Zero means nothing was found** — which is
    /// a real answer, not a missing one.
    static BLOB_AREA: AtomicU32 = AtomicU32::new(0);

    /// Every pin the camera needs, named so two cannot be swapped by
    /// argument order — the same reason [`super::MotorPins`] exists.
    pub struct CameraPins {
        pub i2c: Peri<'static, I2C0>,
        pub sda: Peri<'static, PIN_4>,
        pub scl: Peri<'static, PIN_5>,
        pub xclk_slice: Peri<'static, PWM_SLICE2>,
        pub xclk: Peri<'static, PIN_21>,
        pub vsync: Peri<'static, PIN_1>,
        pub href: Peri<'static, PIN_22>,
        pub pclk: Peri<'static, PIN_0>,
        /// ⚠️ `PIO0` and `DMA_CH0` are the radio's on a `wifi` build — see
        /// the `compile_error!` above. Two owners of one peripheral is the
        /// shape this repo keeps finding bugs in, so it is refused at
        /// compile time rather than discovered as a corrupt frame.
        pub pio: Peri<'static, PIO0>,
        pub dma: Peri<'static, DMA_CH0>,
        /// `D0`..`D7`. **All eight**, not just the base: every pin PIO
        /// reads must have its function switched to PIO, and claiming only
        /// `D0` leaves the other seven reading as nothing. They must also
        /// be consecutive — `in pins` reads a contiguous group, which is
        /// the whole reason the encoders moved.
        pub data: (
            Peri<'static, PIN_13>,
            Peri<'static, PIN_14>,
            Peri<'static, PIN_15>,
            Peri<'static, PIN_16>,
            Peri<'static, PIN_17>,
            Peri<'static, PIN_18>,
            Peri<'static, PIN_19>,
            Peri<'static, PIN_20>,
        ),
    }

    /// QQVGA (160x120) in RGB565 — the format `crates/blob` consumes.
    ///
    /// ⚠️ **A starting point, not a tuned table.** Camera register sets are
    /// order-dependent and vendor-specific. Written as data rather than
    /// code so that adjusting it is editing a table, not editing logic.
    const QQVGA_RGB565: &[(u8, u8)] = &[
        (0x12, 0x04), // COM7   — RGB output
        // ⚠️ Measured, and two of the readings were taken with a broken
        // counter. `/8` puts the internal clock near 3 MHz, under the
        // sensor's ~10 MHz floor, and the frame rate collapses by ~100x
        // rather than the 8x a divider implies.
        //
        // ⚠️ **Not tuned, and cannot be until an image is looked at.**
        // Frame rate matters here only through exposure — exposure cannot
        // exceed the frame period — and brightness is not judgeable from
        // a number. Picking this by arithmetic is the same error as
        // trusting a counter with a silent floor.
        (0x11, 0x01), // CLKRC  — internal clock / 2
        (0x0C, 0x04), // COM3   — enable downsampling (DCW)
        (0x3E, 0x1A), // COM14  — DCW on, PCLK divided by 4
        (0x40, 0xD0), // COM15  — RGB565, full 0-255 range
        // ⚠️ Bit 7 of these two selects the sensor's internal TEST
        // PATTERN, and [`TEST_PATTERN`] flips it.
        //
        //   XSC.7  YSC.7   output
        //     0      0     the lens
        //     0      1     eight-bar colour bars
        //     1      0     fade-to-grey bars
        //     1      1     a shifting "1"
        //
        // Colour bars are generated INSIDE the sensor, after the pixel
        // array and before the output pins. So they travel the whole path
        // being debugged — format, PCLK, HREF, the eight data lines, PIO,
        // DMA — while depending on no lens, no light and no exposure.
        // Bars mean the capture is sound and the problem is optical;
        // noise means the capture is not, and no amount of tuning hue or
        // exposure was ever going to help.
        (0x70, 0x3A), // SCALING_XSC
        (0x71, 0x35), // SCALING_YSC
        (0x72, 0x22), // SCALING_DCWCTR    — /4 horizontally and vertically
        (0x73, 0xF2), // SCALING_PCLK_DIV  — /4
        (0xA2, 0x02), // SCALING_PCLK_DELAY
        // ⚠️ Undocumented, and load-bearing. Without it the OV7670's
        // RGB565 output carries a strong magenta cast — whites render
        // lavender-pink, seen on this very sensor in Rerun (2026-08-14).
        // 0xB0 does not appear in the datasheet at all; 0x84 is the value
        // every working init sequence in the wild converged on, and no
        // more is known about it than that. The cast is upstream of the
        // capture: the bars decoded byte-perfect while the lens image
        // stayed pink, which is what points the finger at the sensor's
        // own colour matrix rather than at anything this repo does.
        (0xB0, 0x84), // undocumented "colour mode" — kills the magenta cast
    ];

    /// Ask the sensor for colour bars instead of the lens.
    ///
    /// ⚠️ A debugging switch, and it should be `false` in anything that
    /// matters. Left in the source rather than deleted because "compare
    /// against something you know" is the move that ended a long guessing
    /// session on 2026-08-14, and the next person deserves the same lever
    /// without having to find the register.
    const TEST_PATTERN: bool = false;

    /// What colour the robot is hunting.
    ///
    /// ⚠️ Green rather than red: red shares a hue neighbourhood with skin,
    /// wood and terracotta, which is most of a room. Tuning belongs here,
    /// in one place, rather than at the call site.
    // ⚠️ Hunting BRIGHTNESS, not hue — third discriminant, and the first
    // with a measurement behind it in both directions.
    //
    // Hue failed twice on this bench (2026-08-14). Green: the post-0xB0
    // white balance tints the whole scene green, so the FLOOR outscored a
    // genuinely green wire — noise ~160, signal ~0. Blue: clean noise
    // floor, but the only blue thing on the bench is a thin wire that
    // must be hand-held in frame, which is a prop-dependent robot.
    //
    // Brightness is immune to the cast — a bright patch is bright
    // whatever colour the sensor believes it is — and it needs no prop:
    // the robot chases the brightest thing it can see, and a phone torch
    // steers it. The threshold is RELATIVE (frame mean + margin), so a
    // dim room and a lit one both have a "brightest patch" rather than a
    // fixed number that works in one room only.
    /// How far above the frame's mean luma a pixel must be to count.
    const BRIGHT_MARGIN: u8 = 50;
    /// Fewer matching pixels than this and nothing is reported.
    const BRIGHT_MIN_PIXELS: u32 = 40;

    /// Bring the camera up and leave it running.
    ///
    /// Returns the PWM guard: **XCLK must keep running** afterwards, and
    /// dropping it would stop the clock. Binding it to a name at the call
    /// site is what keeps it alive.
    #[must_use = "dropping this stops XCLK and the camera goes deaf"]
    pub async fn start(spawner: Spawner, pins: CameraPins) -> Pwm<'static> {
        // ⚠️ XCLK first, before anything touches the bus. The OV7670 has
        // no oscillator of its own, so a transaction issued before this
        // runs fails in a way that looks exactly like bad wiring.
        let mut clock_config = PwmConfig::default();
        clock_config.top = XCLK_TOP;
        // Channel **B**, because GP21 is odd: `slice = (n/2) % 8`,
        // `channel = n % 2`. Embassy encodes that in its types, so
        // `new_output_a` here does not compile rather than silently
        // driving the wrong pin.
        clock_config.compare_b = XCLK_COMPARE;
        let xclk = Pwm::new_output_b(pins.xclk_slice, pins.xclk, clock_config);

        // ⚠️ Async, because the sensor needs the clock to have been
        // running before it answers. A synchronous version identified it
        // microseconds after starting XCLK and reported `BUS ERROR` on a
        // camera that was provably fine.
        Timer::after_millis(SETTLE_MS).await;

        let i2c = I2c::new_blocking(pins.i2c, pins.scl, pins.sda, I2cConfig::default());
        let mut sensor = ov7670_driver::Ov7670::new(i2c);

        // Spaced attempts. One try turns settling-still-in-progress into a
        // permanent verdict, and this verdict is cached for the life of
        // the program.
        let mut result = sensor.identify();
        for _ in 0..ATTEMPTS - 1 {
            if result.as_ref().is_ok_and(ov7670_driver::Identity::is_ov7670) {
                break;
            }
            Timer::after_millis(SETTLE_MS).await;
            result = sensor.identify();
        }

        let mut text: heapless::String<160> = heapless::String::new();
        let mut identity_bytes = (0u8, 0u8, 0u8, 0u8);
        match result {
            Ok(identity) => {
                identity_bytes = (
                    identity.product,
                    identity.version,
                    identity.manufacturer_high,
                    identity.manufacturer_low,
                );
                let _ = write!(
                    text,
                    "# camera pid=0x{:02X} ver=0x{:02X} mid=0x{:02X}{:02X} {}",
                    identity.product,
                    identity.version,
                    identity.manufacturer_high,
                    identity.manufacturer_low,
                    identity.complaint().unwrap_or("OK — this is an OV7670"),
                );
            }
            Err(_) => {
                let _ = text.push_str(
                    "# camera BUS ERROR — nobody acknowledged 0x21; check SIOD/SIOC and RESET",
                );
            }
        }

        // ⚠️ Only attempted once identification succeeded. Writing a
        // register table into silence would look like configuration and be
        // nothing of the kind.
        if identity_bytes.0 == ov7670_driver::EXPECTED_PRODUCT_ID {
            let _ = sensor.reset();
            Timer::after_millis(SETTLE_MS).await;
            // ⚠️ The result is CHECKED, and then the registers are read
            // BACK. Discarding it — `let _ = apply(..)` — is what the
            // first version did, and a sensor left in its power-on
            // default (VGA, YUV) while the host decodes RGB565 produces
            // exactly the full-entropy noise this was debugged from.
            //
            // A write that returns Ok is not proof either: SCCB has no
            // read-after-write guarantee, and a register the sensor
            // refuses still ACKs. Only the read-back is evidence.
            let mut applied = sensor.apply(QQVGA_RGB565).is_ok();
            if TEST_PATTERN {
                // YSC bit 7 on, XSC bit 7 off: the eight-bar pattern.
                applied &= sensor.write_register(0x71, 0x35 | 0x80).is_ok();
            }
            Timer::after_millis(SETTLE_MS).await;
            let com7 = sensor.read_register(0x12).unwrap_or(0xEE);
            let com15 = sensor.read_register(0x40).unwrap_or(0xEE);
            CONFIG_READBACK.store(
                u32::from(applied)
                    | (u32::from(com7) << 8)
                    | (u32::from(com15) << 16),
                Ordering::Relaxed,
            );
        }

        crate::diag::note(&text);
        let packed = [
            identity_bytes.0,
            identity_bytes.1,
            identity_bytes.2,
            identity_bytes.3,
        ];
        CACHED.store(u32::from_be_bytes(packed), Ordering::Relaxed);
        // ⚠️ Only on a real answer. Setting this unconditionally made
        // `announce` restate a cached all-zeroes identity every ten
        // seconds — a failure reported in the voice of a measurement.
        SEEN.store(identity_bytes.0 != 0, Ordering::Relaxed);

        let mut pio = Pio::new(pins.pio, CameraIrqs);
        // ⚠️ HREF and PCLK are PIO pins, not `Input`s.
        //
        // An earlier version made them `Input`s to probe for edges, then
        // `mem::forget` them so the pads survived — because *dropping*
        // them restored the pad and left `wait gpio` reading a dead line
        // forever. `make_pio_pin` is the honest version: PIO configures
        // the pad and owns it, and no diagnostic can leave it in a state
        // the capture then depends on.
        //
        // The probe itself is gone. It asked "is this wired?", and the
        // sensor's own colour bars answer that far better — they travel
        // the whole path and depend on no lens, no light and no exposure.
        let href_pin = pio.common.make_pio_pin(pins.href);
        let _pclk_pin = pio.common.make_pio_pin(pins.pclk);
        // Every one of the eight, switched to PIO function.
        let d0 = pio.common.make_pio_pin(pins.data.0);
        let d1 = pio.common.make_pio_pin(pins.data.1);
        let d2 = pio.common.make_pio_pin(pins.data.2);
        let d3 = pio.common.make_pio_pin(pins.data.3);
        let d4 = pio.common.make_pio_pin(pins.data.4);
        let d5 = pio.common.make_pio_pin(pins.data.5);
        let d6 = pio.common.make_pio_pin(pins.data.6);
        let d7 = pio.common.make_pio_pin(pins.data.7);
        let all_data = [&d0, &d1, &d2, &d3, &d4, &d5, &d6, &d7];

        // ```text
        //   wait 1 gpio 22   ; HREF high  -- this line carries real pixels
        //   wait 0 gpio 0    ; PCLK low   -- so the next wait sees a real edge
        //   wait 1 gpio 0    ; PCLK high  -- data valid on the rising edge
        //   in pins, 8       ; sample D0-D7 (GP13..GP20)
        // ```
        //
        // ⚠️ **Both `PCLK` waits are needed.** With only `wait 1`, a state
        // machine arriving while PCLK is already high samples immediately
        // AND on the next rising edge — one duplicated byte per line,
        // which shifts every subsequent pixel and produces a picture that
        // looks like a wiring fault.
        //
        // When HREF falls the first `wait` blocks until the next line, so
        // horizontal blanking costs nothing and needs no code.
        // ⚠️ A FIXED number of words per line, re-armed at every HREF.
        //
        // Three programs were tried on hardware and the difference is
        // worth keeping:
        //
        //   1. free-running (`wait href` / `wait pclk` / `in`)
        //      -> correct bar ORDER within a row, rows sheared. Lines
        //         were intact; only the stride was wrong.
        //   2. `jmp pin` line loop with `mov isr, null`
        //      -> WORSE. Bar order within a row scrambled, because the
        //         number of words a line yields depends on the line's
        //         length, and that varies.
        //   3. this: count out exactly `WORDS_PER_ROW` words, then block
        //      until the next line. Row alignment cannot drift, because
        //      it is no longer inferred from anything.
        //
        // `Y` holds the count, loaded once at startup; `X` is the working
        // copy, refreshed per line. Four `in`s per iteration because
        // autopush fires at 32 bits and the loop should push whole words.
        // ⚠️ A FIXED word count per line, re-armed at every HREF.
        //
        // Three programs were tried on hardware, and the comparison is
        // worth keeping:
        //
        //   1. free-running (`wait href` / `wait pclk` / `in`)
        //      -> bar ORDER correct within a row, rows sheared. Lines
        //         were intact; only the stride was wrong.
        //   2. `jmp pin` line loop with `mov isr, null`
        //      -> WORSE: order scrambled *within* rows, because the words
        //         a line yields then depend on the line's length.
        //   3. this: count out exactly 6 x 10 = 60 words, then block
        //      until the next line. Alignment is asserted, not inferred.
        //
        // ⚠️ Nested `set` loops rather than a count from the TX FIFO.
        // `capture_frame` calls `clear_fifos()` before every frame to drop
        // stale pixels, and that clears TX as well as RX — so a count
        // pushed once at startup is deleted before the program reads it,
        // and `pull` then blocks forever. That produced a board which
        // captured nothing at all. `set` takes a literal 0..31, so 60 is
        // reached as two loops rather than one.
        //
        // Four `in`s per iteration because autopush fires at 32 bits, and
        // the loop should deal in whole words.
        let program = embassy_rp::pio::program::pio_asm!(
            ".wrap_target",
            "wait 0 gpio 22",       // blanking — the previous line ended
            "mov isr, null",        // drop any partial word it left
            "wait 1 gpio 22",       // this line starts HERE
            "set y, 5",             // outer: 6 passes
            "outer:",
            "set x, 9",             // inner: 10 words each
            "inner:",
            "wait 0 gpio 0",
            "wait 1 gpio 0",
            "in pins, 8",
            "wait 0 gpio 0",
            "wait 1 gpio 0",
            "in pins, 8",
            "wait 0 gpio 0",
            "wait 1 gpio 0",
            "in pins, 8",
            "wait 0 gpio 0",
            "wait 1 gpio 0",
            "in pins, 8",
            "jmp x-- inner",
            "jmp y-- outer",
            ".wrap",
        );

        let mut config = PioConfig::default();
        config.use_program(&pio.common.load_program(&program.program), &[]);
        config.set_in_pins(&all_data);
        // `jmp pin` tests this one. It is what turns "sample forever" into
        // "sample exactly this line".
        config.set_jmp_pin(&href_pin);
        // ⚠️ Shift LEFT, so the first byte sampled lands in the LOW byte of
        // the word. Right-shifting reverses byte order within every word —
        // an image that is subtly, periodically scrambled rather than
        // obviously broken.
        config.shift_in = ShiftConfig {
            threshold: 32,
            direction: ShiftDirection::Left,
            auto_fill: true,
        };
        pio.sm0.set_config(&config);
        pio.sm0.set_pin_dirs(Direction::In, &all_data);

        let dma = embassy_rp::dma::Channel::new(pins.dma, DmaIrqs);
        let vsync = Input::new(pins.vsync, Pull::None);
        let buffer = PICTURE.init([0u32; FRAME_WORDS]);

        // The capture loop gets its own task so that a frame — tens of
        // milliseconds — never sits inside the 10 kHz encoder sampler.
        // Same rule as `diag`: nothing may block the thing being measured.
        spawner.spawn(watch(pio, dma, vsync, buffer).unwrap());
        xclk
    }

    /// Capture frames forever, and look for the target colour in each.
    #[embassy_executor::task]
    async fn watch(
        mut pio: Pio<'static, PIO0>,
        mut dma: embassy_rp::dma::Channel<'static>,
        mut vsync: Input<'static>,
        buffer: &'static mut [u32; FRAME_WORDS],
    ) -> ! {
        loop {
            if !capture_frame(&mut pio, &mut dma, &mut vsync, buffer).await {
                // A frame that never arrived. Say nothing rather than
                // publish a stale blob as though it were current.
                BLOB_AREA.store(0, Ordering::Relaxed);
                Timer::after_millis(100).await;
                continue;
            }
            FRAMES.fetch_add(1, Ordering::Relaxed);

            let (mut low, mut high) = (u8::MAX, u8::MIN);
            for word in buffer.iter() {
                for byte in word.to_le_bytes() {
                    low = low.min(byte);
                    high = high.max(byte);
                }
            }
            PIXEL_RANGE.store(u32::from(low) | (u32::from(high) << 8), Ordering::Relaxed);

            // RGB565 is two bytes per pixel, little-endian within the word
            // as PIO packed them.
            // ⚠️ Big-endian pairing — same measured fact as `pixel_at`,
            // and it must match, or the blob hunts pixels the thumbnail
            // does not show.
            let pixels = || {
                buffer.iter().flat_map(|word| {
                    let [a, b, c, d] = word.to_le_bytes();
                    [u16::from_be_bytes([a, b]), u16::from_be_bytes([c, d])]
                })
            };
            // Pass one: the frame's mean brightness. Pass two: the blob
            // of pixels well above it. Two passes over 19,200 pixels is
            // cheap next to the frame that took milliseconds to arrive —
            // and a relative threshold is what makes this work in any
            // light rather than the light it was tuned in.
            let mean_luma = (pixels().map(|p| u32::from(blob::luma(p))).sum::<u32>()
                / (FRAME_WIDTH as u32 * FRAME_HEIGHT as u32)) as u8;
            let floor = mean_luma.saturating_add(BRIGHT_MARGIN);
            match blob::find_matching(pixels(), FRAME_WIDTH, BRIGHT_MIN_PIXELS, |p| {
                blob::luma(p) > floor
            }) {
                Some(found) => {
                    BLOB_POS.store(
                        ((found.centroid_x as u32) << 16) | (found.centroid_y as u32),
                        Ordering::Relaxed,
                    );
                    BLOB_AREA.store(found.area.max(1), Ordering::Relaxed);
                }
                // ⚠️ Zero is a real answer — "looked, found nothing" —
                // and must not be confused with "did not look".
                None => BLOB_AREA.store(0, Ordering::Relaxed),
            }

            // A picture, periodically. Rare enough not to crowd out the
            // pose stream, often enough to watch a scene change.
            if FRAMES.load(Ordering::Relaxed) % THUMBNAIL_EVERY == 0 {
                send_thumbnail(buffer).await;
            }
        }
    }


    /// Thumbnail geometry — a quarter of the frame in each axis.
    ///
    /// ⚠️ Small on purpose. The full frame is 38,400 bytes; as hex over a
    /// line-based protocol that is 600 lines, twelve seconds at 50 Hz.
    /// 40x30 is 2,400 bytes, thirty lines, well under a second — and it
    /// is *enough to look at*, which is the entire point. Every remaining
    /// uncertainty here (byte order, RGB565-vs-YUV, exposure, and what
    /// the sensor is actually pointed at) is settled by seeing a picture,
    /// and none of them is settled by another statistic.
    const THUMB_WIDTH: usize = FRAME_WIDTH as usize / 4;
    const THUMB_HEIGHT: usize = 30;
    /// Pixels per thumbnail pixel, per axis.
    const THUMB_STEP: usize = FRAME_WIDTH as usize / THUMB_WIDTH;

    /// Send a downsampled copy of `buffer` to the host, as hex.
    ///
    /// Nearest-neighbour rather than averaging: averaging RGB565 needs
    /// unpacking every channel, and a thumbnail exists to be *looked at*,
    /// not measured. Sampling is also honest about aliasing in a way an
    /// average is not — a smooth wrong picture is harder to distrust.
    async fn send_thumbnail(buffer: &[u32]) {
        let mut header: heapless::String<32> = heapless::String::new();
        let _ = write!(header, "# IMG {THUMB_WIDTH} {THUMB_HEIGHT} rgb565");
        crate::diag::note_blocking(&header).await;
        let pixel_at = |x: usize, y: usize| -> u16 {
            let index = y * FRAME_WIDTH as usize + x;
            let word = buffer[index / 2];
            let [a, b, c, d] = word.to_le_bytes();
            // ⚠️ BIG-endian pairing: the OV7670 sends the HIGH byte of each
            // RGB565 pixel first. Measured 2026-08-14 against the sensor's
            // own colour bars: the little-endian pairing produced `6CF7`
            // where the known-good yellow is `F76C` — every value byte-
            // swapped, positions all correct.
            //
            // The earlier free-running capture decoded "correctly" with
            // the wrong pairing because its DMA started at an arbitrary
            // byte offset, and an odd offset re-pairs every pixel — two
            // errors cancelling. Aligning the capture surfaced this one.
            if index % 2 == 0 {
                u16::from_be_bytes([a, b])
            } else {
                u16::from_be_bytes([c, d])
            }
        };
        for row in 0..THUMB_HEIGHT {
            let mut line: heapless::String<192> = heapless::String::new();
            let _ = line.push_str("# ");
            for column in 0..THUMB_WIDTH {
                let pixel = pixel_at(column * THUMB_STEP, row * THUMB_STEP);
                let _ = write!(line, "{pixel:04X}");
            }
            crate::diag::note_blocking(&line).await;
        }
    }

    /// Capture one whole frame, starting at a real frame boundary.
    ///
    /// ```text
    ///   wait for VSYNC high   -- frame ending, blanking begins
    ///   wait for VSYNC low    -- THIS is the start of a new frame
    ///   enable the state machine
    /// ```
    ///
    /// ⚠️ The state machine is **enabled** at the boundary rather than
    /// merely unblocked. A PIO program that waits internally still holds
    /// whatever was in its FIFO from the previous frame, and that stale
    /// word becomes the first pixel of this one — a picture shifted by a
    /// few bytes, which looks like a scaling bug rather than a
    /// synchronisation one.
    async fn capture_frame(
        pio: &mut Pio<'static, PIO0>,
        dma: &mut embassy_rp::dma::Channel<'static>,
        vsync: &mut Input<'static>,
        buffer: &mut [u32],
    ) -> bool {
        pio.sm0.set_enable(false);
        pio.sm0.clear_fifos();

        let synced = embassy_time::with_timeout(Duration::from_millis(200), async {
            vsync.wait_for_high().await;
            vsync.wait_for_low().await;
        })
        .await
        .is_ok();
        if !synced {
            return false;
        }

        pio.sm0.set_enable(true);
        // ⚠️ Bounded. A frame that never completes — HREF stuck low, PCLK
        // dead — would otherwise hang this task forever, reporting
        // nothing, which reads exactly like a crash.
        let filled = embassy_time::with_timeout(
            Duration::from_millis(500),
            pio.sm0.rx().dma_pull(dma, buffer, false),
        )
        .await
        .is_ok();
        pio.sm0.set_enable(false);
        filled
    }


    /// Frames captured since the previous call.
    ///
    /// ⚠️ A **rate**, not a total. A running total that stops climbing is
    /// indistinguishable at a glance from a report that stopped arriving,
    /// and this session has already lost hours to counters that could not
    /// say "nothing happened".
    fn frames_since_last_announce() -> u32 {
        let now = FRAMES.load(Ordering::Relaxed);
        let previous = LAST_ANNOUNCED_FRAMES.swap(now, Ordering::Relaxed);
        now.wrapping_sub(previous)
    }

    /// Frame count at the previous announce.
    static LAST_ANNOUNCED_FRAMES: AtomicU32 = AtomicU32::new(0);

    /// Total frames captured since boot. The chase loop watches this to
    /// tell a live image from a wedged one — a count that stops advancing
    /// is the camera's equivalent of a host gone silent.
    pub fn frame_total() -> u32 {
        FRAMES.load(Ordering::Relaxed)
    }

    /// Where the target colour is, as offsets from frame centre in
    /// **−1..+1**, plus how many pixels matched.
    ///
    /// `None` means the last frame contained no match — a real answer.
    pub fn blob_error() -> Option<(f32, f32, u32)> {
        let area = BLOB_AREA.load(Ordering::Relaxed);
        if area == 0 {
            return None;
        }
        let packed = BLOB_POS.load(Ordering::Relaxed);
        let found = blob::Blob {
            centroid_x: (packed >> 16) as f32,
            centroid_y: (packed & 0xFFFF) as f32,
            area,
            bounds: (0, 0, 0, 0),
            frame: (FRAME_WIDTH, FRAME_HEIGHT),
        };
        let (x, y) = found.error_from_centre();
        Some((x, y, area))
    }

    /// Say what the camera is, again.
    ///
    /// # ⚠️ Why a boot-time fact gets repeated
    ///
    /// A note said once is a note nobody hears: `diag::NOTES` is drained
    /// only while a host is attached, and the camera is identified
    /// milliseconds after power-up. Worse, *opening* a port is enough to
    /// drain the queue, so a tool that probes and closes (`stty` does
    /// exactly this) consumes the line on the way past. Two flashes and
    /// two reads went by with the line never once seen, while the camera
    /// itself was provably fine.
    pub fn announce() {
        if !SEEN.load(Ordering::Relaxed) {
            return;
        }
        let [product, version, mid_high, mid_low] = CACHED.load(Ordering::Relaxed).to_be_bytes();
        let identity = ov7670_driver::Identity {
            product,
            version,
            manufacturer_high: mid_high,
            manufacturer_low: mid_low,
        };
        let range = PIXEL_RANGE.load(Ordering::Relaxed);
        let (low, high) = (range & 0xFF, (range >> 8) & 0xFF);

        let mut text: heapless::String<160> = heapless::String::new();
        let _ = write!(
            text,
            "# camera pid=0x{product:02X} mid=0x{mid_high:02X}{mid_low:02X} {} | px {low}..{high}",
            identity.complaint().unwrap_or("OK"),
        );
        // Frames per second, from the delta over the announce interval.
        let captured = frames_since_last_announce();
        let _ = write!(text, " | {} fps", captured / ANNOUNCE_SECONDS);

        // What the sensor says its format actually is. COM7 bit 2 and
        // COM15 bits 4-5 are the RGB565 selection; if these read back as
        // anything else, every pixel above is being decoded as a format
        // the sensor is not producing.
        let config = CONFIG_READBACK.load(Ordering::Relaxed);
        let (com7, com15) = ((config >> 8) & 0xFF, (config >> 16) & 0xFF);
        let _ = write!(
            text,
            " | com7=0x{com7:02X} com15=0x{com15:02X} {}",
            if config & 1 == 0 {
                "⚠️ WRITES FAILED"
            } else if com7 == 0x04 && com15 == 0xD0 {
                "format confirmed"
            } else {
                "⚠️ FORMAT NOT SET — sensor ignored the table"
            }
        );
        let _ = match blob_error() {
            Some((x, y, area)) => write!(text, " | blob x{x:+.2} y{y:+.2} area {area}"),
            None => write!(text, " | no blob"),
        };
        crate::diag::note(&text);
    }
}


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
async fn chase_forever(mut motors: Motors, spec: RobotSpec) -> ! {
    /// Forward/backward creep, metres per second. "Slightly" made a number.
    const CREEP_M_PER_S: f64 = 0.06;
    /// Turn nudge, radians per second.
    const TURN_RAD_PER_S: f64 = 0.5;
    /// How far off-centre (−1..+1) the blob may sit before turning.
    const CENTRE_DEADBAND: f32 = 0.20;
    /// Blob smaller than this (pixels) → it is far → creep forward.
    const AREA_FAR: u32 = 250;
    /// Blob bigger than this → too close → back away.
    const AREA_NEAR: u32 = 1500;
    /// No new frame for this long → the image has stopped → so do we.
    const FRESH_MS: u64 = 1000;

    // Nothing moves until a host opens the port — the same invariant the
    // sweep and teleop keep. A robot that starts hunting on power-up,
    // with nobody watching and no way to stop it, is the surprise
    // `HOST_WATCHING` exists to prevent.
    while !HOST_WATCHING.load(Ordering::Relaxed) {
        Timer::after_millis(50).await;
    }

    let mut cfg = PwmConfig::default();
    cfg.top = PWM_TOP;
    motors.left.set_signed(&mut cfg, 0);
    motors.right.set_signed(&mut cfg, 0);

    let mut enabled = false;
    let mut ticks_at_last_check = TOTAL_TICKS.load(Ordering::Relaxed);
    let mut stalled_polls: u32 = 0;
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

        let wheels = spec.fit_wheels(spec.drive().inverse(BodyTwist {
            forward_speed: v,
            turn_rate: w,
        }));
        let (mut left, mut right) = (spec.duty(wheels.left), spec.duty(wheels.right));

        // ---- commanded, but not moving — same guard as teleop ----
        let ticks_now = TOTAL_TICKS.load(Ordering::Relaxed);
        let commanded_hard =
            left.unsigned_abs().max(right.unsigned_abs()) > STALL_DUTY_FLOOR.unsigned_abs();
        if commanded_hard && ticks_now == ticks_at_last_check {
            stalled_polls += 1;
            if stalled_polls >= STALL_POLLS {
                STALLED.store(true, Ordering::Relaxed);
            }
        } else {
            stalled_polls = 0;
            if ticks_now != ticks_at_last_check {
                STALLED.store(false, Ordering::Relaxed);
            }
        }
        ticks_at_last_check = ticks_now;
        if STALLED.load(Ordering::Relaxed) {
            left = 0;
            right = 0;
        }

        let should_run = left != 0 || right != 0;
        if should_run != enabled {
            motors.standby.set_level(Level::from(should_run));
            enabled = should_run;
            STALLED.store(false, Ordering::Relaxed);
        }
        // The same mounting fact as everywhere else, applied on the way
        // out — see `firmware_support::motor::DRIVETRAIN_SIGN`.
        let facing = firmware_support::motor::DRIVETRAIN_SIGN;
        motors.left.set_signed(&mut cfg, left * facing);
        motors.right.set_signed(&mut cfg, right * facing);

        DUTY_PERCENT.store(
            (left.unsigned_abs().max(right.unsigned_abs()) * 100 / DUTY_FULL.unsigned_abs()) as u16,
            Ordering::Relaxed,
        );
    }
}

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
    #[cfg(not(any(feature = "teleop", feature = "chase")))]
    use embassy_futures::join::join;
    use embassy_rp::bind_interrupts;
    use embassy_rp::peripherals::USB;
    use embassy_rp::usb::{Driver, InterruptHandler};
    use embassy_time::{with_timeout, Duration};
    #[cfg(any(feature = "teleop", feature = "chase"))]
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
        #[cfg(feature = "camera")]
        let _camera_clock = camera::start(spawner, camera::CameraPins {
            i2c: p.I2C0,
            sda: p.PIN_4,
            scl: p.PIN_5,
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
        #[cfg(not(any(feature = "teleop", feature = "chase")))]
        {
            spawner.spawn(drive_sweep(motors).unwrap());
            join(usb.run(), odometry_forever(encoders, &mut out)).await;
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
