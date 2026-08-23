//! The PCA9685 and the arm servos behind it — from bus probe to
//! camera-driven motion.
//!
//! ```text
//!   blob x    ─▶  ch0   pan toward the target
//!   blob y    ─▶  ch1   tilt toward it
//!   blob area ─▶  ch2   "gripper" closes as it nears
//!   no blob / stale frames  ─▶  HOLD
//! ```
//!
//! This is `crates/blob`'s founding promise made physical: *one blob
//! yields three independent errors, which is exactly the three motions
//! the rig has*. The wheels consume `x` and `area` in the chase build;
//! the arm consumes all three its own way.
//!
//! # ⚠️ Hold, not release, on target loss
//!
//! The wheels stop when the image dies — a stopped base is safe. A
//! released servo is back-driveable, and on an assembled arm that means
//! falling (`crates/arm/src/safety.rs`). Nothing hangs on these horns
//! yet, but the habit is set now: **losing the target freezes the arm
//! where it is.** Slew limiting does the rest: no target, no motion; a
//! new target, gentle motion toward it.
//!
//! # ⚠️ Signs are bench-arbitrary until a linkage exists
//!
//! "Pan toward" assumes a mounting nobody has built. The signs below are
//! placeholders chosen to be *consistent*, and the day a real arm holds
//! the camera they get measured against it — the same promotion
//! `LEFT_ENCODER_SIGN` went through.

use core::fmt::Write as _;

use embassy_rp::i2c::{Async, I2c};
use embassy_rp::peripherals::I2C0;
use embassy_time::Timer;
use pca9685_driver::asynch::Pca9685;
use pca9685_driver::{Channel, SERVO_FRAME_HZ};

/// Assembly jig: `true` holds EVERY channel at the 1500 µs centre and
/// does nothing else — the pose you attach horns in.
///
/// ⚠️ A servo has no centre marking because centre is not a place on the
/// case: it is where the shaft GOES when commanded 1500 µs, defined by
/// the internal potentiometer. The kit workflow is therefore: hold the
/// channel centred, plug the BARE servo in (the snap-to-centre twitch is
/// the centring), then attach the link in the manual's assembly pose and
/// screw it down. From then on 1500 = that pose, permanently. The spline
/// seats in ~18° steps; the remainder becomes a per-joint trim constant,
/// which is software's job, not a reason to lever a powered horn.
const ASSEMBLY_CENTRE: bool = true;

/// Channels held during assembly — the full five-joint arm, not just the
/// three servos currently on hand, so the MG90S get the same jig.
const ASSEMBLY_CHANNELS: usize = 5;

/// With [`ASSEMBLY_CENTRE`], `true` runs the gentle per-joint exercise
/// after the initial centre hold instead of holding forever.
const ASSEMBLY_EXERCISE: bool = true;

/// Exercise amplitude around centre. ±150 µs ≈ ±13°: enough to see every
/// joint move both ways, nowhere near any mechanical stop.
const EXERCISE_US: i32 = 150;

/// Per-40 ms slew step during the exercise — ~9°/s at the horn.
const EXERCISE_STEP_US: i32 = 4;

/// Bring-up lever: `true` restores the blind lockstep sweep that proved
/// the servos on 2026-08-15. Kept for the same reason the camera keeps
/// `TEST_PATTERN` — comparing against a known motion is the move that
/// ends guessing sessions, and the next person deserves the lever.
const BRINGUP_SWEEP: bool = false;

/// Sweep bounds, 100 µs inside even the conservative nominal span. A
/// servo commanded past its stop buzzes at stall current, and nothing on
/// this rail can see the shaft.
const SWEEP_LOW_US: u32 = 1100;
const SWEEP_HIGH_US: u32 = 1900;
/// Sweep step per 20 ms tick — ~8 s per half-sweep.
const SWEEP_STEP_US: u32 = 2;

/// Tracking centre and swing: full blob error deflects ±400 µs, staying
/// inside the sweep bounds by construction.
const CENTRE_US: f32 = 1500.0;
const SWING_US: f32 = 400.0;
/// Slew per 20 ms tick — 400 µs/s, two seconds lock-to-lock. The blob
/// updates at ~4 Hz; slew is what turns those steps into motion.
const TRACK_SLEW_US: u32 = 8;
/// Blob area at which the "gripper" is fully closed.
const AREA_NEAR: u32 = 1500;
/// No new frame for this long → the image has stopped → the arm holds.
const FRESH_MS: u64 = 1000;

/// How many channels the tracker drives — the three SG90s on hand.
const SERVO_COUNT: usize = 3;

/// The control tick, one servo frame at 50 Hz. Every per-tick rate in
/// this file (slew, sweep step) is calibrated against this number, so
/// it exists exactly once.
const TICK_MS: u64 = 20;
/// Announce the commanded pulses every this many ticks — twice a second
/// at the 20 ms tick, enough for the viewer, cheap on the notes queue.
const NOTE_EVERY_TICKS: u32 = 25;
/// The datasheet's ≥500 µs oscillator settle after wake, rounded up to
/// the timer's comfortable resolution.
const OSCILLATOR_SETTLE_MS: u64 = 1;

/// The channels themselves, proven at COMPILE time. `Channel::new` is
/// `const`, so a `SERVO_COUNT` beyond the chip's sixteen fails the build
/// here instead of surfacing as a runtime `else return` buried in a
/// control loop — the error path deleted rather than handled.
const CHANNELS: [Channel; SERVO_COUNT] = {
    let mut channels = [match Channel::new(0) {
        Ok(channel) => channel,
        Err(_) => panic!("channel 0 always exists"),
    }; SERVO_COUNT];
    let mut index = 0;
    while index < SERVO_COUNT {
        channels[index] = match Channel::new(index as u8) {
            Ok(channel) => channel,
            Err(_) => panic!("SERVO_COUNT exceeds the PCA9685's sixteen channels"),
        };
        index += 1;
    }
    channels
};

/// Ask the PCA9685 who it is; say so on the report stream.
pub async fn probe(bus: I2c<'static, I2C0, Async>) -> I2c<'static, I2C0, Async> {
    let mut driver = Pca9685::new(bus);
    let mut text: heapless::String<96> = heapless::String::new();
    match driver.read_register(pca9685_driver::REG_MODE1).await {
        Ok(mode1) => {
            let _ = write!(
                text,
                "# pca mode1=0x{mode1:02X} {}",
                if mode1 == 0x11 {
                    "OK — power-on default, asleep as shipped"
                } else if mode1 & 0x10 != 0 {
                    "present, asleep (non-default flags)"
                } else {
                    "present and AWAKE — something already started it"
                }
            );
        }
        Err(_) => {
            let _ = text.push_str(
                "# pca ABSENT — nobody acknowledged 0x40; check VCC, GND, and the 6b/7b taps",
            );
        }
    }
    crate::diag::note(&text);
    driver.free()
}

/// Wake the chip, verify the frame rate by read-back, report.
///
/// ⚠️ The read-back is the test, not the ACK: at the power-on 200 Hz
/// default every pulse computed for 50 Hz goes out 4x too narrow, under
/// the floor a servo even acknowledges — everything ACKs, nothing moves.
async fn wake(driver: &mut Pca9685<I2c<'static, I2C0, Async>>) -> bool {
    if driver.start(SERVO_FRAME_HZ).await.is_err() {
        crate::diag::note("# servo start FAILED — bus error waking the PCA9685");
        return false;
    }
    // Oscillator settle the driver cannot wait out itself.
    Timer::after_millis(OSCILLATOR_SETTLE_MS).await;
    let mut text: heapless::String<96> = heapless::String::new();
    match driver.read_register(pca9685_driver::REG_PRESCALE).await {
        Ok(121) => {
            let _ = text.push_str("# servo prescale=121 — 50 Hz confirmed by read-back");
        }
        Ok(other) => {
            let _ = write!(
                text,
                "# servo ⚠️ PRESCALE={other} not 121 — pulses too narrow, servos will NOT move"
            );
        }
        Err(_) => {
            let _ = text.push_str("# servo ⚠️ prescale read-back failed");
        }
    }
    crate::diag::note(&text);
    true
}

/// Drive the three arm channels forever — from the camera, or (with
/// [`BRINGUP_SWEEP`]) the blind sweep that first proved them.
#[embassy_executor::task]
pub async fn run(bus: I2c<'static, I2C0, Async>) {
    let mut driver = Pca9685::new(bus);
    if !wake(&mut driver).await {
        return;
    }
    if ASSEMBLY_CENTRE {
        centre_hold(&mut driver).await;
    } else if BRINGUP_SWEEP {
        sweep(&mut driver).await;
    } else {
        track(&mut driver).await;
    }
}

/// Hold every assembly channel at centre — then, if the exercise flag
/// is up, gently wave each joint in turn: the assembled arm's first
/// commanded motion. One joint at a time, ±EXERCISE_US around centre,
/// slewed in small steps — slow enough to grab the power lead if a
/// linkage binds, small enough that nothing can reach a hard stop.
async fn centre_hold(driver: &mut Pca9685<I2c<'static, I2C0, Async>>) {
    let mut pulses = [CENTRE_US as u32; ASSEMBLY_CHANNELS];
    let Ok(first) = Channel::new(0) else { return };
    if driver.set_pulses(first, &pulses).await.is_err() {
        crate::diag::note("# servo bus error — centre hold failed");
        return;
    }
    if !ASSEMBLY_EXERCISE {
        loop {
            crate::diag::note(
                "# servo ASSEMBLY MODE — ch0-4 held at 1500us, attach horns now",
            );
            Timer::after_millis(5000).await;
        }
    }
    Timer::after_millis(3000).await;
    loop {
        for joint in 0..ASSEMBLY_CHANNELS {
            crate::diag::note("# servo EXERCISE — next joint");
            // centre -> +EXERCISE_US -> -EXERCISE_US -> centre, slewed.
            let centre = CENTRE_US as i32;
            for target in [centre + EXERCISE_US, centre - EXERCISE_US, centre] {
                loop {
                    let now = pulses[joint] as i32;
                    let step = (target - now).clamp(-EXERCISE_STEP_US, EXERCISE_STEP_US);
                    if step == 0 {
                        break;
                    }
                    pulses[joint] = (now + step) as u32;
                    if driver.set_pulses(first, &pulses).await.is_err() {
                        crate::diag::note("# servo bus error — exercise aborted");
                        return;
                    }
                    Timer::after_millis(40).await;
                }
                Timer::after_millis(600).await;
            }
        }
        Timer::after_millis(2000).await;
    }
}

/// Camera-driven tracking: three errors in, three positions out.
async fn track(driver: &mut Pca9685<I2c<'static, I2C0, Async>>) {
    crate::diag::note("# servo TRACKING the camera: x->ch0 y->ch1 area->ch2");
    let mut current = [CENTRE_US as u32; SERVO_COUNT];
    // ⚠️ Forced first write. The skip-when-converged rule below would
    // otherwise mean a bootup with no blob never commands the servos at
    // all — outputs off, horns limp, looking exactly like the power
    // faults this rig spent a night chasing.
    let mut dirty = true;
    let mut last_frame_total = crate::camera::frame_total();
    let mut last_frame_at = firmware_support::now_ms();
    let mut tick = 0u32;

    loop {
        Timer::after_millis(TICK_MS).await;
        let now = firmware_support::now_ms();
        let frames = crate::camera::frame_total();
        if frames != last_frame_total {
            last_frame_total = frames;
            last_frame_at = now;
        }
        let image_alive = now.saturating_sub(last_frame_at) < FRESH_MS;

        // No blob, or no image: targets = where we already are. HOLD.
        let target = match crate::camera::blob_error() {
            Some((x, y, area)) if image_alive => [
                // ⚠️ Clamped at the point of production, Tier-0 style:
                // the blob's errors are bounded by construction TODAY,
                // but the envelope must not depend on that staying true
                // through every future edit to the law above it. A servo
                // commanded past its stop buzzes at stall current, and
                // nothing on this rail can see the shaft.
                ((CENTRE_US - x * SWING_US) as u32).clamp(SWEEP_LOW_US, SWEEP_HIGH_US),
                ((CENTRE_US + y * SWING_US) as u32).clamp(SWEEP_LOW_US, SWEEP_HIGH_US),
                SWEEP_LOW_US + area.min(AREA_NEAR) * (SWEEP_HIGH_US - SWEEP_LOW_US) / AREA_NEAR,
            ],
            _ => current,
        };

        // Slew toward the target rather than jumping: the blob steps at
        // ~4 Hz and a servo snapped 800 µs per step is a rattle, not a
        // motion.
        let mut moved = false;
        for (position, goal) in current.iter_mut().zip(target) {
            let next = if goal > *position {
                (*position + TRACK_SLEW_US).min(goal)
            } else {
                (*position).saturating_sub(TRACK_SLEW_US).max(goal)
            };
            moved |= next != *position;
            *position = next;
        }

        // ⚠️ The bus stays SILENT while nothing changes. The PCA holds
        // its outputs in hardware, so a converged tracker needs no
        // traffic — and every skipped transaction is ~0.5 ms the 10 kHz
        // encoder sampler is not blinded, which is the leading suspect
        // for the decode-error climb logged on 2026-08-15. One batched
        // write when something did change, three separate ones never.
        if moved || dirty {
            if driver.set_pulses(CHANNELS[0], &current).await.is_err() {
                crate::diag::note("# servo bus error mid-track — stopping");
                return;
            }
            dirty = false;
        }

        tick += 1;
        if tick.is_multiple_of(NOTE_EVERY_TICKS) {
            let mut text: heapless::String<64> = heapless::String::new();
            let _ = hil_protocol::arm_pulses::write_note(&mut text, &current);
            crate::diag::note(&text);
        }
    }
}

/// The blind lockstep sweep — the bring-up instrument of 2026-08-15.
async fn sweep(driver: &mut Pca9685<I2c<'static, I2C0, Async>>) {
    crate::diag::note("# servo ch0-2 sweeping TOGETHER 1100-1900us");
    let centre = (SWEEP_LOW_US + SWEEP_HIGH_US) / 2;
    let mut pulse = centre;
    let mut rising = true;
    let mut tick = 0u32;
    loop {
        // One transaction for all three — see `Pca9685::set_pulses`.
        if driver.set_pulses(CHANNELS[0], &[pulse; SERVO_COUNT]).await.is_err() {
            crate::diag::note("# servo bus error mid-sweep — stopping");
            return;
        }
        pulse = if rising {
            pulse + SWEEP_STEP_US
        } else {
            pulse - SWEEP_STEP_US
        };
        if pulse >= SWEEP_HIGH_US || pulse <= SWEEP_LOW_US {
            rising = !rising;
        }
        tick += 1;
        if tick.is_multiple_of(NOTE_EVERY_TICKS) {
            let mut text: heapless::String<64> = heapless::String::new();
            let _ = hil_protocol::arm_pulses::write_note(&mut text, &[pulse; SERVO_COUNT]);
            crate::diag::note(&text);
        }
        Timer::after_millis(TICK_MS).await;
    }
}
