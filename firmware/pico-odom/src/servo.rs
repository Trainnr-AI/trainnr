//! The PCA9685 and the arm servos behind it — bring-up order enforced.
//!
//! # Step one is a read-back, with the servo rail UNPOWERED
//!
//! The chip's logic side runs from the Pico's 3.3 V; `V+` — the servo
//! power — can stay disconnected while every register is exercised. So
//! the bus proof costs nothing that can move: [`probe`] asks MODE1 for
//! its power-on value and reports what came back, next to the camera's
//! identity on the same two wires.
//!
//! ⚠️ `MODE1` after power-on reads **0x11** (`SLEEP | ALLCALL`) — the
//! chip boots asleep, exactly like the MPU6050 and for the same reason.
//! Any ACK is not the test; *that value* is. A stuck bus, a wrong
//! address strap or a solder bridge each fail it differently, and the
//! note says which byte arrived rather than only that one did.

use core::fmt::Write as _;

use embassy_rp::i2c::{Blocking, I2c};
use embassy_rp::peripherals::I2C0;
use embassy_time::Timer;
use pca9685_driver::{Channel, Pca9685, SERVO_FRAME_HZ};

/// Ask the PCA9685 who it is; say so on the report stream.
///
/// Takes the shared bus after the camera is done with it and gives it
/// back for whatever comes next — the same handoff discipline as
/// [`crate::camera::start`], because GP4/GP5 are one wire pair with
/// three owners' worth of traffic.
pub fn probe(mut bus: I2c<'static, I2C0, Blocking>) -> I2c<'static, I2C0, Blocking> {
    let mut driver = pca9685_driver::Pca9685::new(bus);
    let mut text: heapless::String<96> = heapless::String::new();
    match driver.read_register(pca9685_driver::REG_MODE1) {
        // Bit 4 is SLEEP: set on a chip that has power-cycled, clear on
        // one already started. Both are "present"; the byte says which.
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
    bus = driver.free();
    bus
}

/// The bring-up sweep: one channel, slow, and INSIDE the nominal span.
///
/// # ⚠️ Why the sweep never touches the span's ends
///
/// [`PulseSpan::SG90_NOMINAL`] is already conservative, and this stays
/// 100 µs inside even that. A servo commanded past its mechanical stop
/// buzzes at stall current until something gives, and nothing on this
/// rail can see the shaft — the real endpoints get MEASURED per servo,
/// widening from safe, never guessed wide and walked back.
const SWEEP_LOW_US: u32 = 1100;
const SWEEP_HIGH_US: u32 = 1900;
/// One microsecond-step per tick at 50 Hz ≈ 8 s per half-sweep — slow
/// enough to watch, gentle enough that the horn never snaps.
const SWEEP_STEP_US: u32 = 2;

/// Wake the chip and sweep channel 0 forever.
///
/// Owns the bus from here on: the camera finished with it in
/// [`crate::camera::start`], and nothing else asks until the AS5600s
/// arrive — at which point this task becomes the bus's steward rather
/// than its terminus.
#[embassy_executor::task]
pub async fn sweep(bus: I2c<'static, I2C0, Blocking>) {
    let mut driver = Pca9685::new(bus);
    // Sleep → prescale → wake → restart, in the only order the chip
    // accepts: the prescaler register is writable ONLY while asleep.
    if driver.start(SERVO_FRAME_HZ).is_err() {
        crate::diag::note("# servo start FAILED — bus error waking the PCA9685");
        return;
    }
    // The datasheet wants 500 µs of oscillator settle after wake; the
    // driver cannot wait (it has no clock), so its caller does.
    Timer::after_millis(1).await;

    // ⚠️ Read the prescaler BACK. It accepts writes only while the chip
    // sleeps — if the sleep sequence silently failed, the chip stays at
    // its power-on 200 Hz frame and every pulse computed for 50 Hz goes
    // out 4x too narrow, under the ~500 µs floor a servo even
    // acknowledges. The symptom is a servo that sits perfectly still
    // while every write ACKs: no error anywhere, nothing moving.
    let mut text: heapless::String<96> = heapless::String::new();
    match driver.read_register(pca9685_driver::REG_PRESCALE) {
        Ok(121) => {
            let _ = text.push_str("# servo prescale=121 — 50 Hz confirmed by read-back");
        }
        Ok(other) => {
            let _ = write!(
                text,
                "# servo ⚠️ PRESCALE={other} not 121 — frame rate wrong, pulses too narrow, \
                 servos will NOT move"
            );
        }
        Err(_) => {
            let _ = text.push_str("# servo ⚠️ prescale read-back failed");
        }
    }
    crate::diag::note(&text);

    // ⚠️ One servo in motion at any moment. The 3xAA servo pack is at
    // the SG90's voltage floor already; three starting together is a
    // surge the single pack cannot hold, and the sweep needs no
    // simultaneity — each channel gets the stage alone, then releases.
    // `release` is safe here precisely because nothing hangs on a horn:
    // on an assembled arm it would be the wrong call (a released joint
    // falls — crates/arm/src/safety.rs), and this loop must be replaced
    // before any linkage is attached.
    let mut active = 0u8;
    loop {
        let Ok(channel) = Channel::new(active) else {
            return;
        };
        let mut text: heapless::String<64> = heapless::String::new();
        let _ = write!(text, "# servo ch{active} sweeping {SWEEP_LOW_US}-{SWEEP_HIGH_US}us");
        crate::diag::note(&text);

        // Centre -> high -> low -> centre: one full excursion, ~16 s,
        // ending where it began so the release leaves the horn centred.
        let centre = (SWEEP_LOW_US + SWEEP_HIGH_US) / 2;
        let mut pulse = centre;
        let mut rising = true;
        let mut turnarounds = 0u8;
        while turnarounds < 2 || pulse != centre {
            if driver.set_pulse(channel, pulse).is_err() {
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
                turnarounds += 1;
            }
            Timer::after_millis(20).await;
        }
        // Park the horn loose before the next channel takes the stage.
        let _ = driver.release(channel);
        active = (active + 1) % SERVO_COUNT;
    }
}

/// How many channels the cycle visits — the three SG90s on hand.
///
/// (Held at 1 during bring-up so the debug had no quiet phases; motion
/// was proven on ch0 2026-08-15, and the cycle returned to 3.)
const SERVO_COUNT: u8 = 3;
