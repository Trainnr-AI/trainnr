//! The PCA9685, and the pulse widths that point a hobby servo.
//!
//! Sixteen PWM channels behind two I2C pins. It exists here because the
//! pin budget says it must: five servos wired straight to the Pico would
//! need 30 pins and the board has 26. Through this chip they cost **two**,
//! shared with the camera's SCCB.
//!
//! # What this crate is careful about
//!
//! A servo is commanded by the *width* of a pulse repeated every 20 ms,
//! and this chip thinks in twelfths-of-a-percent of a frame. Getting from
//! one to the other is three conversions, and every one of them has been a
//! bug in somebody's robot:
//!
//! ```text
//!   angle (radians)  ──▶  pulse width (µs)  ──▶  counts (0..4096)
//!                    ^                      ^
//!                    │                      └── needs the FRAME RATE
//!                    └── needs the servo's MEASURED endpoints
//! ```
//!
//! # ⚠️ The endpoints are measured, not looked up
//!
//! [`PulseSpan::SG90_NOMINAL`] is the datasheet's story. Real SG90s vary
//! by tens of percent, and a servo commanded past its mechanical stop
//! buzzes, draws stall current and cooks itself — silently, because
//! nothing here can see the shaft.
//!
//! So the nominal span is deliberately **conservative** and deliberately
//! **named `_NOMINAL`**. Replace it with a measured one per servo, the way
//! `MEASURED_DUTY_DEADBAND` replaced a guess on the drivetrain.
//!
//! # ⚠️ There is no feedback, and the types say so
//!
//! This driver can only ever implement `arm::Joint`. Not `SensingJoint`,
//! not `Torque`, not `Homing` — all three need `measured()`, and an SG90
//! has no way to report where it is. That is not an omission to fix later;
//! it is the reason `docs/20-video-to-vla-data.md` says the arm cannot
//! produce training data.

#![cfg_attr(not(test), no_std)]

use embedded_hal::i2c::I2c;

/// The 7-bit bus address with every address-select pad open, which is how
/// the common breakout ships. Soldering A0..A5 moves it up from here.
pub const DEFAULT_ADDRESS: u8 = 0x40;

/// Mode register 1: sleep, auto-increment, restart.
pub const REG_MODE1: u8 = 0x00;
/// First channel's first byte. Each channel occupies **four** consecutive
/// registers — on-low, on-high, off-low, off-high — so channel `n` starts
/// at `REG_LED0 + 4n`. See [`Channel::register`].
pub const REG_LED0: u8 = 0x06;
/// Frame-rate divider. ⚠️ Writable **only while the chip is asleep**,
/// which is the single most common bring-up bug with this part.
pub const REG_PRESCALE: u8 = 0xFE;

/// MODE1 bit 4. Set on power-up: the chip boots with its oscillator
/// stopped and outputs idle, exactly like the MPU6050 boots asleep.
const MODE1_SLEEP: u8 = 0x10;
/// MODE1 bit 5 — auto-increment the register pointer, so all four bytes
/// of a channel go out in one transaction and cannot be split by another
/// device's traffic mid-update.
const MODE1_AUTO_INCREMENT: u8 = 0x20;
/// MODE1 bit 7 — restart the PWM channels that were halted by sleep.
const MODE1_RESTART: u8 = 0x80;

/// PWM steps per frame. The chip's counter is 12-bit, so one frame is
/// split into 4096 slices whatever the frame rate.
pub const COUNTS_PER_FRAME: u32 = 4096;

/// The internal oscillator, in hertz, per the datasheet. Only correct
/// while `EXTCLK` is unset — which this driver never sets.
pub const OSCILLATOR_HZ: u32 = 25_000_000;

/// Servos expect a pulse every 20 ms. Slower and they twitch; much faster
/// and an analogue servo's electronics cannot keep up.
pub const SERVO_FRAME_HZ: u32 = 50;

/// Errors this driver can produce that are not the bus's fault.
///
/// The bus's own errors pass through as `I2C::Error`; these are the ones
/// where the *request* was wrong, and each names a specific refusal rather
/// than a generic failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServoError {
    /// A channel index above 15. The chip has sixteen.
    NoSuchChannel(u8),
    /// A frame rate the prescaler cannot express (24–1526 Hz).
    UnreachableFrameRate(u32),
    /// A pulse width outside the span this servo was declared to have.
    /// Carries the width asked for, in microseconds, so the log says what
    /// was wanted rather than only that something was refused.
    PulseOutOfSpan(u32),
}

/// One of the chip's sixteen outputs, validated once at construction.
///
/// A plain `u8` would let `set_angle(99, ..)` typecheck and fail at the
/// bus. Validating on the way in means the failure lands where the mistake
/// is, and every later use is infallible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Channel(u8);

impl Channel {
    /// The highest channel index the chip has.
    pub const LAST: u8 = 15;

    /// Refuses anything the chip does not have.
    pub const fn new(index: u8) -> Result<Self, ServoError> {
        if index > Self::LAST {
            return Err(ServoError::NoSuchChannel(index));
        }
        Ok(Channel(index))
    }

    /// Which index this is, for logging.
    pub const fn index(self) -> u8 {
        self.0
    }

    /// The first of this channel's four registers.
    pub const fn register(self) -> u8 {
        REG_LED0 + 4 * self.0
    }
}

/// The pulse widths at which one particular servo sits at each end of its
/// travel, in microseconds, together with how far it actually swings.
///
/// # ⚠️ Why the travel is stored and not assumed to be 180°
///
/// "180° servo" is a marketing claim. A servo that swings 160° between the
/// endpoints you measured, driven as though it swings 180°, is wrong by
/// 11% *everywhere except the middle* — and it is wrong smoothly, which is
/// exactly the kind of error that reads as a calibration problem somewhere
/// else entirely.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PulseSpan {
    /// Pulse width at one mechanical end, microseconds.
    pub shortest_microseconds: u32,
    /// Pulse width at the other, microseconds.
    pub longest_microseconds: u32,
    /// Total swing between those two, radians.
    pub travel_radians: f32,
}

impl PulseSpan {
    /// ⚠️ **A starting point, not a measurement.** The datasheet's
    /// 1000–2000 µs over 180°, which is conservative: most SG90s travel
    /// further than this and a few travel less.
    ///
    /// Being *inside* the true span means the servo never reaches its
    /// stops, which is the safe direction to be wrong in. Widening this
    /// without measuring the individual servo is how a gearbox gets
    /// stripped.
    pub const SG90_NOMINAL: PulseSpan = PulseSpan {
        shortest_microseconds: 1000,
        longest_microseconds: 2000,
        travel_radians: core::f32::consts::PI,
    };

    /// Microseconds per radian across this span.
    fn microseconds_per_radian(&self) -> f32 {
        let span = self.longest_microseconds as f32 - self.shortest_microseconds as f32;
        span / self.travel_radians
    }

    /// Where the servo's middle sits, in microseconds. Zero radians means
    /// centred, so that a joint's zero is the pose it is safe to park in.
    fn centre_microseconds(&self) -> f32 {
        (self.shortest_microseconds as f32 + self.longest_microseconds as f32) / 2.0
    }

    /// Angle in radians (0 = centred) to a pulse width in microseconds.
    ///
    /// ⚠️ Refuses anything outside the span rather than clamping. A clamp
    /// would silently accept a command it could not carry out, which is
    /// the same class of lie as reporting a position that was commanded
    /// rather than measured.
    pub fn pulse_for(&self, radians: f32) -> Result<u32, ServoError> {
        if !radians.is_finite() {
            return Err(ServoError::PulseOutOfSpan(0));
        }
        let microseconds = self.centre_microseconds() + radians * self.microseconds_per_radian();
        // Round before range-checking, so a value that rounds onto the
        // boundary is accepted rather than rejected by a fraction.
        let rounded = round_to_nearest(microseconds);
        if rounded < self.shortest_microseconds as f32 || rounded > self.longest_microseconds as f32
        {
            // Saturating cast: a wildly out-of-range angle still reports a
            // number rather than wrapping to something plausible.
            return Err(ServoError::PulseOutOfSpan(rounded.max(0.0) as u32));
        }
        Ok(rounded as u32)
    }
}

/// Round to nearest, without `std`.
///
/// ⚠️ `f32::round` and `f32::trunc` are **std-only** — they live in the
/// floating-point runtime, not in core. A driver crate that compiles on
/// the Mac and fails on the chip because of exactly this is a rite of
/// passage, so it is worth naming.
///
/// The cast to `i32` truncates toward zero, which is what `trunc` does,
/// and Rust defines it to *saturate* rather than wrap: a wild input lands
/// on `i32::MAX` and is then refused by the span check, instead of
/// wrapping to a small plausible number.
///
/// Kept as a named function because an open-coded `(x + 0.5) as u32` is
/// wrong for negatives, and that is not obvious at the call site.
fn round_to_nearest(value: f32) -> f32 {
    if value < 0.0 {
        -(((-value) + 0.5) as i32) as f32
    } else {
        ((value + 0.5) as i32) as f32
    }
}

/// The prescaler value that produces `frame_hz`.
///
/// Datasheet: `prescale = round(osc / (4096 × rate)) − 1`, valid for
/// 3..=255, which is 24 Hz to about 1526 Hz.
pub fn prescale_for(frame_hz: u32) -> Result<u8, ServoError> {
    if frame_hz == 0 {
        return Err(ServoError::UnreachableFrameRate(frame_hz));
    }
    let divisor = COUNTS_PER_FRAME * frame_hz;
    // Integer round-to-nearest: add half the divisor before dividing.
    let scaled = (OSCILLATOR_HZ + divisor / 2) / divisor;
    let prescale = scaled.saturating_sub(1);
    if !(3..=255).contains(&prescale) {
        return Err(ServoError::UnreachableFrameRate(frame_hz));
    }
    Ok(prescale as u8)
}

/// A pulse width in microseconds to the chip's 0..4096 counts, at a given
/// frame rate.
///
/// ⚠️ The frame rate is an argument rather than a constant because the
/// same microsecond count means a different fraction of a frame at every
/// rate. Holding the rate in one place and the conversion in another is
/// the shape of bug this repository keeps meeting.
pub fn counts_for(pulse_microseconds: u32, frame_hz: u32) -> u32 {
    let frame_microseconds = 1_000_000 / frame_hz.max(1);
    let counts =
        (pulse_microseconds as u64 * COUNTS_PER_FRAME as u64) / frame_microseconds.max(1) as u64;
    // The counter compares against 0..4095; 4096 would never match.
    (counts as u32).min(COUNTS_PER_FRAME - 1)
}

/// The four register writes that wake the chip, in the only order it
/// accepts — the prescaler is writable only while asleep. Pure data, so
/// the blocking and async drivers cannot drift on the one sequence that
/// has already produced a "everything ACKs, nothing moves" night.
fn start_sequence(frame_hz: u32) -> Result<[(u8, u8); 4], ServoError> {
    let prescale = prescale_for(frame_hz)?;
    Ok([
        (REG_MODE1, MODE1_SLEEP | MODE1_AUTO_INCREMENT),
        (REG_PRESCALE, prescale),
        (REG_MODE1, MODE1_AUTO_INCREMENT),
        (REG_MODE1, MODE1_AUTO_INCREMENT | MODE1_RESTART),
    ])
}

/// Build the one-transaction frame for a contiguous run of channels into
/// `frame`, returning how many bytes to send. Shared by both drivers —
/// the byte layout is chip knowledge, not transport knowledge.
fn pulses_frame(
    first: Channel,
    pulses: &[u32],
    frame_hz: u32,
    frame: &mut [u8; 1 + 4 * 16],
) -> Result<usize, ServoError> {
    let last = usize::from(first.index()) + pulses.len();
    if pulses.is_empty() || last > usize::from(Channel::LAST) + 1 {
        return Err(ServoError::NoSuchChannel(last as u8));
    }
    frame[0] = first.register();
    for (slot, pulse) in pulses.iter().enumerate() {
        let off = counts_for(*pulse, frame_hz);
        frame[1 + 4 * slot] = 0;
        frame[2 + 4 * slot] = 0;
        frame[3 + 4 * slot] = (off & 0xFF) as u8;
        frame[4 + 4 * slot] = (off >> 8) as u8;
    }
    Ok(1 + 4 * pulses.len())
}

/// The driver. Generic over any `embedded-hal` I2C bus, so the same code
/// runs against embassy on the chip and against a mock in these tests.
pub struct Pca9685<I2C> {
    i2c: I2C,
    address: u8,
    frame_hz: u32,
}

impl<I2C: I2c> Pca9685<I2C> {
    /// Wrap a bus. Does not talk to the chip — [`Self::start`] does.
    pub fn new(i2c: I2C) -> Self {
        Pca9685 {
            i2c,
            address: DEFAULT_ADDRESS,
            frame_hz: SERVO_FRAME_HZ,
        }
    }

    /// Use an address other than the default, for a second board.
    pub fn with_address(mut self, address: u8) -> Self {
        self.address = address;
        self
    }

    /// Wake the chip and set its frame rate.
    ///
    /// # ⚠️ The order here is the whole bring-up
    ///
    /// The prescaler is writable **only while asleep**, and the chip boots
    /// asleep — so the sequence is sleep-anyway, set the rate, wake, then
    /// restart. Writing the prescaler to an awake chip is silently ignored
    /// and leaves the outputs running at the power-on 200 Hz, which makes
    /// every servo sit at an angle nobody asked for.
    pub fn start(&mut self, frame_hz: u32) -> Result<(), Error<I2C::Error>> {
        // ⚠️ The datasheet requires 500 µs for the oscillator to settle
        // before RESTART. This driver does not sleep — it has no clock —
        // so the caller must delay. `pico-odom`'s servo module does.
        self.frame_hz = frame_hz;
        for (register, value) in start_sequence(frame_hz).map_err(Error::Servo)? {
            self.write(register, value)?;
        }
        Ok(())
    }

    /// Drive a run of CONSECUTIVE channels in one bus transaction.
    ///
    /// # ⚠️ Why this exists: the sampler-starvation budget
    ///
    /// Every blocking I2C transaction blinds the whole executor — and on
    /// the rig that includes the 10 kHz encoder sampler, whose decode
    /// errors were measured climbing whenever the servo task ran. Three
    /// channels as three transactions is ~1.5 ms of blindness per tick;
    /// as ONE auto-increment write it is a third of that. The channel
    /// registers are contiguous (four per channel from [`REG_LED0`]),
    /// which is what makes the single write possible.
    pub fn set_pulses(&mut self, first: Channel, pulses: &[u32]) -> Result<(), Error<I2C::Error>> {
        let mut frame = [0u8; 1 + 4 * 16];
        let length =
            pulses_frame(first, pulses, self.frame_hz, &mut frame).map_err(Error::Servo)?;
        self.i2c
            .write(self.address, &frame[..length])
            .map_err(Error::Bus)
    }

    /// Drive one channel to a pulse width, in microseconds.
    ///
    /// The pulse always starts at count 0, so `on` is 0 and `off` carries
    /// the width. Staggering the starts would spread the current draw of
    /// several servos, which matters on a battery — deliberately not done
    /// here, because it would make the off-count depend on the channel and
    /// this is the layer where that should stay simple.
    pub fn set_pulse(
        &mut self,
        channel: Channel,
        pulse_microseconds: u32,
    ) -> Result<(), Error<I2C::Error>> {
        let off = counts_for(pulse_microseconds, self.frame_hz);
        let [off_low, off_high] = [(off & 0xFF) as u8, (off >> 8) as u8];
        self.i2c
            .write(self.address, &[channel.register(), 0, 0, off_low, off_high])
            .map_err(Error::Bus)
    }

    /// Point a servo at an angle, given the span that servo was measured
    /// to have. Refuses angles outside the span rather than clamping.
    pub fn set_angle(
        &mut self,
        channel: Channel,
        radians: f32,
        span: &PulseSpan,
    ) -> Result<(), Error<I2C::Error>> {
        let pulse = span.pulse_for(radians).map_err(Error::Servo)?;
        self.set_pulse(channel, pulse)
    }

    /// Stop driving a channel entirely, by setting its full-off bit.
    ///
    /// ⚠️ This is **not** the arm's safe state. A released servo is
    /// back-driveable, so an unbalanced arm falls — which is the inverted
    /// failsafe `crates/arm/src/safety.rs` exists to enforce. Use it to
    /// park a servo that is already resting, not to stop one mid-air.
    pub fn release(&mut self, channel: Channel) -> Result<(), Error<I2C::Error>> {
        // Bit 4 of the off-high byte is "full off", which the datasheet
        // gives priority over any count.
        self.i2c
            .write(self.address, &[channel.register(), 0, 0, 0, 0x10])
            .map_err(Error::Bus)
    }

    /// Read one register back.
    ///
    /// ⚠️ The PCA9685 is real I2C — repeated-START works — so this is one
    /// `write_read`, unlike the OV7670's two-transaction SCCB dance. The
    /// house rule stands regardless of bus dialect: **a write that ACKs
    /// is not evidence, only a read-back is** — MODE1 after power-on
    /// reads `0x11` (SLEEP | ALLCALL), which is how a probe tells "the
    /// chip is there" from "something ACKed".
    pub fn read_register(&mut self, register: u8) -> Result<u8, Error<I2C::Error>> {
        let mut value = [0u8; 1];
        self.i2c
            .write_read(self.address, &[register], &mut value)
            .map_err(Error::Bus)?;
        Ok(value[0])
    }

    /// Hand the bus back — for tests, and for sharing it with the camera.
    pub fn free(self) -> I2C {
        self.i2c
    }

    fn write(&mut self, register: u8, value: u8) -> Result<(), Error<I2C::Error>> {
        self.i2c
            .write(self.address, &[register, value])
            .map_err(Error::Bus)
    }
}

/// The async driver — same chip knowledge, `await`-shaped transport.
///
/// # ⚠️ Why this exists at all
///
/// A blocking I2C write inside a cooperative executor blinds every other
/// task for its duration. On the rig that includes the 10 kHz encoder
/// sampler, and the blindness was *measured*: decode errors climbing
/// whenever the servo task ran (progress log, 2026-08-15). Awaiting the
/// bus instead hands those microseconds back to whoever needs them.
///
/// Every register value, sequence and frame layout comes from the same
/// pure functions the blocking driver uses — the two cannot disagree
/// about what the chip needs, only about how the bytes travel.
#[cfg(feature = "async")]
pub mod asynch {
    use embedded_hal_async::i2c::I2c;

    use crate::{pulses_frame, start_sequence, Channel, Error, SERVO_FRAME_HZ};

    /// The PCA9685 over an async bus. See [`crate::Pca9685`] for the
    /// chip's story; this type only changes how the bytes get there.
    pub struct Pca9685<I2C> {
        i2c: I2C,
        address: u8,
        frame_hz: u32,
    }

    impl<I2C: I2c> Pca9685<I2C> {
        /// Wrap a bus. Does not talk to the chip — [`Self::start`] does.
        pub fn new(i2c: I2C) -> Self {
            Pca9685 {
                i2c,
                address: crate::DEFAULT_ADDRESS,
                frame_hz: SERVO_FRAME_HZ,
            }
        }

        /// Wake the chip and set its frame rate — the caller still owes
        /// the 500 µs oscillator settle before trusting outputs.
        pub async fn start(&mut self, frame_hz: u32) -> Result<(), Error<I2C::Error>> {
            self.frame_hz = frame_hz;
            for (register, value) in start_sequence(frame_hz).map_err(Error::Servo)? {
                self.i2c
                    .write(self.address, &[register, value])
                    .await
                    .map_err(Error::Bus)?;
            }
            Ok(())
        }

        /// Drive a run of consecutive channels in one bus transaction.
        pub async fn set_pulses(
            &mut self,
            first: Channel,
            pulses: &[u32],
        ) -> Result<(), Error<I2C::Error>> {
            let mut frame = [0u8; 1 + 4 * 16];
            let length =
                pulses_frame(first, pulses, self.frame_hz, &mut frame).map_err(Error::Servo)?;
            self.i2c
                .write(self.address, &frame[..length])
                .await
                .map_err(Error::Bus)
        }

        /// Read one register back — real I2C, one `write_read`.
        pub async fn read_register(&mut self, register: u8) -> Result<u8, Error<I2C::Error>> {
            let mut value = [0u8; 1];
            self.i2c
                .write_read(self.address, &[register], &mut value)
                .await
                .map_err(Error::Bus)?;
            Ok(value[0])
        }

        /// Hand the bus back.
        pub fn free(self) -> I2C {
            self.i2c
        }
    }
}

/// Either the bus failed or the request was wrong. Kept apart because the
/// responses differ: a bus error is worth retrying, a bad request is not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error<BusError> {
    /// The I2C transaction itself failed — wiring, address, pull-ups.
    Bus(BusError),
    /// The command was impossible before it reached the bus.
    Servo(ServoError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use embedded_hal_mock::eh1::i2c::{Mock as I2cMock, Transaction as I2cTransaction};

    #[test]
    fn channel_refuses_pins_the_chip_does_not_have() {
        assert!(Channel::new(0).is_ok());
        assert!(Channel::new(15).is_ok());
        assert_eq!(Channel::new(16), Err(ServoError::NoSuchChannel(16)));
    }

    #[test]
    fn channel_registers_are_four_apart() {
        assert_eq!(Channel::new(0).unwrap().register(), 0x06);
        assert_eq!(Channel::new(1).unwrap().register(), 0x0A);
        // The last channel must still land inside the register map.
        assert_eq!(Channel::new(15).unwrap().register(), 0x42);
    }

    #[test]
    fn prescale_matches_the_datasheet_worked_example() {
        // 25 MHz / (4096 x 50 Hz) - 1 = 121.07 -> 121, the value every
        // Arduino library also arrives at.
        assert_eq!(prescale_for(50).unwrap(), 121);
        // 1000 Hz: 25e6 / 4.096e6 = 6.10 -> 6 - 1 = 5.
        assert_eq!(prescale_for(1000).unwrap(), 5);
    }

    #[test]
    fn prescale_refuses_rates_the_divider_cannot_express() {
        assert_eq!(prescale_for(0), Err(ServoError::UnreachableFrameRate(0)));
        // Too fast: prescale would fall below 3.
        assert!(prescale_for(2000).is_err());
        // Too slow: prescale would exceed 255.
        assert!(prescale_for(20).is_err());
    }

    #[test]
    fn counts_for_a_servo_pulse_at_fifty_hertz() {
        // A 20 ms frame in 4096 steps is 4.883 us per count, so 1500 us
        // (centre) is 307 counts.
        assert_eq!(counts_for(1500, 50), 307);
        assert_eq!(counts_for(1000, 50), 204);
        assert_eq!(counts_for(2000, 50), 409);
    }

    #[test]
    fn counts_never_reach_the_top_of_the_counter() {
        // The comparator matches 0..4095; a full-frame pulse must clamp
        // below 4096 or it would never fire at all.
        assert_eq!(counts_for(20_000, 50), COUNTS_PER_FRAME - 1);
        assert_eq!(counts_for(999_999, 50), COUNTS_PER_FRAME - 1);
    }

    #[test]
    fn centre_is_the_middle_of_the_span() {
        let span = PulseSpan::SG90_NOMINAL;
        assert_eq!(span.pulse_for(0.0).unwrap(), 1500);
    }

    #[test]
    fn ends_of_travel_reach_the_ends_of_the_span() {
        let span = PulseSpan::SG90_NOMINAL;
        let half_travel = span.travel_radians / 2.0;
        assert_eq!(span.pulse_for(half_travel).unwrap(), 2000);
        assert_eq!(span.pulse_for(-half_travel).unwrap(), 1000);
    }

    #[test]
    fn past_the_stop_is_refused_not_clamped() {
        let span = PulseSpan::SG90_NOMINAL;
        let past = span.travel_radians / 2.0 + 0.1;
        // ⚠️ The whole point: a clamp here would drive the servo into its
        // mechanical stop and stall it, with nothing able to notice.
        assert!(matches!(
            span.pulse_for(past),
            Err(ServoError::PulseOutOfSpan(_))
        ));
        assert!(span.pulse_for(-past).is_err());
    }

    #[test]
    fn non_finite_angles_are_refused() {
        let span = PulseSpan::SG90_NOMINAL;
        assert!(span.pulse_for(f32::NAN).is_err());
        assert!(span.pulse_for(f32::INFINITY).is_err());
        assert!(span.pulse_for(f32::NEG_INFINITY).is_err());
    }

    #[test]
    fn a_narrower_travel_changes_every_angle_but_the_centre() {
        // Same pulses, but the servo really swings 160 degrees. The centre
        // is unmoved and everything else shifts — the error that reads as
        // a calibration problem somewhere else.
        let measured = PulseSpan {
            travel_radians: 160.0_f32.to_radians(),
            ..PulseSpan::SG90_NOMINAL
        };
        assert_eq!(measured.pulse_for(0.0).unwrap(), 1500);
        let quarter = core::f32::consts::FRAC_PI_4;
        assert_ne!(
            measured.pulse_for(quarter).unwrap(),
            PulseSpan::SG90_NOMINAL.pulse_for(quarter).unwrap()
        );
    }

    #[test]
    fn start_sleeps_before_writing_the_prescaler() {
        // ⚠️ This ordering IS the bring-up bug. If the prescaler write
        // moves after the wake, the chip ignores it and every servo sits
        // wrong. Pinned as a transaction sequence so a reorder fails here.
        let expectations = [
            I2cTransaction::write(
                DEFAULT_ADDRESS,
                vec![REG_MODE1, MODE1_SLEEP | MODE1_AUTO_INCREMENT],
            ),
            I2cTransaction::write(DEFAULT_ADDRESS, vec![REG_PRESCALE, 121]),
            I2cTransaction::write(DEFAULT_ADDRESS, vec![REG_MODE1, MODE1_AUTO_INCREMENT]),
            I2cTransaction::write(
                DEFAULT_ADDRESS,
                vec![REG_MODE1, MODE1_AUTO_INCREMENT | MODE1_RESTART],
            ),
        ];
        let mut driver = Pca9685::new(I2cMock::new(&expectations));
        driver.start(SERVO_FRAME_HZ).unwrap();
        driver.free().done();
    }

    #[test]
    fn set_pulse_writes_four_bytes_starting_at_zero() {
        // 1500 us at 50 Hz is 307 counts = 0x0133.
        let expectations = [I2cTransaction::write(
            DEFAULT_ADDRESS,
            vec![0x06, 0, 0, 0x33, 0x01],
        )];
        let mut driver = Pca9685::new(I2cMock::new(&expectations));
        driver.set_pulse(Channel::new(0).unwrap(), 1500).unwrap();
        driver.free().done();
    }

    #[test]
    fn release_sets_the_full_off_bit() {
        let expectations = [I2cTransaction::write(
            DEFAULT_ADDRESS,
            vec![0x0A, 0, 0, 0, 0x10],
        )];
        let mut driver = Pca9685::new(I2cMock::new(&expectations));
        driver.release(Channel::new(1).unwrap()).unwrap();
        driver.free().done();
    }

    #[test]
    fn a_refused_angle_never_reaches_the_bus() {
        // No expectations at all: the mock panics on any transaction, so
        // this proves the range check happens BEFORE the write.
        let mut driver = Pca9685::new(I2cMock::new(&[]));
        let past = PulseSpan::SG90_NOMINAL.travel_radians;
        assert!(driver
            .set_angle(Channel::new(0).unwrap(), past, &PulseSpan::SG90_NOMINAL)
            .is_err());
        driver.free().done();
    }

    #[test]
    fn read_register_uses_one_write_read() {
        // Real I2C: repeated-START is legal, so one transaction — the
        // opposite decision from ov7670-driver, deliberately.
        let expectations = [I2cTransaction::write_read(
            DEFAULT_ADDRESS,
            vec![REG_MODE1],
            vec![0x11],
        )];
        let mut driver = Pca9685::new(I2cMock::new(&expectations));
        assert_eq!(driver.read_register(REG_MODE1).unwrap(), 0x11);
        driver.free().done();
    }

    #[test]
    fn set_pulses_is_one_transaction_for_a_contiguous_run() {
        // Three channels, ONE write: register 0x06 then 4 bytes each.
        // 1500 us @ 50 Hz = 307 = 0x0133; 1100 -> 225 = 0xE1; 1900 -> 389 = 0x0185.
        let expectations = [I2cTransaction::write(
            DEFAULT_ADDRESS,
            vec![
                0x06, 0, 0, 0x33, 0x01, // ch0: 1500
                0, 0, 0xE1, 0x00, // ch1: 1100
                0, 0, 0x85, 0x01, // ch2: 1900
            ],
        )];
        let mut driver = Pca9685::new(I2cMock::new(&expectations));
        driver
            .set_pulses(Channel::new(0).unwrap(), &[1500, 1100, 1900])
            .unwrap();
        driver.free().done();
    }

    #[test]
    fn set_pulses_refuses_a_run_past_the_last_channel() {
        // 15 + 2 channels would reach 16; the chip stops at 15. Refused
        // before the bus, like every impossible request in this driver.
        let mut driver = Pca9685::new(I2cMock::new(&[]));
        assert!(driver
            .set_pulses(Channel::new(15).unwrap(), &[1500, 1500])
            .is_err());
        driver.free().done();
    }

    /// A minimal executor for the async tests: the mock's futures are
    /// always immediately ready, so one poll with a no-op waker is the
    /// whole runtime. No dev-dependency earns its keep against ten lines.
    #[cfg(feature = "async")]
    fn block_on<F: core::future::Future>(future: F) -> F::Output {
        let mut future = core::pin::pin!(future);
        let waker = std::task::Waker::noop();
        let mut context = core::task::Context::from_waker(waker);
        loop {
            if let core::task::Poll::Ready(output) = future.as_mut().poll(&mut context) {
                return output;
            }
        }
    }

    #[cfg(feature = "async")]
    #[test]
    fn async_driver_sends_the_same_bytes_as_the_blocking_one() {
        // The property the shared core guarantees, pinned: identical
        // start sequence, identical batched frame. If the two drivers
        // ever drift on chip knowledge, one of these expectations breaks.
        let expectations = [
            I2cTransaction::write(DEFAULT_ADDRESS, vec![REG_MODE1, 0x30]),
            I2cTransaction::write(DEFAULT_ADDRESS, vec![REG_PRESCALE, 121]),
            I2cTransaction::write(DEFAULT_ADDRESS, vec![REG_MODE1, 0x20]),
            I2cTransaction::write(DEFAULT_ADDRESS, vec![REG_MODE1, 0xA0]),
            I2cTransaction::write(
                DEFAULT_ADDRESS,
                vec![0x06, 0, 0, 0x33, 0x01, 0, 0, 0xE1, 0x00, 0, 0, 0x85, 0x01],
            ),
        ];
        let mut driver = crate::asynch::Pca9685::new(I2cMock::new(&expectations));
        block_on(async {
            driver.start(SERVO_FRAME_HZ).await.unwrap();
            driver
                .set_pulses(Channel::new(0).unwrap(), &[1500, 1100, 1900])
                .await
                .unwrap();
        });
        driver.free().done();
    }
}
