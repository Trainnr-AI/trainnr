//! A minimal MPU6050 driver — our first sensor.
//!
//! The MPU6050 is the classic hobby IMU: 3-axis accelerometer + 3-axis
//! gyro behind an I2C interface. Talking to it means reading/writing
//! *registers* — numbered mailboxes inside the chip. Everything below
//! comes straight from its datasheet's register map (the skill of Stage 2:
//! datasheet → constants → driver).
//!
//! `#![cfg_attr(not(test), no_std)]`: no_std when compiled for the chip,
//! but full std when compiled for `cargo test` on the Mac — the standard
//! trick that makes driver crates host-testable.

#![cfg_attr(not(test), no_std)]

use embedded_hal::i2c::I2c;

/// The MPU6050's 7-bit bus address (0x69 if its AD0 pin is pulled high).
pub const DEFAULT_ADDRESS: u8 = 0x68;
/// Identity register: always reads back 0x68. The "are you even there?"
/// handshake every I2C session starts with.
pub const REG_WHO_AM_I: u8 = 0x75;
/// Power management: the chip BOOTS ASLEEP; writing 0 here wakes it.
/// (Classic first-hour-with-an-IMU bug: all readings zero because nobody
/// woke the chip.)
pub const REG_PWR_MGMT_1: u8 = 0x6B;
/// First of six accel data registers: XH, XL, YH, YL, ZH, ZL — each axis
/// is a 16-bit signed value split across two 8-bit registers, high byte
/// first ("big-endian").
pub const REG_ACCEL_XOUT_H: u8 = 0x3B;

/// At the default ±2g range, 1 g = 16384 counts (datasheet, table 6.2).
pub const COUNTS_PER_G: f32 = 16384.0;

/// The driver. Generic over ANY type implementing the embedded-hal I2c
/// trait — real embassy-rp hardware, a mock in tests, an emulator bus:
/// the driver can't tell and doesn't care. (Same swappable-world idea as
/// sim-core's RobotWorld.)
pub struct Mpu6050<I2C> {
    i2c: I2C,
    address: u8,
}

impl<I2C: I2c> Mpu6050<I2C> {
    pub fn new(i2c: I2C) -> Self {
        Mpu6050 {
            i2c,
            address: DEFAULT_ADDRESS,
        }
    }

    /// Ask the chip who it is. Anything but 0x68 means wiring trouble.
    pub fn who_am_i(&mut self) -> Result<u8, I2C::Error> {
        let mut buf = [0u8; 1];
        self.i2c
            .write_read(self.address, &[REG_WHO_AM_I], &mut buf)?;
        Ok(buf[0])
    }

    /// Wake the chip from its power-on sleep.
    pub fn wake(&mut self) -> Result<(), I2C::Error> {
        self.i2c.write(self.address, &[REG_PWR_MGMT_1, 0x00])
    }

    /// Raw accelerometer counts [x, y, z] — one burst read of 6 bytes.
    pub fn read_accel_raw(&mut self) -> Result<[i16; 3], I2C::Error> {
        let mut buf = [0u8; 6];
        self.i2c
            .write_read(self.address, &[REG_ACCEL_XOUT_H], &mut buf)?;
        Ok([
            combine_be(buf[0], buf[1]),
            combine_be(buf[2], buf[3]),
            combine_be(buf[4], buf[5]),
        ])
    }

    /// Acceleration in g's [x, y, z] (±2g range).
    pub fn read_accel_g(&mut self) -> Result<[f32; 3], I2C::Error> {
        let raw = self.read_accel_raw()?;
        Ok([raw_to_g(raw[0]), raw_to_g(raw[1]), raw_to_g(raw[2])])
    }

    /// Hand the bus back (needed by tests to verify expectations; on real
    /// hardware, useful when several chips share one bus).
    pub fn release(self) -> I2C {
        self.i2c
    }
}

/// EXERCISE H2.1 — reassemble a 16-bit signed value from two bytes.
///
/// The sensor sends each axis as high byte then low byte. Gluing them back
/// is two bit-operations — but there's a trap. The recipe:
///
/// 1. Widen both bytes to u16 (`as u16`) — UNSIGNED first.
/// 2. Shift the high byte left 8 bits (`<< 8`) and OR (`|`) the low byte in.
/// 3. Reinterpret the combined u16 as i16 (`as i16`) — this is where
///    two's complement kicks in: 0xFFFF becomes -1, for free.
///
/// The trap (and why step 1 says UNSIGNED): casting the LOW byte straight
/// to i16 *sign-extends* it — 0xFF would become 0xFFFF and OR garbage over
/// the high byte. Widen unsigned, combine, THEN reinterpret. The tests
/// below check exactly this case.
///
/// Math background: docs/learning/math-08-binary-and-twos-complement.md
pub fn combine_be(high: u8, low: u8) -> i16 {
    (((high as u16) << 8) | (low as u16)) as i16
}

/// EXERCISE H2.2 — raw counts → physical units.
///
/// At ±2g range the scale is 16384 counts per g (`COUNTS_PER_G`). One
/// line: widen raw to f32 (`as f32`) and divide. A resting, level chip
/// should read about [0, 0, +1.0] — gravity on the z axis.
pub fn raw_to_g(raw: i16) -> f32 {
    raw as f32 / COUNTS_PER_G
}

#[cfg(test)]
mod tests {
    use super::*;
    use embedded_hal_mock::eh1::i2c::{Mock as I2cMock, Transaction as I2cTransaction};

    // ---- exercise tests ----

    #[test]
    fn combine_positive() {
        assert_eq!(combine_be(0x00, 0x01), 1);
        assert_eq!(combine_be(0x01, 0x00), 256);
        assert_eq!(combine_be(0x40, 0x00), 16384); // exactly +1 g
    }

    #[test]
    fn combine_negative_twos_complement() {
        assert_eq!(combine_be(0xFF, 0xFF), -1);
        assert_eq!(combine_be(0x80, 0x00), -32768); // most negative
        assert_eq!(combine_be(0xC0, 0x00), -16384); // exactly -1 g
    }

    #[test]
    fn combine_low_byte_must_not_sign_extend() {
        // THE trap: low byte 0xFF with a small high byte. If the low byte
        // sign-extends, this comes out wildly wrong (-1) instead of 511.
        assert_eq!(combine_be(0x01, 0xFF), 511);
    }

    #[test]
    fn raw_to_g_scale() {
        assert_eq!(raw_to_g(16384), 1.0);
        assert_eq!(raw_to_g(-16384), -1.0);
        assert_eq!(raw_to_g(0), 0.0);
        assert_eq!(raw_to_g(8192), 0.5);
    }

    // ---- driver tests (mocked I2C bus — no hardware, no emulator) ----

    #[test]
    fn who_am_i_reads_identity_register() {
        let expectations = [I2cTransaction::write_read(
            DEFAULT_ADDRESS,
            vec![REG_WHO_AM_I],
            vec![0x68],
        )];
        let mut sensor = Mpu6050::new(I2cMock::new(&expectations));
        assert_eq!(sensor.who_am_i().unwrap(), 0x68);
        sensor.release().done(); // verify every expected transaction ran
    }

    #[test]
    fn wake_clears_power_register() {
        let expectations = [I2cTransaction::write(
            DEFAULT_ADDRESS,
            vec![REG_PWR_MGMT_1, 0x00],
        )];
        let mut sensor = Mpu6050::new(I2cMock::new(&expectations));
        sensor.wake().unwrap();
        sensor.release().done();
    }

    #[test]
    fn read_accel_parses_burst() {
        // Chip lying flat: x=0, y=0, z=+1g (0x4000 = 16384).
        let expectations = [I2cTransaction::write_read(
            DEFAULT_ADDRESS,
            vec![REG_ACCEL_XOUT_H],
            vec![0x00, 0x00, 0x00, 0x00, 0x40, 0x00],
        )];
        let mut sensor = Mpu6050::new(I2cMock::new(&expectations));
        assert_eq!(sensor.read_accel_raw().unwrap(), [0, 0, 16384]);
        sensor.release().done();
    }
}
