//! The OV7670, and the two facts that make it different from every other
//! chip on this bus.
//!
//! # ⚠️ Fact one: it has no oscillator, and is dead without XCLK
//!
//! The MPU6050 and the PCA9685 keep their own time. **This part does
//! not.** Every internal state machine — including the one that answers
//! on the control bus — is clocked from an `XCLK` signal the *host* must
//! generate, typically 10–24 MHz.
//!
//! So with power, ground and both bus wires perfect, a camera with no
//! XCLK answers nothing at all. The symptom is indistinguishable from a
//! dead chip, a wrong address or a missing pull-up, which is why hours go
//! into it. **Start the clock before the first transaction.**
//!
//! # ⚠️ Fact two: SCCB is not quite I2C, and reads are the difference
//!
//! The control bus is Omnivision's SCCB. It looks like I2C and mostly is,
//! with one omission that matters: **it has no repeated-START.**
//!
//! ```text
//!   ordinary I2C read       [S] addr+W  reg  [Sr] addr+R  data  [P]
//!                                             ^^^^ repeated START
//!
//!   SCCB read               [S] addr+W  reg  [P]        <- one transaction
//!                           [S] addr+R  data [P]        <- and another
//! ```
//!
//! `embedded_hal::i2c::I2c::write_read` emits the first form. On an
//! OV7670 it returns whatever was on the bus rather than the register
//! asked for — often plausible, occasionally correct by luck, never
//! reliable. This driver therefore **never calls `write_read`**, and
//! there is a test whose only job is to fail if someone adds one.
//!
//! # What this crate does not do
//!
//! Pixels. The plain OV7670 has **no FIFO**, so frames arrive as a
//! 12-wire parallel stream that has to be caught in real time — PIO work,
//! and a separate concern from telling the chip what to be. This crate is
//! the control plane: identify, reset, configure.

#![cfg_attr(not(test), no_std)]

use embedded_hal::i2c::I2c;

/// The 7-bit SCCB address.
///
/// ⚠️ Datasheets and forum posts usually quote **0x42 / 0x43**, which are
/// the 8-bit write and read forms. `embedded-hal` wants the 7-bit
/// address, which is those shifted right by one. Passing 0x42 here talks
/// to nothing and is the second-most-common bring-up mistake after XCLK.
pub const DEFAULT_ADDRESS: u8 = 0x21;

/// Product ID, high byte. Reads **0x76** on every OV7670.
pub const REG_PRODUCT_ID: u8 = 0x0A;
/// Product ID, low byte. Reads **0x73** on an OV7670; an OV7675 reads
/// 0x73 here too, so the pair is a family check rather than an exact one.
pub const REG_VERSION: u8 = 0x0B;
/// Manufacturer ID, high byte. **0x7F** for Omnivision.
pub const REG_MANUFACTURER_HIGH: u8 = 0x1C;
/// Manufacturer ID, low byte. **0xA2** for Omnivision.
pub const REG_MANUFACTURER_LOW: u8 = 0x1D;
/// Common control 7: output format, and the software-reset bit.
pub const REG_COM7: u8 = 0x12;

/// What [`Identity::is_ov7670`] expects to see.
pub const EXPECTED_PRODUCT_ID: u8 = 0x76;
/// Omnivision's manufacturer ID, as the two bytes it arrives in.
pub const EXPECTED_MANUFACTURER: (u8, u8) = (0x7F, 0xA2);

/// COM7 bit 7 — a software reset. Sets every register back to default,
/// including this one, so it always reads back as 0.
const COM7_RESET: u8 = 0x80;

/// Who the chip says it is.
///
/// All four bytes are kept rather than reduced to a boolean, because
/// *which* one is wrong tells you what is broken: all-0x00 or all-0xFF is
/// wiring or a stopped XCLK, whereas a plausible-but-wrong product ID is
/// a different sensor on the same footprint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Identity {
    /// [`REG_PRODUCT_ID`] — 0x76 expected.
    pub product: u8,
    /// [`REG_VERSION`] — 0x73 on an OV7670.
    pub version: u8,
    /// [`REG_MANUFACTURER_HIGH`] — 0x7F expected.
    pub manufacturer_high: u8,
    /// [`REG_MANUFACTURER_LOW`] — 0xA2 expected.
    pub manufacturer_low: u8,
}

impl Identity {
    /// Whether this is the sensor we think it is.
    pub fn is_ov7670(&self) -> bool {
        self.product == EXPECTED_PRODUCT_ID
            && (self.manufacturer_high, self.manufacturer_low) == EXPECTED_MANUFACTURER
    }

    /// A plain-words reading of a failed identification, for the log line
    /// you want at the bench rather than a hex dump you have to decode.
    ///
    /// Returns `None` when the identity is correct.
    pub fn complaint(&self) -> Option<&'static str> {
        if self.is_ov7670() {
            return None;
        }
        let bytes = [
            self.product,
            self.version,
            self.manufacturer_high,
            self.manufacturer_low,
        ];
        if bytes.iter().all(|byte| *byte == 0x00) {
            // Every line low: the chip is not driving the bus at all.
            Some("all zeroes — no XCLK, no power, or SDA shorted to ground")
        } else if bytes.iter().all(|byte| *byte == 0xFF) {
            // Nothing pulls the bus down: nobody is answering.
            Some("all ones — nothing is acknowledging; check the address and the pull-ups")
        } else if (self.manufacturer_high, self.manufacturer_low) == EXPECTED_MANUFACTURER {
            // Omnivision answered, but it is not this part.
            Some("an Omnivision sensor, but not an OV7670 — check what was shipped")
        } else {
            Some("a device answered with something else entirely — wrong address?")
        }
    }
}

/// One canonical way to print an identity, shared by every firmware that
/// reports one.
///
/// Three call sites grew three hand-rolled formats within a day of each
/// other — and two had already drifted (one printed `ver=`, one did not).
/// The format that appears on a serial console at 2 a.m. is part of this
/// driver's interface, so it lives here, once.
impl core::fmt::Display for Identity {
    fn fmt(&self, out: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            out,
            "pid=0x{:02X} ver=0x{:02X} mid=0x{:02X}{:02X} {}",
            self.product,
            self.version,
            self.manufacturer_high,
            self.manufacturer_low,
            self.complaint().unwrap_or("OK — this is an OV7670"),
        )
    }
}

/// The control-plane driver. Generic over any `embedded-hal` I2C bus.
pub struct Ov7670<I2C> {
    i2c: I2C,
    address: u8,
}

impl<I2C: I2c> Ov7670<I2C> {
    /// Wrap a bus. Talks to nothing until asked.
    ///
    /// ⚠️ Whoever constructs this must already be generating **XCLK**, or
    /// every call below fails in a way that looks like bad wiring.
    pub fn new(i2c: I2C) -> Self {
        Ov7670 {
            i2c,
            address: DEFAULT_ADDRESS,
        }
    }

    /// Read one register.
    ///
    /// ⚠️ **Two transactions, deliberately.** SCCB has no repeated-START,
    /// so this is a write that ends in a STOP followed by a separate
    /// read. Collapsing it into `write_read` is the change that makes this
    /// driver return convincing rubbish.
    pub fn read_register(&mut self, register: u8) -> Result<u8, I2C::Error> {
        self.i2c.write(self.address, &[register])?;
        let mut received = [0u8; 1];
        self.i2c.read(self.address, &mut received)?;
        Ok(received[0])
    }

    /// Write one register.
    pub fn write_register(&mut self, register: u8, value: u8) -> Result<(), I2C::Error> {
        self.i2c.write(self.address, &[register, value])
    }

    /// Ask the chip who it is — the handshake every session should start
    /// with, and the only test that distinguishes "wired wrong" from
    /// "configured wrong".
    pub fn identify(&mut self) -> Result<Identity, I2C::Error> {
        Ok(Identity {
            product: self.read_register(REG_PRODUCT_ID)?,
            version: self.read_register(REG_VERSION)?,
            manufacturer_high: self.read_register(REG_MANUFACTURER_HIGH)?,
            manufacturer_low: self.read_register(REG_MANUFACTURER_LOW)?,
        })
    }

    /// Software-reset every register to its power-on default.
    ///
    /// ⚠️ The caller must then wait — the datasheet asks for **1 ms** —
    /// before the chip answers again. This driver has no clock and cannot
    /// wait for you, the same division of labour as `pca9685-driver`'s
    /// oscillator settle.
    pub fn reset(&mut self) -> Result<(), I2C::Error> {
        self.write_register(REG_COM7, COM7_RESET)
    }

    /// Apply a register table, in order.
    ///
    /// Camera configuration is conventionally a list of register/value
    /// pairs, and order matters — `COM7` in particular resets others. So
    /// the table stays a table rather than becoming a builder that could
    /// reorder it.
    pub fn apply(&mut self, table: &[(u8, u8)]) -> Result<(), I2C::Error> {
        for (register, value) in table {
            self.write_register(*register, *value)?;
        }
        Ok(())
    }

    /// Hand the bus back — this camera shares GP4/5 with the PCA9685.
    pub fn free(self) -> I2C {
        self.i2c
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use embedded_hal_mock::eh1::i2c::{Mock as I2cMock, Transaction as I2cTransaction};

    /// ⚠️ The test that exists to fail if someone "optimises" the read.
    ///
    /// The mock is told to expect a write and then a *separate* read. A
    /// `write_read` would be a different transaction entirely and this
    /// panics — which is the point, because on real silicon the mistake
    /// returns believable numbers instead of failing.
    #[test]
    fn a_read_is_two_transactions_because_sccb_has_no_repeated_start() {
        let expectations = [
            I2cTransaction::write(DEFAULT_ADDRESS, vec![REG_PRODUCT_ID]),
            I2cTransaction::read(DEFAULT_ADDRESS, vec![0x76]),
        ];
        let mut camera = Ov7670::new(I2cMock::new(&expectations));
        assert_eq!(camera.read_register(REG_PRODUCT_ID).unwrap(), 0x76);
        camera.free().done();
    }

    #[test]
    fn identify_reads_all_four_bytes_in_order() {
        let expectations = [
            I2cTransaction::write(DEFAULT_ADDRESS, vec![REG_PRODUCT_ID]),
            I2cTransaction::read(DEFAULT_ADDRESS, vec![0x76]),
            I2cTransaction::write(DEFAULT_ADDRESS, vec![REG_VERSION]),
            I2cTransaction::read(DEFAULT_ADDRESS, vec![0x73]),
            I2cTransaction::write(DEFAULT_ADDRESS, vec![REG_MANUFACTURER_HIGH]),
            I2cTransaction::read(DEFAULT_ADDRESS, vec![0x7F]),
            I2cTransaction::write(DEFAULT_ADDRESS, vec![REG_MANUFACTURER_LOW]),
            I2cTransaction::read(DEFAULT_ADDRESS, vec![0xA2]),
        ];
        let mut camera = Ov7670::new(I2cMock::new(&expectations));
        let identity = camera.identify().unwrap();
        assert!(identity.is_ov7670());
        assert_eq!(identity.complaint(), None);
        camera.free().done();
    }

    #[test]
    fn all_zeroes_names_the_clock_first() {
        // The most likely cause, named first, because it is the one that
        // is invisible on a multimeter.
        let dead = Identity {
            product: 0,
            version: 0,
            manufacturer_high: 0,
            manufacturer_low: 0,
        };
        assert!(!dead.is_ov7670());
        assert!(dead.complaint().unwrap().contains("XCLK"));
    }

    #[test]
    fn all_ones_names_the_pull_ups() {
        let absent = Identity {
            product: 0xFF,
            version: 0xFF,
            manufacturer_high: 0xFF,
            manufacturer_low: 0xFF,
        };
        assert!(absent.complaint().unwrap().contains("pull-ups"));
    }

    #[test]
    fn a_different_omnivision_part_is_called_out_as_such() {
        let cousin = Identity {
            product: 0x77,
            version: 0x73,
            manufacturer_high: 0x7F,
            manufacturer_low: 0xA2,
        };
        assert!(!cousin.is_ov7670());
        assert!(cousin.complaint().unwrap().contains("not an OV7670"));
    }

    #[test]
    fn version_is_not_part_of_the_identity_check() {
        // OV7670 and OV7675 share 0x73 here, so the check deliberately
        // ignores it. Pinned so nobody "tightens" it into a false alarm.
        let odd_version = Identity {
            product: EXPECTED_PRODUCT_ID,
            version: 0x00,
            manufacturer_high: 0x7F,
            manufacturer_low: 0xA2,
        };
        assert!(odd_version.is_ov7670());
    }

    #[test]
    fn reset_sets_the_com7_reset_bit() {
        let expectations = [I2cTransaction::write(
            DEFAULT_ADDRESS,
            vec![REG_COM7, COM7_RESET],
        )];
        let mut camera = Ov7670::new(I2cMock::new(&expectations));
        camera.reset().unwrap();
        camera.free().done();
    }

    #[test]
    fn apply_writes_the_table_in_order() {
        let expectations = [
            I2cTransaction::write(DEFAULT_ADDRESS, vec![0x11, 0x01]),
            I2cTransaction::write(DEFAULT_ADDRESS, vec![0x12, 0x14]),
        ];
        let mut camera = Ov7670::new(I2cMock::new(&expectations));
        camera.apply(&[(0x11, 0x01), (0x12, 0x14)]).unwrap();
        camera.free().done();
    }

    #[test]
    fn display_is_the_console_line() {
        let good = Identity {
            product: 0x76,
            version: 0x73,
            manufacturer_high: 0x7F,
            manufacturer_low: 0xA2,
        };
        assert_eq!(
            format!("{good}"),
            "pid=0x76 ver=0x73 mid=0x7FA2 OK — this is an OV7670"
        );
        let dead = Identity {
            product: 0,
            version: 0,
            manufacturer_high: 0,
            manufacturer_low: 0,
        };
        // The complaint rides along, so a failure prints its diagnosis.
        assert!(format!("{dead}").contains("XCLK"));
    }
}
