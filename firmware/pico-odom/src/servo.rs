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
