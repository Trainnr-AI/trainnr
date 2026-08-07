//! H2b — the firmware reads a real sensor.
//!
//! The full sense path on emulated silicon:
//!   embassy-rp I2C peripheral  ->  YOUR mpu6050-driver  ->  UART report
//!
//! The same driver crate that `cargo test` exercises with a mocked bus on
//! the Mac is compiled here against actual hardware registers. That is the
//! entire point of writing drivers against embedded-hal traits.

#![no_std]
#![no_main]
// No unsafe anywhere in this firmware. `forbid`, not `deny`: it cannot be
// switched off locally with an `#[allow]`. Embassy's HAL already wraps the
// peripheral access that would otherwise need it — if that ever stops being
// true, the argument belongs in a commit that changes this line.
#![forbid(unsafe_code)]

use core::fmt::Write as _;
use embassy_executor::Spawner;
use embassy_rp::gpio::{Level, Output};
use embassy_rp::i2c::{Config as I2cConfig, I2c};
use embassy_rp::uart::{Config as UartConfig, Uart};
use embassy_time::Timer;
use mpu6050_driver::Mpu6050;
use panic_halt as _;

/// Heartbeat: proof the firmware is alive even if the sensor misbehaves.
#[embassy_executor::task]
async fn heartbeat(mut led: Output<'static>) {
    loop {
        led.toggle();
        Timer::after_millis(500).await;
    }
}

#[embassy_executor::main]
async fn main(_spawner: Spawner) {
    let p = embassy_rp::init(Default::default());

    // UART0 on GP0/GP1 — our console. The emulator prints these bytes.
    let mut uart = Uart::new_blocking(p.UART0, p.PIN_0, p.PIN_1, UartConfig::default());
    // I2C0 on GP5 (SCL) and GP4 (SDA) — the Pico's conventional I2C pins.
    let i2c = I2c::new_blocking(p.I2C0, p.PIN_5, p.PIN_4, I2cConfig::default());

    let led = Output::new(p.PIN_25, Level::Low);
    _spawner.spawn(heartbeat(led).unwrap());

    let mut sensor = Mpu6050::new(i2c);
    let mut line: heapless::String<160> = heapless::String::new();

    // 1. "Are you there?" — every I2C session starts with a handshake.
    match sensor.who_am_i() {
        Ok(id) => {
            let _ = write!(line, "WHO_AM_I = 0x{id:02x} (expect 0x68)\r\n");
        }
        Err(_) => {
            let _ = write!(line, "WHO_AM_I failed: no ACK — chip absent or miswired\r\n");
        }
    }
    let _ = uart.blocking_write(line.as_bytes());
    line.clear();

    // 2. The chip boots ASLEEP. Wake it, or every reading stays zero.
    match sensor.wake() {
        Ok(()) => {
            let _ = write!(line, "wake: PWR_MGMT_1 cleared\r\n");
        }
        Err(_) => {
            let _ = write!(line, "wake: FAILED\r\n");
        }
    }
    let _ = uart.blocking_write(line.as_bytes());
    line.clear();

    // 3. Read forever. `combine_be` and `raw_to_g` — Prakhar's functions —
    //    turn the six wire bytes into physical acceleration on every pass.
    loop {
        match sensor.read_accel_g() {
            Ok([x, y, z]) => {
                let _ = write!(
                    line,
                    "accel  x={:+.3}g  y={:+.3}g  z={:+.3}g\r\n",
                    x, y, z
                );
            }
            Err(_) => {
                let _ = write!(line, "accel read failed\r\n");
            }
        }
        let _ = uart.blocking_write(line.as_bytes());
        line.clear();
        Timer::after_millis(250).await;
    }
}
