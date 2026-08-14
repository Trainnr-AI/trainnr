//! Ask the OV7670 who it is.
//!
//! ```sh
//! tools/build-pico2.sh pico-camera usb
//! # hold BOOTSEL, plug in, drop the .uf2 on RPI-RP2, then:
//! screen /dev/cu.usbmodem11 115200
//! ```
//!
//! # What this proves, and what it deliberately does not
//!
//! It answers one question: **is the camera wired correctly?** It reads
//! the four identity registers and says, in words, what is wrong when
//! they are not what they should be.
//!
//! It captures no pixels. The plain OV7670 has no FIFO, so a frame is a
//! 12-wire parallel stream caught in real time by PIO — a different piece
//! of work with a hardware prerequisite this board does not yet meet
//! (see the pin note below). Identification needs **five wires** and is
//! worth having on its own, because every later problem is easier to read
//! once "the chip answers" is not in doubt.
//!
//! # The wiring
//!
//! ```text
//!   OV7670          Pico 2 W        why
//!   ──────          ────────        ───
//!   3V3     ──────  3V3 (pin 36)    ~60 mA of ~300 mA available
//!   GND     ──────  GND             common with everything else
//!   SIOD    ──────  GP4             SCCB data  — shared with the PCA9685
//!   SIOC    ──────  GP5             SCCB clock — shared with the PCA9685
//!   XCLK    ──────  GP21            ⚠️ the clock the camera cannot make
//!   RESET   ──────  3V3             active-low; tie high or it never runs
//!   PWDN    ──────  GND             active-high power-down; tie low
//! ```
//!
//! ⚠️ **`RESET` and `PWDN` are not optional.** Left floating they drift,
//! and a camera held in reset is indistinguishable at the bus from one
//! that is not there. They are the third and fourth things to check after
//! XCLK and the address.
//!
//! ⚠️ **Never 5 V.** This is a 3.3 V part with no level shifting anywhere
//! in this design, and a 5 V module on these pins is out of spec for the
//! RP2350's inputs.
//!
//! # ⚠️ Why XCLK is GP21 and not something tidier
//!
//! Any GPIO can output PWM, but the RP2350's PWM has **eight slices, each
//! serving a fixed pair of pins**: GPIO *n* drives slice `(n/2) % 8`,
//! channel A for even pins and B for odd. Two pins that map to the same
//! slice *and* channel are the same output and cannot be driven
//! independently.
//!
//! The rig already uses **GP6** and **GP10** for motor PWM, which are
//! slice 3A and slice 5A. That rules out GP22 (also 3A) and GP26 (also
//! 5A) — the two pins an eye would otherwise pick. **GP21 is slice 2B**,
//! free, and clear of the pins a future data bus will want.

#![no_std]
#![no_main]
// No unsafe anywhere in this firmware. `forbid`, not `deny`: it cannot be
// switched off locally with an `#[allow]`. Embassy's HAL already wraps the
// peripheral access that would otherwise need it — if that ever stops being
// true, the argument belongs in a commit that changes this line.
#![forbid(unsafe_code)]

use embassy_executor::Spawner;
use embassy_rp::gpio::{Level, Output};
use embassy_rp::i2c::{Config as I2cConfig, I2c};
use embassy_rp::pwm::{Config as PwmConfig, Pwm};
use embassy_time::{Duration, Timer};
use ov7670_driver::Ov7670;
use panic_halt as _;

/// XCLK divider. The OV7670 wants 10–24 MHz; a PWM counter that wraps
/// every 12 system clocks gives ~12.5 MHz on an RP2350's 150 MHz core and
/// ~10.4 MHz on an RP2040's 125 MHz. **Both are inside the window**,
/// which is why one constant serves two chips — the frequency only has to
/// be in range, not exact, because it is a reference and not a data rate.
///
/// ⚠️ Deliberately DIFFERENT from `pico-odom`'s camera module, which runs
/// 25 MHz (top = 5) because frame rate — and through it, exposure — was
/// measured to matter there. This firmware only identifies the sensor,
/// any in-window clock does that, and the two values are separate
/// decisions. Do not "fix" one to match the other.
const XCLK_TOP: u16 = 11;

/// Square wave: high for half the period. The camera clocks on an edge,
/// so the duty barely matters, but a lopsided clock narrows the timing
/// margin for no gain.
const XCLK_COMPARE: u16 = 6;

/// How long the chip is given to wake after a reset. The datasheet asks
/// for 1 ms; this is generous because the cost of waiting is nothing and
/// the cost of reading a register too early is a wrong diagnosis.
const SETTLE_MILLISECONDS: u64 = 10;

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    let p = embassy_rp::init(Default::default());
    transport::run(p, spawner).await
}

/// One line of the report — the driver's own `Display`, so this firmware
/// and `pico-odom`'s camera module cannot drift into different formats.
/// (`write!` into a fixed buffer truncates rather than panics — the
/// discipline the whole firmware tree follows.)
fn describe(identity: &ov7670_driver::Identity) -> heapless::String<160> {
    use core::fmt::Write as _;
    let mut line = heapless::String::new();
    let _ = write!(line, "{identity}");
    line
}

#[cfg(feature = "usb")]
mod transport {
    use super::*;
    use embassy_futures::join::join;
    use embassy_rp::bind_interrupts;
    use embassy_rp::peripherals::USB;
    use embassy_rp::usb::{Driver, InterruptHandler};
    use embassy_usb::class::cdc_acm::CdcAcmClass;
    use embassy_usb::driver::Driver as UsbDriver;

    bind_interrupts!(struct Irqs {
        USBCTRL_IRQ => InterruptHandler<USB>;
    });

    /// USB full-speed endpoints carry 64 bytes; a longer line is split.
    const MAX_PACKET: usize = 64;

    async fn send<'d, D: UsbDriver<'d>>(class: &mut CdcAcmClass<'d, D>, bytes: &[u8]) {
        if !class.dtr() {
            return;
        }
        for chunk in bytes.chunks(MAX_PACKET) {
            if class.write_packet(chunk).await.is_err() {
                return;
            }
        }
    }

    pub async fn run(p: embassy_rp::Peripherals, _spawner: Spawner) -> ! {
        // ⚠️ FIRST OF ALL: hold the motor driver in standby.
        //
        // This firmware runs on a board that also carries a TB6612 and a
        // 6 V pack. It has no interest in the motors — and that is exactly
        // the problem, because a pin nobody claims is a pin left floating,
        // and the TB6612's `STBY` has no internal pull. Floating high with
        // undefined direction inputs drives the motors, which on a robot
        // with wheels on the ground means it leaves the desk.
        //
        // Driving GP9 low costs one line and asserts the safe state rather
        // than assuming it. The same argument `crates/arm/src/safety.rs`
        // makes about hold-versus-cut: the safe state is the one you
        // command, never the one you leave behind.
        let _standby = Output::new(p.PIN_9, Level::Low);

        // ⚠️ XCLK next, before anything touches the bus. The camera's
        // control interface is clocked from this pin, so a transaction
        // issued before the clock runs fails in a way that looks exactly
        // like bad wiring.
        let mut clock_config = PwmConfig::default();
        clock_config.top = XCLK_TOP;
        // ⚠️ Channel **B**, because GP21 is odd. The slice/channel mapping
        // is `slice = (n/2) % 8`, `channel = n % 2` — and embassy encodes
        // it in the type system, so `new_output_a` here does not compile
        // rather than silently driving the wrong pin. That is the check
        // this repo kept wishing for while inventing pin numbers.
        clock_config.compare_b = XCLK_COMPARE;
        let _xclk = Pwm::new_output_b(p.PWM_SLICE2, p.PIN_21, clock_config);

        let i2c = I2c::new_blocking(p.I2C0, p.PIN_5, p.PIN_4, I2cConfig::default());
        let mut camera = Ov7670::new(i2c);

        let driver = Driver::new(p.USB, Irqs);
        // 0x000d — distinct from pico-encoder's 0x000b and pico-robot's
        // 0x000a, so several boards can be plugged in at once and still
        // be told apart in `ioreg`.
        let (mut usb, mut class) = firmware_support::usb::cdc(driver, "pico-camera", 0x000d);

        // ⚠️ No heartbeat LED here, deliberately. On a Pico 2 W the user
        // LED is on the CYW43 radio and **GP25 is that radio's chip
        // select** — driving it as a GPIO pokes the radio. Other firmware
        // in this tree does exactly that and gets away with it because
        // the radio is idle, but this build has a USB line saying what is
        // happening, so there is nothing to buy and a pin to leave alone.

        let report = async {
            // Wait for a terminal before saying anything: unlike the
            // encoder, nothing is being measured here, so there is no
            // cost to waiting and every line is worth reading.
            class.wait_connection().await;
            Timer::after(Duration::from_millis(SETTLE_MILLISECONDS)).await;

            loop {
                let line = match camera.identify() {
                    Ok(identity) => describe(&identity),
                    // A bus error is a different failure from a wrong
                    // answer, and saying so saves checking the wrong end.
                    Err(_) => {
                        let mut line = heapless::String::new();
                        let _ = core::fmt::Write::write_str(
                            &mut line,
                            "BUS ERROR — nobody acknowledged 0x21. Check SIOD/SIOC, \
                             pull-ups, and that RESET is tied HIGH",
                        );
                        line
                    }
                };
                send(&mut class, line.as_bytes()).await;
                send(&mut class, b"\r\n").await;
                Timer::after(Duration::from_secs(1)).await;
            }
        };

        join(usb.run(), report).await;
        loop {
            Timer::after_secs(1).await;
        }
    }
}

/// Without `--features usb` there is no transport at all.
///
/// ⚠️ Deliberately not a UART fallback. The emulator has no OV7670, so a
/// UART build would report a bus error forever and look like a failure
/// when it is an absence. Refusing to build is honest; a build that runs
/// and always says the same wrong thing is not.
#[cfg(not(feature = "usb"))]
mod transport {
    use super::*;

    pub async fn run(_p: embassy_rp::Peripherals, _spawner: Spawner) -> ! {
        loop {
            Timer::after_secs(1).await;
        }
    }
}
