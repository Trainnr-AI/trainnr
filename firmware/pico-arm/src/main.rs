//! One arm joint, on a motor that is really there.
//!
//! ```sh
//! tools/build-pico2.sh pico-arm usb     # then copy the .uf2 after BOOTSEL
//! screen /dev/cu.usbmodem11 115200
//! ```
//!
//! # What this is for
//!
//! `crates/arm` is 45 tests and no hardware. `crates/n20-joint` closes a
//! position loop against a mock. This is where both meet a motor, and it
//! exists to answer questions that only a motor can answer:
//!
//! ```text
//!   does homing find a hard stop, or stall on friction on the way?
//!   does a 153:1 N20 HOLD a link with STBY low, or sag?
//!   what does authorise() cost on an M33, per tick, measured?
//!   pull the USB cable mid-motion — what does the chip actually do?
//! ```
//!
//! The last one is the reason the guard was made `no_std` at all. It
//! cannot be asked from the laptop, because the laptop is the thing being
//! taken away.
//!
//! # The failsafe here is NOT the base's
//!
//! `pico-odom` loses its host and drops STBY: the wheels coast, the robot
//! sits there, safe. Do that to a joint and it goes limp. So this firmware
//! **keeps holding** when the host goes quiet — `Guard` returns a hold
//! verdict, the position loop keeps running against the last authorised
//! target, and STBY stays up.
//!
//! ⚠️ That is the correct *immediate* answer and a bad *indefinite* one:
//! holding is what cooks a servo, and an N20 has no thermometer to notice.
//! `Thermal::trust_joints_without_a_thermometer` is what decides, and its
//! default is to trust. On this hardware that means **there is no thermal
//! protection at all** — which is stated here rather than discovered.
//!
//! # One joint, not two
//!
//! The TB6612 has two channels and the bench has two motors, so a 2-DOF
//! arm is buildable. This drives channel A only. A second joint adds
//! coordination — arriving together, one guard over both — and none of
//! that is worth debugging at the same time as the first motor that has
//! ever turned under `Joint::command`.

#![no_std]
#![no_main]
#![forbid(unsafe_code)]

use arm::homing::{HomingRun, Seek};
use arm::{Joint, JointError, Torque};
use embassy_executor::Spawner;
use embassy_rp::gpio::{Input, Level, Output, Pull};
use embassy_rp::pwm::{Config as PwmConfig, Pwm};
use embassy_time::Timer;
use firmware_support::motor::{Channel, PWM_TOP};
use n20_joint::{MotorPort, N20Joint};
use quad_encoder::QuadratureDecoder;
use sim_core::Pid;

use panic_halt as _;

/// 50 Hz, the same tick as everything else in this project.
const TICK_US: u64 = 20_000;
const DT: f64 = 0.02;

/// Measured on these motors: 4290 counts per output revolution, which is
/// `4290 / 28 = 153:1` — a ratio no supplier published. Read from
/// `RobotSpec::REAL_BOT` rather than repeated here, so retuning the
/// simulator cannot leave this behind.
fn ticks_per_revolution() -> f64 {
    f64::from(sim_core::RobotSpec::REAL_BOT.ticks_per_revolution)
}

/// The TB6612 channel and its encoder, behind the seam `n20-joint` wants.
///
/// Note what is NOT here: a position loop, a target, or any notion of an
/// angle. This is a duty, a standby line and a count — the driver owns
/// everything above that, and owns it identically on the laptop.
struct Port {
    channel: Channel,
    pwm_config: PwmConfig,
    standby: Output<'static>,
    encoder_a: Input<'static>,
    encoder_b: Input<'static>,
    decoder: QuadratureDecoder,
    ticks: i64,
}

impl Port {
    /// Read the encoder pins and fold the change into the running total.
    ///
    /// Polled rather than interrupt-driven, at 50 Hz, which is a **known
    /// limitation**: `pico-encoder` measured aliasing at speed on exactly
    /// this hardware. A joint creeping toward a hard stop moves slowly
    /// enough for it; a joint tracking a fast target does not, and this is
    /// where the first missed counts will appear.
    fn poll_encoder(&mut self) {
        let delta = self
            .decoder
            .update(self.encoder_a.is_high(), self.encoder_b.is_high());
        self.ticks += i64::from(delta);
    }
}

impl MotorPort for Port {
    fn set_duty(&mut self, duty: i32) -> Result<(), JointError> {
        self.channel.set_signed(&mut self.pwm_config, duty);
        Ok(())
    }

    fn set_standby(&mut self, powered: bool) -> Result<(), JointError> {
        if powered {
            self.standby.set_high();
        } else {
            self.standby.set_low();
        }
        Ok(())
    }

    fn ticks(&mut self) -> Result<i64, JointError> {
        Ok(self.ticks)
    }
}

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    let p = embassy_rp::init(Default::default());

    let mut pwm_config = PwmConfig::default();
    pwm_config.top = PWM_TOP;

    // ⚠️ Motor ① / channel A only, and every one of these is the wiring
    // recorded in `docs/learning/hw-01-bench-rig.md` and used by
    // `pico-odom`, NOT a fresh choice. The first draft of this file
    // invented six pin numbers that all looked plausible and were all
    // wrong; a motor that does not move is the *kind* outcome of that.
    //
    //   GP6  PWMA (slice 3, channel A)      GP16  encoder A
    //   GP7  AIN1                           GP17  encoder B
    //   GP8  AIN2
    //   GP9  STBY  (gates BOTH bridges)
    let mut port = Port {
        channel: Channel {
            pwm: Pwm::new_output_a(p.PWM_SLICE3, p.PIN_6, pwm_config.clone()),
            in1: Output::new(p.PIN_7, Level::Low),
            in2: Output::new(p.PIN_8, Level::Low),
        },
        pwm_config,
        standby: Output::new(p.PIN_9, Level::Low),
        encoder_a: Input::new(p.PIN_16, Pull::Up),
        encoder_b: Input::new(p.PIN_17, Pull::Up),
        decoder: QuadratureDecoder::new(false, false),
        ticks: 0,
    };
    port.decoder = QuadratureDecoder::new(port.encoder_a.is_high(), port.encoder_b.is_high());

    spawner.spawn(firmware_support::heartbeat(Output::new(p.PIN_25, Level::Low), 500).unwrap());

    // Gains are a starting guess and will be wrong. Tuning them against a
    // real motor is the point of this firmware, and the numbers that come
    // out belong in one place afterwards, not scattered per call site.
    let mut joint = N20Joint::new(port, ticks_per_revolution(), Pid::new(400.0, 0.0, 8.0, 200.0));
    joint.write_torque(true).ok();

    // ---- Act 1: find zero. Nothing above can run until this succeeds.
    //
    // A joint that never stalls times out and adopts NOTHING, which is why
    // the outcome is checked rather than assumed: an arm with a wrong zero
    // drives itself into its own hard stops at full commanded speed.
    let mut homing = HomingRun::new(-0.25, 1e-4, 25, 750);
    let homed = loop {
        joint.port_mut().poll_encoder();
        let state = match homing.step(&mut joint) {
            Ok(state) => state,
            Err(_) => break false,
        };
        joint.tick(DT).ok();
        if state != Seek::Creeping {
            break state == Seek::Homed;
        }
        Timer::after_micros(TICK_US).await;
    };

    if !homed {
        // No zero, so no angle means anything. Cut the outputs and stop —
        // on a joint this is the ONE case where going limp is right,
        // because nothing here knows where the joint is to hold it.
        joint.write_torque(false).ok();
        loop {
            Timer::after_millis(1_000).await;
        }
    }

    // ---- Act 2: hold, and keep holding.
    //
    // No host, no plan, no `Guard` yet — one joint told to stay at zero,
    // so that "does it hold?" is answered before anything more interesting
    // is layered on top. The next commit is the guard and a cable to pull.
    joint.command(0.0).ok();
    loop {
        joint.port_mut().poll_encoder();
        joint.tick(DT).ok();
        Timer::after_micros(TICK_US).await;
    }
}
