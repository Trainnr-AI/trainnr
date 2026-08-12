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
use arm::{Joint, JointError, SensingJoint, Torque};
use firmware_support::Report;
use hil_protocol::{JointPhase, Message};
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
    /// The last duty written. Kept so telemetry can report what the loop
    /// actually did rather than a placeholder — a reported zero that is
    /// really "not filled in" is a sentinel, and this repo has been bitten
    /// by one of those already.
    last_duty: i32,
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
        self.last_duty = duty;
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

/// Home, then hold — narrating both.
///
/// Generic over [`Report`] for the same reason `pico-encoder`'s sampler
/// is: the behaviour is written once and the transport is chosen at the
/// edge. Without this the first hardware run would be a blinking LED and
/// a guess, which is how this project previously lost an evening to a
/// dark board.
async fn home_then_hold(mut joint: N20Joint<Port>, out: &mut impl Report) -> ! {
    let mut line: heapless::String<160> = heapless::String::new();

    joint.write_torque(true).ok();

    // ---- Act 1: find zero. Nothing above can run until this succeeds.
    let mut homing = HomingRun::new(CREEP_EFFORT, STILL_RADIANS, STALL_TICKS, LIMIT_TICKS);
    let mut ticks: u32 = 0;
    let homed = loop {
        joint.port_mut().poll_encoder();
        let raw = joint.port_mut().ticks;
        let duty = joint.port_mut().last_duty;
        let Ok(state) = homing.step(&mut joint) else {
            break false;
        };
        joint.tick(DT).ok();
        ticks += 1;

        // Every 10th tick: often enough to watch a creep, rare enough not
        // to flood a 115200 terminal at 50 Hz.
        // Every 10th tick: often enough to watch a creep, rare enough not
        // to flood the link at 50 Hz. The transitions are never skipped.
        if ticks % 10 == 0 || state != Seek::Creeping {
            report(
                out,
                &mut line,
                raw,
                0,
                duty,
                match state {
                    Seek::Creeping => JointPhase::Homing,
                    Seek::Homed => JointPhase::Homed,
                    Seek::TimedOut => JointPhase::NoZero,
                },
            )
            .await;
        }
        if state != Seek::Creeping {
            break state == Seek::Homed;
        }
        Timer::after_micros(TICK_US).await;
    };

    if !homed {
        // No zero, so no angle means anything. Cut the outputs and say so
        // — on a joint this is the ONE case where going limp is right,
        // because nothing here knows where the joint is to hold it.
        joint.write_torque(false).ok();
        loop {
            let raw = joint.port_mut().ticks;
            report(out, &mut line, raw, 0, 0, JointPhase::NoZero).await;
            Timer::after_millis(1_000).await;
        }
    }

    // ---- Act 2: hold, and keep holding.
    joint.command(0.0).ok();
    let mut ticks: u32 = 0;
    loop {
        joint.port_mut().poll_encoder();
        joint.tick(DT).ok();
        ticks += 1;

        if ticks % 10 == 0 {
            let milliradians = (joint.measured().unwrap_or(0.0) * 1000.0) as i32;
            let raw = joint.port_mut().ticks;
            let duty = joint.port_mut().last_duty;
            report(out, &mut line, raw, milliradians, duty, JointPhase::Holding).await;
        }
        Timer::after_micros(TICK_US).await;
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
        last_duty: 0,
    };
    port.decoder = QuadratureDecoder::new(port.encoder_a.is_high(), port.encoder_b.is_high());

    spawner.spawn(firmware_support::heartbeat(Output::new(p.PIN_25, Level::Low), 500).unwrap());

    // Gains are a starting guess and will be wrong. Tuning them against a
    // real motor is the point of this firmware, and the numbers that come
    // out belong in one place afterwards, not scattered per call site.
    let joint = N20Joint::new(port, ticks_per_revolution(), Pid::new(400.0, 0.0, 8.0, 200.0));

    #[cfg(feature = "usb")]
    {
        let driver = embassy_rp::usb::Driver::new(p.USB, link::Irqs);
        // 0x000c: distinct from pico-robot's 0x000a and pico-encoder's
        // 0x000b, so three boards can be plugged in and still be told
        // apart in `ioreg`.
        let (mut usb, class) = firmware_support::usb::cdc(driver, "pico-arm", 0x000c);
        let mut out = link::UsbReport(class);
        embassy_futures::join::join(usb.run(), home_then_hold(joint, &mut out)).await;
    }
    #[cfg(not(feature = "usb"))]
    home_then_hold(joint, &mut Silent).await;
}

/// Emit one `J` line — the shared wire format, not a `write!` of this
/// firmware's own devising.
///
/// `hil-protocol`'s own doc records what the alternative costs: a status
/// line that WAS a `write!` here grew two independent host parsers, one
/// of them drew an empty screen for hours while the gate stayed green,
/// and the fix was a third parser. One definition, both ends, round-trip
/// tested — and `--record`/`--replay` then work on arm sessions for free.
async fn report(
    out: &mut impl Report,
    line: &mut heapless::String<160>,
    raw_ticks: i64,
    milliradians: i32,
    duty: i32,
    phase: JointPhase,
) {
    line.clear();
    let _ = Message::Joint {
        index: 0,
        raw_ticks,
        milliradians,
        duty,
        phase,
    }
    .write_into(line);
    out.send(line.as_bytes()).await;
}

/// Homing parameters. Gentle, because a creep that is too fast stalls on
/// friction partway and calls THAT the hard stop.
const CREEP_EFFORT: f64 = 0.25;
const STILL_RADIANS: f64 = 1e-4;
const STALL_TICKS: u32 = 25;
const LIMIT_TICKS: u32 = 750;

/// A [`Report`] that discards everything, for builds with no host link.
struct Silent;
impl Report for Silent {
    async fn send(&mut self, _bytes: &[u8]) {}
}

/// USB CDC, for a board on a cable. Same shape as `pico-encoder`'s — the
/// timeout exists so a terminal that stops reading cannot stall the
/// control loop. Reports are best-effort; ticks are not.
#[cfg(feature = "usb")]
mod link {
    use embassy_rp::bind_interrupts;
    use embassy_rp::peripherals::USB;
    use embassy_rp::usb::InterruptHandler;
    use embassy_time::{with_timeout, Duration};
    use embassy_usb::class::cdc_acm::CdcAcmClass;
    use embassy_usb::driver::Driver as UsbDriver;
    use firmware_support::usb::MAX_PACKET;
    use firmware_support::Report;

    bind_interrupts!(pub struct Irqs {
        USBCTRL_IRQ => InterruptHandler<USB>;
    });

    /// Long enough that a busy host is tolerated, short enough that a
    /// terminal nobody is reading costs one tick rather than the run.
    const REPORT_TIMEOUT_MS: u64 = 5;

    pub struct UsbReport<'d, D: UsbDriver<'d>>(pub CdcAcmClass<'d, D>);

    impl<'d, D: UsbDriver<'d>> Report for UsbReport<'d, D> {
        async fn send(&mut self, bytes: &[u8]) {
            // `dtr()` is raised when something opens the port. Checking it
            // first means a board on a charger costs nothing at all.
            if !self.0.dtr() {
                return;
            }
            let write_all = async {
                for chunk in bytes.chunks(MAX_PACKET) {
                    if self.0.write_packet(chunk).await.is_err() {
                        return;
                    }
                }
            };
            let _ = with_timeout(Duration::from_millis(REPORT_TIMEOUT_MS), write_all).await;
        }
    }
}
