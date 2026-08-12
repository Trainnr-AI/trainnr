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

use arm::homing::Homing;
use arm::{ArmSpec, Guard, Joint, JointError, SensingJoint, Torque, Verdict};
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

/// Encoder sampling period. 100 µs = 10 kHz, exactly as `pico-odom` does
/// it — see `math-09` on aliasing.
///
/// ⚠️ This was 20 ms (one poll per control tick) in the first version, and
/// the first hardware run measured what that costs: **404 encoder ticks
/// where ~17,000 were expected.** `QuadratureDecoder::update` returns at
/// most ±1 per call, so polling at 50 Hz caps counting at 50 ticks/second
/// no matter how fast the motor turns. The encoder was fine; the loop
/// asking it was two orders of magnitude too slow.
const POLL_US: u64 = 100;

/// Control runs at 50 Hz, the same tick as everything else here — one act
/// per 200 encoder polls.
const POLLS_PER_TICK: u32 = 200;
const DT: f64 = 0.02;

/// How long the guard tolerates silence before it stops authorising.
/// 500 ms, matching the laptop-side watchdog on the base.
const SOURCE_TIMEOUT_MS: u64 = 500;

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
    /// The shared STBY line — `Some` on exactly one port. See the note
    /// at the construction site.
    standby: Option<Output<'static>>,
    encoder_a: Input<'static>,
    encoder_b: Input<'static>,
    decoder: QuadratureDecoder,
    /// ⚠️ `-1` on the mirrored assembly. One motor on this rig has its
    /// leads swapped relative to the other, so positive duty drives it
    /// backwards; flipping the ENCODER makes the reported motion agree
    /// with the command without reversing a shaft, which keeps every
    /// speed and deadband already measured valid. See
    /// `firmware_support::motor::LEFT_ENCODER_SIGN`.
    ///
    /// Getting this wrong is not a small error: the loop commands
    /// `measured + step`, so a joint that moves the wrong way is chased
    /// downward forever. Measured on the bench before this was applied:
    /// **-148 rad and still accelerating, duty pinned at 1000.**
    encoder_sign: i64,
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
        self.ticks += i64::from(delta) * self.encoder_sign;
    }
}

impl MotorPort for Port {
    fn set_duty(&mut self, duty: i32) -> Result<(), JointError> {
        self.channel.set_signed(&mut self.pwm_config, duty);
        self.last_duty = duty;
        Ok(())
    }

    fn set_standby(&mut self, powered: bool) -> Result<(), JointError> {
        if let Some(standby) = &mut self.standby {
            if powered {
                standby.set_high();
            } else {
                standby.set_low();
            }
        }
        Ok(())
    }

    fn ticks(&mut self) -> Result<i64, JointError> {
        Ok(self.ticks)
    }
}

/// Drive TWO joints through the real `Guard`, on real motors.
///
/// # Zero is where the shafts were at boot
///
/// Homing is skipped deliberately: the bench shafts have nothing to stall
/// against, so there is no hard stop to find. Adopting the power-up
/// position is a legitimate bench calibration — it is what a servo
/// effectively does when nobody sets `Homing_Offset` — and it is stated
/// here rather than implied, because a zero nobody chose is exactly the
/// thing `Seek::TimedOut` refuses to invent.
///
/// # The four acts, matching `cargo run -p arm --example watch`
///
/// ```text
///    0- 3 s   settle at zero                 the loop is stable at all
///    3- 8 s   step to +0.6 / -0.6 rad        the STEP LIMITER makes a ramp
///    8-13 s   stop feeding the guard         HoldStale -> both joints hold
///   13-20 s   feed again, return to zero     it recovers, not just stops
/// ```
///
/// Act 3 is the one worth watching. The commander stops, the guard says
/// hold, and the joints keep their angle under power — on a base the same
/// silence would cut the outputs and let it coast.
async fn drive_two_joints(
    spec: ArmSpec,
    mut a: N20Joint<Port>,
    mut b: N20Joint<Port>,
    out: &mut impl Report,
) -> ! {
    let mut line: heapless::String<160> = heapless::String::new();
    let mut guard = Guard::new(spec, DT, SOURCE_TIMEOUT_MS);

    // Zero at boot. Both shafts read 0 ticks here by construction.
    a.adopt_zero(0.0).ok();
    b.adopt_zero(0.0).ok();
    a.write_torque(true).ok();
    b.write_torque(true).ok();

    let mut tick: u32 = 0;
    loop {
        // Poll fast, act slow — 10 kHz on the encoders, 50 Hz control.
        for _ in 0..POLLS_PER_TICK {
            a.port_mut().poll_encoder();
            b.port_mut().poll_encoder();
            Timer::after_micros(POLL_US).await;
        }

        let seconds = tick / 50;
        let (wanted, feeding) = match seconds {
            0..=2 => ([0.0, 0.0], true),
            3..=7 => ([0.6, -0.6], true),
            // The commander dies. Nothing feeds the guard.
            8..=12 => ([0.6, -0.6], false),
            _ => ([0.0, 0.0], true),
        };

        let now = firmware_support::now_ms();
        if feeding {
            guard.fed(now);
        }

        let measured = [
            a.measured().unwrap_or(0.0),
            b.measured().unwrap_or(0.0),
        ];
        let verdict = guard.authorise(now, &measured, &wanted);
        if let Verdict::Move { radians } = &verdict {
            a.command(radians[0]).ok();
            b.command(radians[1]).ok();
        }
        // ⚠️ `tick` runs the position loop against whatever target the
        // joint last accepted. On a hold, no new target is written and
        // the joint keeps driving to its old one — which IS the hold.
        a.tick(DT).ok();
        b.tick(DT).ok();

        if tick % 10 == 0 {
            // `Held` and `NoZero` are different failures: one says the
            // commander stopped, the other says the joint has no zero.
            // Collapsing them would send you to the wrong place.
            let phase = match verdict {
                Verdict::Move { .. } => JointPhase::Holding,
                _ => JointPhase::Held,
            };
            for (index, joint) in [&mut a, &mut b].into_iter().enumerate() {
                let milliradians = (joint.measured().unwrap_or(0.0) * 1000.0) as i32;
                let raw = joint.port_mut().ticks;
                let duty = joint.port_mut().last_duty;
                line.clear();
                let _ = Message::Joint {
                    index: index as u8,
                    raw_ticks: raw,
                    milliradians,
                    duty,
                    phase,
                }
                .write_into(&mut line);
                out.send(line.as_bytes()).await;
            }
        }
        tick = (tick + 1) % (20 * 50);
    }
}

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    let p = embassy_rp::init(Default::default());

    let mut pwm_config = PwmConfig::default();
    pwm_config.top = PWM_TOP;

    // ⚠️ Both channels, and every pin is the wiring recorded in
    // `docs/learning/hw-01-bench-rig.md` and used by `pico-odom`, NOT a
    // fresh choice. An earlier draft of this file invented six pin
    // numbers that all looked plausible and were all wrong.
    //
    //   channel A   GP6 PWMA  GP7 AIN1  GP8 AIN2   encoder GP16/GP17
    //   channel B   GP10 PWMB GP11 BIN1 GP12 BIN2  encoder GP18/GP19
    //   GP9 STBY — gates BOTH bridges, so it lives on neither channel
    let mut standby = Output::new(p.PIN_9, Level::Low);
    standby.set_high();

    let port_a = Port {
        channel: Channel {
            pwm: Pwm::new_output_a(p.PWM_SLICE3, p.PIN_6, pwm_config.clone()),
            in1: Output::new(p.PIN_7, Level::Low),
            in2: Output::new(p.PIN_8, Level::Low),
        },
        pwm_config: pwm_config.clone(),
        // ⚠️ Only channel A owns the shared STBY pin. Handing a copy to
        // both would be two owners of one line, and the TB6612 offers no
        // per-channel disable however the struct is shaped.
        standby: Some(standby),
        encoder_a: Input::new(p.PIN_16, Pull::Up),
        encoder_b: Input::new(p.PIN_17, Pull::Up),
        decoder: QuadratureDecoder::new(false, false),
        encoder_sign: i64::from(firmware_support::motor::LEFT_ENCODER_SIGN),
        ticks: 0,
        last_duty: 0,
    };
    let port_b = Port {
        channel: Channel {
            // GP10 is slice 5 channel **A**: the slice is `n / 2` and the
            // channel is A for even `n`, so the pin and the slice cannot
            // be chosen independently. Naming `new_output_b` here — as a
            // draft of this file did — silently drives the wrong output.
            pwm: Pwm::new_output_a(p.PWM_SLICE5, p.PIN_10, pwm_config.clone()),
            in1: Output::new(p.PIN_11, Level::Low),
            in2: Output::new(p.PIN_12, Level::Low),
        },
        pwm_config,
        standby: None,
        encoder_a: Input::new(p.PIN_18, Pull::Up),
        encoder_b: Input::new(p.PIN_19, Pull::Up),
        decoder: QuadratureDecoder::new(false, false),
        encoder_sign: 1,
        ticks: 0,
        last_duty: 0,
    };

    spawner.spawn(firmware_support::heartbeat(Output::new(p.PIN_25, Level::Low), 500).unwrap());

    // Gains in DUTY PER RADIAN, sized to the STEP LIMIT rather than to
    // large errors — because with a limiter upstream, large errors never
    // reach the loop.
    //
    // `Guard` will not command further than `max_speed * dt` ahead of
    // where the joint is: 1.5 rad/s at 20 ms = 0.03 rad. So 0.03 rad is
    // the biggest error this loop can ever see, and the gain must make
    // *that* worth a useful duty.
    //
    //   900 x 0.03  =  27 duty  + 43 deadband =  70   ← below breakaway
    // 13000 x 0.03  = 390 duty  + 43 deadband = 433   ← moves
    //
    // ⚠️ The first hardware run sized this for a 0.05 rad error that the
    // step limiter makes unreachable, and the joints sat still pushing
    // 70 duty for five seconds. Static friction also exceeds the running
    // deadband, so "just above the measured deadband" is not enough to
    // start from rest.
    //
    // No integral: on a joint that cannot reach its target — blocked, or
    // held by a hand — an integrator winds up and then lunges.
    let gains = || Pid::new(13_000.0, 0.0, 200.0, 0.0);
    let spec = ArmSpec::bench_two_joint();
    let ticks_per_rev = ticks_per_revolution();
    // The measured deadband, read from the one place it is recorded
    // rather than repeated here. Below this duty the motor makes no
    // torque, so the loop must add it back or small corrections vanish.
    let deadband =
        (sim_core::MEASURED_DUTY_DEADBAND * f64::from(sim_core::DUTY_FULL)) as i32;
    let a = N20Joint::new(port_a, ticks_per_rev, gains(), deadband);
    let b = N20Joint::new(port_b, ticks_per_rev, gains(), deadband);

    #[cfg(feature = "usb")]
    {
        let driver = embassy_rp::usb::Driver::new(p.USB, link::Irqs);
        let (mut usb, class) = firmware_support::usb::cdc(driver, "pico-arm", 0x000c);
        let mut out = link::UsbReport(class);
        embassy_futures::join::join(usb.run(), drive_two_joints(spec, a, b, &mut out)).await;
    }
    #[cfg(not(feature = "usb"))]
    drive_two_joints(spec, a, b, &mut Silent).await;
}


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
