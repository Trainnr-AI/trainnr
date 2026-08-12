//! The TB6612FNG and the encoders on the other end of it.
//!
//! Moved here verbatim from `pico-odom`, which was the only firmware that
//! could drive a motor because it was the only one that had written this
//! out. `pico-arm` needs exactly the same three types, and a second copy
//! of a signed-duty H-bridge driver is the last thing this repo needs —
//! the direction pins and the deadband note are the sort of detail that
//! diverges silently once there are two of them.

use embassy_rp::gpio::{Input, Output};
use embassy_rp::pwm::{Config as PwmConfig, Pwm};
use sim_core::DUTY_FULL;

/// PWM counter top. 125 MHz / 5000 = 25 kHz — above hearing, below the
/// TB6612's 100 kHz ceiling.
pub const PWM_TOP: u16 = 5000;

/// The four encoder pins, named so left and right cannot be swapped by
/// argument order. They were four positional `Input`s in a row, which is
/// exactly the shape that lets `ra` and `lb` trade places silently.
pub struct Encoders {
    pub left_a: Input<'static>,
    pub left_b: Input<'static>,
    pub right_a: Input<'static>,
    pub right_b: Input<'static>,
}

/// One TB6612 channel: speed, and the two pins that pick direction.
///
/// `STBY` is deliberately NOT in here — it is shared by both channels, and
/// putting it in a per-channel struct would suggest otherwise.
pub struct Channel {
    pub pwm: Pwm<'static>,
    pub in1: Output<'static>,
    pub in2: Output<'static>,
}

impl Channel {
    /// Forward, stopped. Direction fixed — one variable at a time, and
    /// reverse is a sign rather than a separate experiment.
    pub fn arm(&mut self, cfg: &mut PwmConfig) {
        self.in1.set_high();
        self.in2.set_low();
        cfg.compare_a = 0;
        self.pwm.set_config(cfg);
    }

    /// Duty as a percentage, direction untouched. For calibration sweeps,
    /// where the direction is fixed and only the magnitude varies.
    pub fn set_duty(&mut self, cfg: &mut PwmConfig, percent: u16) {
        cfg.compare_a = PWM_TOP / 100 * percent;
        self.pwm.set_config(cfg);
    }

    /// Drive at a signed command in `±DUTY_FULL` units — the scale
    /// `RobotSpec::duty` produces and `hil-protocol` carries.
    ///
    /// Sign picks the direction pins; magnitude picks the duty. **Zero
    /// coasts** (both inputs low) rather than braking (both high): a
    /// stopped command should let the wheels turn freely, so a robot that
    /// has lost its host can be pushed off whatever it is against.
    ///
    /// ⚠️ **The measured ~4.3% deadband is not compensated here.** A
    /// command under about 43 units produces no motion at all, silently.
    /// That is deliberate and it is the same decision `sim_core::Motor`
    /// records: the deadband gets modelled once, together, in
    /// `RobotSpec::REAL_BOT` — not patched into one call site where the
    /// simulator would then disagree with the robot.
    pub fn set_signed(&mut self, cfg: &mut PwmConfig, duty: i32) {
        let magnitude = duty.unsigned_abs().min(DUTY_FULL.unsigned_abs());
        match duty.signum() {
            1 => {
                self.in1.set_high();
                self.in2.set_low();
            }
            -1 => {
                self.in1.set_low();
                self.in2.set_high();
            }
            _ => {
                self.in1.set_low();
                self.in2.set_low();
            }
        }
        // u32 throughout: `magnitude * PWM_TOP` reaches 5,000,000 and
        // would overflow the u16 the register finally takes.
        cfg.compare_a = (magnitude * u32::from(PWM_TOP) / DUTY_FULL.unsigned_abs()) as u16;
        self.pwm.set_config(cfg);
    }
}

/// The TB6612 as one object: both channels and the enable they share.
///
/// `STBY` is in here and NOT in [`Channel`] because it gates both bridges
/// at once — a per-channel copy would suggest each could be disabled
/// alone, which the chip does not offer.
pub struct Motors {
    pub left: Channel,
    pub right: Channel,
    pub standby: Output<'static>,
}
