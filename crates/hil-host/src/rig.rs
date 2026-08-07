//! The host loop, with the plumbing taken out.
//!
//! # Why this is not in `main`
//!
//! It was, and it measured **21.9% line coverage** — the lowest in the
//! workspace by a wide margin — while containing the exchange every
//! hardware run depends on. Not because it is hardware-gated: the loop
//! itself is arithmetic and message handling. It was untested because it
//! sat in `main()` beside a serial port, an emulator subprocess and a
//! Rerun window, and there is no way to call `main()` from a test.
//!
//! Split out, the rig takes a [`Wire`] — and `Wire::Replay` is a
//! transcript in memory. So the whole loop runs in `cargo test`: no chip,
//! no emulator, no serial port, no viewer.
//!
//! `main` keeps what genuinely needs the outside world: arguments, opening
//! the port or spawning the emulator, drawing, and printing.

use crate::wire::Wire;
use hil_protocol::Message;
use sim_core::{Pose, WheelSpeeds};
use sim_run::{Mission, MissionConfig, Outcome, Tick};

/// The rig: a simulated body, a real brain somewhere on the far end of a
/// [`Wire`], and the lockstep exchange between them.
pub struct Rig {
    pub mission: Mission,
    wire: Wire,
    /// Duty ±`DUTY_FULL` maps to ±`max_wheel_speed`. Read off the
    /// mission's own spec so there is exactly one robot in this program.
    duty_scale: f64,
    /// The chip's own reported pose. For the viewer and the summary — the
    /// host never steers on it.
    pub belief_from_chip: Pose,
    /// Worst control-loop compute time the chip has reported, µs.
    pub worst_us: u32,
    pub ticks: usize,
}

impl Rig {
    pub fn new(config: MissionConfig, wire: Wire) -> Rig {
        let start = config.start;
        let mission = Mission::new(config);
        let duty_scale = mission.config.spec.max_wheel_speed / f64::from(sim_core::DUTY_FULL);
        Rig {
            mission,
            wire,
            duty_scale,
            belief_from_chip: start,
            worst_us: 0,
            ticks: 0,
        }
    }

    /// Tell the chip where this session begins, and wait for it to agree.
    ///
    /// Both halves matter. Without the message the chip carries its
    /// previous run's belief into this one — measured: 0/1 waypoints,
    /// 4.97 m drift, 2036 wall bumps. Without the wait, this line and the
    /// first goal land in its 32-byte UART FIFO together (46 bytes) and
    /// the goal is shredded.
    pub fn start(&mut self) -> std::io::Result<()> {
        let s = self.mission.config.start;
        self.wire.send(Message::Start {
            x: s.x,
            y: s.y,
            heading: s.heading,
        })?;
        while let Some(line) = self.wire.recv_line()? {
            if let Ok(Message::Pose { x, y, heading }) = Message::parse(line.trim_end()) {
                self.belief_from_chip = Pose::new(x, y, heading);
                break;
            }
        }
        Ok(())
    }

    /// One full exchange: decide here, control there, physics here.
    ///
    /// `None` ends the run — the mission finished, or the chip stopped
    /// answering.
    pub fn tick(&mut self) -> std::io::Result<Option<Tick>> {
        let Some(obs) = self.mission.observe() else {
            return Ok(None);
        };

        // 1. Send the planner's decision. Note what is NOT here: a second
        //    copy of the policy. This used to rebuild the messages from the
        //    observation, mirroring `Mission::decide` by hand, with the
        //    digital-twin claim resting on the two staying identical.
        let directive = self.mission.plan(&obs);
        self.wire.send(directive.into())?;

        // 2. Wait for its motor command. Anything else is the chip's
        //    belief (for the viewer), its health, or human-facing noise.
        let mut duty = None;
        while duty.is_none() {
            let Some(line) = self.wire.recv_line()? else {
                eprintln!("[host] chip closed the connection");
                break;
            };
            match Message::parse(line.trim_end()) {
                Ok(Message::Motor { duty_l, duty_r }) => duty = Some((duty_l, duty_r)),
                Ok(Message::Pose { x, y, heading }) => {
                    self.belief_from_chip = Pose::new(x, y, heading)
                }
                Ok(Message::Health { worst_us }) => self.worst_us = self.worst_us.max(worst_us),
                Ok(_) => {}
                Err(_) => eprintln!("[host] {}", line.trim_end()),
            }
        }
        let Some((duty_l, duty_r)) = duty else {
            return Ok(None);
        };

        // 3. Physics — the SAME advance() sim-run calls. Duty scaled to
        //    commanded wheel speeds is the only translation.
        let tick = self.mission.advance(
            obs,
            WheelSpeeds::new(
                f64::from(duty_l) * self.duty_scale,
                f64::from(duty_r) * self.duty_scale,
            ),
        );

        // 4. Hand back what the encoders saw.
        self.wire.send(Message::Sensors {
            dl: tick.dticks.0,
            dr: tick.dticks.1,
        })?;

        self.ticks += 1;
        Ok(Some(tick))
    }

    /// Everything worth judging the run by — computed, not printed.
    pub fn verdict(&self) -> Verdict {
        Verdict {
            outcome: self.mission.outcome(),
            belief_from_chip: self.belief_from_chip,
            ticks: self.ticks,
            tick_seconds: self.mission.config.tick_seconds,
            worst_us: self.worst_us,
            budget_us: (self.mission.config.tick_seconds * 1e6) as u32,
            divergences: self.wire.divergences().to_vec(),
            unconsumed: self.wire.unconsumed(),
        }
    }
}

/// How the run went. Separated from printing so a test can assert on it.
#[derive(Debug, Clone)]
pub struct Verdict {
    pub outcome: Outcome,
    pub belief_from_chip: Pose,
    pub ticks: usize,
    pub tick_seconds: f64,
    /// 0 when the chip never reported — an old firmware, or a replay of a
    /// log recorded before health existed.
    pub worst_us: u32,
    pub budget_us: u32,
    pub divergences: Vec<String>,
    pub unconsumed: usize,
}

impl Verdict {
    /// Did the chip miss its control period?
    ///
    /// Not a warning. A robot that misses its period is integrating stale
    /// sensor data and steering on it.
    pub fn missed_deadline(&self) -> bool {
        self.worst_us > self.budget_us
    }

    /// Did today's code command anything the recording did not?
    pub fn diverged(&self) -> bool {
        !self.divergences.is_empty() || self.unconsumed > 0
    }

    /// Should the process exit non-zero?
    pub fn failed(&self) -> bool {
        self.diverged() || self.missed_deadline()
    }

    /// The human-facing summary.
    pub fn report(&self, replaying: bool) -> String {
        use std::fmt::Write as _;
        let o = &self.outcome;
        let mut s = String::new();
        let _ = write!(
            s,
            "\n{} ticks ({:.1} s simulated).\n  waypoints:   {}/{}\n  \
             true pose:   x={:.3} y={:.3}\n  chip belief: x={:.3} y={:.3}\n  \
             drift: {:.3} m,  wall bumps: {}",
            self.ticks,
            self.ticks as f64 * self.tick_seconds,
            o.waypoints_reached,
            o.waypoints_total,
            o.final_pose.x,
            o.final_pose.y,
            self.belief_from_chip.x,
            self.belief_from_chip.y,
            o.drift,
            o.bumps
        );

        if self.diverged() {
            let _ = write!(s, "\n\nREPLAY DIVERGED from the recorded session:");
            for d in self.divergences.iter().take(10) {
                let _ = write!(s, "\n  - {d}");
            }
            if self.divergences.len() > 10 {
                let _ = write!(s, "\n  ... and {} more", self.divergences.len() - 10);
            }
            if self.unconsumed > 0 {
                let _ = write!(
                    s,
                    "\n  - {} recorded lines were never reached — this run \
                     ended earlier than the one captured",
                    self.unconsumed
                );
            }
            let _ = write!(
                s,
                "\n\nThe robot would behave differently from the recording. \
                 If that is intended, re-record; if not, this is the regression."
            );
        } else if replaying {
            let _ = write!(s, "\n  replay:      matched the recording exactly");
        }

        match self.worst_us {
            0 => {
                let _ = write!(s, "\n  timing:      chip reported none (old firmware?)");
            }
            w => {
                let _ = write!(
                    s,
                    "\n  timing:      worst {w} us of {} us ({:.2}%), {:.0}x headroom",
                    self.budget_us,
                    100.0 * f64::from(w) / f64::from(self.budget_us),
                    f64::from(self.budget_us) / f64::from(w)
                );
                if self.missed_deadline() {
                    let _ = write!(
                        s,
                        "\n\nFAIL: the chip missed its {} us control deadline \
                         (worst {w} us). The loop grew past what the chip can \
                         do at {:.0} Hz.",
                        self.budget_us,
                        1.0 / self.tick_seconds
                    );
                }
            }
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A chip that answers every goal with the same duty, forever.
    ///
    /// Built as a transcript because that is exactly what `Wire::Replay`
    /// consumes — which is the whole reason the rig takes a `Wire` rather
    /// than opening its own port.
    fn transcript(ticks: usize, duty: (i32, i32), health: Option<u32>) -> String {
        let mut s = String::new();
        // The session handshake: host says I, chip acknowledges with P.
        s.push_str("> I 1.0000 3.0000 0.0000\n< P 1.0000 3.0000 0.0000\n");
        for i in 0..ticks {
            s.push_str("> ?\n"); // the goal (content checked separately)
                                 // Health arrives at the START of a chip block, not after `M`.
                                 // The host stops reading the moment it has an `M`, so a health
                                 // line the chip emitted just after one is not seen until the
                                 // next tick's read — and the recording, which logs in the
                                 // order the HOST acts, shows it there. Copied from a real
                                 // capture rather than reasoned about, after guessing wrong.
            if let Some(h) = health {
                if i == 1 {
                    s.push_str(&format!("< H {h}\n"));
                }
            }
            s.push_str("< P 1.0000 3.0000 0.0000\n");
            s.push_str(&format!("< M {} {}\n", duty.0, duty.1));
            s.push_str("> ?\n"); // the sensor report
        }
        s
    }

    /// Drive the rig and return its verdict, ignoring what the host says.
    ///
    /// Divergences are expected here: the transcript cannot know the exact
    /// goal lines. The tests that care about divergence build their
    /// transcript from a real run instead.
    fn run(config: MissionConfig, log: &str) -> Verdict {
        let mut rig = Rig::new(config, Wire::from_log(log).expect("valid transcript"));
        rig.start().expect("handshake");
        while rig.tick().expect("tick").is_some() {}
        rig.verdict()
    }

    fn short() -> MissionConfig {
        MissionConfig {
            duration: 0.4, // 20 ticks
            control_on_belief: true,
            ..MissionConfig::default()
        }
    }

    #[test]
    fn the_handshake_adopts_the_pose_the_chip_reports() {
        let mut rig = Rig::new(
            short(),
            Wire::from_log("> I 1.0000 3.0000 0.0000\n< P 2.5000 4.5000 1.2000\n").unwrap(),
        );
        rig.start().unwrap();
        assert_eq!(rig.belief_from_chip.x, 2.5);
        assert_eq!(rig.belief_from_chip.y, 4.5);
        assert_eq!(rig.belief_from_chip.heading, 1.2);
    }

    #[test]
    fn a_run_advances_physics_and_counts_ticks() {
        let v = run(short(), &transcript(20, (500, 500), None));
        assert_eq!(v.ticks, 20, "every M should produce one tick");
        assert!(
            v.outcome.final_pose.x > 1.0,
            "equal duty should drive it forward from x=1.0, got {:.3}",
            v.outcome.final_pose.x
        );
    }

    /// Duty is scaled by the mission's own spec. Full duty must command
    /// the motor's full speed — not some other robot's.
    #[test]
    fn full_duty_drives_at_the_specs_top_speed() {
        let cfg = short();
        let spec = cfg.spec;
        let v = run(
            cfg,
            &transcript(20, (sim_core::DUTY_FULL, sim_core::DUTY_FULL), None),
        );
        // 20 ticks x 0.02 s at up to max_body_speed, minus motor lag.
        let travelled = v.outcome.final_pose.x - 1.0;
        let ceiling = spec.max_body_speed() * 0.4;
        assert!(
            travelled > 0.0 && travelled <= ceiling,
            "travelled {travelled:.3} m; the spec allows at most {ceiling:.3}"
        );
    }

    #[test]
    fn a_chip_that_goes_silent_ends_the_run_rather_than_hanging() {
        // Handshake, one exchange, then nothing.
        let log = "> I 1.0000 3.0000 0.0000\n< P 1.0000 3.0000 0.0000\n\
                   > ?\n< M 500 500\n> ?\n";
        let v = run(short(), log);
        assert_eq!(v.ticks, 1, "one tick completed, then the log ran dry");
    }

    // ---- the verdict, which is what CI and the operator act on ----

    #[test]
    fn a_healthy_run_passes() {
        let v = Verdict {
            worst_us: 300,
            budget_us: 20_000,
            divergences: vec![],
            unconsumed: 0,
            ..run(short(), &transcript(20, (500, 500), Some(300)))
        };
        assert!(!v.failed());
        assert!(v.report(true).contains("matched the recording exactly"));
        assert!(v.report(true).contains("67x headroom"));
    }

    #[test]
    fn missing_the_control_deadline_fails_the_run() {
        let mut v = run(short(), &transcript(20, (500, 500), Some(25_000)));
        v.divergences.clear();
        v.unconsumed = 0;
        assert_eq!(v.worst_us, 25_000, "the chip's report should be kept");
        assert!(v.missed_deadline());
        assert!(v.failed(), "a missed period must fail the run");
        assert!(v
            .report(false)
            .contains("missed its 20000 us control deadline"));
    }

    #[test]
    fn a_divergence_fails_the_run_and_names_both_sides() {
        let mut v = run(short(), &transcript(4, (500, 500), None));
        v.unconsumed = 0;
        v.divergences = vec!["would send \"G 1\", recording has \"G 2\"".into()];
        assert!(v.diverged() && v.failed());
        let r = v.report(true);
        assert!(r.contains("REPLAY DIVERGED"), "{r}");
        assert!(r.contains("G 1") && r.contains("G 2"), "{r}");
        assert!(!r.contains("matched the recording"), "cannot be both");
    }

    /// Ending earlier than the recording is a divergence even with no
    /// mismatched line: the run stopped somewhere the captured one did not.
    #[test]
    fn stopping_early_is_a_divergence() {
        let mut v = run(short(), &transcript(4, (500, 500), None));
        v.divergences.clear();
        v.unconsumed = 12;
        assert!(v.diverged() && v.failed());
        assert!(v
            .report(true)
            .contains("12 recorded lines were never reached"));
    }

    /// A silent "0 µs" must not read as a passing timing check.
    #[test]
    fn a_chip_that_reports_no_timing_says_so_rather_than_passing_quietly() {
        let mut v = run(short(), &transcript(4, (500, 500), None));
        v.divergences.clear();
        v.unconsumed = 0;
        assert_eq!(v.worst_us, 0);
        assert!(!v.missed_deadline(), "unknown is not a miss");
        assert!(v.report(false).contains("chip reported none"));
    }

    #[test]
    fn long_divergence_lists_are_truncated_with_a_count() {
        let mut v = run(short(), &transcript(4, (500, 500), None));
        v.unconsumed = 0;
        v.divergences = (0..25).map(|i| format!("line {i}")).collect();
        let r = v.report(true);
        assert!(r.contains("... and 15 more"), "{r}");
        assert!(!r.contains("line 20"), "should stop after ten");
    }
}
