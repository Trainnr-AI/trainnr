//! The Stage 0 mission: sense → map → plan → decide → act → observe.
//!
//! Pure simulation. Knows nothing about Rerun, threads, or wall-clock
//! time. See the module docs in `lib.rs` for why.

use sim_core::{
    lookahead_point, plan, summarize_scan, AvoidHysteresis, BodyTwist, ControlGains, DepthCamera,
    DiffDrive, Directive, Encoders, GotoController, Mode, Motor, OccupancyGrid, Odometry, Point,
    Pose, Rng, Robot, RobotSpec, Segment, StuckMonitor, WheelSpeeds, World,
};

/// Everything the mission needs to be reproducible.
///
/// A struct rather than module constants so a test can vary one parameter
/// — a different seed, a shorter deadline — without editing the binary.
#[derive(Debug, Clone)]
pub struct MissionConfig {
    /// Control-loop period, seconds. 50 Hz.
    pub tick_seconds: f64,
    /// Give up after this much simulated time.
    pub duration: f64,
    pub seed: u64,
    /// Geometry the robot *believes*; shared with the firmware.
    pub spec: RobotSpec,
    pub gains: ControlGains,
    /// Ratio of true wheel radius to believed. Below 1.0 the robot's
    /// wheels are smaller than it thinks, so odometry over-reports
    /// distance. **This gap is the odometry-drift lesson** — see
    /// docs/learning/math-03.
    pub wheel_wear: f64,
    pub motor_tau: f64,
    /// Fraction of full command below which the motors do not turn.
    ///
    /// **0.0 for `sim-run`**, whose Stage 0 baseline is a fixed point of
    /// this repo. `hil-host` sets the measured value, because it is a twin
    /// of hardware rather than a teaching simulator.
    pub motor_deadband_fraction: f64,
    /// Steer on odometry belief instead of ground truth. `false` is the
    /// honest simulator default; `true` shows how drift compounds.
    pub control_on_belief: bool,
    pub cam_rays: usize,
    pub cam_fov: f64,
    pub cam_range: f64,
    pub map_res: f64,
    /// Re-run A* every N ticks — the map grows constantly.
    pub replan_ticks: usize,
    /// Cells of clearance kept from mapped walls (the robot has a body).
    pub inflation_cells: usize,
    pub lookahead: f64,
    pub avoid: AvoidHysteresis,
    pub avoid_v: f64,
    pub avoid_w: f64,
    pub robot_radius: f64,
    pub world_size: (f64, f64),
    pub start: Pose,
    pub waypoints: Vec<Point>,
    /// Extra walls beyond the room's perimeter.
    pub obstacles: Vec<Segment>,
}

impl Default for MissionConfig {
    /// The M5 scenario: **the U-trap**.
    ///
    /// Three walls forming a U with its mouth facing the robot, the goal
    /// directly behind the back wall, and flanking spurs at the mouth so
    /// that wall-following around an arm tip shoves the robot back toward
    /// the entrance. A purely reactive avoider cannot solve this; a robot
    /// that maps and plans routes around the outside.
    fn default() -> Self {
        MissionConfig {
            tick_seconds: 0.02,
            duration: 60.0,
            seed: 7,
            spec: RobotSpec::SIM_BOT,
            gains: ControlGains::WAYPOINT,
            wheel_wear: 0.99,
            motor_tau: 0.15,
            motor_deadband_fraction: 0.0,
            control_on_belief: false,
            cam_rays: 21,
            cam_fov: 1.22,
            cam_range: 2.5,
            map_res: 0.1,
            replan_ticks: 25,
            inflation_cells: 2,
            lookahead: 0.35,
            avoid: AvoidHysteresis {
                enter: 0.35,
                exit: 0.6,
            },
            avoid_v: 0.12,
            avoid_w: 1.8,
            robot_radius: 0.09,
            world_size: (8.0, 6.0),
            start: Pose::new(1.0, 3.0, 0.0), // facing into the U's mouth
            waypoints: vec![Point::new(6.5, 3.0)], // dead centre behind the back wall
            obstacles: vec![
                Segment {
                    start: Point::new(5.0, 2.2),
                    end: Point::new(5.0, 3.8), // back wall
                },
                Segment {
                    start: Point::new(2.6, 3.8),
                    end: Point::new(5.0, 3.8), // top arm (deep!)
                },
                Segment {
                    start: Point::new(2.6, 2.2),
                    end: Point::new(5.0, 2.2), // bottom arm (deep!)
                },
                Segment {
                    start: Point::new(2.6, 3.8),
                    end: Point::new(2.6, 5.2), // spur up from the top arm tip
                },
                Segment {
                    start: Point::new(2.6, 0.8),
                    end: Point::new(2.6, 2.2), // spur down from the bottom arm tip
                },
            ],
        }
    }
}

/// One completed control tick — everything a viewer needs to draw it.
///
/// **This is the [`Observation`] plus what the physics did with it**, and
/// it says so structurally rather than by copying. It used to restate
/// eight of `Observation`'s fields — `index`, `t`, `scan`, `mode`,
/// `mode_changes`, `target`, `goal`, `path` — plus `min_dist`, which
/// duplicated `summary.nearest`. `advance` then moved each one across by
/// hand. Nine chances for the two to disagree, and nine lines of moving
/// that carried no information.
#[derive(Debug, Clone)]
pub struct Tick {
    /// What the robot saw and steered on, verbatim.
    pub obs: Observation,
    /// Did the body overlap a wall, so the translation was refused?
    pub bumped: bool,
    pub true_pose: Pose,
    pub belief_pose: Pose,
    /// Distance between truth and belief — the accumulated odometry error.
    pub drift: f64,
    /// Encoder counts this tick. The only thing the chip gets to see of
    /// the physics — everything else here is for the viewer.
    pub dticks: (i64, i64),
    /// What the wheels were **asked** for, before motor lag and slip.
    ///
    /// Kept because it is the actuator command — the signal that matters
    /// most once a real H-bridge is on the end of it. It was plotted until
    /// the digital-twins refactor moved drawing into the shared `viz`,
    /// where duty had no home; the panels went blank and nobody noticed
    /// until Prakhar saw two empty plots in the viewer.
    pub commanded: WheelSpeeds,
}

/// Everything a decider needs for one tick, and nothing about how the
/// decision gets made.
///
/// This is the seam that makes the simulator and the HIL rig **digital
/// twins**: both run the same world and the same physics, and differ only
/// in who fills the gap between [`Mission::observe`] and
/// [`Mission::advance`]. In `sim-run` that gap is a function call; in
/// `hil-host` it is a serial cable with a real chip on the far end.
#[derive(Debug, Clone)]
pub struct Observation {
    pub index: usize,
    /// Simulated seconds since the mission started.
    pub elapsed_seconds: f64,
    /// The pose the controller should steer on — truth, or belief if
    /// `control_on_belief` is set.
    pub pose: Pose,
    /// The waypoint currently being driven to.
    pub goal: Point,
    /// Where to actually aim: the path lookahead point, or the goal if
    /// there is no plan. **This is what gets sent to the chip.**
    pub target: Point,
    pub dist_to_goal: f64,
    pub mode: Mode,
    pub mode_changes: Vec<(Mode, Mode)>,
    pub scan: Vec<f64>,
    pub summary: sim_core::ScanSummary,
    pub path: Option<Vec<Point>>,
}

/// How the mission ended.
#[derive(Debug, Clone, PartialEq)]
pub struct Outcome {
    /// Simulated time at which the last waypoint was reached, if it was.
    pub completed_at: Option<f64>,
    pub waypoints_reached: usize,
    pub waypoints_total: usize,
    /// Final truth-vs-belief distance, metres.
    pub drift: f64,
    /// Ticks on which the robot was pressed against a wall.
    pub bumps: usize,
    pub ticks: usize,
    pub final_pose: Pose,
}

impl Outcome {
    pub fn succeeded(&self) -> bool {
        self.waypoints_reached == self.waypoints_total
    }
}

/// The running simulation.
pub struct Mission {
    pub config: MissionConfig,
    pub world: World,
    pub camera: DepthCamera,
    pub map: OccupancyGrid,
    pub robot: Robot,
    pub odometry: Odometry,

    nominal: DiffDrive,
    /// Notices the robot is commanded to move and is not, and backs it
    /// out. See `sim_core::StuckMonitor` for what it cost to not have.
    stuck: StuckMonitor,
    /// True forward speed last tick, in m/s — what the monitor compares
    /// the command against.
    last_true_speed: f64,
    encoders: Encoders,
    rng: Rng,
    motor_l: Motor,
    motor_r: Motor,
    controller: GotoController,

    path: Option<Vec<Point>>,
    mode: Mode,
    wp_index: usize,
    index: usize,
    max_steps: usize,
    bumps: usize,
    completed_at: Option<f64>,
    /// The controller must drop accumulated state before the next Steer.
    ///
    /// Set where the mission used to call `controller.reset()` directly.
    /// The difference is that this travels: it becomes `Directive::fresh`
    /// and reaches the chip, which previously had no way to learn that a
    /// waypoint boundary had been crossed.
    pending_reset: bool,
}

impl Mission {
    /// # Panics
    ///
    /// If the spec and gains are physically incompatible — see
    /// [`RobotSpec::check`](sim_core::RobotSpec::check).
    ///
    /// Deliberately a panic rather than a `Result`. This is a
    /// configuration error in a `const` written by hand, not a runtime
    /// condition: every caller passes a compile-time constant, so it
    /// either always fires or never does, and a test catches it the first
    /// time the suite runs. `check` spent its whole life with no callers
    /// outside its own unit tests, which is how it went unnoticed that it
    /// never looked at the turn axis.
    pub fn new(config: MissionConfig) -> Mission {
        if let Err(why) = config.spec.check(&config.gains) {
            panic!("MissionConfig is not physically achievable: {why}");
        }
        // `is_valid` documented itself as "callers can assert on this
        // rather than discovering it in the viewer" and had no callers,
        // exactly like `check` before it. `exit <= enter` collapses the
        // dead zone and the robot judders between Goto and Avoid.
        assert!(
            config.avoid.is_valid(),
            "avoid hysteresis is not hysteretic: exit {} must exceed enter {}",
            config.avoid.exit,
            config.avoid.enter
        );
        let mut world = World::room(config.world_size.0, config.world_size.1);
        world.walls.extend(config.obstacles.iter().copied());

        // What the robot believes it is...
        let nominal = config.spec.drive();
        // ...and what it actually is. Do NOT unify these.
        let true_model = DiffDrive {
            wheel_radius: config.spec.wheel_radius * config.wheel_wear,
            track_width: config.spec.track_width,
        };

        Mission {
            camera: DepthCamera {
                n_rays: config.cam_rays,
                fov: config.cam_fov,
                max_range: config.cam_range,
            },
            map: OccupancyGrid::new(config.world_size.0, config.world_size.1, config.map_res),
            robot: Robot {
                model: true_model,
                pose: config.start,
            },
            odometry: Odometry {
                model: nominal,
                ticks_per_revolution: config.spec.ticks_per_revolution,
                pose: config.start,
            },
            encoders: Encoders::new(config.spec.ticks_per_revolution),
            rng: Rng::new(config.seed),
            // Saturation comes from the spec, not a separate field.
            // `motor_max` used to duplicate it — and was hardcoded to
            // `SIM_BOT` rather than derived from `config.spec`, so a
            // mission on a different robot clamped at one limit while
            // `fit_wheels` scaled against another. The scaler would then
            // hand over a "safe" pair the motor still clipped, distorting
            // the very arc the scaling exists to preserve.
            // Half a second of "asked to move, didn't" before backing
            // out, and one second of backing out. `still_speed` is 1 cm/s
            // — below any real commanded motion, above the numerical
            // noise of a robot pressed against a wall.
            // One definition of the policy, in sim-core. This used to be
            // built here AND in `chase`, each converting seconds to its
            // own tick rate by hand.
            stuck: StuckMonitor::for_robot(&config.spec, 1.0 / config.tick_seconds),
            last_true_speed: 0.0,
            motor_l: Motor::with_deadband(
                config.motor_tau,
                config.spec.max_wheel_speed,
                config.motor_deadband_fraction,
            ),
            motor_r: Motor::with_deadband(
                config.motor_tau,
                config.spec.max_wheel_speed,
                config.motor_deadband_fraction,
            ),
            controller: GotoController::new(config.gains),
            nominal,
            path: None,
            mode: Mode::Goto,
            wp_index: 0,
            index: 0,
            max_steps: (config.duration / config.tick_seconds) as usize,
            bumps: 0,
            completed_at: None,
            pending_reset: false,
            world,
            config,
        }
    }

    /// SENSE, MAP, PLAN — everything up to the decision.
    ///
    /// Returns `None` when the mission is over (all waypoints reached, or
    /// the deadline passed). Ticks on which a waypoint is *reached* do
    /// this work and then loop, exactly as the original `continue` did, so
    /// a caller never sees a half-finished tick.
    pub fn observe(&mut self) -> Option<Observation> {
        let dt = self.config.tick_seconds;
        let mut mode_changes = Vec::new();

        loop {
            if self.index >= self.max_steps {
                return None; // out of time
            }
            let index = self.index;
            self.index += 1;
            let elapsed_seconds = index as f64 * dt;

            if self.wp_index >= self.config.waypoints.len() {
                self.completed_at = Some(elapsed_seconds);
                return None; // mission complete
            }
            let goal = self.config.waypoints[self.wp_index];

            // ---- SENSE: the camera sees the PHYSICAL world (true pose).
            let scan = self.camera.scan(&self.robot.pose, &self.world);
            let summary = summarize_scan(&scan);
            let min_dist = summary.nearest;

            // ---- The reflex state machine, with hysteresis.
            let prev_mode = self.mode;
            self.mode = self.config.avoid.next(self.mode, min_dist);
            if self.mode != prev_mode {
                self.pending_reset = true; // stale momentum doesn't cross modes
                mode_changes.push((prev_mode, self.mode));
            }

            // The controller's own world-view (truth, or belief if enabled).
            let pose = if self.config.control_on_belief {
                self.odometry.pose
            } else {
                self.robot.pose
            };

            // ---- MAP: burn every camera ray into memory.
            for (idx, &d) in scan.iter().enumerate() {
                let a = pose.heading + self.camera.ray_angle(idx);
                self.map
                    .mark_ray(pose.x, pose.y, a, d, self.config.cam_range);
            }

            // ---- PLAN: re-run A* over the growing map.
            if index.is_multiple_of(self.config.replan_ticks) || self.path.is_none() {
                self.path = plan(
                    &self.map,
                    pose.position(),
                    goal,
                    self.config.inflation_cells,
                );
            }

            let dist_to_goal = pose.position().distance_to(goal);
            if dist_to_goal < self.config.gains.arrive_radius {
                self.wp_index += 1;
                self.pending_reset = true;
                self.path = None;
                continue; // no physics on an arrival tick
            }

            let target = match &self.path {
                Some(p) => lookahead_point(p, &pose, self.config.lookahead, goal),
                None => goal,
            };

            return Some(Observation {
                index,
                elapsed_seconds,
                pose,
                goal,
                target,
                dist_to_goal,
                mode: self.mode,
                mode_changes,
                scan,
                summary,
                path: self.path.clone(),
            });
        }
    }

    /// **THE POLICY** — the one place this robot decides what to do.
    ///
    /// Returns a [`Directive`] rather than a twist, which is the whole
    /// point: `sim-run` hands it straight to its own controller, and
    /// `hil-host` puts it on the wire. Both therefore send the *same
    /// decision* rather than each rebuilding it from the observation, and
    /// the chip cannot interpret it differently because it runs the same
    /// [`GotoController::execute`].
    pub fn plan(&mut self, obs: &Observation) -> Directive {
        // Consume the flag: a reset applies to the next command only.
        let fresh = core::mem::take(&mut self.pending_reset);
        match obs.mode {
            // Steer at the LOOKAHEAD point but throttle on distance to the
            // real goal, so the robot doesn't crawl just because the next
            // path node happens to be close.
            Mode::Goto => Directive::Steer {
                target: obs.target,
                budget: self.config.gains.distance_proportional * obs.dist_to_goal,
                fresh,
            },
            // A reflex from a depth scan the controller does not have.
            Mode::Avoid => Directive::Twist {
                v: self.config.avoid_v,
                w: obs.summary.turn_direction() * self.config.avoid_w,
            },
        }
    }

    /// The decision this mission's own controller would make.
    ///
    /// `sim-run` uses it; `hil-host` sends `plan()` down the wire and lets
    /// the chip run the identical `execute` instead. Those two paths must
    /// agree — see `crates/hil-host/tests/twin_decision.rs`, which drives
    /// the whole mission both ways and requires identical outcomes.
    pub fn decide(&mut self, obs: &Observation) -> BodyTwist {
        let directive = self.plan(obs);
        self.controller
            .execute(directive, &obs.pose, self.config.tick_seconds)
    }

    /// ACT and OBSERVE — the physics, given *commanded wheel speeds*.
    ///
    /// Takes wheel speeds rather than a body twist because that is the
    /// narrowest thing both callers can produce: `sim-run` gets there via
    /// `DiffDrive::inverse`, `hil-host` via duty × scale from the chip.
    /// One physics implementation, two sources of command.
    pub fn advance(&mut self, obs: Observation, commanded: WheelSpeeds) -> Tick {
        let dt = self.config.tick_seconds;

        // Each motor lags its command independently — that asymmetry is
        // part of why the robot drifts.
        let actual = WheelSpeeds {
            left: self.motor_l.step(commanded.left, dt),
            right: self.motor_r.step(commanded.right, dt),
        };
        let slip_l = 1.0 - 0.01 * self.rng.uniform();
        let slip_r = 1.0 - 0.01 * self.rng.uniform();
        let before = self.robot.pose;
        self.robot.step(
            WheelSpeeds::new(actual.left * slip_l, actual.right * slip_r),
            dt,
        );

        // Walls are solid: if the body would overlap one, the translation
        // is refused (rotation survives — a bumped robot can still pivot).
        // The wheels DID spin, so odometry keeps integrating: grinding
        // against a wall drifts fast, exactly like a real robot pushing
        // on one.
        let bumped = self.world.collides(
            self.robot.pose.x,
            self.robot.pose.y,
            self.config.robot_radius,
        );
        if bumped {
            self.robot.pose.x = before.x;
            self.robot.pose.y = before.y;
            self.bumps += 1;
        }

        // What the robot ACTUALLY achieved, for `StuckMonitor`. Computed
        // after the collision revert, so a robot held by a wall reports
        // zero however fast its wheels are turning — which is the entire
        // signal. On hardware the encoders would still report motion
        // here, and a wheel stopped by a wall reports zero for real.
        let moved_x = self.robot.pose.x - before.x;
        let moved_y = self.robot.pose.y - before.y;
        self.last_true_speed = (moved_x * moved_x + moved_y * moved_y).sqrt() / dt;

        // ---- OBSERVE: belief from encoder ticks alone.
        let (dticks_l, dticks_r) = self.encoders.advance(actual, dt);
        self.odometry.update(dticks_l, dticks_r);

        Tick {
            obs,
            commanded,
            bumped,
            true_pose: self.robot.pose,
            belief_pose: self.odometry.pose,
            drift: self.robot.pose.distance_to(&self.odometry.pose),
            dticks: (dticks_l, dticks_r),
        }
    }

    /// One full tick with this mission's own controller in the loop.
    ///
    /// `observe` -> `decide` -> `advance`. `hil-host` runs the same first
    /// and last calls with a serial cable in the middle.
    pub fn step(&mut self) -> Option<Tick> {
        let obs = self.observe()?;
        let mut twist = self.decide(&obs);

        // ---- did the last command actually move us? ----
        //
        // Measured against TRUE motion, not odometry: a wedged robot's
        // wheels keep turning, so dead reckoning agrees with the command
        // and would never notice. That is not cheating in a simulator —
        // on hardware the encoders play this role, and a wheel held still
        // by a wall reports zero.
        if let Some(escape) = self.stuck.update(twist.forward_speed, self.last_true_speed) {
            twist = escape;
        }
        // Scale, don't clip: a saturated turn keeps its arc.
        let commanded = self.config.spec.fit_wheels(self.nominal.inverse(twist));
        Some(self.advance(obs, commanded))
    }

    /// Run to completion, discarding per-tick telemetry.
    pub fn run(&mut self) -> Outcome {
        while self.step().is_some() {}
        self.outcome()
    }

    pub fn outcome(&self) -> Outcome {
        Outcome {
            completed_at: self.completed_at,
            waypoints_reached: self.wp_index,
            waypoints_total: self.config.waypoints.len(),
            drift: self.robot.pose.distance_to(&self.odometry.pose),
            bumps: self.bumps,
            ticks: self.index,
            final_pose: self.robot.pose,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **THE REGRESSION TEST.**
    ///
    /// This is the project's headline result, asserted rather than
    /// watched: mapping plus A* defeats the U-trap that beat the purely
    /// reactive robot (0/1 waypoints, 5.32 m drift — see docs/07,
    /// 2026-07-29).
    ///
    /// The tolerances are loose on purpose. Tightening them to the exact
    /// f64 output would make this fail on any harmless refactor —
    /// reordering two additions changes the last bits — while a real
    /// regression (driving into the trap, ghosting through a wall,
    /// never arriving) moves these numbers by whole seconds and metres.
    #[test]
    fn mapping_and_planning_beat_the_u_trap() {
        let outcome = Mission::new(MissionConfig::default()).run();

        assert!(
            outcome.succeeded(),
            "robot failed the U-trap: reached {}/{} waypoints in {} ticks",
            outcome.waypoints_reached,
            outcome.waypoints_total,
            outcome.ticks
        );

        let t = outcome
            .completed_at
            .expect("completed missions have a time");
        assert!(
            (20.0..25.0).contains(&t),
            "took {t:.1} s; the recorded baseline is 22.5 s"
        );

        assert!(
            outcome.drift < 0.10,
            "odometry drift {:.3} m exceeds the 0.052 m baseline by too much",
            outcome.drift
        );
    }

    /// The other half of the headline: it must not get there by cheating.
    ///
    /// Prakhar caught the robot driving *through* a wall on screen when
    /// the simulator had no collision detection at all. A mission that
    /// "succeeds" while grinding along walls is the same bug wearing a
    /// disguise.
    #[test]
    fn the_route_is_found_without_scraping_the_walls() {
        let outcome = Mission::new(MissionConfig::default()).run();
        assert!(
            outcome.bumps < 20,
            "robot bumped walls on {} ticks — it is feeling its way, not planning",
            outcome.bumps
        );
    }

    #[test]
    fn the_robot_actually_arrives_at_the_goal() {
        // Guards against "waypoints_reached" being incremented by a bug
        // rather than by arriving: check the final position independently.
        let config = MissionConfig::default();
        let goal = config.waypoints[0];
        let outcome = Mission::new(config).run();
        let distance = (outcome.final_pose.x - goal.x).hypot(outcome.final_pose.y - goal.y);
        assert!(
            distance < 0.2,
            "claims success but stopped {distance:.2} m from the goal"
        );
    }

    /// **THE DIGITAL-TWIN PROPERTY, as a test.**
    ///
    /// `hil-host` does not call `step()`. It calls `observe()`, ships the
    /// observation to a chip, gets a motor command back, and calls
    /// `advance()`. If that path can drift from `step()`, the simulator
    /// and the rig stop being twins — and the only thing that would notice
    /// is a human running the emulator and squinting at the numbers.
    ///
    /// So: drive the mission BOTH ways and require identical outcomes.
    /// The manual path here mirrors `hil-host` exactly, minus the wire.
    #[test]
    fn driving_it_by_hand_matches_step_exactly() {
        let in_process = Mission::new(MissionConfig::default()).run();

        // The hil-host path: observe, decide outside, advance.
        let mut m = Mission::new(MissionConfig::default());
        while let Some(obs) = m.observe() {
            let twist = m.decide(&obs);
            // Mirror what the chip does: inverse, then fit_wheels, then
            // (on the wire) duty. Miss the fit and the paths diverge —
            // which is exactly what this test caught when `step()` gained
            // the scaling and this did not.
            let commanded = m
                .config
                .spec
                .fit_wheels(m.config.spec.drive().inverse(twist));
            m.advance(obs, commanded);
        }
        let manual = m.outcome();

        assert_eq!(
            in_process, manual,
            "observe/decide/advance diverged from step() — the simulator \
             and the HIL rig are no longer digital twins"
        );
    }

    /// The wire carries *duty*, not wheel speeds. Two separate questions:
    /// is the quantisation lossless, and does anything saturate?
    ///
    /// # History, kept because the numbers are the point
    ///
    /// This test used to *document* an unfixed bug: peak wheel overshoot
    /// of **~169 rad/s against a 30 rad/s motor**, from the PID's D term.
    /// `Kd · de/dt` with `dt = 0.02` multiplies any error jump by 50, and
    /// the error jumps whenever the planner hops to the next lookahead
    /// node — even though the robot has not moved.
    ///
    /// Fixed 2026-08-07 by `Pid::derivative_limit` (bounding the derivative's
    /// contribution) plus `RobotSpec::fit_wheels` (scaling a saturated
    /// pair instead of clipping each wheel). Peak is **~33 rad/s** now,
    /// and what remains is the P term: a large heading error genuinely
    /// warrants a turn the robot cannot physically make that fast.
    ///
    /// The bound below is deliberately loose. Its job is to catch the D
    /// clamp being removed or mis-sized — which puts this back near 169 —
    /// not to pin a figure a harmless retune would break.
    #[test]
    fn duty_quantisation_is_lossless_but_the_controller_saturates() {
        let spec = MissionConfig::default().spec;
        let step = spec.max_wheel_speed / 1000.0; // one duty count, rad/s
        let mut m = Mission::new(MissionConfig::default());
        let mut worst_quantisation: f64 = 0.0;
        let mut worst_overshoot: f64 = 0.0;

        for _ in 0..300 {
            let Some(obs) = m.observe() else { break };
            let twist = m.decide(&obs);
            let commanded = spec.drive().inverse(twist);
            for cmd in [commanded.left, commanded.right] {
                let over = cmd.abs() - spec.max_wheel_speed;
                if over > 0.0 {
                    // Saturated: the duty cannot represent this at all.
                    worst_overshoot = worst_overshoot.max(over);
                } else {
                    // In range: round-tripping must cost less than one count.
                    let back = f64::from(spec.duty(cmd)) * step;
                    worst_quantisation = worst_quantisation.max((back - cmd).abs());
                }
            }
            m.advance(obs, commanded);
        }

        assert!(
            worst_quantisation <= step,
            "unsaturated duty lost {worst_quantisation} rad/s, more than \
             one count ({step})"
        );
        // Peak overshoot was ~169 rad/s before `heading_derivative_limit` existed.
        // It is ~33 now, and what remains is the P term: a large heading
        // error genuinely warrants a hard turn, and the robot genuinely
        // cannot make it that fast. That saturation is honest, and
        // `fit_wheels` handles it by scaling both wheels so the ARC is
        // preserved rather than the curvature distorted.
        //
        // The bound is deliberately loose. Its job is to catch the D clamp
        // being removed or mis-sized — which would put this back near 169
        // — not to pin an exact figure that a harmless retune would break.
        assert!(
            worst_overshoot < 60.0,
            "peak wheel overshoot {worst_overshoot:.1} rad/s. It was ~33 \
             with heading_derivative_limit in place and ~169 without; this looks \
             like the derivative clamp is gone or mis-sized."
        );
    }

    /// **An H4 risk, measured and pinned rather than discovered on a bench.**
    ///
    /// The duty plots are not smooth. The commanded duty reaches full
    /// scale on about 2% of ticks and can swing **1500 counts inside one
    /// 20 ms tick** — a full-scale reversal, the motor slammed from one
    /// direction to the other.
    ///
    /// In simulation this is harmless: `Motor::step` is a first-order lag
    /// with tau = 0.15 s, so the wheel cannot follow a 20 ms step and the
    /// mission completes cleanly. **A real motor has no such courtesy.** A
    /// commanded reversal means back-EMF opposing the drive, a current
    /// spike toward stall, battery sag on six AA cells, and mechanical
    /// shock through a small plastic gearbox.
    ///
    /// The source is legitimate: the lookahead point hops to the next path
    /// node when A* replans, the heading error steps, and the D term
    /// responds. `derivative_limit` already bounds it to 12 rad/s — the
    /// robot's physical spin limit — but 12 rad/s of turn rate *is* full
    /// duty on this geometry, so the clamp alone cannot prevent this.
    ///
    /// Deliberately NOT fixed here. The remedy is a slew-rate limit on the
    /// duty, and the right ramp is a number to **measure against the real
    /// motor**, not guess against a simulated one. Fixing it now would move
    /// the Stage 0 baseline on a guess.
    ///
    /// This test pins today's figures so the change is visible when it
    /// happens, in either direction.
    #[test]
    fn commanded_duty_slews_hard_enough_to_matter_on_real_motors() {
        let spec = MissionConfig::default().spec;
        let mut mission = Mission::new(MissionConfig::default());
        let (mut previous, mut worst_slew, mut saturated, mut ticks) =
            ((0i32, 0i32), 0i32, 0usize, 0usize);

        while let Some(tick) = mission.step() {
            let now = (
                spec.duty(tick.commanded.left),
                spec.duty(tick.commanded.right),
            );
            if now.0.abs() == sim_core::DUTY_FULL || now.1.abs() == sim_core::DUTY_FULL {
                saturated += 1;
            }
            worst_slew = worst_slew
                .max((now.0 - previous.0).abs())
                .max((now.1 - previous.1).abs());
            previous = now;
            ticks += 1;
        }

        // Loose bounds: this records the shape of the problem, not an
        // exact figure a harmless retune would break.
        assert!(
            saturated * 100 / ticks <= 5,
            "{saturated}/{ticks} ticks at full duty — saturation has grown"
        );
        assert!(
            (1000..=2000).contains(&worst_slew),
            "worst duty slew is {worst_slew} counts per 20 ms tick; it was \
             ~1500 when measured. A large drop means a slew limiter landed \
             (good — record the new baseline); a rise means something is \
             commanding even harder reversals."
        );
    }

    /// Determinism is what makes every assertion above meaningful.
    #[test]
    fn the_same_seed_produces_the_same_run() {
        let a = Mission::new(MissionConfig::default()).run();
        let b = Mission::new(MissionConfig::default()).run();
        assert_eq!(a, b, "mission is not reproducible");
    }

    #[test]
    fn a_different_seed_produces_a_different_trajectory() {
        // If the seed had no effect, the drift assertions above would be
        // testing a fixed constant rather than the physics.
        let base = MissionConfig::default();
        let mut other = base.clone();
        other.seed = 12345;

        let a = Mission::new(base).run();
        let b = Mission::new(other).run();
        assert_ne!(a.drift, b.drift, "the seed does not affect wheel slip");
        // Both should still solve it — the trap is not luck.
        assert!(
            a.succeeded() && b.succeeded(),
            "outcome depends on the seed"
        );
    }

    // ---- the mechanism, tested separately from the outcome ----

    #[test]
    fn odometry_drifts_because_the_wheels_are_worn() {
        // With perfect wheels there is still quantisation and slip, but
        // the systematic component vanishes and drift shrinks a lot.
        let perfect = MissionConfig {
            wheel_wear: 1.0,
            ..MissionConfig::default()
        };

        let worn = Mission::new(MissionConfig::default()).run();
        let exact = Mission::new(perfect).run();
        assert!(
            exact.drift < worn.drift,
            "worn wheels ({:.3} m) should drift more than exact ones ({:.3} m)",
            worn.drift,
            exact.drift
        );
    }

    #[test]
    fn an_impossible_deadline_reports_failure_rather_than_hanging() {
        let config = MissionConfig {
            duration: 1.0, // nowhere near enough
            ..MissionConfig::default()
        };

        let outcome = Mission::new(config).run();
        assert!(!outcome.succeeded());
        assert_eq!(outcome.completed_at, None);
        assert_eq!(outcome.ticks, 50, "should stop at the deadline");
    }

    #[test]
    fn a_goal_at_the_start_is_reached_immediately() {
        let mut config = MissionConfig::default();
        config.waypoints = vec![Point::new(config.start.x, config.start.y)];

        let outcome = Mission::new(config).run();
        assert!(outcome.succeeded());
        assert_eq!(outcome.ticks, 2, "one tick to arrive, one to notice");
    }

    /// The actuator command must survive into the `Tick`.
    ///
    /// It did not, for a while: the digital-twins refactor moved drawing
    /// into the shared `viz` and duty had no home there, so the plots went
    /// blank. Nothing failed, because nothing asserted on it — the symptom
    /// was two empty panels in the viewer, spotted by eye.
    #[test]
    fn the_tick_reports_what_the_wheels_were_asked_for() {
        let mut mission = Mission::new(MissionConfig::default());
        let mut moved = false;
        for _ in 0..200 {
            let Some(tick) = mission.step() else { break };
            assert!(
                tick.commanded.left.is_finite() && tick.commanded.right.is_finite(),
                "a command must always be a real number"
            );
            assert!(
                tick.commanded.peak() <= mission.config.spec.max_wheel_speed + 1e-9,
                "fit_wheels should have kept this within the motor: {:?}",
                tick.commanded
            );
            if tick.commanded.peak() > 0.0 {
                moved = true;
            }
        }
        assert!(moved, "the robot never commanded any wheel speed at all");
    }

    #[test]
    fn ticks_carry_the_telemetry_a_viewer_needs() {
        let mut mission = Mission::new(MissionConfig::default());
        let tick = mission.step().expect("first tick");

        assert_eq!(tick.obs.index, 0);
        assert_eq!(tick.obs.scan.len(), mission.config.cam_rays);
        assert!(tick.obs.summary.nearest.is_finite());
        assert!(!tick.bumped, "should not start inside a wall");
        assert_eq!(tick.obs.goal, Point::new(6.5, 3.0));
        assert!(tick.drift >= 0.0);
    }

    #[test]
    fn the_robot_starts_facing_the_trap_and_still_gets_around_it() {
        // The start pose points straight into the U's mouth: the naive
        // path is the wrong one, which is the entire scenario.
        let config = MissionConfig::default();
        assert!((config.start.heading - 0.0).abs() < 1e-12, "should face +x");
        assert!(config.start.x < 2.6, "should start outside the trap mouth");

        let outcome = Mission::new(config).run();
        assert!(outcome.succeeded());
    }
}

#[cfg(test)]
mod the_stack_finishes_from_routes_it_was_not_tuned_on {
    use super::*;

    /// Four configurations that each wedged the robot permanently before
    /// `sim_core::StuckMonitor` existed, and now do not.
    ///
    /// The U-trap had always passed, and on 2026-08-10 it turned out to be
    /// passing by luck: `WAYPOINT` on `SIM_BOT` happened to take a route
    /// that never wedged. Measuring the real motor changed the route, the
    /// robot nosed into a corner at (2.52, 0.76) and stayed there — pose
    /// identical to two decimals for 40 s, 28,000 ticks of wall contact.
    ///
    /// Isolating one variable at a time showed it was never about speed.
    /// Dropping only the derivative gain wedged it. So did cruise
    /// 0.45 -> 0.30 with nothing else touched. **A differential drive that
    /// only ever drives forward has no move that gets it out of a corner**,
    /// and nothing in the stack ever reversed.
    ///
    /// These are not four speeds anyone ships. They are four routes the
    /// tuning never saw, which is the only thing that distinguishes a
    /// recovery behaviour from a lucky one.
    #[test]
    fn four_routes_that_used_to_wedge_forever() {
        let slower = ControlGains {
            max_forward_speed: 0.30,
            ..ControlGains::WAYPOINT
        };
        let slowest = ControlGains {
            max_forward_speed: 0.20,
            ..ControlGains::WAYPOINT
        };
        let underdamped = ControlGains {
            heading_derivative: 0.155,
            ..ControlGains::WAYPOINT
        };
        for (name, spec, gains) in [
            ("the measured robot", RobotSpec::REAL_BOT, ControlGains::HIL),
            ("cruise 0.30", RobotSpec::SIM_BOT, slower),
            ("cruise 0.20", RobotSpec::SIM_BOT, slowest),
            ("weak derivative", RobotSpec::SIM_BOT, underdamped),
        ] {
            let config = MissionConfig {
                spec,
                gains,
                // The measured robot is 3.86x slower, so the same journey
                // takes proportionally longer. Not a fudge factor.
                duration: 300.0,
                ..Default::default()
            };
            let outcome = Mission::new(config).run();
            assert_eq!(
                outcome.waypoints_reached, outcome.waypoints_total,
                "{name} wedged: {outcome:?}"
            );
        }
    }
}
