//! The Stage 0 mission: sense → map → plan → decide → act → observe.
//!
//! Pure simulation. Knows nothing about Rerun, threads, or wall-clock
//! time. See the module docs in `lib.rs` for why.

use sim_core::exercises::shortest_turn;
use sim_core::{
    lookahead_point, plan, summarize_scan, AvoidHysteresis, ControlGains, DepthCamera, DiffDrive,
    Encoders, GotoController, Mode, Motor, OccupancyGrid, Odometry, Pose, Rng, Robot, RobotSpec,
    Segment, World,
};

/// Everything the mission needs to be reproducible.
///
/// A struct rather than module constants so a test can vary one parameter
/// — a different seed, a shorter deadline — without editing the binary.
#[derive(Debug, Clone)]
pub struct MissionConfig {
    /// Control-loop period, seconds. 50 Hz.
    pub dt: f64,
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
    /// Motor saturation, rad/s. Defaults to `spec.max_wheel_rad_s` — the
    /// same physical quantity, and a third copy of `30.0` until 2026-08-02.
    pub motor_max: f64,
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
    pub waypoints: Vec<(f64, f64)>,
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
            dt: 0.02,
            duration: 60.0,
            seed: 7,
            spec: RobotSpec::SIM_BOT,
            gains: ControlGains::WAYPOINT,
            wheel_wear: 0.99,
            motor_tau: 0.15,
            motor_max: RobotSpec::SIM_BOT.max_wheel_rad_s,
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
            waypoints: vec![(6.5, 3.0)],     // dead centre behind the back wall
            obstacles: vec![
                Segment {
                    a: (5.0, 2.2),
                    b: (5.0, 3.8), // back wall
                },
                Segment {
                    a: (2.6, 3.8),
                    b: (5.0, 3.8), // top arm (deep!)
                },
                Segment {
                    a: (2.6, 2.2),
                    b: (5.0, 2.2), // bottom arm (deep!)
                },
                Segment {
                    a: (2.6, 3.8),
                    b: (2.6, 5.2), // spur up from the top arm tip
                },
                Segment {
                    a: (2.6, 0.8),
                    b: (2.6, 2.2), // spur down from the bottom arm tip
                },
            ],
        }
    }
}

/// One completed control tick — everything a viewer needs to draw it.
#[derive(Debug, Clone)]
pub struct Tick {
    pub index: usize,
    /// Simulated seconds since the start.
    pub t: f64,
    /// Depth reading per camera ray, right to left.
    pub scan: Vec<f64>,
    pub mode: Mode,
    /// Mode transitions that happened this tick, as `(from, to)`.
    pub mode_changes: Vec<(Mode, Mode)>,
    pub min_dist: f64,
    /// Did the body overlap a wall, so the translation was refused?
    pub bumped: bool,
    /// The point being steered at (path lookahead, or the goal).
    pub target: (f64, f64),
    pub goal: (f64, f64),
    pub true_pose: Pose,
    pub belief_pose: Pose,
    /// Distance between truth and belief — the accumulated odometry error.
    pub drift: f64,
    pub path: Option<Vec<(f64, f64)>>,
    /// Encoder counts this tick. The only thing the chip gets to see of
    /// the physics — everything else here is for the viewer.
    pub dticks: (i64, i64),
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
    pub t: f64,
    /// The pose the controller should steer on — truth, or belief if
    /// `control_on_belief` is set.
    pub pose: Pose,
    /// The waypoint currently being driven to.
    pub goal: (f64, f64),
    /// Where to actually aim: the path lookahead point, or the goal if
    /// there is no plan. **This is what gets sent to the chip.**
    pub target: (f64, f64),
    pub dist_to_goal: f64,
    pub mode: Mode,
    pub mode_changes: Vec<(Mode, Mode)>,
    pub scan: Vec<f64>,
    pub summary: sim_core::ScanSummary,
    pub path: Option<Vec<(f64, f64)>>,
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
    encoders: Encoders,
    rng: Rng,
    motor_l: Motor,
    motor_r: Motor,
    controller: GotoController,

    path: Option<Vec<(f64, f64)>>,
    mode: Mode,
    wp_index: usize,
    index: usize,
    max_steps: usize,
    bumps: usize,
    completed_at: Option<f64>,
}

impl Mission {
    pub fn new(config: MissionConfig) -> Mission {
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
                ticks_per_rev: config.spec.ticks_per_rev,
                pose: config.start,
            },
            encoders: Encoders::new(config.spec.ticks_per_rev),
            rng: Rng::new(config.seed),
            motor_l: Motor::new(config.motor_tau, config.motor_max),
            motor_r: Motor::new(config.motor_tau, config.motor_max),
            controller: GotoController::new(config.gains),
            nominal,
            path: None,
            mode: Mode::Goto,
            wp_index: 0,
            index: 0,
            max_steps: (config.duration / config.dt) as usize,
            bumps: 0,
            completed_at: None,
            world,
            config,
        }
    }

    /// Advance one control tick.
    ///
    /// Returns `None` when the mission is over — all waypoints reached, or
    /// the deadline passed. Ticks on which a waypoint is *reached* do the
    /// sense/map/plan work and then skip physics, exactly as the original
    /// loop's `continue` did; `step` absorbs those internally so a caller
    /// never sees a half-finished tick.
    /// SENSE, MAP, PLAN — everything up to the decision.
    ///
    /// Returns `None` when the mission is over (all waypoints reached, or
    /// the deadline passed). Ticks on which a waypoint is *reached* do
    /// this work and then loop, exactly as the original `continue` did, so
    /// a caller never sees a half-finished tick.
    pub fn observe(&mut self) -> Option<Observation> {
        let dt = self.config.dt;
        let mut mode_changes = Vec::new();

        loop {
            if self.index >= self.max_steps {
                return None; // out of time
            }
            let index = self.index;
            self.index += 1;
            let t = index as f64 * dt;

            if self.wp_index >= self.config.waypoints.len() {
                self.completed_at = Some(t);
                return None; // mission complete
            }
            let goal = self.config.waypoints[self.wp_index];

            // ---- SENSE: the camera sees the PHYSICAL world (true pose).
            let scan = self.camera.scan(&self.robot.pose, &self.world);
            let summary = summarize_scan(&scan);
            let min_dist = summary.min;

            // ---- The reflex state machine, with hysteresis.
            let prev_mode = self.mode;
            self.mode = self.config.avoid.next(self.mode, min_dist);
            if self.mode != prev_mode {
                self.controller.reset(); // stale momentum doesn't cross modes
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
                let a = pose.theta + self.camera.ray_angle(idx);
                self.map
                    .mark_ray(pose.x, pose.y, a, d, self.config.cam_range);
            }

            // ---- PLAN: re-run A* over the growing map.
            if index.is_multiple_of(self.config.replan_ticks) || self.path.is_none() {
                self.path = plan(
                    &self.map,
                    (pose.x, pose.y),
                    goal,
                    self.config.inflation_cells,
                );
            }

            let dist_to_goal = (goal.0 - pose.x).hypot(goal.1 - pose.y);
            if dist_to_goal < self.config.gains.arrive_radius {
                self.wp_index += 1;
                self.controller.reset();
                self.path = None;
                continue; // no physics on an arrival tick
            }

            let target = match &self.path {
                Some(p) => lookahead_point(p, &pose, self.config.lookahead, goal),
                None => goal,
            };

            return Some(Observation {
                index,
                t,
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

    /// The decision this mission's own controller would make.
    ///
    /// `sim-run` uses it; `hil-host` ignores it and asks the chip instead.
    pub fn decide(&mut self, obs: &Observation) -> (f64, f64) {
        match obs.mode {
            Mode::Goto => {
                // Steer at the LOOKAHEAD point but throttle on distance to
                // the real goal, so the robot doesn't crawl just because
                // the next path node happens to be close.
                let bearing = (obs.target.1 - obs.pose.y).atan2(obs.target.0 - obs.pose.x);
                let heading_error = shortest_turn(obs.pose.theta, bearing);
                self.controller.steer(
                    heading_error,
                    obs.pose.theta,
                    self.config.gains.kp_dist * obs.dist_to_goal,
                    self.config.dt,
                )
            }
            Mode::Avoid => (
                self.config.avoid_v,
                obs.summary.turn_direction() * self.config.avoid_w,
            ),
        }
    }

    /// ACT and OBSERVE — the physics, given *commanded wheel speeds*.
    ///
    /// Takes wheel speeds rather than a body twist because that is the
    /// narrowest thing both callers can produce: `sim-run` gets there via
    /// `DiffDrive::inverse`, `hil-host` via duty × scale from the chip.
    /// One physics implementation, two sources of command.
    pub fn advance(&mut self, obs: Observation, cmd_l: f64, cmd_r: f64) -> Tick {
        let dt = self.config.dt;

        let act_l = self.motor_l.step(cmd_l, dt);
        let act_r = self.motor_r.step(cmd_r, dt);
        let slip_l = 1.0 - 0.01 * self.rng.uniform();
        let slip_r = 1.0 - 0.01 * self.rng.uniform();
        let before = self.robot.pose;
        self.robot.step(act_l * slip_l, act_r * slip_r, dt);

        // Walls are solid: if the body would overlap one, the translation
        // is refused (rotation survives — a bumped robot can still pivot).
        // The wheels DID spin, so odometry keeps integrating: grinding
        // against a wall drifts fast, exactly like a real robot pushing
        // on one.
        let bumped =
            self.world
                .collides(self.robot.pose.x, self.robot.pose.y, self.config.robot_radius);
        if bumped {
            self.robot.pose.x = before.x;
            self.robot.pose.y = before.y;
            self.bumps += 1;
        }

        // ---- OBSERVE: belief from encoder ticks alone.
        let (dticks_l, dticks_r) = self.encoders.advance(act_l, act_r, dt);
        self.odometry.update(dticks_l, dticks_r);

        Tick {
            index: obs.index,
            t: obs.t,
            scan: obs.scan,
            mode: obs.mode,
            mode_changes: obs.mode_changes,
            min_dist: obs.summary.min,
            bumped,
            target: obs.target,
            goal: obs.goal,
            true_pose: self.robot.pose,
            belief_pose: self.odometry.pose,
            drift: self.robot.pose.distance_to(&self.odometry.pose),
            path: obs.path,
            dticks: (dticks_l, dticks_r),
        }
    }

    /// One full tick with this mission's own controller in the loop.
    ///
    /// `observe` -> `decide` -> `advance`. `hil-host` runs the same first
    /// and last calls with a serial cable in the middle.
    pub fn step(&mut self) -> Option<Tick> {
        let obs = self.observe()?;
        let (v, w) = self.decide(&obs);
        // Scale, don't clip: a saturated turn keeps its arc.
        let (cmd_l, cmd_r) = self.config.spec.fit_wheels_of(self.nominal.inverse(v, w));
        Some(self.advance(obs, cmd_l, cmd_r))
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
        let distance = (outcome.final_pose.x - goal.0).hypot(outcome.final_pose.y - goal.1);
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
            let (v, w) = m.decide(&obs);
            // Mirror what the chip does: inverse, then fit_wheels, then
            // (on the wire) duty. Miss the fit and the paths diverge —
            // which is exactly what this test caught when `step()` gained
            // the scaling and this did not.
            let (cmd_l, cmd_r) = m.config.spec.fit_wheels_of(m.config.spec.drive().inverse(v, w));
            m.advance(obs, cmd_l, cmd_r);
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
    /// # A real finding, deliberately recorded rather than fixed
    ///
    /// The second one is not benign. The controller regularly commands
    /// wheel speeds the motors cannot deliver — this test measured a peak
    /// **overshoot of ~169 rad/s against a 30 rad/s motor**, i.e. asking
    /// for over six times what exists.
    ///
    /// The source is the PID's D term. `Kd · de/dt` with `dt = 0.02` means
    /// a target that jumps sideways gives `de/dt ≈ 157`, so `w ≈ 94 rad/s`
    /// of turn rate, which is ~235 rad/s at the wheel.
    ///
    /// It does not break the simulation: `Motor::step` clamps, the mission
    /// still completes in 22.5 s. But the controller is **relying on
    /// saturation to clean up after it**, which is a different thing from
    /// being correct, and on real hardware it means the turn's shape
    /// differs from what the controller computed — with no feedback path
    /// telling the PID its output was ignored.
    ///
    /// Not fixed here because clamping `steer`'s output would move the
    /// recorded Stage 0 baseline, and that is a decision to make
    /// deliberately rather than inside a test. Note also that
    /// `RobotSpec::check` claims to catch "commanding speeds the robot
    /// cannot reach" and misses this entirely — it only validates `v_max`,
    /// never the turn component.
    #[test]
    fn duty_quantisation_is_lossless_but_the_controller_saturates() {
        let spec = MissionConfig::default().spec;
        let step = spec.max_wheel_rad_s / 1000.0; // one duty count, rad/s
        let mut m = Mission::new(MissionConfig::default());
        let mut worst_quantisation: f64 = 0.0;
        let mut worst_overshoot: f64 = 0.0;

        for _ in 0..300 {
            let Some(obs) = m.observe() else { break };
            let (v, w) = m.decide(&obs);
            let (cmd_l, cmd_r) = spec.drive().inverse(v, w);
            for cmd in [cmd_l, cmd_r] {
                let over = cmd.abs() - spec.max_wheel_rad_s;
                if over > 0.0 {
                    // Saturated: the duty cannot represent this at all.
                    worst_overshoot = worst_overshoot.max(over);
                } else {
                    // In range: round-tripping must cost less than one count.
                    let back = f64::from(spec.duty(cmd)) * step;
                    worst_quantisation = worst_quantisation.max((back - cmd).abs());
                }
            }
            m.advance(obs, cmd_l, cmd_r);
        }

        assert!(
            worst_quantisation <= step,
            "unsaturated duty lost {worst_quantisation} rad/s, more than \
             one count ({step})"
        );
        // Peak overshoot was ~169 rad/s before `heading_d_limit` existed.
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
             with heading_d_limit in place and ~169 without; this looks \
             like the derivative clamp is gone or mis-sized."
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
        config.waypoints = vec![(config.start.x, config.start.y)];

        let outcome = Mission::new(config).run();
        assert!(outcome.succeeded());
        assert_eq!(outcome.ticks, 2, "one tick to arrive, one to notice");
    }

    #[test]
    fn ticks_carry_the_telemetry_a_viewer_needs() {
        let mut mission = Mission::new(MissionConfig::default());
        let tick = mission.step().expect("first tick");

        assert_eq!(tick.index, 0);
        assert_eq!(tick.scan.len(), mission.config.cam_rays);
        assert!(tick.min_dist.is_finite());
        assert!(!tick.bumped, "should not start inside a wall");
        assert_eq!(tick.goal, (6.5, 3.0));
        assert!(tick.drift >= 0.0);
    }

    #[test]
    fn the_robot_starts_facing_the_trap_and_still_gets_around_it() {
        // The start pose points straight into the U's mouth: the naive
        // path is the wrong one, which is the entire scenario.
        let config = MissionConfig::default();
        assert!((config.start.theta - 0.0).abs() < 1e-12, "should face +x");
        assert!(config.start.x < 2.6, "should start outside the trap mouth");

        let outcome = Mission::new(config).run();
        assert!(outcome.succeeded());
    }
}
