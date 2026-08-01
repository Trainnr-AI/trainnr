//! LEVEL 3 — the robot's entire brain, running on the chip.
//!
//! This firmware is a complete autonomy stack with no idea it is driving a
//! simulated robot. Every tick it:
//!
//!   1. reports its believed pose            (P line, for the host's viewer)
//!   2. computes and sends motor commands    (M line)
//!   3. BLOCKS until the host reports encoder ticks (S line)
//!   4. updates its belief from those ticks
//!
//! Everything in the decide path came from the laptop simulator:
//!   sim_core::Odometry        — belief from ticks (Stage 0 exercise 3)
//!   sim_core::GotoController  — the steering law  (Pid, exercise 4, +
//!                               shortest_turn, exercise 2, + the
//!                               alignment throttle from M3)
//!   sim_core::RobotSpec       — the geometry, shared with the host
//!
//! Note what is NOT here any more: the control law itself. It used to be
//! written out inline in this file, in the simulator, and in the camera
//! chase — three copies of the robot's most important code. Now the chip
//! and the laptop call the same function (docs/13-architecture-review.md).
//!
//! The host owns physics: motor lag, wheel slip, walls, collisions. This
//! side owns *decisions*. That is exactly the Stage 3 architecture, and
//! swapping the emulator for a real Pico changes nothing here.
//!
//! Protocol: docs/10-hil-protocol.md — encoded and parsed by the shared
//! `hil-protocol` crate, so the host cannot drift from the chip. The
//! hand-rolled `read_line`/`parse_sensor` that used to live here are gone;
//! that format now has twenty tests it never had as a private fn.

#![no_std]
#![no_main]

use embassy_executor::Spawner;
use embassy_rp::gpio::{Level, Output};
use embassy_rp::uart::{Config as UartConfig, Uart};
use embassy_time::Timer;
use hil_protocol::{LineReader, Message};
use panic_halt as _;
use sim_core::{ControlGains, GotoController, Odometry, Pose, RobotSpec};

/// Control period — must match the host's physics step.
const DT: f64 = 0.02; // 50 Hz
/// The robot this firmware is driving. `REAL_BOT` — not `SIM_BOT` —
/// because this code runs on the physical machine: when the measured
/// values land in `spec.rs`, they take effect here with no edit.
const SPEC: RobotSpec = RobotSpec::REAL_BOT;

/// Steering profile. The waypoint gains, with a slightly wider arrival
/// radius: this loop runs against the host's motor lag over a serial link,
/// so it overshoots a little more than the pure simulator does.
const GAINS: ControlGains = ControlGains {
    arrive_radius: 0.18,
    ..ControlGains::WAYPOINT
};

/// The tour. The robot has no map — it just drives to these in order,
/// and the host has put walls in the way.
const WAYPOINTS: [(f64, f64); 4] = [(2.2, 1.0), (2.2, 3.6), (1.0, 3.6), (1.0, 1.0)];

#[embassy_executor::task]
async fn heartbeat(mut led: Output<'static>) {
    loop {
        led.toggle();
        Timer::after_millis(400).await;
    }
}

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    let p = embassy_rp::init(Default::default());
    spawner.spawn(heartbeat(Output::new(p.PIN_25, Level::Low)).unwrap());

    let mut uart = Uart::new_blocking(p.UART0, p.PIN_0, p.PIN_1, UartConfig::default());

    let mut odom = Odometry {
        model: SPEC.drive(),
        ticks_per_rev: SPEC.ticks_per_rev,
        pose: Pose::new(1.0, 1.0, 0.0), // told where it starts, once
    };
    let mut controller = GotoController::new(GAINS);

    let mut wp = 0usize;
    let mut out: heapless::String<96> = heapless::String::new();
    let mut reader: LineReader<64> = LineReader::new();

    loop {
        let pose = odom.pose;

        // ---- 1. report the belief (visualization only) ----
        out.clear();
        let _ = Message::Pose {
            x: pose.x,
            y: pose.y,
            theta: pose.theta,
        }
        .write_into(&mut out);
        let _ = uart.blocking_write(out.as_bytes());

        // ---- 2. decide ----
        let (mut duty_l, mut duty_r) = (0i32, 0i32);
        if wp < WAYPOINTS.len() {
            let goal = WAYPOINTS[wp];
            if controller.arrived(&pose, goal) {
                wp += 1;
                controller.reset();
            } else {
                // The shared steering law — identical to the simulator's.
                let (v, w) = controller.goto_point(&pose, goal, DT);
                // body twist -> wheel speeds -> duty (±1000)
                let (wl, wr) = SPEC.drive().inverse(v, w);
                duty_l = SPEC.duty(wl);
                duty_r = SPEC.duty(wr);
            }
        }
        out.clear();
        let _ = Message::Motor { duty_l, duty_r }.write_into(&mut out);
        let _ = uart.blocking_write(out.as_bytes());

        // ---- 3. block for the host's physics update ----
        // Read bytes until a complete line arrives, then parse it with the
        // SAME code the host uses (hil-protocol). Anything that isn't a
        // sensor line is ignored rather than guessed at.
        loop {
            let mut byte = [0u8; 1];
            if uart.blocking_read(&mut byte).is_err() {
                continue;
            }
            if let Some(line) = reader.push(byte[0]) {
                if let Ok(Message::Sensors { dl, dr }) = Message::parse(line) {
                    // ---- 4. update belief (Stage 0's odometry) ----
                    odom.update(dl, dr);
                }
                break;
            }
        }
    }
}
