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
//!   sim_core::Odometry  — belief from ticks   (Stage 0 exercise 3)
//!   sim_core::Pid       — heading control     (Stage 0 exercise 4)
//!   shortest_turn       — the error term      (Stage 0 exercise 2)
//!
//! The host owns physics: motor lag, wheel slip, walls, collisions. This
//! side owns *decisions*. That is exactly the Stage 3 architecture, and
//! swapping the emulator for a real Pico changes nothing here.
//!
//! Protocol: docs/10-hil-protocol.md

#![no_std]
#![no_main]

use core::fmt::Write as _;
use embassy_executor::Spawner;
use embassy_rp::gpio::{Level, Output};
use embassy_rp::uart::{Config as UartConfig, Uart};
use embassy_time::Timer;
use panic_halt as _;
use sim_core::exercises::shortest_turn;
use sim_core::{DiffDrive, Float as _, Odometry, Pid, Pose};

/// Control period — must match the host's physics step.
const DT: f64 = 0.02; // 50 Hz
/// Robot geometry (the same numbers the host simulates).
const WHEEL_RADIUS: f64 = 0.03;
const TRACK_WIDTH: f64 = 0.15;
const TICKS_PER_REV: f64 = 1024.0;
/// Heading PID gains, in the units of this loop: error in radians →
/// turn rate in rad/s. Tuned in the Rerun viewer during Stage 0's M3.
const HEADING_KP: f64 = 6.0;
const HEADING_KI: f64 = 0.0;
const HEADING_KD: f64 = 0.6;
/// Forward speed: proportional to distance, capped, throttled while
/// badly misaligned (turn first, then drive).
const KP_DIST: f64 = 0.8;
const V_MAX: f64 = 0.45;
const ARRIVE_RADIUS: f64 = 0.18;
/// Motor command scale: what body velocity a duty of 1000 corresponds to.
/// (The host's motor model turns duty into wheel speed.)
const MAX_WHEEL_RAD_S: f64 = 30.0;

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

/// Read one line from the UART into `buf`, returning its length.
/// Blocking: the loop cannot proceed until the host answers.
fn read_line<const N: usize>(
    uart: &mut Uart<'static, embassy_rp::uart::Blocking>,
    buf: &mut [u8; N],
) -> usize {
    let mut n = 0;
    loop {
        let mut byte = [0u8; 1];
        if uart.blocking_read(&mut byte).is_err() {
            continue;
        }
        match byte[0] {
            b'\n' => return n,
            b'\r' => {}
            b => {
                if n < N {
                    buf[n] = b;
                    n += 1;
                }
            }
        }
    }
}

/// Parse "S <int> <int>" — the host's encoder report.
fn parse_sensor(line: &[u8]) -> Option<(i64, i64)> {
    let text = core::str::from_utf8(line).ok()?;
    let mut parts = text.split_whitespace();
    if parts.next()? != "S" {
        return None;
    }
    let l = parts.next()?.parse::<i64>().ok()?;
    let r = parts.next()?.parse::<i64>().ok()?;
    Some((l, r))
}

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    let p = embassy_rp::init(Default::default());
    spawner.spawn(heartbeat(Output::new(p.PIN_25, Level::Low)).unwrap());

    let mut uart = Uart::new_blocking(p.UART0, p.PIN_0, p.PIN_1, UartConfig::default());

    let model = DiffDrive {
        wheel_radius: WHEEL_RADIUS,
        track_width: TRACK_WIDTH,
    };
    let mut odom = Odometry {
        model,
        ticks_per_rev: TICKS_PER_REV,
        pose: Pose::new(1.0, 1.0, 0.0), // told where it starts, once
    };
    let mut heading_pid = Pid::new(HEADING_KP, HEADING_KI, HEADING_KD, 1.0);

    let mut wp = 0usize;
    let mut out: heapless::String<96> = heapless::String::new();
    let mut line_buf = [0u8; 64];

    loop {
        let pose = odom.pose;

        // ---- 1. report the belief (visualization only) ----
        out.clear();
        let _ = write!(out, "P {:.4} {:.4} {:.4}\n", pose.x, pose.y, pose.theta);
        let _ = uart.blocking_write(out.as_bytes());

        // ---- 2. decide ----
        let (mut duty_l, mut duty_r) = (0i32, 0i32);
        if wp < WAYPOINTS.len() {
            let (gx, gy) = WAYPOINTS[wp];
            let dist = (gx - pose.x).hypot(gy - pose.y);
            if dist < ARRIVE_RADIUS {
                wp += 1;
                heading_pid.reset();
            } else {
                let bearing = (gy - pose.y).atan2(gx - pose.x);
                let err = shortest_turn(pose.theta, bearing);
                let w = heading_pid.update(err, DT);
                let align = (1.0 - err.abs() / core::f64::consts::FRAC_PI_2).max(0.0);
                let v = (KP_DIST * dist).min(V_MAX) * align;
                // body twist -> wheel speeds -> duty (±1000)
                let (wl, wr) = model.inverse(v, w);
                duty_l = ((wl / MAX_WHEEL_RAD_S) * 1000.0).clamp(-1000.0, 1000.0) as i32;
                duty_r = ((wr / MAX_WHEEL_RAD_S) * 1000.0).clamp(-1000.0, 1000.0) as i32;
            }
        }
        out.clear();
        let _ = write!(out, "M {} {}\n", duty_l, duty_r);
        let _ = uart.blocking_write(out.as_bytes());

        // ---- 3. block for the host's physics update ----
        let n = read_line(&mut uart, &mut line_buf);
        if let Some((dl, dr)) = parse_sensor(&line_buf[..n]) {
            // ---- 4. update belief (Stage 0's odometry) ----
            odom.update(dl, dr);
        }
    }
}
