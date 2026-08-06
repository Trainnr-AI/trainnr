//! Host half of the cross-platform self-test.
//!
//! Computes exactly what `firmware/pico-selftest` computes on the chip,
//! and prints it in exactly the same format. Diff the two and any
//! disagreement between the Mac and the ARM build shows up immediately.
//!
//! ```sh
//! cargo run -p sim-core --example selftest > /tmp/host.txt
//! # flash pico-selftest, capture its serial output to /tmp/chip.txt
//! diff /tmp/host.txt /tmp/chip.txt
//! ```
//!
//! **Why this can differ at all.** The Mac does `f64` in hardware. The
//! Cortex-M33 has only a *single-precision* FPU, so every `f64` operation
//! is emulated in software — a completely different code path. And `libm`
//! on bare metal is a different implementation of `sin`/`cos`/`atan2`/`exp`
//! from the system one. The project claims the same source runs in both
//! places; this checks whether the same source produces the same
//! *numbers*.

use sim_core::{
    ControlGains, Encoders, GotoController, Motor, Odometry, Pid, Pose, RobotSpec,
};

const SPEC: RobotSpec = RobotSpec::SIM_BOT;
const DT: f64 = 0.02;

fn main() {
    println!("\n=== robotiq self-test on real silicon ===\r");
    println!("\n[1] sim-core outputs (compare with the host)\r");

    let drive = SPEC.drive();

    let (v, w) = drive.forward(3.0, 7.0);
    println!("  forward       v={v:.17e} w={w:.17e}\r");

    let (ol, or) = drive.inverse(v, w);
    println!("  inverse       l={ol:.17e} r={or:.17e}\r");

    let moved = Pose::new(1.0, 3.0, 0.0).integrate(0.45, 1.2, DT);
    println!("  integrate     x={:.17e}\r", moved.x);
    println!("                y={:.17e} th={:.17e}\r", moved.y, moved.theta);

    let mut odom = Odometry {
        model: drive,
        ticks_per_rev: SPEC.ticks_per_rev,
        pose: Pose::new(1.0, 1.0, 0.0),
    };
    for _ in 0..50 {
        odom.update(37, 41);
    }
    println!("  odom x50      x={:.17e}\r", odom.pose.x);
    println!(
        "                y={:.17e} th={:.17e}\r",
        odom.pose.y, odom.pose.theta
    );

    let mut pid = Pid::new(6.0, 0.5, 0.6, 1.0);
    let mut out = 0.0;
    for i in 0..50 {
        out = pid.update(0.3 - 0.004 * i as f64, DT);
    }
    println!("  pid x50       out={out:.17e}\r");

    let mut motor = Motor::new(0.15, SPEC.max_wheel_rad_s);
    let mut speed = 0.0;
    for _ in 0..50 {
        speed = motor.step(20.0, DT);
    }
    println!("  motor x50     w={speed:.17e}\r");

    let mut enc = Encoders::new(SPEC.ticks_per_rev);
    let (mut tl, mut tr) = (0i64, 0i64);
    for _ in 0..50 {
        let (a, b) = enc.advance(12.5, 13.25, DT);
        tl += a;
        tr += b;
    }
    println!("  encoders x50  l={tl} r={tr}\r");

    let mut ctrl = GotoController::new(ControlGains::WAYPOINT);
    let pose = Pose::new(1.0, 3.0, 0.2);
    let (cv, cw) = ctrl.goto_point(&pose, (6.5, 3.0), DT);
    println!("  goto_point    v={cv:.17e}\r");
    println!("                w={cw:.17e}\r");
}
