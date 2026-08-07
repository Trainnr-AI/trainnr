//! LEVEL 1 PROOF — the laptop robot's brain, running on the chip.
//!
//! Every line of dead reckoning here comes from `sim-core`: the SAME
//! `Odometry`, `Pose` and `DiffDrive` that drive the Rerun simulator,
//! compiled with `--no-default-features` for bare metal. Nothing was
//! rewritten or ported. The two islands are now one codebase:
//!
//!   quad-encoder  (H3)  ->  sim-core::Odometry  (Stage 0 exercise 3)
//!   pin transitions      ->  believed pose, computed on emulated ARM
//!
//! Two encoders (left GP16/17, right GP18/19) feed the same tick-to-pose
//! math the simulator uses, and the firmware reports x/y/heading over UART.

#![no_std]
#![no_main]
// No unsafe anywhere in this firmware. `forbid`, not `deny`: it cannot be
// switched off locally with an `#[allow]`. Embassy's HAL already wraps the
// peripheral access that would otherwise need it — if that ever stops being
// true, the argument belongs in a commit that changes this line.
#![forbid(unsafe_code)]

use core::fmt::Write as _;
use embassy_executor::Spawner;
use embassy_rp::gpio::{Input, Level, Output, Pull};
// ⚠️ RP2350-E9 (see docs/09): our physical board is stepping A2, where an
// input with an internal PULL-DOWN can latch high instead of reading a
// clean low. GP16–GP19 below use exactly that configuration. It may
// never show — a push-pull encoder drives the line itself — but if ticks
// stick or counts only ever rise, add an external pull-down <=4.7k before
// suspecting this code. The emulated RP2040 is unaffected.
use embassy_rp::uart::{Blocking, Config as UartConfig, Uart, UartTx};
use embassy_time::{Instant, Timer};
use panic_halt as _;
use quad_encoder::QuadratureDecoder;
use sim_core::{DiffDrive, Odometry, Pose};

/// Encoder sampling period. 100 µs = 10 kHz (see math-09 on aliasing).
const POLL_US: u64 = 100;
/// Robot geometry — identical numbers to the Stage 0 simulator.
const WHEEL_RADIUS: f64 = 0.03;
const TRACK_WIDTH: f64 = 0.15;
const TICKS_PER_REV: f64 = 1024.0;

#[embassy_executor::task]
async fn heartbeat(mut led: Output<'static>) {
    loop {
        led.toggle();
        Timer::after_millis(500).await;
    }
}

#[embassy_executor::task]
async fn odometry_task(
    la: Input<'static>,
    lb: Input<'static>,
    ra: Input<'static>,
    rb: Input<'static>,
    mut uart: UartTx<'static, Blocking>,
) {
    let mut left = QuadratureDecoder::new(la.is_high(), lb.is_high());
    let mut right = QuadratureDecoder::new(ra.is_high(), rb.is_high());

    // sim-core's odometry, unmodified, on a microcontroller.
    let mut odom = Odometry {
        model: DiffDrive {
            wheel_radius: WHEEL_RADIUS,
            track_width: TRACK_WIDTH,
        },
        ticks_per_revolution: TICKS_PER_REV,
        pose: Pose::ORIGIN,
    };

    let mut line: heapless::String<192> = heapless::String::new();
    let mut last_report = Instant::now();
    let mut last_l = 0i32;
    let mut last_r = 0i32;

    loop {
        // ---- sample both encoders (H3's decoder) ----
        left.update(la.is_high(), lb.is_high());
        right.update(ra.is_high(), rb.is_high());

        let now = Instant::now();
        if now.duration_since(last_report).as_millis() >= 100 {
            // ---- feed the tick deltas to Stage 0's odometry ----
            let dl = left.count - last_l;
            let dr = right.count - last_r;
            last_l = left.count;
            last_r = right.count;
            odom.update(dl as i64, dr as i64);

            let p = odom.pose;
            let _ = write!(
                line,
                "pose x={:+.3} y={:+.3} th={:+.3}  ticks L={} R={}  err={}\r\n",
                p.x,
                p.y,
                p.heading,
                left.count,
                right.count,
                left.errors + right.errors
            );
            let _ = uart.blocking_write(line.as_bytes());
            line.clear();
            last_report = now;
        }

        Timer::after_micros(POLL_US).await;
    }
}

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    let p = embassy_rp::init(Default::default());
    let uart = Uart::new_blocking(p.UART0, p.PIN_0, p.PIN_1, UartConfig::default());
    let (tx, _rx) = uart.split();

    spawner.spawn(heartbeat(Output::new(p.PIN_25, Level::Low)).unwrap());
    spawner.spawn(
        odometry_task(
            Input::new(p.PIN_16, Pull::Down),
            Input::new(p.PIN_17, Pull::Down),
            Input::new(p.PIN_18, Pull::Down),
            Input::new(p.PIN_19, Pull::Down),
            tx,
        )
        .unwrap(),
    );
}
