//! SELF-TEST — run the shared maths on real silicon and report.
//!
//! Answers two questions this project has been asserting without evidence.
//!
//! # 1. Does `sim-core` give the SAME answers on ARM?
//!
//! The claim throughout these docs is *"the PID you tune in Rerun is
//! byte-for-byte the PID on the chip"*. That is true of the **source**. It
//! has never been checked of the **output**, and there are real reasons it
//! might not hold:
//!
//! - the Mac does `f64` in hardware; the M0+ has **no FPU at all** and
//!   emulates every one in software
//! - the M33 in the RP2350 has an FPU, but a **single-precision** one, so
//!   `f64` is still emulated — a different code path again
//! - `libm` on bare metal and the system libm on the host are *different
//!   implementations* of `sin`, `cos`, `atan2`, `exp`
//!
//! Any of those could shift a low bit, and over a 50 Hz loop low bits
//! accumulate. So: push fixed inputs through `Odometry`, `Pid`,
//! `DiffDrive`, `Motor`, `Encoders` and `GotoController`, print to more
//! digits than anyone would eyeball, and compare against the same values
//! computed on the Mac.
//!
//! # 2. Does a control tick fit in 20 ms?
//!
//! `docs/15-testing-and-coverage.md` lists timing as genuinely untested:
//! *"nothing here catches a control loop that misses its 50 Hz
//! deadline."* Now it can be measured — the exact work `pico-robot` does
//! every tick, timed on the real chip.
//!
//! Output goes over **USB serial**: no Debug Probe, no soldering, the same
//! cable that powers the board.
//!
//! ```sh
//! tools/build-pico2.sh pico-selftest
//! picotool load -x firmware/pico-selftest/pico-selftest-pico2.uf2
//! ls /dev/tty.usbmodem*          # then: screen <that port> 115200
//! ```

#![no_std]
#![no_main]
// No unsafe anywhere in this firmware. `forbid`, not `deny`: it cannot be
// switched off locally with an `#[allow]`. Embassy's HAL already wraps the
// peripheral access that would otherwise need it — if that ever stops being
// true, the argument belongs in a commit that changes this line.
#![forbid(unsafe_code)]

use core::fmt::Write as _;

use embassy_executor::Spawner;
use embassy_futures::join::join;
use embassy_rp::bind_interrupts;
use embassy_rp::peripherals::USB;
use embassy_rp::usb::{Driver, InterruptHandler};
use embassy_time::{Instant, Timer};
use embassy_usb::class::cdc_acm::{CdcAcmClass, State};
use embassy_usb::{Builder, Config};
use panic_halt as _;
use static_cell::StaticCell;

use sim_core::{
    ControlGains, DiffDrive, Encoders, GotoController, Motor, Odometry, Pid, Pose, RobotSpec,
};

bind_interrupts!(struct Irqs {
    USBCTRL_IRQ => InterruptHandler<USB>;
});

const SPEC: RobotSpec = RobotSpec::SIM_BOT;
const DT: f64 = 0.02;

/// Big enough for our longest report line.
type Line = heapless::String<160>;

#[embassy_executor::main]
async fn main(_spawner: Spawner) {
    let p = embassy_rp::init(Default::default());
    let driver = Driver::new(p.USB, Irqs);

    let mut config = Config::new(0x2e8a, 0x0009);
    config.manufacturer = Some("robotiq");
    config.product = Some("pico-selftest");
    config.serial_number = Some("1");
    config.max_power = 100;
    config.max_packet_size_0 = 64;

    // Descriptor buffers must outlive the Builder. StaticCell hands out a
    // `&'static mut` exactly once, with no allocator involved.
    static CONFIG_DESC: StaticCell<[u8; 256]> = StaticCell::new();
    static BOS_DESC: StaticCell<[u8; 256]> = StaticCell::new();
    static CONTROL_BUF: StaticCell<[u8; 64]> = StaticCell::new();
    static STATE: StaticCell<State> = StaticCell::new();

    let state = STATE.init(State::new());
    let mut builder = Builder::new(
        driver,
        config,
        CONFIG_DESC.init([0; 256]),
        BOS_DESC.init([0; 256]),
        &mut [], // no Microsoft OS descriptors
        CONTROL_BUF.init([0; 64]),
    );

    let mut class = CdcAcmClass::new(&mut builder, state, 64);
    let mut usb = builder.build();

    // Two jobs at once: the USB stack servicing the host, and our report
    // writing into it. `join` polls both on one stack — no second task.
    let usb_fut = usb.run();
    let report_fut = async {
        loop {
            class.wait_connection().await;
            let _ = run_report(&mut class).await;
            // Reconnecting the terminal re-runs it, which helps when you
            // miss the first burst.
            Timer::after_secs(1).await;
        }
    };
    join(usb_fut, report_fut).await;
}

async fn run_report<'d, D: embassy_usb::driver::Driver<'d>>(
    class: &mut CdcAcmClass<'d, D>,
) -> Result<(), embassy_usb::driver::EndpointError> {
    let mut l = Line::new();

    say(class, "\r\n=== robotiq self-test on real silicon ===\r\n").await?;

    // ---- 1. Does the shared maths agree with the Mac? ----
    say(class, "\r\n[1] sim-core outputs (compare with the host)\r\n").await?;

    let drive: DiffDrive = SPEC.drive();

    let (v, w) = drive.forward(3.0, 7.0);
    l.clear();
    let _ = write!(l, "  forward       v={v:.17e} w={w:.17e}\r\n");
    say(class, &l).await?;

    let (ol, or) = drive.inverse(v, w);
    l.clear();
    let _ = write!(l, "  inverse       l={ol:.17e} r={or:.17e}\r\n");
    say(class, &l).await?;

    // Arc integration — trig-heavy, so most likely to differ between libm
    // implementations.
    let moved = Pose::new(1.0, 3.0, 0.0).integrate(0.45, 1.2, DT);
    l.clear();
    let _ = write!(l, "  integrate     x={:.17e}\r\n", moved.x);
    say(class, &l).await?;
    l.clear();
    let _ = write!(
        l,
        "                y={:.17e} th={:.17e}\r\n",
        moved.y, moved.theta
    );
    say(class, &l).await?;

    let mut odom = Odometry {
        model: drive,
        ticks_per_rev: SPEC.ticks_per_rev,
        pose: Pose::new(1.0, 1.0, 0.0),
    };
    for _ in 0..50 {
        odom.update(37, 41);
    }
    l.clear();
    let _ = write!(l, "  odom x50      x={:.17e}\r\n", odom.pose.x);
    say(class, &l).await?;
    l.clear();
    let _ = write!(
        l,
        "                y={:.17e} th={:.17e}\r\n",
        odom.pose.y, odom.pose.theta
    );
    say(class, &l).await?;

    let mut pid = Pid::new(6.0, 0.5, 0.6, 1.0);
    let mut out = 0.0;
    for i in 0..50 {
        out = pid.update(0.3 - 0.004 * i as f64, DT);
    }
    l.clear();
    let _ = write!(l, "  pid x50       out={out:.17e}\r\n");
    say(class, &l).await?;

    // Motor lag uses exp().
    let mut motor = Motor::new(0.15, SPEC.max_wheel_rad_s);
    let mut speed = 0.0;
    for _ in 0..50 {
        speed = motor.step(20.0, DT);
    }
    l.clear();
    let _ = write!(l, "  motor x50     w={speed:.17e}\r\n");
    say(class, &l).await?;

    // Encoders quantise to whole ticks — integers, so these must match
    // exactly or something is very wrong.
    let mut enc = Encoders::new(SPEC.ticks_per_rev);
    let (mut tl, mut tr) = (0i64, 0i64);
    for _ in 0..50 {
        let (a, b) = enc.advance(12.5, 13.25, DT);
        tl += a;
        tr += b;
    }
    l.clear();
    let _ = write!(l, "  encoders x50  l={tl} r={tr}\r\n");
    say(class, &l).await?;

    let mut ctrl = GotoController::new(ControlGains::WAYPOINT);
    let pose = Pose::new(1.0, 3.0, 0.2);
    let (cv, cw) = ctrl.goto_point(&pose, (6.5, 3.0), DT);
    l.clear();
    let _ = write!(l, "  goto_point    v={cv:.17e}\r\n");
    say(class, &l).await?;
    l.clear();
    let _ = write!(l, "                w={cw:.17e}\r\n");
    say(class, &l).await?;

    // ---- the saturation fix (2026-08-07) ----
    // Both of these are branch-heavy rather than trig-heavy, so the risk
    // is not a low bit — it is a *comparison* landing differently once
    // f64 is emulated in software. `fit_wheels` branches on
    // `peak <= max_wheel_rad_s`, and `d_limit` on a clamp boundary. A
    // divergence here would not be a rounding difference; it would be the
    // chip taking the other branch.
    let (fl, fr) = SPEC.fit_wheels(40.0, 10.0);
    l.clear();
    let _ = write!(l, "  fit_wheels    l={fl:.17e} r={fr:.17e}\r\n");
    say(class, &l).await?;

    // A setpoint STEP — precisely the derivative-kick case d_limit exists
    // to bound. Without the clamp this term runs away.
    let mut kick = Pid::with_d_limit(6.0, 0.0, 0.6, 1.0, 12.0);
    let mut k = 0.0;
    for i in 0..50 {
        k = kick.update(if i < 25 { 0.05 } else { 3.0 }, DT);
    }
    l.clear();
    let _ = write!(l, "  pid d_limit   out={k:.17e}\r\n");
    say(class, &l).await?;

    let mut st = GotoController::new(ControlGains::WAYPOINT);
    let (sv, sw) = st.steer(2.5, 1.5, DT);
    l.clear();
    let _ = write!(l, "  steer         v={sv:.17e} w={sw:.17e}\r\n");
    say(class, &l).await?;

    // ---- 2. Does a control tick fit in the 20 ms budget? ----
    say(class, "\r\n[2] timing on this chip\r\n").await?;

    let mut ctrl = GotoController::new(ControlGains::WAYPOINT);
    let mut odom = Odometry {
        model: drive,
        ticks_per_rev: SPEC.ticks_per_rev,
        pose: Pose::new(1.0, 1.0, 0.0),
    };
    const ITERS: u32 = 1000;
    let t0 = Instant::now();
    for _ in 0..ITERS {
        // Exactly what pico-robot does per tick.
        odom.update(37, 41);
        let (v, w) = ctrl.goto_point(&odom.pose, (2.2, 1.0), DT);
        let (a, b) = drive.inverse(v, w);
        core::hint::black_box((a, b));
    }
    let elapsed_us = t0.elapsed().as_micros();
    let per_tick_ns = elapsed_us as f64 * 1000.0 / ITERS as f64;

    l.clear();
    let _ = write!(l, "  {ITERS} ticks in {elapsed_us} us\r\n");
    say(class, &l).await?;
    l.clear();
    let _ = write!(l, "  per tick      {per_tick_ns:.0} ns\r\n");
    say(class, &l).await?;
    l.clear();
    let _ = write!(
        l,
        "  budget used   {:.4}%  (of 20 ms)\r\n",
        per_tick_ns / (DT * 1e9) * 100.0
    );
    say(class, &l).await?;
    l.clear();
    let _ = write!(l, "  headroom      {:.0}x\r\n", (DT * 1e9) / per_tick_ns);
    say(class, &l).await?;

    say(class, "\r\n=== done ===\r\n").await?;
    Ok(())
}

/// Write a string in <=64-byte chunks — the endpoint's packet limit.
async fn say<'d, D: embassy_usb::driver::Driver<'d>>(
    class: &mut CdcAcmClass<'d, D>,
    s: &str,
) -> Result<(), embassy_usb::driver::EndpointError> {
    for chunk in s.as_bytes().chunks(64) {
        class.write_packet(chunk).await?;
    }
    Ok(())
}
