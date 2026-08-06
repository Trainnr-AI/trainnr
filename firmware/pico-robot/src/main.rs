//! LEVEL 3 — the robot's entire brain, running on the chip.
//!
//! This firmware is a complete autonomy stack with no idea it is driving a
//! simulated robot. Every tick it:
//!
//!   1. BLOCKS until the host says where to steer   (G line)
//!   2. reports its believed pose                   (P line, for the viewer)
//!   3. computes and sends motor commands           (M line)
//!   4. BLOCKS until the host reports encoder ticks (S line)
//!   5. updates its belief from those ticks
//!
//! It holds no map and no waypoint list. The host plans; this controls.
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
//! side owns *decisions*.
//!
//! # Two transports, one control loop
//!
//! ```text
//!   default          UART on GP0/GP1   → the rp2040js emulator
//!   --features usb   USB CDC serial    → a REAL Pico on a USB cable
//! ```
//!
//! The decision logic is written **once**, in [`control_loop`], and takes a
//! [`Link`]. Only the plumbing differs. That is the same discipline as
//! `sim-core` itself: when a second copy is about to appear, give the
//! shared thing a home instead.
//!
//! Protocol: docs/10-hil-protocol.md — encoded and parsed by the shared
//! `hil-protocol` crate, so the host cannot drift from the chip.

#![no_std]
#![no_main]
#![allow(async_fn_in_trait)]

use embassy_executor::Spawner;
use embassy_rp::gpio::{Level, Output};
use embassy_time::Timer;
use hil_protocol::{LineReader, Message};
use panic_halt as _;
use sim_core::exercises::shortest_turn;
// `Float` gives f64 its trig methods on bare metal, where they come from
// libm rather than from std.
use sim_core::{ControlGains, Float as _, GotoController, Odometry, Pose, RobotSpec};

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

// The waypoint list that used to live here is gone. The chip is a
// CONTROLLER, not a planner: the host senses, maps, runs A* and sends a
// single point to steer at (`G` in hil-protocol). That is the two-tier
// split docs/00 describes, and it is why the chip could never have solved
// the U-trap on its own — it has no allocator, so no map and no A*.

/// What the host asked for this tick.
#[derive(Clone, Copy)]
enum Command {
    /// Steer at this point, at most this fast. The chip's controller runs.
    Steer((f64, f64), f64),
    /// Apply this body twist verbatim — a host-side reflex.
    Twist(f64, f64),
}

/// A byte pipe to the host. The whole difference between running against
/// the emulator and running on real hardware lives behind these two
/// methods.
trait Link {
    /// Send bytes. Partial writes are the implementation's problem.
    async fn send(&mut self, bytes: &[u8]);
    /// Receive whatever has arrived, into `buf`. Returns how many bytes.
    /// May return 0.
    async fn recv(&mut self, buf: &mut [u8]) -> usize;
}

/// The robot's brain. Transport-agnostic on purpose — this is the code we
/// care about, and it must not be duplicated per wire.
async fn control_loop<L: Link>(link: &mut L) -> ! {
    let mut odom = Odometry {
        model: SPEC.drive(),
        ticks_per_rev: SPEC.ticks_per_rev,
        pose: Pose::new(1.0, 3.0, 0.0), // told where it starts, once
    };
    let mut controller = GotoController::new(GAINS);

    let mut out: heapless::String<96> = heapless::String::new();
    let mut reader: LineReader<64> = LineReader::new();
    let mut rx = [0u8; 64];

    loop {
        // ---- 1. wait for the host to say where to aim ----
        let command = loop {
            let n = link.recv(&mut rx).await;
            let mut found = None;
            for &b in &rx[..n] {
                if let Some(line) = reader.push(b) {
                    match Message::parse(line) {
                        Ok(Message::Goal { x, y, budget }) => {
                            found = Some(Command::Steer((x, y), budget))
                        }
                        Ok(Message::Twist { v, w }) => found = Some(Command::Twist(v, w)),
                        _ => {}
                    }
                }
            }
            if let Some(g) = found {
                break g;
            }
        };

        let pose = odom.pose;

        // ---- 2. report the belief (visualization only) ----
        out.clear();
        let _ = Message::Pose {
            x: pose.x,
            y: pose.y,
            theta: pose.theta,
        }
        .write_into(&mut out);
        link.send(out.as_bytes()).await;

        // ---- 3. decide ----
        let (v, w) = match command {
            // The shared steering law, identical to the simulator's.
            //
            // `steer`, not `goto_point`: the host's point is a LOOKAHEAD
            // and is deliberately close, so deriving speed from it would
            // make the robot crawl. The budget comes down the wire —
            // which is exactly why `steer` takes one.
            Command::Steer(target, budget) => {
                let bearing = (target.1 - pose.y).atan2(target.0 - pose.x);
                let error = shortest_turn(pose.theta, bearing);
                controller.steer(error, pose.theta, budget, DT)
            }
            // A reflex the host computed from sensors we do not have.
            // Obey it, and drop any accumulated PID state so the next
            // Steer does not inherit an integral from before the swerve.
            Command::Twist(v, w) => {
                controller.reset();
                (v, w)
            }
        };
        // Scale, don't clip — a saturated turn keeps its arc (docs/13).
        let (wl, wr) = SPEC.fit_wheels_of(SPEC.drive().inverse(v, w));
        out.clear();
        let _ = Message::Motor {
            duty_l: SPEC.duty(wl),
            duty_r: SPEC.duty(wr),
        }
        .write_into(&mut out);
        link.send(out.as_bytes()).await;

        // ---- 4. block for the host's physics update ----
        'wait: loop {
            let n = link.recv(&mut rx).await;
            for &b in &rx[..n] {
                if let Some(line) = reader.push(b) {
                    if let Ok(Message::Sensors { dl, dr }) = Message::parse(line) {
                        // ---- 5. update belief (Stage 0's odometry) ----
                        odom.update(dl, dr);
                        break 'wait;
                    }
                }
            }
        }
    }
}

#[embassy_executor::task]
async fn heartbeat(mut led: Output<'static>) {
    loop {
        led.toggle();
        Timer::after_millis(400).await;
    }
}

// ---------------------------------------------------------------------
// Transport A — UART on GP0/GP1. What the emulator speaks.
// ---------------------------------------------------------------------
#[cfg(not(feature = "usb"))]
mod transport {
    use super::*;
    use embassy_rp::uart::{Blocking, Config as UartConfig, Uart};

    pub struct UartLink(pub Uart<'static, Blocking>);

    impl Link for UartLink {
        async fn send(&mut self, bytes: &[u8]) {
            let _ = self.0.blocking_write(bytes);
        }
        async fn recv(&mut self, buf: &mut [u8]) -> usize {
            // One byte at a time: the emulator's bridge is line-oriented
            // and this keeps the reader logic identical for both links.
            let mut one = [0u8; 1];
            match self.0.blocking_read(&mut one) {
                Ok(()) => {
                    buf[0] = one[0];
                    1
                }
                Err(_) => 0,
            }
        }
    }

    pub async fn run(p: embassy_rp::Peripherals, spawner: Spawner) -> ! {
        if let Ok(t) = heartbeat(Output::new(p.PIN_25, Level::Low)) {
            spawner.spawn(t);
        }
        let uart = Uart::new_blocking(p.UART0, p.PIN_0, p.PIN_1, UartConfig::default());
        let mut link = UartLink(uart);
        control_loop(&mut link).await
    }
}

// ---------------------------------------------------------------------
// Transport B — USB CDC serial. What a real Pico on a cable speaks.
// ---------------------------------------------------------------------
#[cfg(feature = "usb")]
mod transport {
    use super::*;
    use embassy_futures::join::join;
    use embassy_rp::bind_interrupts;
    use embassy_rp::peripherals::USB;
    use embassy_rp::usb::{Driver, InterruptHandler};
    use embassy_usb::class::cdc_acm::{CdcAcmClass, State};
    use embassy_usb::driver::Driver as UsbDriver;
    use embassy_usb::{Builder, Config};
    use static_cell::StaticCell;

    bind_interrupts!(struct Irqs {
        USBCTRL_IRQ => InterruptHandler<USB>;
    });

    pub struct UsbLink<'d, D: UsbDriver<'d>>(pub CdcAcmClass<'d, D>);

    impl<'d, D: UsbDriver<'d>> Link for UsbLink<'d, D> {
        async fn send(&mut self, bytes: &[u8]) {
            // 64 is the endpoint's max packet size; longer writes must be
            // split or the transfer is rejected.
            for chunk in bytes.chunks(64) {
                let _ = self.0.write_packet(chunk).await;
            }
        }
        async fn recv(&mut self, buf: &mut [u8]) -> usize {
            self.0.read_packet(buf).await.unwrap_or(0)
        }
    }

    pub async fn run(p: embassy_rp::Peripherals, spawner: Spawner) -> ! {
        if let Ok(t) = heartbeat(Output::new(p.PIN_25, Level::Low)) {
            spawner.spawn(t);
        }

        let driver = Driver::new(p.USB, Irqs);
        let mut config = Config::new(0x2e8a, 0x000a);
        config.manufacturer = Some("robotiq");
        config.product = Some("pico-robot");
        config.serial_number = Some("1");
        config.max_power = 100;
        config.max_packet_size_0 = 64;

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
            &mut [],
            CONTROL_BUF.init([0; 64]),
        );
        let class = CdcAcmClass::new(&mut builder, state, 64);
        let mut usb = builder.build();

        let mut link = UsbLink(class);
        // The USB stack and the control loop must both run. `join` polls
        // them on one stack — the control loop is the one that never
        // returns, so this never returns either.
        join(usb.run(), async {
            link.0.wait_connection().await;
            control_loop(&mut link).await
        })
        .await;
        // `join` on a `!` future is unreachable, but the compiler wants a
        // value for the `-> !` signature.
        loop {
            embassy_time::Timer::after_secs(1).await;
        }
    }
}

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    let p = embassy_rp::init(Default::default());
    transport::run(p, spawner).await
}
