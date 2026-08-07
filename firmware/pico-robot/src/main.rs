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
use embassy_time::{Duration, Instant};
use hil_protocol::{LineReader, Message};
use panic_halt as _;
use sim_core::{CommandWatchdog, ControlGains, GotoController, Odometry, Pose, RobotSpec};

/// Control period — must match the host's physics step.
const DT: f64 = 0.02; // 50 Hz

/// Stop if the planner goes quiet for this long.
///
/// Ten missed ticks at 50 Hz, and about 9 cm of travel at `v_max` — well
/// beyond any real scheduling hiccup, well short of a table edge.
const COMMAND_TIMEOUT_MS: u64 = 200;

/// How long a read waits before coming up for air to check the watchdog.
///
/// Shorter than [`COMMAND_TIMEOUT_MS`], so staleness is noticed within one
/// poll of becoming true. Whether a transport can honour it is the
/// transport's business — see [`Link::recv`].
const POLL: Duration = Duration::from_millis(50);

/// Milliseconds since boot, the unit [`CommandWatchdog`] speaks.
fn now_ms() -> u64 {
    Instant::now().as_millis()
}
/// The robot this firmware is driving. `REAL_BOT` — not `SIM_BOT` —
/// because this code runs on the physical machine: when the measured
/// values land in `spec.rs`, they take effect here with no edit.
const SPEC: RobotSpec = RobotSpec::REAL_BOT;

/// Steering profile — shared, not defined here.
///
/// This used to be written out inline, which meant `RobotSpec::check`
/// never saw it and no test could reach it. It now lives beside the other
/// profiles in `sim-core`, where `every_shipped_profile_is_physically_
/// achievable` validates it against both robot specs.
const GAINS: ControlGains = ControlGains::HIL;

// The waypoint list that used to live here is gone. The chip is a
// CONTROLLER, not a planner: the host senses, maps, runs A* and sends a
// single point to steer at (`G` in hil-protocol). That is the two-tier
// split docs/00 describes, and it is why the chip could never have solved
// the U-trap on its own — it has no allocator, so no map and no A*.

// The local `Command` enum that used to live here is gone. It was a
// third copy of the host→chip vocabulary — `Mission::decide` had one,
// `hil-host` built messages from a second, this matched on a third — and
// the digital-twin property depended on all three agreeing by hand.
// `sim_core::Directive` is now the single definition, and
// `Message::directive()` the single translation from the wire.

/// A byte pipe to the host. The whole difference between running against
/// the emulator and running on real hardware lives behind these two
/// methods.
trait Link {
    /// Send bytes. Partial writes are the implementation's problem.
    async fn send(&mut self, bytes: &[u8]);

    /// Receive whatever has arrived, into `buf`. Returns how many bytes,
    /// possibly 0.
    ///
    /// `poll` is how long to wait before returning empty-handed so the
    /// caller can check its watchdog. **A transport that cannot honour it
    /// may block for longer** — see each impl.
    ///
    /// The deadline lives here, in the transport, rather than being
    /// wrapped around `recv` by the control loop. That was the first
    /// attempt and it was wrong twice over: on USB it belongs *inside* the
    /// driver's cancel-safe read, and on the blocking UART it cannot work
    /// at all — `blocking_read` never yields, so the timer is pure
    /// overhead. Measured: wrapping the UART path in `with_timeout`
    /// registered an emulated alarm per poll and dropped the emulator
    /// from ~1100 ticks to fewer than 50 in 90 seconds.
    async fn recv(&mut self, buf: &mut [u8], poll: Duration) -> usize;
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
    // Starts stale — never fed is not "no news is good news".
    let mut watchdog = CommandWatchdog::new(COMMAND_TIMEOUT_MS);

    loop {
        // ---- 1. wait for the host to say where to aim ----
        let command = loop {
            // Bounded by the transport, not by a timer wrapped around it.
            let n = link.recv(&mut rx, POLL).await;
            let mut found = None;
            for &b in &rx[..n] {
                if let Some(line) = reader.push(b) {
                    // Telemetry parses fine and yields `None` here, so the
                    // chip cannot be commanded by a stray status line.
                    if let Some(d) = Message::parse(line).ok().and_then(|m| m.directive()) {
                        found = Some(d);
                    }
                }
            }
            if let Some(g) = found {
                watchdog.feed(now_ms());
                break g;
            }
            if watchdog.is_stale(now_ms()) {
                // The planner is gone. Nothing is sent — the host is not
                // waiting on us, and injecting an unrequested `M` line
                // would be read as the answer to its *next* goal.
                //
                // ⚠️ AT H4 THIS IS WHERE THE H-BRIDGE GETS WRITTEN TO
                // ZERO. Today the host owns the motors, so the failsafe
                // that matters here is the gate below: any duty we go on
                // to report while stale is already zeroed.
                controller.reset();
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
        // Literally the same call `Mission::decide` makes on the laptop.
        // Not "the same algorithm" — the same function.
        let (v, w) = controller.execute(command, &pose, DT);
        // Scale, don't clip — a saturated turn keeps its arc (docs/13).
        let (wl, wr) = SPEC.fit_wheels_of(SPEC.drive().inverse(v, w));
        // THE FAILSAFE. Everything the robot does physically passes through
        // this line. While the planner is fresh it is the identity; once it
        // goes quiet the duty is zero, whatever the controller computed.
        let (duty_l, duty_r) = watchdog.gate(now_ms(), (SPEC.duty(wl), SPEC.duty(wr)));
        out.clear();
        let _ = Message::Motor { duty_l, duty_r }.write_into(&mut out);
        link.send(out.as_bytes()).await;

        // ---- 4. wait for the host's physics update ----
        //
        // # This wait must NOT be abandoned, and that is a protocol fact
        //
        // The link is strictly lockstep: `G` → `P`+`M` → `S`. We have
        // already sent `M`, so the host is committed to sending `S`.
        // Giving up here and looping back to wait for a `G` **deadlocks**:
        // the host is waiting for our `M` (already sent, so it never comes
        // again) and we are waiting for a `G` (already sent, likewise).
        //
        // That is not hypothetical — a `COMMAND_TIMEOUT_MS` deadline here
        // hung the emulator run indefinitely. A bounded read protects
        // against a dead host by breaking a live one, which is the worse
        // trade.
        //
        // So: poll forever, but keep checking staleness. The failsafe is
        // not "stop waiting", it is "stop driving" — and that distinction
        // is the whole reason `gate` sits on the output rather than here.
        'wait: loop {
            let n = link.recv(&mut rx, POLL).await;
            if n == 0 {
                // ⚠️ AT H4 THIS IS A MOTOR-OFF SITE. The wheels are
                // turning on the duty just reported and the planner has
                // gone quiet; the H-bridge must be written to zero here
                // while we keep waiting for it to come back.
                continue;
            }
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

// A heartbeat LED task used to live here, driving GP25. It never once
// blinked: on a Pico **W** board GP25 is the CYW43 wireless chip's
// **chip-select**, not an LED — `firmware/pico-led` says so in as many
// words. So it toggled an unpowered radio's CS line at 2.5 Hz while
// looking, in the source, exactly like a liveness indicator.
//
// That is worse than no LED. Anyone debugging a wedged robot looks for a
// blinking light, does not find one, and concludes the firmware crashed
// when it is running fine.
//
// A real status light on a W board costs 231 KB of CYW43 firmware, a PIO
// block and a background task (see `pico-led`) — absurd for a heartbeat.
// When the robot is assembled, wire an LED to a spare GPIO and bring this
// back honestly.

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

        /// **Ignores `poll` and blocks.** `blocking_read` busy-waits with
        /// no await inside, so there is nothing for a timer to interrupt.
        ///
        /// # Why not `BufferedUart`, which would fix this properly
        ///
        /// It is the right answer and it does not run on our emulator.
        /// `BufferedUart` services the hardware FIFO from an interrupt,
        /// which would remove both this limitation *and* the FIFO
        /// constraint described on `Message` — but rp2040js's UART raises
        /// `UARTTXINTR` unconditionally inside its own `checkInterrupts`
        /// (a `TODO` in its source), so clearing the interrupt re-asserts
        /// it immediately. Measured: an interrupt storm, 0 ticks in 210 s.
        ///
        /// Acceptable because it matches the hazard: this transport exists
        /// for the emulator, which has no motors and a host that is always
        /// alive. Real hardware uses USB, which honours the poll.
        /// **At H4, a UART-driven robot must use `BufferedUart`** — a
        /// failsafe that cannot fire is worse than none, because the code
        /// reads as though the robot is protected.
        async fn recv(&mut self, buf: &mut [u8], _poll: Duration) -> usize {
            // One byte at a time: the bridge is line-oriented and this
            // keeps the reader logic identical for both links.
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

    pub async fn run(p: embassy_rp::Peripherals) -> ! {
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
    use embassy_time::with_timeout;
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
        /// **Honours `poll`.** This is the transport that reaches real
        /// motors, so this is the one that has to work.
        ///
        /// Cancelling is safe here, checked in `embassy-rp`'s source
        /// rather than assumed: the endpoint read is a `poll_fn` that only
        /// registers a waker and tests a hardware bit, and every side
        /// effect — copying the packet out, re-arming the endpoint —
        /// happens after the await with no further await points. Dropping
        /// it while `Pending` consumes nothing, so no byte is lost and the
        /// protocol cannot desync.
        async fn recv(&mut self, buf: &mut [u8], poll: Duration) -> usize {
            match with_timeout(poll, self.0.read_packet(buf)).await {
                Ok(Ok(n)) => n,
                Ok(Err(_)) | Err(_) => 0,
            }
        }
    }

    pub async fn run(p: embassy_rp::Peripherals) -> ! {
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
async fn main(_spawner: Spawner) {
    let p = embassy_rp::init(Default::default());
    transport::run(p).await
}
