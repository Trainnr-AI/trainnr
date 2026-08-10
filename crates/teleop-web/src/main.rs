//! Drive the robot from a phone on the same WiFi.
//!
//! ```sh
//! tools/build-pico2.sh pico-odom teleop      # then BOOTSEL + picotool load
//! cargo run --release -p teleop-web -- /dev/cu.usbmodem11
//! ```
//!
//! It prints a URL. Open it on a phone on the same network.
//!
//! ```text
//!   phone browser ──WebSocket──▶ this ──USB──▶ pico-odom teleop ──▶ motors
//!                                        │
//!                              CommandWatchdog 200 ms, on the chip
//! ```
//!
//! # What makes this safe enough to hand to a phone
//!
//! Nothing in this program. The failsafe is `CommandWatchdog` in
//! `firmware/pico-odom`, which zeroes the duty **and drops STBY** 200 ms
//! after the twists stop arriving. Phone out of range, tab closed, laptop
//! asleep, this process killed, cable pulled — all the same event to the
//! chip, and all stop the wheels in hardware.
//!
//! That is why this can be a small program. It does not have to be
//! trustworthy; it has to be *unable to keep the robot moving by
//! accident*, which is a much weaker requirement and is enforced
//! elsewhere.
//!
//! # ⚠️ Opening the port starts the robot's clock
//!
//! `link::open` raises DTR, which is what un-gates the chip. From that
//! line on, a powered H-bridge can move. See `hil_protocol::link`.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::extract::ws::{Message as WsMessage, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::{Html, IntoResponse};
use axum::routing::get;
use axum::Router;
use hil_protocol::Message;
use sim_core::RobotSpec;
use teleop_web::{Pilot, Stick, SEND_HZ};

/// The page, compiled in rather than read from disk.
///
/// A file path would be one more thing that can be missing at the moment
/// someone is standing in a room holding a phone over a robot.
const PAGE: &str = include_str!("index.html");

/// Shared between the socket handlers and the serial writer.
///
/// A `std::sync::Mutex` rather than tokio's: every critical section here
/// is a few field assignments with no `.await` inside, so an async mutex
/// would add machinery to guard nothing.
type Shared = Arc<Mutex<Pilot>>;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let port = std::env::args()
        .nth(1)
        .ok_or("usage: teleop-web <serial-port>   (e.g. /dev/cu.usbmodem11)")?;

    let pilot: Shared = Arc::new(Mutex::new(Pilot::new(RobotSpec::REAL_BOT)));

    // The serial writer runs on a blocking thread rather than a tokio
    // task: `serialport` is a blocking API, and a blocking write inside
    // the async runtime would stall every other task on that worker —
    // including the socket that is the only way to say "stop".
    let writer = pilot.clone();
    std::thread::spawn(move || {
        if let Err(e) = drive_forever(&port, writer) {
            eprintln!("⚠️  serial link stopped: {e}");
            eprintln!("    the chip's watchdog stops the wheels within 200 ms");
        }
    });

    let app = Router::new()
        .route("/", get(|| async { Html(PAGE) }))
        .route("/drive", get(upgrade))
        .with_state(pilot);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await?;
    announce();
    axum::serve(listener, app).await?;
    Ok(())
}

/// Print the URL to type into a phone.
///
/// Worth the dependency: the alternative is telling someone to go and
/// find their laptop's IP while holding a robot.
fn announce() {
    match local_ip_address::local_ip() {
        Ok(ip) => println!("open  http://{ip}:8080  on a phone on this WiFi"),
        Err(_) => println!("open  http://<this-machine>:8080  on a phone on this WiFi"),
    }
    println!("⚠️  the motors can move as soon as the port opens");
}

async fn upgrade(ws: WebSocketUpgrade, State(pilot): State<Shared>) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle(socket, pilot))
}

/// One phone, for as long as it is connected.
async fn handle(mut socket: WebSocket, pilot: Shared) {
    let started = Instant::now();
    while let Some(Ok(msg)) = socket.recv().await {
        let WsMessage::Text(text) = msg else { continue };
        // A malformed frame is ignored rather than fatal, and crucially it
        // does NOT feed the watchdog — so a phone sending only garbage
        // times out exactly like a phone sending nothing.
        let Ok(stick) = serde_json::from_str::<Incoming>(&text) else {
            continue;
        };
        let now_ms = started.elapsed().as_millis() as u64;
        if let Ok(mut p) = pilot.lock() {
            p.steer(
                Stick {
                    forward: stick.forward,
                    turn: stick.turn,
                },
                now_ms,
            );
        }
    }
    // The socket closed — thumb lifted, tab shut, phone locked, or out of
    // range. All mean stop, and all are KNOWN to mean stop, so this does
    // not wait out the phone timeout.
    if let Ok(mut p) = pilot.lock() {
        p.halt();
    }
    println!("phone disconnected — commanding stop");
}

#[derive(serde::Deserialize)]
struct Incoming {
    forward: f64,
    turn: f64,
}

/// Send the current twist to the chip at [`SEND_HZ`], forever.
///
/// ⚠️ **Unconditional.** It sends whether or not a phone is connected,
/// because "stop" is a command and not an absence of one — and because a
/// chip fed a steady stream of zeros is a chip whose watchdog is being
/// exercised continuously rather than only when something goes wrong.
fn drive_forever(port: &str, pilot: Shared) -> Result<(), Box<dyn std::error::Error>> {
    let mut serial = hil_protocol::link::open(port, Duration::from_millis(200))?;

    // Read the chip's side of the conversation back.
    //
    // Not decoration: without it this program is write-only, and "the
    // joystick moved" and "the robot heard it" are indistinguishable from
    // the laptop. `duty` is the chip's own account of what it is doing
    // with the twists being sent, which is the only thing that closes the
    // loop when the robot has no wheels to watch.
    //
    // A CLONE of the handle, because macOS gives a serial port to one
    // opener — the reader cannot open the port a second time, and this
    // thread must not block the 50 Hz writer below.
    let listening = serial.try_clone()?;
    std::thread::spawn(move || report_forever(listening));

    let started = Instant::now();
    let period = Duration::from_micros(1_000_000 / SEND_HZ);
    let mut next = Instant::now();
    let mut line = String::new();
    // Consecutive writes the chip did not take. See the timeout branch.
    let mut unheard: u64 = 0;

    loop {
        next += period;
        let now_ms = started.elapsed().as_millis() as u64;
        let twist = match pilot.lock() {
            Ok(p) => p.command(now_ms),
            // A poisoned mutex means a socket handler panicked. Command
            // zero rather than reusing a stale twist: the one thing worse
            // than stopping is continuing on state nobody trusts.
            Err(_) => sim_core::BodyTwist {
                forward_speed: 0.0,
                turn_rate: 0.0,
            },
        };

        line.clear();
        // Built through `hil-protocol`, never formatted by hand — the chip
        // parses with the same crate, so the two cannot drift.
        Message::Twist {
            v: twist.forward_speed,
            w: twist.turn_rate,
        }
        .write_into(&mut line)?;
        // ⚠️ A write timeout is NOT fatal, and treating it as fatal was a
        // bug worth keeping the note for.
        //
        // Writing to a USB CDC device blocks when the firmware is not
        // draining its OUT endpoint. A firmware built WITHOUT `teleop`
        // never reads commands at all — `class.split()` leaves the
        // receiver unused — so every write fills a buffer nobody empties
        // and times out after 200 ms. Ending the thread there killed the
        // command path permanently while the page cheerfully went on
        // showing "connected".
        //
        // The chip is safe either way: no commands means the watchdog
        // stops the wheels. What was lost was any chance of recovery, and
        // any hint about why.
        if let Err(e) = std::io::Write::write_all(&mut serial, line.as_bytes()) {
            if e.kind() == std::io::ErrorKind::TimedOut {
                unheard += 1;
                // Once, then rarely — a line per failed write at 50 Hz is
                // 3,000 a minute saying the same thing.
                if unheard == 1 || unheard.is_multiple_of(SEND_HZ * 10) {
                    eprintln!(
                        "⚠️  the chip is not reading commands ({unheard} writes timed out).\n    \
                         Is it flashed with `tools/build-pico2.sh pico-odom teleop`? \
                         A plain `usb` build never drains the command endpoint."
                    );
                }
                continue;
            }
            // Anything else — unplugged, port revoked — genuinely ends it.
            return Err(e.into());
        }
        if unheard > 0 {
            println!("chip is reading commands again after {unheard} timed-out writes");
            unheard = 0;
        }

        let sleep_for = next.saturating_duration_since(Instant::now());
        std::thread::sleep(sleep_for);
    }
}

/// Print the chip's reported duty whenever it changes.
///
/// Rate-limited to changes rather than the 50 Hz the chip reports at,
/// because a line per report is 3,000 lines a minute saying nothing
/// happened — the same reasoning that makes `pico-robot` send its
/// worst-case compute time only when the record is beaten.
///
/// ⚠️ On a board with no encoders the tick counts stay at zero and the
/// chip will latch STALLED, correctly: it commanded motion and measured
/// none. That is the stall detector working, not a fault.
fn report_forever(serial: Box<dyn serialport::SerialPort>) {
    use std::io::BufRead;
    let mut reader = std::io::BufReader::new(serial);
    let mut line = String::new();
    let mut last_duty = u64::MAX;
    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => return,
            Ok(_) => {}
            // A timeout is normal — it just means the chip had nothing to
            // say yet. Only a closed port ends this loop.
            Err(_) => continue,
        }
        let Some(status) = hil_protocol::Status::parse(&line) else {
            continue;
        };
        if status.duty_percent != last_duty {
            println!(
                "chip: duty={:>3}%  ticks L={} R={}{}",
                status.duty_percent,
                status.ticks_left,
                status.ticks_right,
                if status.stalled { "  [STALLED]" } else { "" }
            );
            last_duty = status.duty_percent;
        }
    }
}
