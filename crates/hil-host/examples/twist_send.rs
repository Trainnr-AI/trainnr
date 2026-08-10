//! Drive the real motors from the laptop, and prove the failsafe stops them.
//!
//! ```sh
//! # creep forward for 2 s, then go silent and watch the chip stop itself
//! cargo run -p hil-host --example twist_send -- /dev/cu.usbmodem11 0.10 0.0 2
//!
//! # spin on the spot
//! cargo run -p hil-host --example twist_send -- /dev/cu.usbmodem11 0.0 1.5 2
//! ```
//!
//! Needs `firmware/pico-odom` built with `teleop`:
//!
//! ```sh
//! tools/build-pico2.sh pico-odom teleop
//! picotool load -x firmware/pico-odom/pico-odom-pico2.uf2   # after BOOTSEL
//! ```
//!
//! # What this is for
//!
//! It is the step before the camera. The camera→motors loop has two
//! risky halves — a control law that produces twists, and a chip that
//! turns twists into motion without running away. **This proves the
//! second half on its own**, with a human choosing the numbers, so that
//! when the camera is attached there is only one new thing in the room.
//!
//! # The failsafe is the point, not the motion
//!
//! Any program can make a motor turn. The property worth testing is what
//! happens when the program **stops**: this one deliberately falls silent
//! after `seconds` and does *not* send a zero twist. If the wheels keep
//! turning, `CommandWatchdog` is not wired up and the robot has no
//! failsafe — which is exactly the bug that is invisible while everything
//! is working.
//!
//! So: watch the wheels after the countdown ends. They must stop within
//! `COMMAND_TIMEOUT_MS` (200 ms) *without being told to*.

use hil_protocol::Message;
use std::io::Write;
use std::time::{Duration, Instant};

/// Command rate. Matches the chip's 50 Hz control tick, so the watchdog
/// is fed roughly ten times per timeout — jitter cannot trip it, and a
/// genuine stop is caught within 200 ms.
const HZ: u64 = 50;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: twist_send <serial-port> <forward_m_s> <turn_rad_s> <seconds>";
    let (Some(port), Some(forward), Some(turn), Some(seconds)) =
        (args.next(), args.next(), args.next(), args.next())
    else {
        return Err(usage.into());
    };
    let forward_speed: f64 = forward.parse()?;
    let turn_rate: f64 = turn.parse()?;
    let seconds: f64 = seconds.parse()?;

    let mut serial = hil_protocol::link::open(&port, Duration::from_millis(200))?;

    // ⚠️ Opening the port raises DTR, which is what un-gates the chip.
    // From this line on, the motors can move.
    println!("{port}: T {forward_speed} {turn_rate} at {HZ} Hz for {seconds}s");
    println!("then SILENCE — the wheels must stop on their own within 200 ms");

    let mut line = String::new();
    let started = Instant::now();
    let period = Duration::from_micros(1_000_000 / HZ);
    let mut next = Instant::now();

    while started.elapsed().as_secs_f64() < seconds {
        line.clear();
        // Built through `hil-protocol`, not formatted by hand. The chip
        // parses with the same crate, so the two cannot drift — which is
        // the failure `examples/odom_view.rs` documents from the one
        // format in this system that is NOT shared.
        Message::Twist {
            v: forward_speed,
            w: turn_rate,
        }
        .write_into(&mut line)?;
        serial.write_all(line.as_bytes())?;
        serial.flush()?;

        next += period;
        if let Some(nap) = next.checked_duration_since(Instant::now()) {
            std::thread::sleep(nap);
        }
    }

    // Deliberately no zero twist, no close, no goodbye. This is a host
    // that died, and the chip has to notice by itself.
    println!("silent now — watch the wheels");
    std::thread::sleep(Duration::from_millis(1500));
    println!("if they are still turning, the watchdog is not wired up");
    Ok(())
}
