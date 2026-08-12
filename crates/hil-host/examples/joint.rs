//! Watch one arm joint on a real board.
//!
//! ```sh
//! cargo run -p hil-host --example joint -- /dev/cu.usbmodem11
//! ```
//!
//! Reads the `J` lines `firmware/pico-arm` emits and decodes them through
//! the shared parser — deliberately not a second one. `hil-protocol`'s own
//! history is the argument: the status line that grew three independent
//! parsers, one of which drew an empty screen for hours while the gate
//! stayed green.
//!
//! # What each field answers
//!
//! ```text
//!   raw_ticks   does the encoder count AT ALL, and in which direction
//!   milliradians the same fact after the zero and the gear ratio
//!   duty        what the loop actually asked the H-bridge for
//!   phase       homing | homed | holding | nozero
//! ```
//!
//! `raw_ticks` and `milliradians` are the same quantity through two
//! conversions. When they disagree the fault is the zero, the tick count
//! or the gear ratio — which is exactly what a homing bug looks like.

use std::io::{BufRead, BufReader};
use std::time::{Duration, Instant};

use hil_protocol::{JointPhase, Message};

/// Joints tracked separately. Six is `arm::MAX_JOINTS`; anything beyond
/// is folded into the last slot rather than dropped.
const SLOTS: usize = 6;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let port = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: joint <serial-port> [seconds]");
        std::process::exit(2);
    });
    let seconds: u64 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);

    // `--replay <file>` reads a capture instead of a port, so this doubles
    // as a gate step: no hardware, no viewer, and a hardware session that
    // stops parsing becomes a build failure rather than a surprise.
    let replay = port == "--replay";
    let source = if replay {
        std::env::args().nth(2).unwrap_or_default()
    } else {
        port.clone()
    };
    let reader: Box<dyn BufRead> = if replay {
        Box::new(BufReader::new(std::fs::File::open(&source)?))
    } else {
        // ⚠️ Opening raises DTR, which is what un-gates the firmware's
        // reports. See `hil_protocol::link::open`.
        Box::new(BufReader::new(hil_protocol::link::open(
            &source,
            Duration::from_millis(500),
        )?))
    };
    let mut lines = reader.lines();
    println!("watching {source}");

    let deadline = Instant::now() + Duration::from_secs(if replay { 3600 } else { seconds });
    let (mut seen, mut unparsed) = (0u32, 0u32);
    let mut phase_now: Option<JointPhase> = None;
    let mut first_ticks: [Option<i64>; SLOTS] = [None; SLOTS];
    let mut last_ticks = [0i64; SLOTS];

    while Instant::now() < deadline {
        // ⚠️ `continue` here spun for the whole deadline at end of file:
        // on a serial port a read that yields nothing means "wait", and on
        // a file it means "finished". The same expression, two meanings,
        // and only one of them was written down.
        let Some(next) = lines.next() else { break };
        let Ok(line) = next else { continue };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match Message::parse(line) {
            Ok(Message::Joint {
                index,
                raw_ticks,
                milliradians,
                duty,
                phase,
            }) => {
                seen += 1;
                // ⚠️ Per joint. Printing a shared column made two joints
                // holding equal-and-opposite duties look like ONE joint
                // oscillating — a reader that hides the index invents a
                // fault that is not there.
                let slot = usize::from(index).min(SLOTS - 1);
                first_ticks[slot].get_or_insert(raw_ticks);
                last_ticks[slot] = raw_ticks;
                if phase_now != Some(phase) {
                    println!("  → {phase}");
                    phase_now = Some(phase);
                }
                if seen % 10 < 2 {
                    println!(
                        "    joint {index}  ticks={raw_ticks:<8} mrad={milliradians:<7} duty={duty}"
                    );
                }
            }
            Ok(other) => println!("  (non-joint message: {})", other.tag()),
            Err(_) => unparsed += 1,
        }
    }

    println!("\n{seen} joint reports, {unparsed} unparsable");
    for slot in 0..SLOTS {
        if let Some(first) = first_ticks[slot] {
            println!(
                "  joint {slot} moved {} ticks while watching",
                last_ticks[slot] - first
            );
        }
    }
    match phase_now {
        None => println!("NOTHING ARRIVED — check the port, and that DTR un-gated it"),
        Some(JointPhase::NoZero) => {
            println!("phase: nozero — no zero was ever adopted, torque is OFF");
        }
        Some(JointPhase::Held) => {
            println!("phase: held — the guard refused; joints keep their last target");
            println!("the calibration is fine, the COMMANDER is not");
        }
        Some(phase) => println!("phase: {phase}"),
    }
    Ok(())
}
