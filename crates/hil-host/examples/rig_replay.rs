//! Read a rig recording with no viewer, and say exactly what was in it.
//!
//! ```sh
//! cargo run -p hil-host --example rig_replay -- recordings/chase-brightness.wire
//! ```
//!
//! # Why a viewer-less twin of `rig_view` exists
//!
//! The gate must run headless, and `rig_view` spawns Rerun — so the
//! recordings the rig makes could not pin anything until now. This
//! reader consumes the same shared vocabulary (`Status`, `thumbnail`,
//! `arm_pulses`) and reduces a session to one summary line the gate can
//! compare against the counts written beside each fixture. A parser
//! change that breaks any of the three formats stops being a surprise
//! in the viewer and becomes a red build.
//!
//! # What counts as failure
//!
//! `unparsable` — a non-empty line no format claims. A healthy recording
//! has zero, and the gate pins that. `torn` — an image whose header
//! arrived but whose rows never completed; real captures may carry one
//! at end-of-file where the recording stopped mid-picture, so the gate
//! pins the *count* rather than demanding zero.

use std::io::{BufRead, BufReader};

use hil_protocol::{arm_pulses, thumbnail, Status};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: rig_replay <file.wire>");
        std::process::exit(2);
    });
    let reader = BufReader::new(std::fs::File::open(&path)?);

    let (mut statuses, mut stalled) = (0u64, 0u64);
    let (mut images, mut torn) = (0u64, 0u64);
    let (mut servo_lines, mut notes, mut unparsable) = (0u64, 0u64, 0u64);
    let mut expecting: Option<(u32, u32, u64)> = None; // width, height, pixels so far

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        if let Some((width, height)) = thumbnail::parse_header(line) {
            // A new header while a picture is open means the old one
            // never finished.
            if expecting.is_some() {
                torn += 1;
            }
            expecting = Some((width, height, 0));
            continue;
        }
        if let Some((width, height, pixels)) = expecting {
            if let Some(row) = thumbnail::parse_row(line, width) {
                let pixels = pixels + row.count() as u64;
                expecting = if pixels >= u64::from(width) * u64::from(height) {
                    images += 1;
                    None
                } else {
                    Some((width, height, pixels))
                };
                continue;
            }
        }

        if let Some(payload) = line.strip_prefix("# ") {
            if arm_pulses::parse_note(payload).is_some() {
                servo_lines += 1;
            } else {
                notes += 1;
            }
            continue;
        }

        match Status::parse(line) {
            Some(report) => {
                statuses += 1;
                if report.stalled {
                    stalled += 1;
                }
            }
            None => unparsable += 1,
        }
    }
    if expecting.is_some() {
        torn += 1;
    }

    // One line, exact, greppable — the gate pins these numbers.
    println!(
        "{statuses} status ({stalled} stalled), {images} images ({torn} torn), \
         {servo_lines} servo, {notes} notes, {unparsable} unparsable"
    );
    // Unparsable lines are a failing exit as well as a count: a fixture
    // that stops parsing should fail even if nobody greps carefully.
    if unparsable > 0 {
        std::process::exit(1);
    }
    Ok(())
}
