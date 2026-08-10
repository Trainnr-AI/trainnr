//! Conformance checks that can only be run against a **real chip**.
//!
//! ```sh
//! cargo run -p hil-host --example chip_probe -- /dev/cu.usbmodem11
//! ```
//!
//! # Why this exists separately from `tools/verify.sh`
//!
//! The firmware carries no host tests — it is `no_std`, builds for another
//! CPU, and cannot run `cargo test`. `tools/verify.sh --serial` covers the
//! happy path by flying the whole mission, but the interesting failures are
//! the ones a *successful* mission never exercises: a corrupt command, and
//! a host that dies and reconnects.
//!
//! Both of the properties below were **broken on hardware** when this file
//! was written, and neither was visible from the laptop.

use std::io::{BufRead, BufReader, Write};
use std::time::{Duration, Instant};

type Port = BufReader<Box<dyn serialport::SerialPort>>;

/// Read whatever the chip says for `ms`, then stop. Never blocks longer.
fn listen(reader: &mut Port, ms: u64) -> Vec<String> {
    let deadline = Instant::now() + Duration::from_millis(ms);
    let mut lines = Vec::new();
    let mut line = String::new();
    while Instant::now() < deadline {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) => lines.push(line.trim_end().to_string()),
        }
    }
    lines
}

fn said(lines: &[String], tag: char) -> bool {
    lines.iter().any(|l| l.starts_with(tag))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let port = std::env::args()
        .nth(1)
        .ok_or("usage: chip_probe <serial-port>")?;
    let open = || -> Result<(Box<dyn serialport::SerialPort>, Port), Box<dyn std::error::Error>> {
        let sp = hil_protocol::link::open(&port, Duration::from_millis(1200))?;
        let reader = BufReader::new(sp.try_clone()?);
        Ok((sp, reader))
    };

    let mut failures = 0;
    let mut check = |name: &str, ok: bool| {
        println!("  {:<52} {}", name, if ok { "ok" } else { "FAIL" });
        if !ok {
            failures += 1;
        }
    };

    // ---- 1. a session starts, and the chip acknowledges ----
    let (mut w, mut r) = open()?;
    writeln!(w, "I 1.0000 3.0000 0.0000")?;
    w.flush()?;
    check(
        "session start is acknowledged with a pose",
        said(&listen(&mut r, 800), 'P'),
    );

    // ---- 2. a corrupt command is ignored, not obeyed ----
    //
    // `"NaN".parse::<f64>()` succeeds, so this used to parse as a valid
    // directive and poison the PID's integral permanently — every later
    // tick output NaN however good its input, and the robot silently
    // stopped steering while still reporting healthy.
    writeln!(w, "G NaN NaN NaN")?;
    w.flush()?;
    check(
        "a NaN goal produces no motor command",
        !said(&listen(&mut r, 800), 'M'),
    );

    writeln!(w, "G 6.5000 3.0000 4.4000")?;
    w.flush()?;
    let after = listen(&mut r, 800);
    check("a good goal still answers afterwards", said(&after, 'M'));

    // ---- 3. reconnecting mid-tick does not wedge the link ----
    //
    // The chip is now waiting for the `S` that answers the `M` above —
    // exactly the state a host leaves it in when it crashes or is
    // Ctrl-C'd. It used to wait there forever, discarding the next host's
    // `I`, and the link stayed dead until someone power-cycled the board.
    drop(w);
    drop(r);
    let (mut w2, mut r2) = open()?;
    writeln!(w2, "I 1.0000 3.0000 0.0000")?;
    w2.flush()?;
    check(
        "a new session supersedes an abandoned tick",
        said(&listen(&mut r2, 1200), 'P'),
    );

    println!("\n{} checks failed", failures);
    if failures > 0 {
        std::process::exit(1);
    }
    Ok(())
}
