//! Watch the chip's dead reckoning in Rerun, live — over a cable, over the
//! radio, or over both at once so they can be compared.
//!
//! ```sh
//! cargo run -p hil-host --example odom_view -- /dev/cu.usbmodem11
//! cargo run -p hil-host --example odom_view -- --udp
//! cargo run -p hil-host --example odom_view -- /dev/cu.usbmodem11 --udp
//! ```
//!
//! `firmware/pico-odom` computes a pose on the microcontroller using
//! `sim-core`'s `Odometry` — the same integrator the simulator runs — and
//! sends one line every `REPORT_MS` (20 ms). This turns that text into the
//! same picture the other runners draw, so a real robot's belief can be
//! watched the way a simulated one is.
//!
//! # Why a viewer rather than a terminal
//!
//! Reading `th=-0.720` tells you a number changed. Watching the trail
//! curve tells you the *kinematics* are right — that one wheel produces an
//! arc and two produce a line — which is the property under test before
//! any dimension has been measured. This project has already learned once
//! that a bug can be invisible in a passing test and obvious on a screen
//! (docs/07, the blank duty panels).
//!
//! # Running both sources at once
//!
//! Built with `--features wifi,usb`, the firmware sends every status line
//! down **both** wires. Give this both flags and it draws them as two
//! trails and plots what each one delivered.
//!
//! The comparison only means anything because both copies carry the same
//! [`hil_protocol::Status::seq`], written by the same loop iteration. That
//! makes three things measurable that no pair of separate runs could
//! establish:
//!
//! ```text
//!   loss     a sequence number one wire delivered and the other did not
//!   lag      for a line BOTH delivered, how much later the radio arrived
//!   jitter   the spread of gaps between consecutive arrivals
//! ```
//!
//! Lag is the one to watch. A radio that delivers everything 200 ms late
//! is useless for control and fine for logging, and those two verdicts are
//! indistinguishable from a delivery count alone.
//!
//! # Colours match the rest of the repo
//!
//! Blue is *belief* everywhere here — `sim-run` and `hil-host` both draw
//! the odometry estimate blue and ground truth yellow. There is no ground
//! truth on a bench, so everything drawn here is blue on purpose: it is a
//! reminder that dead reckoning is a claim, not a measurement. The radio's
//! trail is drawn amber only to tell the two transports apart; it is not a
//! second opinion about where the robot is, it is the same opinion that
//! took a different road.
//!
//! # ⚠️ This file is coupled to a format string in another crate
//!
//! The firmware's writer and [`Report::parse`] are two halves of one wire
//! format. On 2026-08-09 the firmware split its error counter into
//! `errL`/`errR` and this parser kept looking for `err`, so **every line
//! was rejected and the viewer drew nothing** — with `tools/verify.sh`
//! green at 25/25 throughout, because a parser that agrees with itself
//! compiles fine.
//!
//! Sharing the type with the firmware that writes it fixes the class:
//! `hil-protocol` round-trips writer against parser, so a renamed field is
//! a failing test rather than a blank screen.

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::net::UdpSocket;
use std::sync::mpsc::{channel, Sender};
use std::time::{Duration, Instant};

/// One status line, as the shared [`hil_protocol::Status`] type.
///
/// **This file used to own a parser.** On 2026-08-09 the firmware split
/// its error counter per wheel, this parser kept looking for the old key,
/// and every line was silently dropped — an empty viewer with
/// `tools/verify.sh` green at 25/25, because a parser that agrees with
/// itself compiles fine.
use hil_protocol::Status as Report;

/// Status lines per second, i.e. `1000 / REPORT_MS` in
/// `firmware/pico-odom`. Measured rather than assumed: 300 lines arrived
/// in 6.0 s off the bench board on 2026-08-10.
const REPORTS_PER_SECOND: u64 = 50;

/// Where `firmware/pico-odom --features wifi` broadcasts. One constant on
/// each side, deliberately not configurable: a port that can disagree is
/// one more pair of facts nobody compares.
const TELEMETRY_PORT: u16 = 9870;

/// How many recent sequence numbers to keep while waiting for the other
/// transport's copy to show up.
///
/// A line is only counted as "delivered by one and not the other" once it
/// is this far in the past, so a radio running slightly behind the cable
/// is measured as *late* rather than miscounted as *lost*. Two seconds of
/// reports is far beyond any plausible lag and still bounded.
const MATCH_WINDOW: u64 = REPORTS_PER_SECOND * 2;

/// How long every source may go quiet before the run is declared over and
/// the summary printed. Generous, because a board mid-reflash or a robot
/// waiting for someone to turn a wheel is not a finished run.
const QUIET_BEFORE_GIVING_UP: Duration = Duration::from_secs(5);

/// Which wire a line came down. The whole point of the comparison is that
/// this is the *only* thing that differs between two copies of a report.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Wire {
    Usb,
    Wifi,
}

impl Wire {
    /// Rerun entity prefix, and the name used in messages.
    fn name(self) -> &'static str {
        match self {
            Wire::Usb => "usb",
            Wire::Wifi => "wifi",
        }
    }

    /// Belief blue for the cable, amber for the radio. The radio's colour
    /// is NOT `belief_viz::truth()` despite looking similar — yellow means
    /// ground truth everywhere else, and there is none on a bench.
    fn colour(self) -> rerun::Color {
        match self {
            Wire::Usb => belief_viz::belief(),
            Wire::Wifi => rerun::Color::from_rgb(255, 176, 60),
        }
    }
}

/// What one wire has delivered so far.
#[derive(Default)]
struct Tally {
    delivered: u64,
    /// Sequence numbers that arrived out of order or twice. UDP permits
    /// both; USB cannot produce either, so a non-zero count here on the
    /// cable means something is wrong with this program, not the link.
    out_of_order: u64,
    first_seq: Option<u64>,
    highest_seq: u64,
    last_arrival: Option<Instant>,
}

impl Tally {
    /// Reports the chip says it sent, from the first one we saw. The
    /// denominator for loss — and it comes from the chip's own counter
    /// rather than from elapsed time, so a board that pauses is not
    /// mistaken for a link that dropped.
    fn sent_since_first(&self) -> u64 {
        match self.first_seq {
            Some(first) => self.highest_seq.saturating_sub(first) + 1,
            None => 0,
        }
    }

    fn lost(&self) -> u64 {
        self.sent_since_first().saturating_sub(self.delivered)
    }
}

/// What comparing the two wires concluded about one report.
#[derive(Debug, PartialEq)]
enum Verdict {
    /// Both wires delivered it. Signed: positive means the radio landed
    /// after the cable, which is the expected direction and the number
    /// that decides whether the radio could ever carry commands.
    Lag { seq: u64, wifi_lag_ms: f64 },
    /// This wire did not deliver it, and is now far enough behind that it
    /// never will. Named per wire, because "a line went missing" and "the
    /// radio lost it" are different bug reports.
    Lost { seq: u64, wire: Wire },
}

/// Pairs up the two copies of each report.
///
/// Split out from the draw loop and unit-tested because it is the one
/// piece here with no other way to be checked: the loss counters can be
/// proved against a stream with known drops, but lag needs two wires
/// arriving at controlled times, and the `serialport` crate cannot open a
/// pty on macOS — so the only way to exercise it with real timings is a
/// real board, which is exactly the thing this is meant to measure.
struct Comparison {
    wires: Vec<Wire>,
    /// Reports seen on at least one wire, until they fall out of the
    /// window. Entries stay after matching rather than being deleted — see
    /// [`Tracked::resolved`].
    pending: HashMap<u64, Tracked>,
    /// The furthest ahead any wire has got, which is what the window is
    /// measured back from.
    highest_seq: u64,
}

/// One report's progress through the comparison.
#[derive(Default)]
struct Tracked {
    seen: HashMap<Wire, Instant>,
    /// Set once every wire has delivered this report.
    ///
    /// **The entry is kept, not removed.** Deleting it on completion is
    /// what the duplicate test caught: UDP is allowed to deliver the same
    /// datagram twice, and a second copy arriving after deletion created a
    /// *fresh* entry containing only the radio — which then aged out and
    /// blamed the cable for losing a report it had delivered perfectly.
    /// A resolved entry absorbs duplicates silently and is pruned by the
    /// same window as everything else.
    resolved: bool,
}

impl Comparison {
    fn new(wires: Vec<Wire>) -> Self {
        Comparison {
            wires,
            pending: HashMap::new(),
            highest_seq: 0,
        }
    }

    /// Record one arrival and return whatever became knowable because of
    /// it. Nothing is reported twice for the same sequence number.
    fn observe(&mut self, seq: u64, wire: Wire, at: Instant) -> Vec<Verdict> {
        // With one source there is nothing to compare against, and every
        // report would otherwise be declared lost by the absent wire.
        if self.wires.len() < 2 {
            return Vec::new();
        }
        let mut verdicts = Vec::new();
        self.highest_seq = self.highest_seq.max(seq);
        let cutoff = self.highest_seq.saturating_sub(MATCH_WINDOW);

        // A report from before the window has already been judged. Letting
        // it back in would re-open a decision that was made with more
        // information than this straggler carries.
        if seq < cutoff {
            return verdicts;
        }

        let tracked = self.pending.entry(seq).or_default();
        if tracked.resolved {
            return verdicts;
        }
        let seen = &mut tracked.seen;
        seen.insert(wire, at);
        if seen.len() == self.wires.len() {
            if let (Some(&usb), Some(&wifi)) = (seen.get(&Wire::Usb), seen.get(&Wire::Wifi)) {
                // Signed, and `Instant` subtraction saturates rather than
                // going negative — so the two directions are computed
                // separately instead of trusting one of them to be
                // representable.
                let wifi_lag_ms = if wifi >= usb {
                    wifi.duration_since(usb).as_secs_f64() * 1000.0
                } else {
                    -(usb.duration_since(wifi).as_secs_f64() * 1000.0)
                };
                verdicts.push(Verdict::Lag { seq, wifi_lag_ms });
            }
            tracked.resolved = true;
        }

        // Anything this far behind is never coming. Waiting a window
        // rather than deciding immediately is what makes a slow wire read
        // as *late* instead of being miscounted as *lossy*.
        let wires = &self.wires;
        self.pending.retain(|&old, tracked| {
            if old >= cutoff {
                return true;
            }
            if !tracked.resolved {
                for &missing in wires.iter().filter(|w| !tracked.seen.contains_key(w)) {
                    verdicts.push(Verdict::Lost {
                        seq: old,
                        wire: missing,
                    });
                }
            }
            false
        });
        verdicts
    }
}

/// A parsed line, tagged with where it came from and when it landed.
struct Arrival {
    wire: Wire,
    at: Instant,
    report: Report,
}

/// Reads lines from a serial port until it closes.
fn read_serial(port: String, tx: Sender<Arrival>) -> Result<(), Box<dyn std::error::Error>> {
    let serial = hil_protocol::link::open(&port, Duration::from_millis(2000))?;
    let mut reader = BufReader::new(serial);
    let mut line = String::new();
    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => return Ok(()),
            Ok(_) => {}
            // A timeout is normal when nothing is moving — the firmware
            // still reports, but a disconnect looks the same to
            // `read_line`. Keep going; a frozen viewer is more noticeable
            // than a message here.
            Err(_) => continue,
        }
        // The firmware can send prose as well as poses — `#` lines are how
        // the radio reports what it can see, which used to be knowable
        // only by asking a human to watch an LED.
        if let Some(note) = line.trim_end().strip_prefix('#') {
            println!("  chip:{note}");
            continue;
        }
        if let Some(report) = Report::parse(&line) {
            let arrival = Arrival {
                wire: Wire::Usb,
                at: Instant::now(),
                report,
            };
            if tx.send(arrival).is_err() {
                return Ok(());
            }
        }
    }
}

/// Receives broadcast datagrams until the channel closes.
///
/// One datagram may carry more than one line — the firmware sends them one
/// at a time today, but batching is the obvious optimisation the moment
/// framing overhead matters, so splitting here costs nothing and means the
/// host does not have to change when it happens.
fn read_udp(tx: Sender<Arrival>) -> Result<(), Box<dyn std::error::Error>> {
    let socket = UdpSocket::bind(("0.0.0.0", TELEMETRY_PORT))?;
    let mut buf = [0u8; 2048];
    loop {
        let Ok((len, _from)) = socket.recv_from(&mut buf) else {
            continue;
        };
        let Ok(text) = std::str::from_utf8(&buf[..len]) else {
            continue;
        };
        for line in text.lines() {
            if let Some(report) = Report::parse(line) {
                let arrival = Arrival {
                    wire: Wire::Wifi,
                    at: Instant::now(),
                    report,
                };
                if tx.send(arrival).is_err() {
                    return Ok(());
                }
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let want_udp = args.iter().any(|a| a == "--udp");
    // A fixed-length run, so a measurement can be repeated and compared
    // rather than depending on when someone pressed Ctrl-C. Without it the
    // only way to stop a live board is a kill, which loses the summary
    // that is the entire point of a comparison run.
    let run_for = args
        .iter()
        .position(|a| a == "--seconds")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.parse::<u64>())
        .transpose()?
        .map(Duration::from_secs);
    let serial_port = args
        .iter()
        .enumerate()
        .find(|(i, a)| {
            !a.starts_with("--")
                && args.get(i.wrapping_sub(1)).map(String::as_str) != Some("--seconds")
        })
        .map(|(_, a)| a.clone());
    if serial_port.is_none() && !want_udp {
        return Err(
            "usage: odom_view [<serial-port>] [--udp] [--seconds N]  (at least one source)".into(),
        );
    }

    let (tx, rx) = channel();
    let mut wires = Vec::new();
    if let Some(port) = serial_port {
        println!("reading {port}");
        wires.push(Wire::Usb);
        let tx = tx.clone();
        std::thread::spawn(move || {
            if let Err(e) = read_serial(port, tx) {
                eprintln!("⚠️  serial reader stopped: {e}");
            }
        });
    }
    if want_udp {
        println!("listening for broadcasts on UDP {TELEMETRY_PORT}");
        wires.push(Wire::Wifi);
        let tx = tx.clone();
        std::thread::spawn(move || {
            if let Err(e) = read_udp(tx) {
                eprintln!("⚠️  udp reader stopped: {e}");
            }
        });
    }
    // Both readers hold clones; this one would otherwise keep the channel
    // open forever and the loop below would never see a disconnect.
    drop(tx);

    let mut comparison = Comparison::new(wires.clone());
    let rec = rerun::RecordingStreamBuilder::new("robotiq_odom").spawn()?;
    println!("turn a wheel");

    let started = Instant::now();
    let mut tallies: HashMap<Wire, Tally> = wires.iter().map(|&w| (w, Tally::default())).collect();
    let mut trails: HashMap<Wire, belief_viz::Trail> = HashMap::new();
    let mut last_errors_left = 0u64;
    let mut last_errors_right = 0u64;
    let mut announced_stall = false;
    let mut parsed_any = false;
    // Every lag sample, kept so the summary can quote a distribution
    // rather than a plot. The plot shows the shape; a number is what ends
    // up in the progress log and in an argument about whether the radio
    // could ever carry commands.
    let mut lags: Vec<f64> = Vec::new();

    // `recv_timeout` rather than `rx.iter()`: the UDP reader never returns
    // — a socket with nobody sending to it is indistinguishable from one
    // whose sender is between packets — so waiting for every sender to
    // hang up would mean the summary never printed on a `--udp` run.
    // Silence this long means the board stopped or the run is over.
    loop {
        // Before the first report, wait indefinitely: the bench workflow
        // is to start the viewer and *then* power the board, so a timeout
        // that applied from launch would quit during the reflash it is
        // waiting for. After the first report, silence means the run is
        // over and the summary is owed.
        let arrival = if parsed_any {
            match rx.recv_timeout(QUIET_BEFORE_GIVING_UP) {
                Ok(arrival) => arrival,
                Err(_) => {
                    println!("\nno reports for {QUIET_BEFORE_GIVING_UP:?} — stopping");
                    break;
                }
            }
        } else if let Some(limit) = run_for {
            // `--seconds N` means N seconds, including the wait for the
            // first line. Without this a board that never reports leaves
            // the viewer hanging forever, which is fine at a bench with a
            // human present and useless when something else is driving it.
            match rx.recv_timeout(limit.saturating_sub(started.elapsed())) {
                Ok(arrival) => arrival,
                Err(_) => {
                    println!("\n{limit:?} elapsed with no reports at all");
                    break;
                }
            }
        } else {
            match rx.recv() {
                Ok(arrival) => arrival,
                // Every reader has hung up without ever parsing a line.
                Err(_) => break,
            }
        };
        if let Some(limit) = run_for {
            if started.elapsed() >= limit {
                println!("\n{limit:?} elapsed — stopping");
                break;
            }
        }
        let Arrival { wire, at, report } = arrival;
        if !parsed_any {
            println!("✅ first line from {}", wire.name());
            parsed_any = true;
        }
        let tally = tallies.entry(wire).or_default();

        // Ordering first, because a duplicate must not inflate `delivered`
        // and thereby hide a loss somewhere else in the stream.
        if tally.first_seq.is_none() {
            tally.first_seq = Some(report.seq);
        }
        if report.seq <= tally.highest_seq && tally.delivered > 0 {
            tally.out_of_order += 1;
        }
        tally.highest_seq = tally.highest_seq.max(report.seq);
        tally.delivered += 1;

        rec.set_duration_secs("wall_time", started.elapsed().as_secs_f64());

        // Gap since this wire's previous line. Flat at 20 ms is healthy;
        // a radio parked by power management shows up here as a spike
        // long before it shows up as a loss.
        if let Some(previous) = tally.last_arrival.replace(at) {
            rec.log(
                format!("{}/gap_ms", wire.name()),
                &rerun::Scalars::single(at.duration_since(previous).as_secs_f64() * 1000.0),
            )?;
        }
        rec.log(
            format!("{}/lost_total", wire.name()),
            &rerun::Scalars::single(tally.lost() as f64),
        )?;

        // ---- the cross-transport comparison ----
        for verdict in comparison.observe(report.seq, wire, at) {
            match verdict {
                Verdict::Lag { wifi_lag_ms, .. } => {
                    rec.log("compare/wifi_lag_ms", &rerun::Scalars::single(wifi_lag_ms))?;
                    lags.push(wifi_lag_ms);
                }
                Verdict::Lost { seq, wire } => {
                    rec.log(
                        "events",
                        &rerun::TextLog::new(format!("{} lost report {seq}", wire.name())),
                    )?;
                }
            }
        }

        // ---- the pose itself ----
        trails
            .entry(wire)
            .or_insert_with(|| {
                belief_viz::Trail::new(wire.colour(), belief_viz::Size::BENCH, REPORTS_PER_SECOND)
            })
            .draw(
                &rec,
                &format!("{}/belief", wire.name()),
                sim_core::Pose::new(report.x, report.y, report.heading),
            )?;

        rec.log(
            format!("{}/belief/heading_rad", wire.name()),
            &rerun::Scalars::single(report.heading),
        )?;

        // Plotted beside the ticks, because this is the pair that reveals
        // a stall: commanded duty rising while the tick lines stay flat.
        // docs/07 records the run where restoring exactly this panel is
        // what made a bug visible that the tests could not see.
        rec.log(
            format!("{}/motor/duty_percent", wire.name()),
            &rerun::Scalars::single(report.duty_percent as f64),
        )?;

        // Plotted, not just printed. A rising error count means the tick
        // stream is being undercounted, so every pose after it is wrong by
        // an amount nothing else on screen reveals.
        rec.log(
            format!("{}/ticks/errors_left", wire.name()),
            &rerun::Scalars::single(report.errors_left as f64),
        )?;
        rec.log(
            format!("{}/ticks/errors_right", wire.name()),
            &rerun::Scalars::single(report.errors_right as f64),
        )?;

        // Decode errors and stalls are properties of the ROBOT, not of a
        // wire, so they are announced once across all transports rather
        // than once per transport — otherwise running both sources would
        // double every warning and make the rig look twice as broken.
        for (wheel, was, now) in [
            ("left", &mut last_errors_left, report.errors_left),
            ("right", &mut last_errors_right, report.errors_right),
        ] {
            if now > *was {
                rec.log(
                    "events",
                    &rerun::TextLog::new(format!(
                        "{wheel} decode errors {was} -> {now} — pose is now an UNDERCOUNT"
                    )),
                )?;
                println!("⚠️  {wheel} decode errors: {was} -> {now}");
                *was = now;
            }
        }

        // The firmware stops driving when this latches, so it is reported
        // once rather than 50 times a second for as long as it is parked.
        if report.stalled && !announced_stall {
            rec.log(
                "events",
                &rerun::TextLog::new(
                    "STALLED — commanded but not moving. Check the battery switch.",
                ),
            )?;
            println!("⚠️  STALLED — commanded but not moving. Check the battery switch.");
            announced_stall = true;
        }
    }

    if !parsed_any {
        eprintln!("⚠️  nothing parsed. If the board is running, its format has moved:");
        eprintln!("    a firmware built before status lines carried `n=` is rejected");
        eprintln!("    on purpose — reflash it.");
    }
    summarise(&tallies);
    summarise_lag(&mut lags);
    Ok(())
}

/// How far behind the cable the radio ran, for reports both delivered.
///
/// Percentiles rather than a mean, because the mean of a latency
/// distribution is the one statistic that cannot fail a deadline: what
/// matters for control is the tail, and a radio that is usually 3 ms
/// behind and occasionally 300 ms behind is not a 6 ms radio.
fn summarise_lag(lags: &mut [f64]) {
    if lags.is_empty() {
        return;
    }
    lags.sort_by(|a, b| a.partial_cmp(b).expect("lag samples are never NaN"));
    let at = |q: f64| lags[((lags.len() - 1) as f64 * q) as usize];
    println!();
    println!(
        "wifi lag behind usb, for the {} reports both delivered:",
        lags.len()
    );
    println!(
        "   median {:.1} ms · p95 {:.1} ms · p99 {:.1} ms · worst {:.1} ms",
        at(0.50),
        at(0.95),
        at(0.99),
        lags[lags.len() - 1],
    );
}

/// Printed on exit, because the plots go away with the viewer and the
/// headline numbers are what end up in the progress log.
fn summarise(tallies: &HashMap<Wire, Tally>) {
    let mut wires: Vec<_> = tallies.keys().copied().collect();
    wires.sort_by_key(|w| w.name());
    println!();
    println!(
        "{:<6} {:>10} {:>8} {:>8} {:>12}",
        "wire", "delivered", "lost", "loss%", "out-of-order"
    );
    for wire in wires {
        let tally = &tallies[&wire];
        let sent = tally.sent_since_first();
        let loss_percent = if sent == 0 {
            0.0
        } else {
            100.0 * tally.lost() as f64 / sent as f64
        };
        println!(
            "{:<6} {:>10} {:>8} {:>7.2}% {:>12}",
            wire.name(),
            tally.delivered,
            tally.lost(),
            loss_percent,
            tally.out_of_order,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The parsing tests live in `hil-protocol` now, beside the writer,
    /// where they cover the round trip instead of one direction. Keeping a
    /// second copy of the captured fixture here would recreate exactly the
    /// duplication that emptied this viewer in the first place.
    ///
    /// What is worth checking here is that this file has not quietly grown
    /// its own parser again.
    #[test]
    fn the_viewer_uses_the_shared_protocol_type() {
        let shared = hil_protocol::Status::default();
        let _: Report = shared;
    }

    /// Loss is counted against what the CHIP says it sent, not against
    /// elapsed time — so a board that pauses reads as zero loss, and a
    /// wire that drops the middle of a run reads as exactly the gap.
    #[test]
    fn loss_is_measured_against_the_chips_own_counter() {
        let mut tally = Tally {
            first_seq: Some(100),
            highest_seq: 100,
            delivered: 1,
            ..Tally::default()
        };
        assert_eq!(tally.lost(), 0, "one report, one delivery");

        // Reports 101..=109 were sent; we saw only 110.
        tally.highest_seq = 110;
        tally.delivered = 2;
        assert_eq!(tally.sent_since_first(), 11);
        assert_eq!(tally.lost(), 9);
    }

    /// A wire that has delivered nothing must not report a loss, or a
    /// viewer started before the board reads as 100% broken.
    #[test]
    fn a_silent_wire_reports_no_loss_rather_than_total_loss() {
        let tally = Tally::default();
        assert_eq!(tally.sent_since_first(), 0);
        assert_eq!(tally.lost(), 0);
    }

    fn both_wires() -> Comparison {
        Comparison::new(vec![Wire::Usb, Wire::Wifi])
    }

    /// The headline measurement: when both wires deliver the same report,
    /// how much later the radio landed.
    #[test]
    fn a_report_both_wires_delivered_yields_the_signed_lag() {
        let mut c = both_wires();
        let t0 = Instant::now();
        assert_eq!(c.observe(1, Wire::Usb, t0), vec![]);
        let verdicts = c.observe(1, Wire::Wifi, t0 + Duration::from_millis(30));
        match verdicts.as_slice() {
            [Verdict::Lag {
                seq: 1,
                wifi_lag_ms,
            }] => {
                assert!(
                    (wifi_lag_ms - 30.0).abs() < 0.001,
                    "expected ~30 ms, got {wifi_lag_ms}"
                );
            }
            other => panic!("expected one Lag verdict, got {other:?}"),
        }
    }

    /// `Instant` subtraction saturates at zero, so a radio that somehow
    /// beat the cable would silently read as 0 ms rather than negative —
    /// which would hide the very anomaly worth investigating.
    #[test]
    fn a_radio_that_beats_the_cable_reads_negative_not_zero() {
        let mut c = both_wires();
        let t0 = Instant::now();
        c.observe(1, Wire::Wifi, t0);
        let verdicts = c.observe(1, Wire::Usb, t0 + Duration::from_millis(5));
        match verdicts.as_slice() {
            [Verdict::Lag { wifi_lag_ms, .. }] => {
                assert!(*wifi_lag_ms < 0.0, "expected negative, got {wifi_lag_ms}");
            }
            other => panic!("expected one Lag verdict, got {other:?}"),
        }
    }

    /// A report only one wire ever delivered is blamed on the other — by
    /// name, and only once it is far enough in the past to be certain.
    #[test]
    fn a_report_only_one_wire_delivered_is_blamed_on_the_other() {
        let mut c = both_wires();
        let t0 = Instant::now();
        c.observe(1, Wire::Usb, t0);

        // Still inside the window: the radio may yet be running late, and
        // calling it lost here is the mistake the window exists to avoid.
        for seq in 2..=MATCH_WINDOW {
            let verdicts = c.observe(seq, Wire::Usb, t0);
            assert!(
                !verdicts.iter().any(|v| matches!(v, Verdict::Lost { .. })),
                "report 1 declared lost after only {seq} reports"
            );
        }

        let verdicts = c.observe(MATCH_WINDOW + 2, Wire::Usb, t0);
        assert!(
            verdicts.contains(&Verdict::Lost {
                seq: 1,
                wire: Wire::Wifi
            }),
            "the radio lost report 1 and nothing said so: {verdicts:?}"
        );
    }

    /// With one source there is nothing to compare against. Without this
    /// guard every report would be declared lost by the wire that was
    /// never connected — turning a perfectly good cable-only run into a
    /// screen full of loss events.
    #[test]
    fn a_single_wire_run_produces_no_verdicts_at_all() {
        let mut c = Comparison::new(vec![Wire::Usb]);
        let t0 = Instant::now();
        for seq in 1..=(MATCH_WINDOW * 2) {
            assert_eq!(c.observe(seq, Wire::Usb, t0), vec![]);
        }
    }

    /// Neither a lag nor a loss may be announced twice, or the plots
    /// double-count and the event log becomes unreadable.
    #[test]
    fn nothing_is_reported_twice_for_one_sequence_number() {
        let mut c = both_wires();
        let t0 = Instant::now();
        c.observe(7, Wire::Usb, t0);
        c.observe(7, Wire::Wifi, t0 + Duration::from_millis(10));
        // A duplicate datagram — UDP permits it — must not re-emit.
        assert_eq!(
            c.observe(7, Wire::Wifi, t0 + Duration::from_millis(11)),
            vec![]
        );

        let mut lost_count = 0;
        for seq in 8..=(MATCH_WINDOW * 2) {
            lost_count += c
                .observe(seq, Wire::Usb, t0)
                .iter()
                .filter(|v| matches!(v, Verdict::Lost { seq: 7, .. }))
                .count();
        }
        assert_eq!(lost_count, 0, "report 7 arrived on both wires");
    }
}
