//! Recording and replaying the **seam between seeing and steering**.
//!
//! # Why
//!
//! The HIL rig learned this the hard way: the bugs that survive a test
//! suite are the ones that live *between* components, and the only way to
//! see them is to capture what actually crossed the boundary. Recording
//! the wire found a stale-pose bug on its first hardware run, from four
//! lines of log.
//!
//! The camera path has the same boundary and, until now, nothing watching
//! it — while being the *more* non-deterministic half: real frames, real
//! lighting, real inference. Every nasty bug here was found by a human
//! watching a screen. A detector that chased **NaN**, an open-vocabulary
//! model that found a mug on a blank wall, a backend that silently put all
//! 731 nodes on the CPU, overlays from three minutes ago painted over live
//! video. None of those had a test; all of them had a moment you could
//! have replayed.
//!
//! ```sh
//! cargo run -p vision --bin chase -- --record chase.perc
//! cargo run -p vision --bin chase -- --replay chase.perc   # no camera
//! ```
//!
//! # What is captured, and what is not
//!
//! One line per frame plus one per detection — a few hundred bytes a
//! second, not video. That is deliberate: the question this answers is
//! *"given what the detector saw, did the control law do the right
//! thing?"*, and that is the half that has to keep working when the
//! motors arrive.
//!
//! **Target selection is recorded, not re-run.** `TargetLock::pick`
//! matches on hue and needs the frame's pixels, which are not here. So the
//! chosen detection is stored by index and replay honours it. Re-running
//! selection offline would need frames — a different, heavier facility,
//! and a different question.

use crate::detect::Detection;
use std::fmt::Write as _;
use std::fs::File;
use std::io::Write as _;
use std::path::Path;

/// One frame's perception result: what was seen, and which one is being
/// chased.
#[derive(Debug, Clone, PartialEq)]
pub struct Perceived {
    /// Seconds since the previous frame — real, measured, and part of the
    /// recording because the PID differentiates by it. Replaying with a
    /// synthetic `dt` would compute a different `D` term and quietly stop
    /// being the run that was captured.
    pub dt: f64,
    pub frame_w: u32,
    pub frame_h: u32,
    pub detections: Vec<Detection>,
    /// Index into `detections` of the one being chased, if any.
    pub target: Option<usize>,
}

impl Perceived {
    /// The detection being chased.
    pub fn target(&self) -> Option<&Detection> {
        self.target.and_then(|i| self.detections.get(i))
    }

    fn encode(&self, out: &mut String) {
        let _ = writeln!(
            out,
            "F {:.6} {} {} {}",
            self.dt,
            self.frame_w,
            self.frame_h,
            self.target.map_or(-1i64, |i| i as i64)
        );
        for d in &self.detections {
            // The label goes last because it can contain spaces
            // ("cell phone", "potted plant").
            let _ = writeln!(
                out,
                "D {:.3} {:.3} {:.3} {:.3} {:.6} {} {}",
                d.x, d.y, d.width, d.height, d.confidence, d.class_id, d.label
            );
        }
    }
}

/// Appends frames to a file, or does nothing when no `--record` was given.
pub struct Recorder(Option<File>);

impl Recorder {
    pub fn create(path: Option<&Path>) -> std::io::Result<Recorder> {
        Ok(Recorder(path.map(File::create).transpose()?))
    }

    pub fn write(&mut self, p: &Perceived) -> std::io::Result<()> {
        let Some(f) = &mut self.0 else { return Ok(()) };
        let mut buf = String::new();
        p.encode(&mut buf);
        f.write_all(buf.as_bytes())
    }
}

/// Read a recorded session.
///
/// Malformed lines are an error rather than a skip: a partially understood
/// recording would replay a slightly different run and report that it
/// matched, which is worse than refusing to start.
pub fn read(path: &Path) -> Result<Vec<Perceived>, String> {
    let body = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut frames: Vec<Perceived> = Vec::new();

    for (n, line) in body.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let at = |what: &str| format!("line {}: {what}: {line:?}", n + 1);
        let mut f = line.split_whitespace();

        match f.next() {
            Some("F") => {
                let dt = num(f.next(), &at)?;
                let frame_w = num::<u32>(f.next(), &at)?;
                let frame_h = num::<u32>(f.next(), &at)?;
                let idx = num::<i64>(f.next(), &at)?;
                frames.push(Perceived {
                    dt,
                    frame_w,
                    frame_h,
                    detections: Vec::new(),
                    target: (idx >= 0).then_some(idx as usize),
                });
            }
            Some("D") => {
                let frame = frames.last_mut().ok_or_else(|| at("D before any F"))?;
                let d = Detection {
                    x: num(f.next(), &at)?,
                    y: num(f.next(), &at)?,
                    width: num(f.next(), &at)?,
                    height: num(f.next(), &at)?,
                    confidence: num(f.next(), &at)?,
                    class_id: num(f.next(), &at)?,
                    // Rest of the line: labels contain spaces.
                    label: f.collect::<Vec<_>>().join(" "),
                };
                frame.detections.push(d);
            }
            _ => return Err(at("expected a line starting with 'F' or 'D'")),
        }
    }

    // A target index past the end would panic a naive consumer later, far
    // from the cause. Reject it here, where the line number is known.
    for (i, p) in frames.iter().enumerate() {
        if let Some(t) = p.target {
            if t >= p.detections.len() {
                return Err(format!(
                    "frame {i}: target index {t} but only {} detections",
                    p.detections.len()
                ));
            }
        }
    }
    if frames.is_empty() {
        return Err(format!("{} has no frames", path.display()));
    }
    Ok(frames)
}

fn num<T: std::str::FromStr>(tok: Option<&str>, at: &impl Fn(&str) -> String) -> Result<T, String> {
    tok.ok_or_else(|| at("ran out of fields"))?
        .parse()
        .map_err(|_| at("could not parse a number"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(label: &str, conf: f32) -> Detection {
        Detection {
            x: 10.0,
            y: 20.0,
            width: 30.0,
            height: 40.0,
            confidence: conf,
            label: label.to_string(),
            class_id: 41,
        }
    }

    /// `name` must be unique per test: these run in parallel, and keying
    /// the temp file on anything they can share (frame count, say) makes
    /// two tests corrupt each other's file and fail for reasons that have
    /// nothing to do with the code.
    fn round_trip(name: &str, frames: &[Perceived]) -> Vec<Perceived> {
        let mut buf = String::new();
        for p in frames {
            p.encode(&mut buf);
        }
        let path = std::env::temp_dir().join(format!("perc-{name}.perc"));
        std::fs::write(&path, buf).unwrap();
        let back = read(&path).unwrap();
        std::fs::remove_file(&path).ok();
        back
    }

    #[test]
    fn a_session_survives_the_round_trip() {
        let frames = vec![
            Perceived {
                dt: 0.05,
                frame_w: 640,
                frame_h: 480,
                detections: vec![det("cup", 0.9), det("book", 0.4)],
                target: Some(0),
            },
            Perceived {
                dt: 0.048,
                frame_w: 640,
                frame_h: 480,
                detections: vec![],
                target: None,
            },
        ];
        assert_eq!(round_trip("session", &frames), frames);
    }

    /// COCO labels contain spaces. Putting the label anywhere but last
    /// would silently truncate "cell phone" to "cell".
    #[test]
    fn labels_with_spaces_survive() {
        let frames = vec![Perceived {
            dt: 0.05,
            frame_w: 1,
            frame_h: 1,
            detections: vec![det("cell phone", 0.5), det("potted plant", 0.6)],
            target: Some(1),
        }];
        let back = round_trip("labels", &frames);
        assert_eq!(back[0].detections[0].label, "cell phone");
        assert_eq!(back[0].target().unwrap().label, "potted plant");
    }

    /// `dt` drives the PID's derivative. Four decimals of a 50 ms frame is
    /// 1 part in 500; six keeps replay honest.
    #[test]
    fn dt_keeps_enough_precision_for_the_derivative() {
        let frames = vec![Perceived {
            dt: 0.0491234,
            frame_w: 1,
            frame_h: 1,
            detections: vec![],
            target: None,
        }];
        let back = round_trip("dt", &frames);
        assert!(
            (back[0].dt - 0.0491234).abs() < 1e-6,
            "dt came back as {}",
            back[0].dt
        );
    }

    #[test]
    fn a_target_index_past_the_end_is_rejected() {
        let path = std::env::temp_dir().join("perc-bad-index.perc");
        std::fs::write(&path, "F 0.05 640 480 3\nD 1 2 3 4 0.9 41 cup\n").unwrap();
        let err = read(&path).unwrap_err();
        std::fs::remove_file(&path).ok();
        assert!(err.contains("target index 3"), "{err}");
    }

    #[test]
    fn a_detection_before_any_frame_is_rejected() {
        let path = std::env::temp_dir().join("perc-orphan.perc");
        std::fs::write(&path, "D 1 2 3 4 0.9 41 cup\n").unwrap();
        let err = read(&path).unwrap_err();
        std::fs::remove_file(&path).ok();
        assert!(err.contains("D before any F"), "{err}");
    }

    #[test]
    fn a_truncated_line_names_its_line_number() {
        let path = std::env::temp_dir().join("perc-short.perc");
        std::fs::write(&path, "F 0.05 640 480 -1\nF 0.05 640\n").unwrap();
        let err = read(&path).unwrap_err();
        std::fs::remove_file(&path).ok();
        assert!(err.contains("line 2") && err.contains("ran out"), "{err}");
    }

    #[test]
    fn comments_and_blanks_are_ignored() {
        let path = std::env::temp_dir().join("perc-comments.perc");
        std::fs::write(&path, "# notes\n\nF 0.05 640 480 -1\n\n# more\n").unwrap();
        let frames = read(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(frames.len(), 1);
    }

    #[test]
    fn recording_nothing_writes_nothing() {
        let mut r = Recorder::create(None).unwrap();
        r.write(&Perceived {
            dt: 0.05,
            frame_w: 1,
            frame_h: 1,
            detections: vec![],
            target: None,
        })
        .expect("a no-op recorder cannot fail");
    }
}
