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
//! # Two levels, because they answer different questions
//!
//! | mode | cost | answers |
//! |---|---|---|
//! | `--record` | **6 KB/s** | did the control law do the right thing? |
//! | `--record --video` | **0.9 MB/s** | *and* was the detector right? |
//!
//! The default records one line per frame plus one per detection, and
//! nothing else. That is enough to re-run every steering decision, and
//! cheap enough to leave on forever.
//!
//! It is *not* enough to see why a box appeared. Boxes floating on a black
//! background cannot tell you whether the detector was looking at a person
//! or a coat on a chair. `--video` adds JPEG frames beside the log — 150×
//! larger, still only 55 MB a minute, and the difference between watching
//! a replay and merely reading one. (Raw RGB would be 1.1 GB a minute,
//! which is why it is JPEG.)
//!
//! **Target selection is recorded, not re-run.** `TargetLock::pick`
//! matches on hue and needs the frame's pixels, which are not here. So the
//! chosen detection is stored by index and replay honours it. Re-running
//! selection offline would need frames — a different, heavier facility,
//! and a different question.

use crate::camera::Frame;
use crate::detect::Detection;
use sim_core::BodyTwist;
use std::fmt::Write as _;
use std::fs::File;
use std::io::Write as _;
use std::path::{Path, PathBuf};

/// One frame's perception result: what was seen, and which one is being
/// chased.
#[derive(Debug, Clone, PartialEq)]
pub struct Perceived {
    /// Seconds since the previous frame — real, measured, and part of the
    /// recording because the PID differentiates by it. Replaying with a
    /// synthetic `dt` would compute a different `D` term and quietly stop
    /// being the run that was captured.
    pub frame_seconds: f64,
    pub frame_w: u32,
    pub frame_h: u32,
    pub detections: Vec<Detection>,
    /// Index into `detections` of the one being chased, if any.
    pub target: Option<usize>,
    /// The `(v, w)` the controller commanded from this frame.
    ///
    /// **This is what makes a replay a test rather than a viewing.**
    /// Without it a recording says only what the camera saw; replay can
    /// re-run the control law but has nothing to check it against, so a
    /// change to the filter, the deadband or the approach curve replays
    /// happily and silently differently.
    ///
    /// `None` in a log written before this field existed — replay then
    /// says it cannot verify, rather than pretending it did.
    pub command: Option<BodyTwist>,
}

impl Perceived {
    /// The detection being chased.
    pub fn target(&self) -> Option<&Detection> {
        self.target.and_then(|i| self.detections.get(i))
    }

    fn encode(&self, out: &mut String) {
        let _ = write!(
            out,
            "F {:.6} {} {} {}",
            self.frame_seconds,
            self.frame_w,
            self.frame_h,
            self.target.map_or(-1i64, |i| i as i64)
        );
        // Trailing and optional, so a log without commands still parses.
        if let Some(twist) = self.command {
            let _ = write!(out, " {:.6} {:.6}", twist.forward_speed, twist.turn_rate);
        }
        let _ = writeln!(out);
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

/// Where the JPEGs live for a given log: `chase.perc` → `chase.perc.frames/`.
///
/// Derived rather than configured, so a log and its footage cannot be
/// separated by a careless move — and replay can just look.
pub fn frames_dir(log: &Path) -> PathBuf {
    let mut d = log.as_os_str().to_os_string();
    d.push(".frames");
    PathBuf::from(d)
}

fn frame_path(dir: &Path, index: usize) -> PathBuf {
    dir.join(format!("{index:06}.jpg"))
}

/// Appends frames to a file, or does nothing when no `--record` was given.
pub struct Recorder {
    log: Option<File>,
    /// `Some` when `--video` was asked for.
    frames: Option<PathBuf>,
    index: usize,
}

impl Recorder {
    pub fn create(path: Option<&Path>, video: bool) -> std::io::Result<Recorder> {
        let frames = match (path, video) {
            (Some(p), true) => {
                let dir = frames_dir(p);
                std::fs::create_dir_all(&dir)?;
                Some(dir)
            }
            _ => None,
        };
        Ok(Recorder {
            log: path.map(File::create).transpose()?,
            frames,
            index: 0,
        })
    }

    /// Record one frame's perception, and its pixels if `--video`.
    ///
    /// The index advances with the log, not with the images, so the two
    /// stay aligned even if an encode fails.
    pub fn write(&mut self, p: &Perceived, frame: Option<&Frame>) -> std::io::Result<()> {
        if let Some(f) = &mut self.log {
            let mut buf = String::new();
            p.encode(&mut buf);
            f.write_all(buf.as_bytes())?;
        }
        if let (Some(dir), Some(fr)) = (&self.frames, frame) {
            save_jpeg(&frame_path(dir, self.index), fr)?;
        }
        self.index += 1;
        Ok(())
    }
}

/// Quality 80: visually indistinguishable for this purpose, and a third
/// the size of 95.
fn save_jpeg(path: &Path, frame: &Frame) -> std::io::Result<()> {
    let buf: image::RgbImage =
        image::ImageBuffer::from_raw(frame.width, frame.height, frame.rgb.clone()).ok_or_else(
            || {
                std::io::Error::other(format!(
                    "frame is {} bytes, not {}x{}x3",
                    frame.rgb.len(),
                    frame.width,
                    frame.height
                ))
            },
        )?;
    let mut out = std::io::BufWriter::new(File::create(path)?);
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 80)
        .encode_image(&buf)
        .map_err(std::io::Error::other)
}

/// The recorded frame at `index`, if `--video` captured one.
///
/// A missing image is `None`, not an error: a log recorded without
/// `--video` must still replay, just without pictures.
pub fn load_frame(log: &Path, index: usize) -> Option<Frame> {
    let img = image::open(frame_path(&frames_dir(log), index)).ok()?;
    let rgb = img.to_rgb8();
    Some(Frame {
        width: rgb.width(),
        height: rgb.height(),
        rgb: rgb.into_raw(),
    })
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
                let frame_seconds = num(f.next(), &at)?;
                let frame_w = num::<u32>(f.next(), &at)?;
                let frame_h = num::<u32>(f.next(), &at)?;
                let idx = num::<i64>(f.next(), &at)?;
                // Both or neither: half a command is a malformed line, not
                // an old one.
                let command = match (f.next(), f.next()) {
                    (Some(v), Some(w)) => Some(BodyTwist::new(
                        v.parse().map_err(|_| at("bad commanded forward speed"))?,
                        w.parse().map_err(|_| at("bad commanded turn rate"))?,
                    )),
                    (None, None) => None,
                    _ => return Err(at("a command needs both v and w")),
                };
                frames.push(Perceived {
                    frame_seconds,
                    frame_w,
                    frame_h,
                    detections: Vec::new(),
                    target: (idx >= 0).then_some(idx as usize),
                    command,
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

/// How far a recomputed command may drift from the recorded one before it
/// counts as a behaviour change.
///
/// The log carries six decimals, so re-running identical code differs by
/// under 5e-7 — pure formatting. A 1% gain change moves `v` by ~2e-3, three
/// hundred times this bar. Wide enough never to cry wolf, tight enough that
/// nothing real slips through.
pub const COMMAND_TOLERANCE: f64 = 1e-5;

/// Compare a freshly computed command against what the session recorded.
///
/// `Ok(())` when they agree or the log predates commands; `Err` describes
/// the disagreement in the same shape `hil-host` uses, because it is the
/// same question asked of a different boundary.
pub fn check_command(frame: usize, p: &Perceived, got: BodyTwist) -> Result<(), String> {
    let Some(recorded) = p.command else {
        return Ok(()); // nothing recorded to check against
    };
    let speed_differs = (got.forward_speed - recorded.forward_speed).abs() > COMMAND_TOLERANCE;
    let turn_differs = (got.turn_rate - recorded.turn_rate).abs() > COMMAND_TOLERANCE;
    if !speed_differs && !turn_differs {
        return Ok(());
    }
    Err(format!(
        "frame {frame}: would command v={:.6} w={:.6}, recording has v={:.6} w={:.6}",
        got.forward_speed, got.turn_rate, recorded.forward_speed, recorded.turn_rate
    ))
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
                frame_seconds: 0.05,
                frame_w: 640,
                frame_h: 480,
                detections: vec![det("cup", 0.9), det("book", 0.4)],
                target: Some(0),
                command: Some(BodyTwist::new(0.31, -0.42)),
            },
            Perceived {
                frame_seconds: 0.048,
                frame_w: 640,
                frame_h: 480,
                detections: vec![],
                target: None,
                command: Some(BodyTwist::STOPPED),
            },
        ];
        assert_eq!(round_trip("session", &frames), frames);
    }

    /// COCO labels contain spaces. Putting the label anywhere but last
    /// would silently truncate "cell phone" to "cell".
    #[test]
    fn labels_with_spaces_survive() {
        let frames = vec![Perceived {
            frame_seconds: 0.05,
            frame_w: 1,
            frame_h: 1,
            detections: vec![det("cell phone", 0.5), det("potted plant", 0.6)],
            target: Some(1),
            command: None,
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
            frame_seconds: 0.0491234,
            frame_w: 1,
            frame_h: 1,
            detections: vec![],
            target: None,
            command: None,
        }];
        let back = round_trip("dt", &frames);
        assert!(
            (back[0].frame_seconds - 0.0491234).abs() < 1e-6,
            "dt came back as {}",
            back[0].frame_seconds
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
        let mut r = Recorder::create(None, false).unwrap();
        r.write(&blank(), None)
            .expect("a no-op recorder cannot fail");
    }

    fn blank() -> Perceived {
        Perceived {
            frame_seconds: 0.05,
            frame_w: 4,
            frame_h: 2,
            detections: vec![],
            target: None,
            command: Some(BodyTwist::new(0.1, -0.2)),
        }
    }

    fn checkerboard() -> Frame {
        Frame {
            width: 4,
            height: 2,
            rgb: (0..4 * 2 * 3).map(|i| (i * 7 % 256) as u8).collect(),
        }
    }

    /// The point of recording commands at all.
    #[test]
    fn a_changed_command_is_caught() {
        let p = blank(); // commands forward 0.1 m/s, turning -0.2 rad/s
        assert!(
            check_command(0, &p, BodyTwist::new(0.1, -0.2)).is_ok(),
            "identical must pass"
        );

        let err = check_command(7, &p, BodyTwist::new(0.15, -0.2)).unwrap_err();
        assert!(err.contains("frame 7"), "{err}");
        assert!(
            err.contains("0.150000") && err.contains("0.100000"),
            "{err}"
        );

        assert!(
            check_command(0, &p, BodyTwist::new(0.1, -0.25)).is_err(),
            "turn rate matters too"
        );
    }

    /// Six decimals on the wire means re-running identical code differs by
    /// rounding alone. That must not read as a regression.
    #[test]
    fn rounding_is_not_a_divergence() {
        let p = blank();
        assert!(check_command(0, &p, BodyTwist::new(0.1 + 4e-7, -0.2 - 4e-7)).is_ok());
        // ...but a real change, three hundred times larger, is caught.
        assert!(check_command(0, &p, BodyTwist::new(0.1 + 2e-3, -0.2)).is_err());
    }

    /// A log written before commands existed must replay, and must NOT
    /// claim it verified anything.
    #[test]
    fn a_log_without_commands_cannot_be_verified_and_says_so() {
        let p = Perceived {
            command: None,
            ..blank()
        };
        assert!(
            check_command(0, &p, BodyTwist::new(99.0, -99.0)).is_ok(),
            "nothing recorded means nothing to contradict"
        );
    }

    /// Half a command is a corrupt line, not an old one — and the
    /// difference matters, because treating it as old would silently skip
    /// verification for that frame.
    #[test]
    fn half_a_command_is_rejected() {
        let path = std::env::temp_dir().join("perc-halfcmd.perc");
        std::fs::write(&path, "F 0.05 640 480 -1 0.3\n").unwrap();
        let err = read(&path).unwrap_err();
        std::fs::remove_file(&path).ok();
        assert!(err.contains("both v and w"), "{err}");
    }

    #[test]
    fn the_frames_directory_is_derived_from_the_log() {
        assert_eq!(
            frames_dir(Path::new("runs/chase.perc")),
            PathBuf::from("runs/chase.perc.frames")
        );
    }

    #[test]
    fn video_round_trips_and_stays_aligned_with_the_log() {
        let log = std::env::temp_dir().join("perc-video.perc");
        let _ = std::fs::remove_dir_all(frames_dir(&log));

        let mut r = Recorder::create(Some(&log), true).unwrap();
        let fr = checkerboard();
        r.write(&blank(), Some(&fr)).unwrap();
        r.write(&blank(), Some(&fr)).unwrap();
        drop(r);

        // Both frames present, and indexed from zero alongside the log.
        assert_eq!(read(&log).unwrap().len(), 2);
        let back = load_frame(&log, 1).expect("frame 1 should exist");
        assert_eq!((back.width, back.height), (4, 2));
        assert_eq!(back.rgb.len(), 4 * 2 * 3);
        assert!(load_frame(&log, 2).is_none(), "only two were written");

        std::fs::remove_file(&log).ok();
        std::fs::remove_dir_all(frames_dir(&log)).ok();
    }

    /// A log recorded WITHOUT `--video` must still replay — just silently,
    /// without pictures. Erroring here would make the cheap mode useless.
    #[test]
    fn a_log_with_no_video_loads_no_frame_and_does_not_fail() {
        let log = std::env::temp_dir().join("perc-novideo.perc");
        let mut r = Recorder::create(Some(&log), false).unwrap();
        r.write(&blank(), Some(&checkerboard())).unwrap();
        drop(r);
        assert!(!frames_dir(&log).exists(), "no directory should be made");
        assert!(load_frame(&log, 0).is_none());
        std::fs::remove_file(&log).ok();
    }
}
