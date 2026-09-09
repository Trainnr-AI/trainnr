//! The control surface: an agent drives the Studio through files under
//! `<project>/.index/` (mirrored from `rq_pipeline/project/control.py`).
//!
//! - `commands/<id>.json` arrives from the agent; this module polls the
//!   directory at 20 Hz, parses each new file into a [`Command`], and the
//!   frame loop (`main.rs`) applies it and answers with `<id>.ack.json`.
//! - `studio-state.json` is what the window shows right now, written on
//!   change and at least once a second (the heartbeat the agent checks).
//! - `events.jsonl` is what the human did — appended, never rewritten.
//!
//! No sockets and no server: a file is the contract, like the index.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

pub const COMMANDS_RELATIVE: &str = ".index/commands";
pub const STATE_RELATIVE: &str = ".index/studio-state.json";
pub const EVENTS_RELATIVE: &str = ".index/events.jsonl";
pub const STATE_SCHEMA: &str = "trainnr-studio-state/1";
/// How often the commands directory is read: 20 Hz, cheap (one readdir
/// of a small directory).
pub const POLL_EVERY: Duration = Duration::from_millis(50);
/// The state file is rewritten at least this often even when nothing
/// changed, so a stale heartbeat means a dead Studio.
const HEARTBEAT_EVERY: Duration = Duration::from_secs(1);
/// A time cursor moved by the human is reported once it rests this long.
const SCRUB_SETTLE: Duration = Duration::from_millis(250);

/// One command from the agent. `verb` picks the variant; unknown fields
/// are ignored so an older Studio still answers a newer agent by name.
#[derive(Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "verb", rename_all = "lowercase")]
pub enum Command {
    Open {
        #[serde(default)]
        project: Option<String>,
        #[serde(default)]
        section: Option<String>,
        #[serde(default)]
        artifact: Option<String>,
        /// A table of the selected artifact to open in the modal, by its
        /// title; an empty string closes the modal.
        #[serde(default)]
        table: Option<String>,
    },
    Show {
        artifact: String,
    },
    Compare {
        a: String,
        b: String,
    },
    Time {
        #[serde(default)]
        timeline: Option<String>,
        #[serde(default)]
        seconds: Option<f64>,
        #[serde(default)]
        sequence: Option<i64>,
        #[serde(default)]
        play: Option<bool>,
        #[serde(default)]
        speed: Option<f32>,
        #[serde(default)]
        start: Option<f64>,
        #[serde(default)]
        end: Option<f64>,
        #[serde(default)]
        follow: Option<bool>,
        #[serde(default)]
        step: Option<i64>,
    },
    Panels {
        #[serde(default)]
        blueprint: Option<String>,
        #[serde(default)]
        selection: Option<String>,
        #[serde(default)]
        time: Option<String>,
    },
    Screenshot {
        /// The written image's width in pixels; taller frames scale to it.
        #[serde(default)]
        width: Option<u32>,
    },
    Quit,
}

pub const SCREENSHOTS_RELATIVE: &str = ".index/screenshots";
/// The default and the ceiling for a screenshot's width: one image an
/// agent reads whole, never a retina frame verbatim.
pub const SCREENSHOT_WIDTH: u32 = 1600;
pub const SCREENSHOT_WIDTH_MAX: u32 = 4000;

/// A command file read from disk: its id and the parse (an unparseable
/// file is acknowledged as refused, with the reason, never ignored).
pub struct Pending {
    pub id: String,
    pub command: Result<Command, String>,
}

/// The viewer's half of the state: which recording, where in time.
#[derive(Serialize, Default, Clone, PartialEq, Debug)]
pub struct Live {
    pub recording: Option<String>,
    pub timeline: Option<String>,
    /// The cursor in seconds on a duration or timestamp timeline.
    pub seconds: Option<f64>,
    /// The cursor on a sequence timeline (a step, a frame, an episode).
    pub sequence: Option<i64>,
}

#[derive(Serialize, Clone, PartialEq, Debug)]
pub struct StudioState {
    pub schema: &'static str,
    pub pid: u32,
    pub heartbeat: f64,
    pub project: String,
    pub project_name: String,
    pub section: String,
    pub selected: Option<String>,
    /// The table open in the modal, by title.
    pub table: Option<String>,
    pub live: Live,
    pub presenter_running: bool,
    pub jobs_running: usize,
}

pub struct Control {
    root: PathBuf,
    seen: BTreeSet<String>,
    last_poll: Option<Instant>,
    last_state: Option<StudioState>,
    last_written: Option<Instant>,
    /// The live state last observed, and when it last moved, for the
    /// scrub event; `commanded_until` suppresses it after our own command.
    observed_live: Option<Live>,
    live_moved_at: Option<Instant>,
    commanded_until: Option<Instant>,
}

impl Control {
    pub fn new(root: PathBuf) -> Self {
        let mut control = Self {
            root,
            seen: BTreeSet::new(),
            last_poll: None,
            last_state: None,
            last_written: None,
            observed_live: None,
            live_moved_at: None,
            commanded_until: None,
        };
        control.forget_answered();
        control
    }

    /// A project switch: the commands directory moves with it.
    pub fn set_root(&mut self, root: PathBuf) {
        if root != self.root {
            self.root = root;
            self.seen.clear();
            self.last_state = None;
            self.forget_answered();
        }
    }

    /// Commands already acknowledged on disk are not applied again when
    /// the Studio restarts on the same project.
    fn forget_answered(&mut self) {
        let Ok(entries) = std::fs::read_dir(self.root.join(COMMANDS_RELATIVE)) else {
            return;
        };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if let Some(id) = name.strip_suffix(".ack.json") {
                self.seen.insert(id.to_owned());
            }
        }
    }

    /// New command files since the last poll, oldest first; at most once
    /// per [`POLL_EVERY`].
    pub fn poll(&mut self) -> Vec<Pending> {
        let now = Instant::now();
        if self
            .last_poll
            .is_some_and(|t| now.duration_since(t) < POLL_EVERY)
        {
            return Vec::new();
        }
        self.last_poll = Some(now);
        let Ok(entries) = std::fs::read_dir(self.root.join(COMMANDS_RELATIVE)) else {
            return Vec::new();
        };
        let mut fresh: Vec<(String, PathBuf)> = entries
            .flatten()
            .filter_map(|entry| {
                let name = entry.file_name().to_string_lossy().into_owned();
                let id = name.strip_suffix(".json")?;
                if id.ends_with(".ack") || self.seen.contains(id) {
                    return None;
                }
                Some((id.to_owned(), entry.path()))
            })
            .collect();
        fresh.sort();
        fresh
            .into_iter()
            .map(|(id, path)| {
                self.seen.insert(id.clone());
                let command = std::fs::read_to_string(&path)
                    .map_err(|e| format!("unreadable: {e}"))
                    .and_then(|text| parse_command(&text));
                Pending { id, command }
            })
            .collect()
    }

    /// Answer a command: `done`, `refused` or `failed`, with the reason.
    pub fn ack(&self, id: &str, status: &str, reason: Option<&str>) {
        self.ack_with(id, status, reason, serde_json::Map::new());
    }

    /// An answer carrying data (a screenshot's path and size).
    pub fn ack_with(
        &self,
        id: &str,
        status: &str,
        reason: Option<&str>,
        extra: serde_json::Map<String, serde_json::Value>,
    ) {
        let dir = self.root.join(COMMANDS_RELATIVE);
        let _ = std::fs::create_dir_all(&dir);
        let mut body = serde_json::json!({
            "id": id,
            "status": status,
            "reason": reason,
            "t": epoch_seconds(),
        });
        if let Some(into) = body.as_object_mut() {
            into.extend(extra);
        }
        let path = dir.join(format!("{id}.ack.json"));
        let _ = write_atomic(&path, &body.to_string());
    }

    /// Write a captured frame as a PNG under the project, scaled to
    /// `width` (never upscaled); returns the path and the written size.
    pub fn save_screenshot(
        &self,
        id: &str,
        frame: &egui::ColorImage,
        width: u32,
    ) -> Result<(PathBuf, u32, u32), String> {
        let [w, h] = frame.size;
        let (w, h) = (w as u32, h as u32);
        if w == 0 || h == 0 {
            return Err("the captured frame is empty".into());
        }
        let image = image::RgbaImage::from_raw(w, h, frame.as_raw().to_vec())
            .ok_or_else(|| "the captured frame has a wrong byte count".to_owned())?;
        let width = width.clamp(1, SCREENSHOT_WIDTH_MAX).min(w);
        let height = (u64::from(h) * u64::from(width) / u64::from(w)).max(1) as u32;
        let scaled = if width == w {
            image
        } else {
            image::imageops::resize(&image, width, height, image::imageops::FilterType::Triangle)
        };
        let dir = self.root.join(SCREENSHOTS_RELATIVE);
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let path = dir.join(format!("{id}.png"));
        scaled.save(&path).map_err(|e| e.to_string())?;
        Ok((path, width, height))
    }

    /// Write the state when it changed, or when the heartbeat is due.
    pub fn record_state(&mut self, mut state: StudioState) {
        let now = Instant::now();
        let due = self
            .last_written
            .is_none_or(|t| now.duration_since(t) >= HEARTBEAT_EVERY);
        let changed = self
            .last_state
            .as_ref()
            .is_none_or(|last| !same_but_heartbeat(last, &state));
        if !(due || changed) {
            return;
        }
        state.heartbeat = epoch_seconds();
        let _ = write_atomic(
            &self.root.join(STATE_RELATIVE),
            &serde_json::to_string(&state).unwrap_or_default(),
        );
        self.last_state = Some(state);
        self.last_written = Some(now);
    }

    /// Append one event the human caused.
    pub fn event(&self, kind: &str, fields: serde_json::Value) {
        let mut body = serde_json::json!({ "t": epoch_nanos(), "kind": kind });
        if let (Some(into), Some(from)) = (body.as_object_mut(), fields.as_object()) {
            for (k, v) in from {
                into.insert(k.clone(), v.clone());
            }
        }
        let path = self.root.join(EVENTS_RELATIVE);
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        use std::io::Write as _;
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            let _ = writeln!(file, "{body}");
        }
    }

    /// Our own time command is not the human scrubbing: mute the scrub
    /// event for a moment.
    pub fn note_commanded_time(&mut self) {
        self.commanded_until = Some(Instant::now() + SCRUB_SETTLE * 2);
    }

    /// Watch the viewer's cursor: when it moves and then rests, and no
    /// command of ours moved it, the human scrubbed — one event.
    pub fn watch_live(&mut self, live: &Live) -> Option<Live> {
        let now = Instant::now();
        let moved = self.observed_live.as_ref() != Some(live);
        if moved {
            self.observed_live = Some(live.clone());
            self.live_moved_at = Some(now);
            return None;
        }
        let rested = self
            .live_moved_at
            .take_if(|t| now.duration_since(*t) >= SCRUB_SETTLE)?;
        let _ = rested;
        if self.commanded_until.is_some_and(|until| now < until) {
            return None;
        }
        (live.seconds.is_some() || live.sequence.is_some()).then(|| live.clone())
    }
}

pub fn parse_command(text: &str) -> Result<Command, String> {
    serde_json::from_str::<Command>(text).map_err(|e| format!("not a command: {e}"))
}

fn same_but_heartbeat(a: &StudioState, b: &StudioState) -> bool {
    let mut a = a.clone();
    let mut b = b.clone();
    a.heartbeat = 0.0;
    b.heartbeat = 0.0;
    a == b
}

fn write_atomic(path: &Path, body: &str) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, body)?;
    std::fs::rename(tmp, path)
}

pub fn epoch_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or_default()
}

fn epoch_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(name: &str) -> PathBuf {
        let root =
            std::env::temp_dir().join(format!("studio-control-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join(COMMANDS_RELATIVE)).expect("mkdir");
        root
    }

    #[test]
    fn commands_parse_by_verb_and_unknown_verbs_are_refused_by_name() {
        let open = parse_command(r#"{"verb":"open","section":"robots"}"#).expect("parses");
        assert_eq!(
            open,
            Command::Open {
                project: None,
                section: Some("robots".into()),
                artifact: None,
                table: None,
            }
        );
        let time = parse_command(r#"{"verb":"time","seconds":1.5,"play":true}"#).expect("parses");
        assert!(
            matches!(time, Command::Time { seconds: Some(s), play: Some(true), .. } if s == 1.5)
        );
        assert!(matches!(
            parse_command(r#"{"verb":"quit"}"#),
            Ok(Command::Quit)
        ));
        let err = parse_command(r#"{"verb":"dance"}"#).expect_err("refused");
        assert!(err.starts_with("not a command"), "{err}");
    }

    #[test]
    fn new_command_files_are_seen_once_and_acked_ones_are_never_replayed() {
        let root = temp_root("poll");
        let dir = root.join(COMMANDS_RELATIVE);
        std::fs::write(
            dir.join("1-open.json"),
            r#"{"verb":"open","section":"live"}"#,
        )
        .unwrap();
        std::fs::write(dir.join("0-quit.json"), r#"{"verb":"quit"}"#).unwrap();
        std::fs::write(dir.join("0-quit.ack.json"), r#"{"status":"done"}"#).unwrap();
        let mut control = Control::new(root.clone());
        let pending = control.poll();
        assert_eq!(pending.len(), 1, "the acked quit is not replayed");
        assert_eq!(pending[0].id, "1-open");
        control.ack("1-open", "done", None);
        assert!(dir.join("1-open.ack.json").is_file());
        control.last_poll = None;
        assert!(control.poll().is_empty(), "seen once");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn state_is_written_on_change_and_events_append() {
        let root = temp_root("state");
        let mut control = Control::new(root.clone());
        let state = StudioState {
            schema: STATE_SCHEMA,
            pid: 1,
            heartbeat: 0.0,
            project: "p".into(),
            project_name: "p".into(),
            section: "overview".into(),
            selected: None,
            table: None,
            live: Live::default(),
            presenter_running: false,
            jobs_running: 0,
        };
        control.record_state(state.clone());
        let text = std::fs::read_to_string(root.join(STATE_RELATIVE)).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(parsed["section"], "overview");
        assert!(parsed["heartbeat"].as_f64().unwrap() > 0.0);
        control.event("select", serde_json::json!({"artifact": "a@000000000000"}));
        control.event("open", serde_json::json!({"section": "robots"}));
        let lines: Vec<String> = std::fs::read_to_string(root.join(EVENTS_RELATIVE))
            .unwrap()
            .lines()
            .map(String::from)
            .collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("\"kind\":\"select\""));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn a_frame_is_written_as_a_png_scaled_to_the_asked_width_never_up() {
        let root = temp_root("shot");
        let control = Control::new(root.clone());
        let frame = egui::ColorImage::filled([400, 200], egui::Color32::from_rgb(10, 20, 30));
        let (path, w, h) = control
            .save_screenshot("1-screenshot", &frame, 100)
            .expect("saved");
        assert_eq!((w, h), (100, 50));
        assert!(path.ends_with("1-screenshot.png"));
        let back = image::open(&path).expect("a real png");
        assert_eq!((back.width(), back.height()), (100, 50));
        let (_, w, h) = control
            .save_screenshot("2-screenshot", &frame, 9000)
            .expect("saved");
        assert_eq!((w, h), (400, 200), "never upscaled");
        let empty = egui::ColorImage::filled([0, 0], egui::Color32::BLACK);
        assert!(control.save_screenshot("3", &empty, 100).is_err());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn a_scrub_is_reported_once_it_rests_and_never_after_our_own_command() {
        let root = temp_root("scrub");
        let mut control = Control::new(root.clone());
        let at = |s: f64| Live {
            recording: Some("r".into()),
            timeline: Some("time".into()),
            seconds: Some(s),
            sequence: None,
        };
        assert!(control.watch_live(&at(0.0)).is_none(), "first sight");
        assert!(control.watch_live(&at(1.0)).is_none(), "still moving");
        control.live_moved_at = Some(Instant::now() - SCRUB_SETTLE * 2);
        assert_eq!(
            control.watch_live(&at(1.0)),
            Some(at(1.0)),
            "rested: one event"
        );
        assert!(control.watch_live(&at(1.0)).is_none(), "not repeated");
        control.note_commanded_time();
        control.watch_live(&at(2.0));
        control.live_moved_at = Some(Instant::now() - SCRUB_SETTLE * 2);
        assert!(
            control.watch_live(&at(2.0)).is_none(),
            "ours, not the human's"
        );
        let _ = std::fs::remove_dir_all(root);
    }
}
