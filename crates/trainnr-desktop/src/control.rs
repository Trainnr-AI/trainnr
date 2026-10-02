//! The control surface: an agent drives the Studio through files under
//! `<project>/.index/` (mirrored from `trainnr/project/control.py`).
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

use crate::model::now_epoch;

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
        /// How the page lists its artifacts: cards, table or matrix.
        #[serde(default)]
        view: Option<String>,
        /// Open the command palette with this query typed.
        #[serde(default)]
        search: Option<String>,
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
    /// Run a scene in the MuJoCo viewport (a preview task by name), or
    /// stop it when `task` is absent.
    Simulate {
        #[serde(default)]
        task: Option<String>,
    },
    /// The simulate controls: run or pause, step, reset (to a keyframe
    /// or the initial state), speed, manual, one actuator or joint
    /// value, one visualization or rendering flag.
    Simulator {
        #[serde(default)]
        run: Option<bool>,
        #[serde(default)]
        step: Option<u32>,
        #[serde(default)]
        reset: Option<bool>,
        #[serde(default)]
        keyframe: Option<String>,
        #[serde(default)]
        speed: Option<f32>,
        #[serde(default)]
        manual: Option<bool>,
        #[serde(default)]
        actuator: Option<String>,
        #[serde(default)]
        joint: Option<String>,
        #[serde(default)]
        value: Option<f32>,
        #[serde(default)]
        flag: Option<String>,
        #[serde(default)]
        on: Option<bool>,
        /// One group mask bit: the group number with `on`, of `kind` —
        /// one of the kinds the scene reports (`model.groups`), its
        /// first unless said.
        #[serde(default)]
        group: Option<u32>,
        #[serde(default)]
        kind: Option<String>,
        /// One axis of the commanded twist in a walk scene, with `value`:
        /// vx (forward), vy (left) or wz (turn); `own` hands the commands
        /// back to the task. The twist goes to the followed world, else w0.
        #[serde(default)]
        command: Option<String>,
        /// The Inspect drawer: a tab by its word (`simulator::Tab::ALL`:
        /// control, joints, physics, visuals, commands) or close.
        #[serde(default)]
        inspect: Option<String>,
        /// The camera to a named view: front, side, top, reset.
        #[serde(default)]
        view: Option<String>,
        /// Many-worlds: none, worst, failing, cycle, or a world index.
        #[serde(default)]
        follow: Option<String>,
        /// The viewport alone on the page (true), or the Live page as it
        /// is (false) - what the `f` key toggles.
        #[serde(default)]
        fullscreen: Option<bool>,
    },
    Screenshot {
        /// The written image's width in pixels; taller frames scale to it.
        #[serde(default)]
        width: Option<u32>,
    },
    /// Bring a recording to the front of the viewer by its application
    /// id as the tool that streams it named it (the Sources list's name;
    /// Rerun migrates it to an entry name the same way on both sides).
    /// A human's click on a card leaves that card's recording in front,
    /// and a live twin started after it stayed a row in Sources
    /// (2026-09-26).
    Focus {
        recording: String,
    },
    Quit,
}

impl Command {
    /// Whether applying it changes what the window shows — what a
    /// capture must wait for after.
    pub fn moves_the_window(&self) -> bool {
        matches!(
            self,
            Command::Open { .. }
                | Command::Show { .. }
                | Command::Compare { .. }
                | Command::Time { .. }
                | Command::Panels { .. }
                | Command::Simulate { .. }
                | Command::Simulator { .. }
                | Command::Focus { .. }
        )
    }
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
    /// The cursor sits at the recording's newest time. Never written to
    /// the state file; `watch_live` reads it with `tip`.
    #[serde(skip)]
    pub at_tip: bool,
    /// The recording's newest time on the cursor's timeline: when it
    /// advances while the cursor rides it, a live stream is carrying
    /// the cursor, not a hand; a hand that reaches a tip that did not
    /// move is a scrub like any other. Never written to the state file.
    #[serde(skip)]
    pub tip: Option<i64>,
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
    /// The MuJoCo viewport: which preview scene runs, and the frames it
    /// drew to the screen in the last second (the on-screen rate).
    pub viewport_task: Option<String>,
    pub viewport_fps: Option<f32>,
    /// The simulator's clock and mode, from the stream's status.
    pub simulator: Option<SimulatorState>,
    /// The window's content size in logical points and its pixel ratio,
    /// so a capture's pixels can be read back as layout.
    pub window: Option<WindowState>,
    /// Set only in the pointer a project switch leaves behind in the old
    /// project's state file: the root the Studio moved to, so a door
    /// called under the old project finds the live window there instead
    /// of reading a stale heartbeat as a dead Studio (2026-09-27).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub moved_to: Option<String>,
}

#[derive(Serialize, Clone, PartialEq, Debug)]
pub struct WindowState {
    pub width: f32,
    pub height: f32,
    pub pixels_per_point: f32,
}

#[derive(Serialize, Clone, PartialEq, Debug)]
pub struct SimulatorState {
    pub time: f64,
    pub rtf: f64,
    pub paused: bool,
    pub manual: bool,
    pub speed: f64,
    pub render_ms: f64,
    /// Where the camera is: azimuth, elevation, distance, lookat x y z.
    pub camera: Vec<f64>,
}

// -- events: one struct writes `events.jsonl` and reads it back ---------------

/// Who did it, as the event names them.
pub const BY_AGENT: &str = "agent";
pub const BY_USER: &str = "user";
pub const BY_STUDIO: &str = "studio";
/// What was done: the event kinds the window writes.
pub const EVENT_OPEN: &str = "open";
pub const EVENT_SELECT: &str = "select";
pub const EVENT_DESELECT: &str = "deselect";
pub const EVENT_SHOW: &str = "show";
pub const EVENT_TABLE: &str = "table";
pub const EVENT_TIME: &str = "time";
pub const EVENT_KEY: &str = "key";

/// One line of `events.jsonl`: what the human, the agent or the Studio
/// itself did in the window. The same struct writes the line and reads
/// it back (`model::Model::events`), so a key can never drift between
/// the writer and the activity feed — the `show` line once wrote one
/// name and read another, and showed blank.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct Event {
    /// Nanoseconds since the epoch.
    #[serde(default)]
    pub t: f64,
    pub kind: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub by: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub section: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub table: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeline: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seconds: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sequence: Option<i64>,
    /// A key of the picture's (W A S D Q E) pressed: which, whether the
    /// picture had the keys, who had focus (none, picture, other,
    /// text), where the pointer was - so "the keys do nothing" can be
    /// read off the log (2026-09-12).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub picture: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub focus: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pointer: Option<[f32; 2]>,
}

impl Event {
    fn of(kind: &str, by: &str) -> Self {
        Self {
            kind: kind.to_owned(),
            by: by.to_owned(),
            ..Self::default()
        }
    }

    pub fn open(by: &str) -> Self {
        Self::of(EVENT_OPEN, by)
    }

    pub fn select(by: &str) -> Self {
        Self::of(EVENT_SELECT, by)
    }

    pub fn deselect(by: &str) -> Self {
        Self::of(EVENT_DESELECT, by)
    }

    pub fn show(by: &str) -> Self {
        Self::of(EVENT_SHOW, by)
    }

    pub fn table(by: &str) -> Self {
        Self::of(EVENT_TABLE, by)
    }

    pub fn time(by: &str) -> Self {
        Self::of(EVENT_TIME, by)
    }

    /// A pan key pressed over the page, with what decided its fate.
    pub fn key(by: &str, press: &crate::simulator::KeyPress) -> Self {
        let mut event = Self::of(EVENT_KEY, by);
        event.key = Some(press.key.clone());
        event.picture = Some(press.picture);
        event.focus = Some(press.focus.to_owned());
        event.pointer = press.pointer.map(|p| [p.x, p.y]);
        event
    }

    pub fn in_section(mut self, slug: String) -> Self {
        self.section = Some(slug);
        self
    }

    pub fn of_artifact(mut self, stamp: String) -> Self {
        self.artifact = Some(stamp);
        self
    }

    /// The table opened, or `None` when it was closed.
    pub fn on_table(mut self, title: Option<String>) -> Self {
        self.table = title;
        self
    }

    pub fn in_project(mut self, root: String) -> Self {
        self.project = Some(root);
        self
    }

    /// Where the viewer's cursor rested.
    pub fn at(mut self, live: &Live) -> Self {
        self.timeline = live.timeline.clone();
        self.seconds = live.seconds;
        self.sequence = live.sequence;
        self
    }

    pub fn epoch_seconds(&self) -> f64 {
        self.t / 1e9
    }
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
    /// The tip last seen, and whether the cursor's last move rode it.
    observed_tip: Option<i64>,
    following: bool,
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
            observed_tip: None,
            following: false,
        };
        control.forget_answered();
        control
    }

    /// A project switch: the commands directory moves with it, and the
    /// old project's state file becomes a pointer to the new root (the
    /// last state written there, `moved_to` set), never a stale
    /// heartbeat.
    pub fn set_root(&mut self, root: PathBuf) {
        if root != self.root {
            if let Some(mut last) = self.last_state.take() {
                last.moved_to = Some(root.display().to_string());
                last.heartbeat = now_epoch();
                if let Err(e) = write_atomic(
                    &self.root.join(STATE_RELATIVE),
                    &serde_json::to_string(&last).unwrap_or_default(),
                ) {
                    eprintln!("studio: cannot leave a pointer in {STATE_RELATIVE}: {e}");
                }
            }
            self.root = root;
            self.seen.clear();
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
            "t": now_epoch(),
        });
        if let Some(into) = body.as_object_mut() {
            into.extend(extra);
        }
        let path = dir.join(format!("{id}.ack.json"));
        if let Err(e) = write_atomic(&path, &body.to_string()) {
            eprintln!("studio: cannot answer command {id}: {e}");
        }
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
            .is_none_or(|last| discrete(last) != discrete(&state));
        if !(due || changed) {
            return;
        }
        state.heartbeat = now_epoch();
        if let Err(e) = write_atomic(
            &self.root.join(STATE_RELATIVE),
            &serde_json::to_string(&state).unwrap_or_default(),
        ) {
            eprintln!("studio: cannot write {STATE_RELATIVE}: {e}");
        }
        self.last_state = Some(state);
        self.last_written = Some(now);
    }

    /// Append one event: what the human, the agent or the Studio did.
    pub fn event(&self, mut event: Event) {
        event.t = epoch_nanos();
        let Ok(line) = serde_json::to_string(&event) else {
            return;
        };
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
            let _ = writeln!(file, "{line}");
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
            // riding the tip: at it, and the tip itself moved since last seen
            self.following = live.at_tip && live.tip != self.observed_tip;
            self.observed_tip = live.tip;
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
        // A stream that paused between its bursts rested the cursor at a
        // tip it had just carried forward; that is the stream's doing, not
        // the human's (a Brush run logged "you moved the time cursor" eight
        // times a minute, 2026-09-23). A hand that dragged the cursor to a
        // tip that stood still is a scrub and stays one.
        if self.following {
            return None;
        }
        (live.seconds.is_some() || live.sequence.is_some()).then(|| live.clone())
    }
}

pub fn parse_command(text: &str) -> Result<Command, String> {
    serde_json::from_str::<Command>(text).map_err(|e| format!("not a command: {e}"))
}

/// The facts a change of which writes the state at once. The rest — the
/// simulator's clock and rate, the camera, the on-screen frame rate, the
/// viewer's cursor — move every frame while a scene runs and ride the
/// heartbeat instead; comparing whole states wrote and renamed the file
/// at frame rate (2026-09-13).
fn discrete(state: &StudioState) -> impl PartialEq + '_ {
    (
        state.pid,
        &state.project,
        &state.project_name,
        &state.section,
        &state.selected,
        &state.table,
        &state.live.recording,
        state.presenter_running,
        state.jobs_running,
        &state.viewport_task,
        state
            .simulator
            .as_ref()
            .map(|s| (s.paused, s.manual, s.speed.to_bits())),
        state.window.as_ref().map(|w| {
            (
                w.width.to_bits(),
                w.height.to_bits(),
                w.pixels_per_point.to_bits(),
            )
        }),
    )
}

/// Write a file in one replace: a reader never sees a half-written
/// state, ack or intent. The one such routine in the crate.
/// What rerun 0.36 lets an application id contain besides ASCII letters
/// and digits; mirrored from `trainnr/viz.py` (`ENTRY_NAME_EXTRA`),
/// which folds every stream's id the same way before it is sent.
const ENTRY_NAME_EXTRA: &str = "_-. []:";
const ENTRY_NAME_MAX: usize = 180;

/// A stamp or a tool's name as the viewer holds it: every character rerun
/// refuses became `-` on the way in (a stamp's `@`), so the same folding
/// here finds the recording; the raw name drew a "requires migration"
/// toast and then "no recording for it" (2026-09-27).
pub fn entry_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || ENTRY_NAME_EXTRA.contains(c) {
                c
            } else {
                '-'
            }
        })
        .take(ENTRY_NAME_MAX)
        .collect()
}

pub fn write_atomic(path: &Path, body: &str) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, body)?;
    std::fs::rename(tmp, path)
}

/// Nanoseconds since the epoch as the event line carries them (an f64
/// keeps them to within a microsecond, enough to order a feed).
fn epoch_nanos() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as f64)
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
                view: None,
                search: None,
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
        let focus = parse_command(r#"{"verb":"focus","recording":"trainnr-sim-x"}"#)
            .expect("parses");
        assert!(matches!(focus, Command::Focus { recording } if recording == "trainnr-sim-x"));
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
    fn entry_names_fold_what_rerun_refuses() {
        assert_eq!(entry_name("go2-c2@4dc757293b97"), "go2-c2-4dc757293b97");
        assert_eq!(entry_name("a@1 vs b@2"), "a-1 vs b-2");
        assert_eq!(entry_name("trainnr-sim-deploy:go2-c2"), "trainnr-sim-deploy:go2-c2");
        assert_eq!(entry_name(&"x".repeat(200)).len(), ENTRY_NAME_MAX);
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
            viewport_task: None,
            viewport_fps: None,
            simulator: None,
            window: None,
            moved_to: None,
        };
        control.record_state(state.clone());
        let text = std::fs::read_to_string(root.join(STATE_RELATIVE)).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(parsed["section"], "overview");
        assert!(parsed["heartbeat"].as_f64().unwrap() > 0.0);
        assert!(parsed.get("moved_to").is_none(), "a live state carries no pointer");
        control.event(Event::select(BY_USER).of_artifact("a@000000000000".into()));
        control.event(Event::open(BY_AGENT).in_section("robots".into()));
        let lines: Vec<String> = std::fs::read_to_string(root.join(EVENTS_RELATIVE))
            .unwrap()
            .lines()
            .map(String::from)
            .collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("\"kind\":\"select\""));
        // A project switch leaves a pointer behind in the old project.
        let other = temp_root("state-other");
        control.set_root(other.clone());
        let text = std::fs::read_to_string(root.join(STATE_RELATIVE)).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(parsed["moved_to"], other.display().to_string());
        assert_eq!(parsed["pid"], 1);
        control.record_state(state.clone());
        assert!(other.join(STATE_RELATIVE).is_file());
        let _ = std::fs::remove_dir_all(other);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn an_event_reads_back_with_the_keys_it_was_written_with() {
        let root = temp_root("events");
        let control = Control::new(root.clone());
        control.event(Event::show(BY_USER).of_artifact("go2@abc".into()));
        control.event(Event::table(BY_AGENT).on_table(None));
        let rested = Live {
            recording: Some("r".into()),
            timeline: Some("time".into()),
            seconds: Some(1.5),
            sequence: None,
            at_tip: false,
            tip: None,
        };
        control.event(Event::time(BY_USER).at(&rested));
        let text = std::fs::read_to_string(root.join(EVENTS_RELATIVE)).unwrap();
        let back: Vec<Event> = text
            .lines()
            .map(|l| serde_json::from_str(l).expect("the line we wrote"))
            .collect();
        assert_eq!(back[0].kind, EVENT_SHOW);
        assert_eq!(back[0].artifact.as_deref(), Some("go2@abc"));
        assert!(back[0].t > 0.0 && back[0].epoch_seconds() > 1.0e9);
        assert_eq!(back[1].kind, EVENT_TABLE);
        assert_eq!(back[1].table, None, "a closed table is absent, not null");
        assert!(!text.lines().nth(1).unwrap().contains("\"table\":"));
        assert_eq!(back[2].timeline.as_deref(), Some("time"));
        assert_eq!(back[2].seconds, Some(1.5));
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
            at_tip: false,
            tip: None,
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
        // A live stream rests the cursor at a tip it just carried forward.
        let riding = Live {
            at_tip: true,
            tip: Some(3_000_000_000),
            ..at(3.0)
        };
        control.watch_live(&riding);
        control.live_moved_at = Some(Instant::now() - SCRUB_SETTLE * 2);
        assert!(
            control.watch_live(&riding).is_none(),
            "the stream's, not the human's"
        );
        // The stream stopped; the human drags the cursor back, then to the
        // tip that did not move: both are scrubs (our own command's window
        // from above is over by then).
        control.commanded_until = None;
        let back = Live {
            at_tip: false,
            tip: Some(3_000_000_000),
            ..at(1.0)
        };
        control.watch_live(&back);
        control.live_moved_at = Some(Instant::now() - SCRUB_SETTLE * 2);
        assert_eq!(
            control.watch_live(&back),
            Some(back.clone()),
            "a scrub back"
        );
        control.watch_live(&riding);
        control.live_moved_at = Some(Instant::now() - SCRUB_SETTLE * 2);
        assert_eq!(
            control.watch_live(&riding),
            Some(riding.clone()),
            "to a tip that stood still: the human's"
        );
        let _ = std::fs::remove_dir_all(root);
    }
}
