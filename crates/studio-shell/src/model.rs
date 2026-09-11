//! What the Studio knows about a project: the index and the job table,
//! both files the Python side writes and this side polls.
//!
//! The index (`<project>/.index/project.json`) is written by
//! `rq_pipeline.project.write_index` and the `describe_project` MCP tool.
//! The job table (`<project>/mcp-jobs/<id>.json` + `<id>.exit`) is written
//! by `rq_pipeline.mcp_jobs.JobManager` for every act tool the agent
//! calls. The Studio never speaks MCP itself and never walks the project's
//! artifact files: these two are the whole contract between the halves
//! (decision 2026-09-08), re-read whenever their modification times move,
//! so a tool call from the developer's agent updates this window with no
//! protocol in between. The files are the truth; this is a cache.

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::{Duration, SystemTime};

use serde::Deserialize;

pub use crate::control::Event;
use crate::detail::Detail;

/// The index's schema, as `rq_pipeline/project/index.py` writes it
/// (`INDEX_SCHEMA`). The family (before the `/`) must match; a newer
/// minor version is read, its unknown fields ignored.
pub const INDEX_SCHEMA: &str = "trainnr-project-index/1";
/// Mirrored from `rq_pipeline/project/locate.py`.
const INDEX_RELATIVE: &str = ".index/project.json";
pub const MANIFEST_FILE: &str = "project.json";
/// Where projects live in a checkout (`PROJECTS_DIR_NAME`), and the two
/// the Studio opens when none is named.
pub const PROJECTS_HOME: &str = "projects";
const DEFAULT_PROJECT_NAME: &str = "default";
const SAMPLE_PROJECT_NAME: &str = "sample";
/// Mirrored from `rq_pipeline/mcp_jobs.py` (`JOBS_DIR_NAME`).
const JOBS_DIR: &str = "mcp-jobs";
/// Mirrored from `rq_pipeline/project/present.py` (`INTENT_FILE`): the
/// Studio writes `{"stamp": ...}` here; the presenter streams that
/// artifact into the viewer and clears it.
const INTENT_RELATIVE: &str = ".index/present.json";
/// The environment variable both halves honour (`PROJECT_ENV`).
pub const PROJECT_ENV: &str = "TRAINNR_PROJECT";
/// How often the files' mtimes are polled. A tool call rewrites the
/// index in one atomic replace; a second of latency is invisible next to
/// the seconds the tool itself took.
pub const RELOAD_EVERY: Duration = Duration::from_secs(1);
/// How often the projects home is re-walked when nothing moved: every
/// project's index is parsed on a walk, so it is not a per-frame cost.
const PROJECTS_RESCAN_EVERY: Duration = Duration::from_secs(10);
/// What is shown when a fact was never recorded: the reader must not
/// mistake an empty cell for a value. One spelling across the window.
pub const UNRECORDED: &str = "unrecorded";
/// Summary keys never shown as a fact or a table column.
pub const HIDDEN_KEYS: &[&str] = &["files"];

/// `ProjectIndex`, as `rq_pipeline.project.index` writes it. Unknown
/// fields are ignored so an older Studio still opens a newer index.
#[derive(Deserialize, Default)]
pub struct Index {
    #[serde(skip)]
    by_stamp: std::collections::HashMap<String, usize>,
    #[serde(skip)]
    by_hash: std::collections::HashMap<String, Vec<usize>>,
    #[serde(default)]
    pub schema: String,
    #[serde(default)]
    pub project: String,
    #[serde(default)]
    pub root: String,
    #[serde(default)]
    pub indexed: String,
    #[serde(default)]
    pub artifacts: Vec<Artifact>,
    #[serde(default)]
    pub states: Vec<State>,
    #[serde(default)]
    pub next_move: Option<String>,
    #[serde(default)]
    pub refused: Vec<Refused>,
}

#[derive(Deserialize, Clone)]
pub struct Artifact {
    pub kind: String,
    pub stamp: String,
    pub path: String,
    #[serde(default)]
    pub cites: serde_json::Map<String, serde_json::Value>,
    #[serde(default)]
    pub summary: serde_json::Map<String, serde_json::Value>,
    /// A picture, relative to the project root, when the kind has one.
    #[serde(default)]
    pub preview: Option<String>,
    /// The detail view's file, relative to the project root, when the
    /// kind has a writer (`rq_pipeline/project/details.py`).
    #[serde(default)]
    pub detail: Option<String>,
    /// When it entered and when it last changed (ISO 8601 UTC), from the
    /// index (`rq_pipeline/project/index.py`, `_times`); None when the
    /// artifact has no files.
    #[serde(default)]
    pub created: Option<String>,
    #[serde(default)]
    pub updated: Option<String>,
    /// Stamps of the artifacts in the project that cite this one.
    #[serde(default)]
    pub cited_by: Vec<String>,
    /// `updated` parsed once at load (`Index::finish`); an artifact with
    /// no time sorts last.
    #[serde(skip)]
    updated_epoch: Option<f64>,
}

impl Artifact {
    /// Seconds since the epoch of the last change, for ordering; an
    /// artifact with no time sorts last.
    pub fn updated_epoch(&self) -> f64 {
        self.updated_epoch
            .or_else(|| self.updated.as_deref().and_then(epoch_of))
            .unwrap_or(f64::NEG_INFINITY)
    }
}

/// How many events the activity feed keeps in memory.
pub const EVENTS_KEPT: usize = 60;

/// The presenter's last answer (`.index/present-status.json`): what it
/// showed, or why it could not.
#[derive(Deserialize, Clone, Debug, Default, PartialEq)]
pub struct PresentStatus {
    #[serde(default)]
    pub shown: Option<String>,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub stamp: Option<String>,
}

const PRESENT_STATUS_RELATIVE: &str = ".index/present-status.json";

#[derive(Deserialize)]
pub struct State {
    pub name: String,
    #[serde(default)]
    pub proved_by: Vec<String>,
    pub present: bool,
    /// False when this loop never passes through the stage (a
    /// reinforcement-learning loop has no dataset); `note` says why.
    #[serde(default = "yes")]
    pub needed: bool,
    #[serde(default)]
    pub note: Option<String>,
}

fn yes() -> bool {
    true
}

#[derive(Deserialize)]
pub struct Refused {
    pub path: String,
    pub reason: String,
}

/// `JobRecord`, as `rq_pipeline.mcp_jobs` writes it, plus the exit code
/// from the sibling `.exit` file when the job has ended.
#[derive(Deserialize, Clone)]
pub struct Job {
    pub id: String,
    pub tool: String,
    #[serde(default)]
    pub argv: Vec<String>,
    pub started: f64,
    #[serde(default)]
    pub pid: u32,
    #[serde(skip)]
    pub exit: Option<i32>,
}

impl Job {
    /// Running means no exit recorded AND the process still there: a
    /// job whose runner was killed before it could write the exit file
    /// showed as running for an hour after it ended (2026-09-10).
    pub fn running(&self) -> bool {
        self.exit.is_none() && pid_alive(self.pid)
    }
}

/// Whether a process id is alive. Linux answers through `/proc`; other
/// platforms have no dependency-free check here and answer "alive", so
/// there the exit file alone decides (the runner writes it).
fn pid_alive(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    if cfg!(target_os = "linux") {
        return std::path::Path::new(&format!("/proc/{pid}")).exists();
    }
    true
}

impl Index {
    /// Once per load: parse every time, index every stamp and hash, so
    /// a frame never re-parses or scans the artifact list.
    pub fn finish(&mut self) {
        for a in &mut self.artifacts {
            a.updated_epoch = a.updated.as_deref().and_then(epoch_of);
        }
        self.by_stamp = self
            .artifacts
            .iter()
            .enumerate()
            .map(|(i, a)| (a.stamp.clone(), i))
            .collect();
        self.by_hash.clear();
        for (i, a) in self.artifacts.iter().enumerate() {
            let (_, hash) = split_stamp(&a.stamp);
            if !hash.is_empty() {
                self.by_hash.entry(hash.to_owned()).or_default().push(i);
            }
        }
    }

    /// The artifact a stamp names: by the whole stamp, else by its hash
    /// when exactly one artifact carries it (a cite's name half is a
    /// label; the project may hold the same version under another name).
    pub fn artifact(&self, stamp: &str) -> Option<&Artifact> {
        if let Some(&i) = self.by_stamp.get(stamp) {
            return self.artifacts.get(i);
        }
        let (_, hash) = split_stamp(stamp);
        match self.by_hash.get(hash).map(Vec::as_slice) {
            Some([i]) => self.artifacts.get(*i),
            _ => None,
        }
    }

    pub fn by_kind(&self, kind: &str) -> Vec<&Artifact> {
        self.artifacts.iter().filter(|a| a.kind == kind).collect()
    }

    pub fn count(&self, kind: &str) -> usize {
        self.artifacts.iter().filter(|a| a.kind == kind).count()
    }
}

/// One polled file: its path, its last-seen mtime, and the parse.
struct Watched<T> {
    path: PathBuf,
    seen: Option<SystemTime>,
    value: Option<T>,
}

impl<T> Watched<T> {
    fn new(path: PathBuf) -> Self {
        Self {
            path,
            seen: None,
            value: None,
        }
    }

    /// True when the file's mtime moved (or it appeared / vanished).
    fn changed(&mut self) -> bool {
        let modified = std::fs::metadata(&self.path)
            .and_then(|m| m.modified())
            .ok();
        if modified != self.seen {
            self.seen = modified;
            true
        } else {
            false
        }
    }

    /// The value, re-read through `load` only when the file moved: one
    /// stat per call, never a parse.
    fn current(&mut self, load: impl FnOnce(&Path) -> Option<T>) -> Option<&T> {
        if self.changed() {
            self.value = load(&self.path);
        }
        self.value.as_ref()
    }
}

/// The family of a schema string: `trainnr-detail/5` → `trainnr-detail`.
/// A Studio reads any version of the families it knows; a version bump
/// within a family adds fields, which serde ignores.
pub fn schema_family(schema: &str) -> &str {
    schema.split_once('/').map_or(schema, |(family, _)| family)
}

/// Whether a file written for `written` is one this Studio reads
/// (`ours`). An unstamped file is taken as is: older writers wrote none.
pub fn schema_compatible(written: &str, ours: &str) -> bool {
    written.is_empty() || schema_family(written) == schema_family(ours)
}

/// One project under `projects/`, as the Projects page and the switcher
/// list it — read from its own index when it has one.
pub struct ProjectSummary {
    pub name: String,
    pub root: PathBuf,
    pub stages_proved: usize,
    pub stages: usize,
    pub artifacts: usize,
    pub indexed: String,
    /// The first artifact preview found, for the project's tile.
    pub preview: Option<PathBuf>,
}

pub struct Model {
    /// `projects/` in the checkout: where the switcher looks.
    projects_home: PathBuf,
    /// Every project under the home, as last walked (`refresh_projects`).
    projects: Vec<ProjectSummary>,
    projects_home_seen: Option<SystemTime>,
    projects_scanned_at: Option<std::time::Instant>,
    pub project_root: PathBuf,
    index: Watched<Index>,
    /// The selected artifact's detail file, re-read only when it moves:
    /// the drawer asks for it every frame it is open.
    detail: RefCell<Option<Watched<Rc<Detail>>>>,
    jobs_dir: PathBuf,
    jobs_seen: Option<SystemTime>,
    pub jobs: Vec<Job>,
    /// The newest window events (`.index/events.jsonl`), newest first.
    pub events: Vec<Event>,
    events_seen: Option<std::time::SystemTime>,
    /// The presenter's last answer, and when its file last changed.
    pub present_status: Option<PresentStatus>,
    present_seen: Option<std::time::SystemTime>,
    /// Why there is no index, when there is none.
    pub problem: Option<String>,
    last_poll: Option<std::time::Instant>,
}

impl Model {
    /// The project the Studio opens: `$TRAINNR_PROJECT`, else the
    /// checkout's default, else the committed sample — the same rule the
    /// Python side applies, plus the sample so a fresh clone shows a page
    /// instead of an error.
    pub fn open(repo_root: &Path) -> Self {
        let home = repo_root.join(PROJECTS_HOME);
        let root = std::env::var_os(PROJECT_ENV)
            .map(PathBuf::from)
            .filter(|p| p.join(MANIFEST_FILE).is_file())
            .or_else(|| {
                let default = home.join(DEFAULT_PROJECT_NAME);
                default.join(MANIFEST_FILE).is_file().then_some(default)
            })
            .unwrap_or_else(|| home.join(SAMPLE_PROJECT_NAME));
        Self::at(root, home)
    }

    /// Open a specific project directory.
    pub fn at(root: PathBuf, projects_home: PathBuf) -> Self {
        let mut model = Self {
            projects_home,
            projects: Vec::new(),
            projects_home_seen: None,
            projects_scanned_at: None,
            index: Watched::new(root.join(INDEX_RELATIVE)),
            detail: RefCell::new(None),
            jobs_dir: root.join(JOBS_DIR),
            jobs_seen: None,
            jobs: Vec::new(),
            events: Vec::new(),
            events_seen: None,
            present_status: None,
            present_seen: None,
            problem: None,
            project_root: root,
            last_poll: None,
        };
        model.reload_index();
        model.reload_jobs();
        model.rescan_projects();
        model
    }

    pub fn index(&self) -> Option<&Index> {
        self.index.value.as_ref()
    }

    /// Switch to another project directory, keeping the projects home.
    pub fn switch(&mut self, root: PathBuf) {
        *self = Self::at(root, self.projects_home.clone());
    }

    /// An artifact's detail file as an absolute path, when it has one.
    pub fn detail_path(&self, artifact: &Artifact) -> Option<PathBuf> {
        artifact
            .detail
            .as_ref()
            .map(|rel| self.project_root.join(rel))
            .filter(|p| p.is_file())
    }

    /// An artifact's detail, parsed once and re-read when its file moves.
    /// One artifact is watched at a time: the drawer shows one.
    pub fn detail(&self, artifact: &Artifact) -> Option<Rc<Detail>> {
        let path = self.detail_path(artifact)?;
        let mut slot = self.detail.borrow_mut();
        let watched = match slot.as_mut() {
            Some(w) if w.path == path => w,
            _ => slot.insert(Watched::new(path)),
        };
        watched.current(|p| Detail::load(p).map(Rc::new)).cloned()
    }

    /// An artifact's preview as an absolute path, when it has one.
    pub fn preview_path(&self, artifact: &Artifact) -> Option<PathBuf> {
        artifact
            .preview
            .as_ref()
            .map(|rel| self.project_root.join(rel))
            .filter(|p| p.is_file())
    }

    /// Every project under the projects home, by name, with what its own
    /// index says about it — as last walked. The top bar reads this every
    /// frame; the walk itself happens in `refresh_projects`.
    pub fn projects(&self) -> &[ProjectSummary] {
        &self.projects
    }

    /// Re-walk the projects home when it moved, when this project's index
    /// was reloaded, or when the last walk is old — at most once per
    /// [`RELOAD_EVERY`]. The switcher's popup and the Projects page call
    /// it when they open, so a fresh project shows the moment it is asked
    /// for.
    pub fn refresh_projects(&mut self) {
        let now = std::time::Instant::now();
        let recent = self
            .projects_scanned_at
            .is_some_and(|t| now.duration_since(t) < RELOAD_EVERY);
        if recent {
            return;
        }
        self.rescan_projects();
    }

    fn rescan_projects(&mut self) {
        self.projects_scanned_at = Some(std::time::Instant::now());
        self.projects_home_seen = std::fs::metadata(&self.projects_home)
            .and_then(|m| m.modified())
            .ok();
        self.projects = self.walk_projects();
    }

    fn walk_projects(&self) -> Vec<ProjectSummary> {
        let Ok(entries) = std::fs::read_dir(&self.projects_home) else {
            return Vec::new();
        };
        let mut found: Vec<ProjectSummary> = entries
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.join(MANIFEST_FILE).is_file())
            .map(|root| {
                let index: Option<Index> = std::fs::read_to_string(root.join(INDEX_RELATIVE))
                    .ok()
                    .and_then(|t| serde_json::from_str(&t).ok());
                let name = index
                    .as_ref()
                    .map(|i| i.project.clone())
                    .filter(|n| !n.is_empty())
                    .or_else(|| {
                        std::fs::read_to_string(root.join(MANIFEST_FILE))
                            .ok()
                            .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
                            .and_then(|v| v.get("name")?.as_str().map(str::to_owned))
                    })
                    .unwrap_or_else(|| {
                        root.file_name()
                            .map(|n| n.to_string_lossy().into_owned())
                            .unwrap_or_default()
                    });
                let preview = index.as_ref().and_then(|i| {
                    i.artifacts
                        .iter()
                        .filter_map(|a| a.preview.as_ref())
                        .map(|rel| root.join(rel))
                        .find(|p| p.is_file())
                });
                ProjectSummary {
                    name,
                    stages_proved: index
                        .as_ref()
                        .map_or(0, |i| i.states.iter().filter(|s| s.present).count()),
                    stages: index
                        .as_ref()
                        .map_or(0, |i| i.states.iter().filter(|s| s.needed).count()),
                    artifacts: index.as_ref().map_or(0, |i| i.artifacts.len()),
                    indexed: index.as_ref().map_or(String::new(), |i| i.indexed.clone()),
                    preview,
                    root,
                }
            })
            .collect();
        found.sort_by(|a, b| a.name.cmp(&b.name));
        found
    }

    /// The project's display name: the index's, else the directory's.
    pub fn name(&self) -> String {
        self.index()
            .map(|i| i.project.clone())
            .filter(|n| !n.is_empty())
            .unwrap_or_else(|| {
                self.project_root
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default()
            })
    }

    pub fn running_jobs(&self) -> usize {
        self.jobs.iter().filter(|j| j.running()).count()
    }

    /// Ask the presenter to show an artifact: write the intent file. The
    /// presenter (spawned by the shell) streams it and deletes the file.
    pub fn request_show(&self, stamp: &str) -> std::io::Result<()> {
        self.forget_present_status();
        self.write_intent(serde_json::json!({ "stamp": stamp }))
    }

    /// A new request makes the presenter's last answer stale: remove it,
    /// so an old failure is never shown as this request's.
    fn forget_present_status(&self) {
        let _ = std::fs::remove_file(self.project_root.join(PRESENT_STATUS_RELATIVE));
    }

    /// Ask the presenter for two artifacts side by side (`{"stamps": [a, b]}`).
    pub fn request_compare(&self, a: &str, b: &str) -> std::io::Result<()> {
        self.forget_present_status();
        self.write_intent(serde_json::json!({ "stamps": [a, b] }))
    }

    fn write_intent(&self, body: serde_json::Value) -> std::io::Result<()> {
        crate::control::write_atomic(&self.project_root.join(INTENT_RELATIVE), &body.to_string())
    }

    /// The artifact a stamp names (`Index::artifact`).
    pub fn artifact(&self, stamp: &str) -> Option<&Artifact> {
        self.index()?.artifact(stamp)
    }

    /// Poll at most once a second; re-read whatever moved.
    pub fn refresh(&mut self) {
        let now = std::time::Instant::now();
        if self
            .last_poll
            .is_some_and(|t| now.duration_since(t) < RELOAD_EVERY)
        {
            return;
        }
        self.last_poll = Some(now);
        let mut projects_stale = self
            .projects_scanned_at
            .is_none_or(|t| now.duration_since(t) >= PROJECTS_RESCAN_EVERY);
        if self.index.changed() {
            self.reload_index();
            projects_stale = true;
        }
        let home_modified = std::fs::metadata(&self.projects_home)
            .and_then(|m| m.modified())
            .ok();
        if home_modified != self.projects_home_seen {
            projects_stale = true;
        }
        if projects_stale {
            self.rescan_projects();
        }
        let jobs_modified = std::fs::metadata(&self.jobs_dir)
            .and_then(|m| m.modified())
            .ok();
        // A directory's mtime moves when entries are added; an exit file
        // landing is such an entry, so this catches job ends too.
        if jobs_modified != self.jobs_seen || self.jobs.iter().any(Job::running) {
            self.jobs_seen = jobs_modified;
            self.reload_jobs();
        }
        let events_path = self.project_root.join(crate::control::EVENTS_RELATIVE);
        let events_modified = std::fs::metadata(&events_path)
            .and_then(|m| m.modified())
            .ok();
        if events_modified != self.events_seen {
            self.events_seen = events_modified;
            self.reload_events(&events_path);
        }
        let status_path = self.project_root.join(PRESENT_STATUS_RELATIVE);
        let status_modified = std::fs::metadata(&status_path)
            .and_then(|m| m.modified())
            .ok();
        if status_modified != self.present_seen {
            self.present_seen = status_modified;
            self.present_status = std::fs::read_to_string(&status_path)
                .ok()
                .and_then(|text| serde_json::from_str(&text).ok());
        }
    }

    fn reload_index(&mut self) {
        match std::fs::read_to_string(&self.index.path) {
            Ok(text) => match serde_json::from_str::<Index>(&text).map(|mut i| {
                i.finish();
                i
            }) {
                Ok(index) if !schema_compatible(&index.schema, INDEX_SCHEMA) => {
                    self.index.value = None;
                    self.problem = Some(format!(
                        "{} was written for schema {}; this Studio reads {INDEX_SCHEMA}. \
                         Re-index with a matching pipeline, or update the Studio.",
                        self.index.path.display(),
                        index.schema
                    ));
                }
                Ok(index) => {
                    self.index.value = Some(index);
                    self.problem = None;
                }
                Err(err) => {
                    self.problem = Some(format!(
                        "{} is not an index this Studio can read: {err}",
                        self.index.path.display()
                    ));
                }
            },
            Err(_) => {
                self.index.value = None;
                self.problem = Some(if self.project_root.join(MANIFEST_FILE).is_file() {
                    format!(
                        "No index yet at {}.\nAsk your agent to run `describe_project`, \
                         or set {PROJECT_ENV} to another project.",
                        self.index.path.display()
                    )
                } else {
                    format!(
                        "No project at {}.\nAsk your agent to run `create_project`, \
                         or set {PROJECT_ENV} to a project directory.",
                        self.project_root.display()
                    )
                });
            }
        }
    }

    fn reload_events(&mut self, path: &Path) {
        let text = std::fs::read_to_string(path).unwrap_or_default();
        let mut events: Vec<Event> = text
            .lines()
            .rev()
            .take(EVENTS_KEPT)
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect();
        events.sort_by(|a, b| b.t.total_cmp(&a.t));
        self.events = events;
    }

    fn reload_jobs(&mut self) {
        let Ok(entries) = std::fs::read_dir(&self.jobs_dir) else {
            self.jobs.clear();
            return;
        };
        let mut jobs: Vec<Job> = entries
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "json"))
            .filter_map(|p| {
                let text = std::fs::read_to_string(&p).ok()?;
                let mut job: Job = serde_json::from_str(&text).ok()?;
                job.exit = std::fs::read_to_string(p.with_extension("exit"))
                    .ok()
                    .and_then(|s| s.trim().parse().ok());
                Some(job)
            })
            .collect();
        jobs.sort_by(|a, b| b.started.total_cmp(&a.started));
        self.jobs = jobs;
    }
}

/// `name@hash` → (`name`, `hash`); a bare name has an empty hash.
pub fn split_stamp(stamp: &str) -> (&str, &str) {
    stamp.split_once('@').unwrap_or((stamp, ""))
}

/// A one-line gloss of an artifact's summary for a table row.
pub fn summary_line(summary: &serde_json::Map<String, serde_json::Value>) -> Option<String> {
    let parts: Vec<String> = summary
        .iter()
        .filter(|(k, _)| !HIDDEN_KEYS.contains(&k.as_str()))
        // A claim or a success rate reads on its own; a key would be noise.
        // The keys are the index writer's (`project/index.py`,
        // `_summary_finding` and `_summary_certificate`).
        .map(|(k, v)| match k.as_str() {
            "claim" | "success" => render_value(v),
            _ => format!("{k} {}", render_value(v)),
        })
        .collect();
    (!parts.is_empty()).then(|| parts.join(" · "))
}

/// A JSON value as a person reads it: a string as itself, a list joined
/// by commas, nothing for `null` (an absent fact is never the word
/// "null"), an object pretty-printed. The one renderer in the crate.
pub fn render_value(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => String::new(),
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(items) => items
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| v.to_string())
            })
            .collect::<Vec<_>>()
            .join(", "),
        serde_json::Value::Object(_) => serde_json::to_string_pretty(value).unwrap_or_default(),
        other => other.to_string(),
    }
}

/// `2026-09-08T19:34:51+00:00` → `2026-09-08 19:34 UTC`.
pub fn short_time(iso: &str) -> String {
    let date = iso.get(..10).unwrap_or(iso);
    let time = iso.get(11..16).unwrap_or("");
    if time.is_empty() {
        date.to_owned()
    } else {
        format!("{date} {time} UTC")
    }
}

/// A duration in seconds → `4s`, `2m 10s`, `1h 03m`.
pub fn elapsed(seconds: f64) -> String {
    let s = seconds.max(0.0) as u64;
    if s < 60 {
        format!("{s}s")
    } else if s < 3600 {
        format!("{}m {:02}s", s / 60, s % 60)
    } else {
        format!("{}h {:02}m", s / 3600, (s / 60) % 60)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_index_shape_round_trips() {
        let text = r#"{"schema":"trainnr-project-index/1","project":"p","root":"/x",
            "indexed":"2026-09-08T19:34:51+00:00",
            "artifacts":[{"kind":"batch","stamp":"b@000000000000","path":"batches/b",
              "cites":{"expert":"e@111111111111"},"summary":{"episodes":2}}],
            "states":[{"name":"robot known","proved_by":[],"present":false}],
            "next_move":"onboard a robot","refused":[]}"#;
        let index: Index = serde_json::from_str(text).expect("parses");
        assert_eq!(index.project, "p");
        assert_eq!(index.by_kind("batch")[0].stamp, "b@000000000000");
        assert_eq!(index.count("robot"), 0);
        assert!(!index.states[0].present);
        assert_eq!(short_time(&index.indexed), "2026-09-08 19:34 UTC");
    }

    #[test]
    fn unknown_fields_do_not_break_an_older_studio() {
        let text = r#"{"project":"p","future_field":42,"states":[],"artifacts":[]}"#;
        let index: Index = serde_json::from_str(text).expect("tolerant");
        assert_eq!(index.project, "p");
        assert!(index.next_move.is_none());
    }

    #[test]
    fn a_job_record_parses_and_is_running_while_its_process_lives() {
        let alive = std::process::id();
        let text = format!(
            r#"{{"id":"generate-demos-1a2b","tool":"generate-demos",
            "argv":["uv","run"],"cwd":"/p","log":"/p/x.log","pid":{alive},"started":1757000000.5}}"#
        );
        let job: Job = serde_json::from_str(&text).expect("parses");
        assert!(job.running());
        assert_eq!(job.tool, "generate-demos");
        // An exit recorded ends it whatever the pid says.
        let mut ended = job.clone();
        ended.exit = Some(0);
        assert!(!ended.running());
        // No exit and no process (a runner killed before it wrote one):
        // not running on Linux, where the check exists.
        let mut gone = job.clone();
        gone.pid = 0;
        assert!(!gone.running());
    }

    #[test]
    fn summary_lines_skip_file_lists() {
        let mut m = serde_json::Map::new();
        m.insert("files".into(), serde_json::json!(["a", "b"]));
        assert!(summary_line(&m).is_none());
        m.insert("episodes".into(), serde_json::json!(2));
        assert_eq!(summary_line(&m).as_deref(), Some("episodes 2"));
        assert_eq!(render_value(&serde_json::Value::Null), "");
        assert_eq!(render_value(&serde_json::json!(["a", 1])), "a, 1");
    }

    #[test]
    fn a_schema_of_another_family_is_refused_and_a_newer_minor_is_read() {
        assert_eq!(
            schema_family("trainnr-project-index/1"),
            "trainnr-project-index"
        );
        assert!(schema_compatible("trainnr-project-index/2", INDEX_SCHEMA));
        assert!(
            schema_compatible("", INDEX_SCHEMA),
            "an older writer wrote none"
        );
        assert!(!schema_compatible("trainnr-detail/5", INDEX_SCHEMA));
        let root = std::env::temp_dir().join(format!("studio-schema-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join(".index")).unwrap();
        std::fs::write(root.join(MANIFEST_FILE), "{}").unwrap();
        std::fs::write(
            root.join(INDEX_RELATIVE),
            r#"{"schema":"somebody-else/1","project":"p","artifacts":[],"states":[]}"#,
        )
        .unwrap();
        let model = Model::at(root.clone(), root.join(PROJECTS_HOME));
        assert!(model.index().is_none());
        assert!(model
            .problem
            .as_deref()
            .is_some_and(|p| p.contains("somebody-else/1") && p.contains(INDEX_SCHEMA)));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn stamps_split_and_durations_read() {
        assert_eq!(
            split_stamp("a6-golden@fcc3f277c25b"),
            ("a6-golden", "fcc3f277c25b")
        );
        assert_eq!(split_stamp("bare"), ("bare", ""));
        assert_eq!(elapsed(4.0), "4s");
        assert_eq!(elapsed(130.0), "2m 10s");
        assert_eq!(elapsed(3780.0), "1h 03m");
    }
}

// ---------------------------------------------------------------------
// Time, without a calendar crate: the index writes ISO 8601 UTC.

/// `2026-09-08T19:34:51+00:00` (or `...Z`) → seconds since the epoch.
pub fn epoch_of(iso: &str) -> Option<f64> {
    if !iso.is_ascii() {
        return None;
    }
    let num = |r: std::ops::Range<usize>| iso.get(r)?.parse::<i64>().ok();
    let (y, mo, d) = (num(0..4)?, num(5..7)?, num(8..10)?);
    let (h, mi, s) = (num(11..13)?, num(14..16)?, num(17..19)?);
    if !(1..=12).contains(&mo)
        || !(1..=31).contains(&d)
        || !(0..24).contains(&h)
        || !(0..60).contains(&mi)
        || !(0..61).contains(&s)
        || !(YEAR_MIN..=YEAR_MAX).contains(&y)
    {
        return None;
    }
    let days = days_from_civil(y, mo, d);
    let mut secs = days * 86_400 + h * 3600 + mi * 60 + s;
    // A zone offset, when one is written; the index writes +00:00.
    let tail = iso.get(19..).unwrap_or("");
    if let Some(sign_at) = tail.find(['+', '-']) {
        let z = &tail[sign_at..];
        let zn = |r: std::ops::Range<usize>| z.get(r)?.parse::<i64>().ok();
        if let (Some(zh), Some(zm)) = (zn(1..3), zn(4..6)) {
            let sign = if z.starts_with('-') { -1 } else { 1 };
            secs -= sign * (zh * 3600 + zm * 60);
        }
    }
    Some(secs as f64)
}

/// The years a timestamp may name: nothing before Unix time, nothing
/// past what a project could plausibly record.
const YEAR_MIN: i64 = 1970;
const YEAR_MAX: i64 = 9999;
/// Epoch seconds outside this window are treated as garbage by the
/// date helpers rather than overflowing the calendar arithmetic.
const EPOCH_MAX: f64 = 253_402_300_799.0; // 9999-12-31T23:59:59Z

/// Days since 1970-01-01 for a proleptic Gregorian date (Howard Hinnant's
/// `days_from_civil`).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

pub fn now_epoch() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// How long ago, in a person's words: `just now`, `4 min ago`,
/// `3 h ago`, `2 d ago`, then the date.
pub fn ago(epoch_seconds: f64) -> String {
    let dt = (now_epoch() - epoch_seconds).max(0.0);
    if dt < 60.0 {
        "just now".to_owned()
    } else if dt < 3600.0 {
        format!("{} min ago", (dt / 60.0) as u64)
    } else if dt < 86_400.0 {
        format!("{} h ago", (dt / 3600.0) as u64)
    } else if dt < 7.0 * 86_400.0 {
        format!("{} d ago", (dt / 86_400.0) as u64)
    } else {
        date_of(epoch_seconds)
    }
}

/// The ISO time's `ago`, or `unrecorded`.
pub fn ago_iso(iso: Option<&str>) -> String {
    iso.and_then(epoch_of)
        .map(ago)
        .unwrap_or_else(|| UNRECORDED.to_owned())
}

/// `YYYY-MM-DD` (UTC) of an epoch time.
pub fn date_of(epoch_seconds: f64) -> String {
    if !(0.0..=EPOCH_MAX).contains(&epoch_seconds) {
        return UNRECORDED.to_owned();
    }
    let (y, m, d) = civil_from_days((epoch_seconds / 86_400.0).floor() as i64);
    format!("{y:04}-{m:02}-{d:02}")
}

/// The day an artifact belongs to on a page grouped by day: `Today`,
/// `Yesterday`, else the date (UTC days; the index writes UTC).
pub fn day_label(epoch_seconds: f64) -> String {
    if !(0.0..=EPOCH_MAX).contains(&epoch_seconds) {
        return "undated".to_owned();
    }
    let today = (now_epoch() / 86_400.0).floor() as i64;
    let day = (epoch_seconds / 86_400.0).floor() as i64;
    match today - day {
        0 => "Today".to_owned(),
        1 => "Yesterday".to_owned(),
        _ => date_of(epoch_seconds),
    }
}

fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod time_tests {
    use super::*;

    #[test]
    fn iso_round_trips_through_the_epoch() {
        let t = epoch_of("2026-09-04T00:00:00+00:00").unwrap();
        assert_eq!(date_of(t), "2026-09-04");
        assert_eq!(epoch_of("1970-01-01T00:00:00Z"), Some(0.0));
        assert_eq!(
            epoch_of("2026-09-08T19:34:51+00:00"),
            epoch_of("2026-09-08T21:34:51+02:00")
        );
        assert_eq!(epoch_of("junk"), None);
        assert_eq!(epoch_of("2026-13-40T00:00:00Z"), None);
        assert_eq!(epoch_of("2026-09-0ʘT00:00:00Z"), None);
        assert_eq!(date_of(1e30), UNRECORDED);
        assert_eq!(date_of(-5.0), UNRECORDED);
    }

    #[test]
    fn ago_speaks_plainly() {
        let now = now_epoch();
        assert_eq!(ago(now), "just now");
        assert_eq!(ago(now - 300.0), "5 min ago");
        assert_eq!(ago(now - 7200.0), "2 h ago");
        assert_eq!(ago(now - 3.0 * 86_400.0), "3 d ago");
        assert_eq!(day_label(now), "Today");
        assert_eq!(day_label(now - 86_400.0), "Yesterday");
    }
}

#[cfg(test)]
mod lookup_tests {
    use super::*;

    #[test]
    fn artifact_lookup_by_hash_is_unambiguous() {
        let mut index: Index = serde_json::from_value(serde_json::json!({
            "schema": "x", "project": "p", "root": "/p", "indexed": "2026-09-09T00:00:00+00:00",
            "artifacts": [
                {"kind": "task", "stamp": "tray-far@abc", "path": "t"},
                {"kind": "task", "stamp": "kitting-default@def", "path": "u"},
                {"kind": "policy", "stamp": "other@def", "path": "v"}
            ],
            "states": [], "next_move": null, "refused": []
        }))
        .unwrap();
        index.finish();
        assert_eq!(
            index.artifact("tray-far@abc").map(|a| a.path.as_str()),
            Some("t")
        );
        assert_eq!(
            index.artifact("kitting@abc").map(|a| a.path.as_str()),
            Some("t")
        );
        assert!(
            index.artifact("kitting@def").is_none(),
            "two artifacts share def"
        );
        assert!(index.artifact("ghost@zzz").is_none());
    }
}
