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

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use serde::Deserialize;

/// Mirrored from `rq_pipeline/project/locate.py`.
const INDEX_RELATIVE: &str = ".index/project.json";
const MANIFEST_FILE: &str = "project.json";
/// Mirrored from `rq_pipeline/mcp_jobs.py` (`JOBS_DIR_NAME`).
const JOBS_DIR: &str = "mcp-jobs";
/// Mirrored from `rq_pipeline/project/present.py` (`INTENT_FILE`): the
/// Studio writes `{"stamp": ...}` here; the presenter streams that
/// artifact into the viewer and clears it.
const INTENT_RELATIVE: &str = ".index/present.json";
/// The environment variable both halves honour (`PROJECT_ENV`).
pub const PROJECT_ENV: &str = "TRAINNR_PROJECT";
const DEFAULT_PROJECT: &str = "projects/default";
const SAMPLE_PROJECT: &str = "projects/sample";
/// How often the files' mtimes are polled. A tool call rewrites the
/// index in one atomic replace; a second of latency is invisible next to
/// the seconds the tool itself took.
pub const RELOAD_EVERY: Duration = Duration::from_secs(1);

/// `ProjectIndex`, as `rq_pipeline.project.index` writes it. Unknown
/// fields are ignored so an older Studio still opens a newer index.
#[derive(Deserialize, Default)]
pub struct Index {
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
}

#[derive(Deserialize)]
pub struct State {
    pub name: String,
    #[serde(default)]
    pub proved_by: Vec<String>,
    pub present: bool,
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
    #[serde(skip)]
    pub exit: Option<i32>,
}

impl Job {
    pub fn running(&self) -> bool {
        self.exit.is_none()
    }
}

impl Index {
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
    pub project_root: PathBuf,
    index: Watched<Index>,
    jobs_dir: PathBuf,
    jobs_seen: Option<SystemTime>,
    pub jobs: Vec<Job>,
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
        let root = std::env::var_os(PROJECT_ENV)
            .map(PathBuf::from)
            .filter(|p| p.join(MANIFEST_FILE).is_file())
            .or_else(|| {
                let default = repo_root.join(DEFAULT_PROJECT);
                default.join(MANIFEST_FILE).is_file().then_some(default)
            })
            .unwrap_or_else(|| repo_root.join(SAMPLE_PROJECT));
        Self::at(root, repo_root.join("projects"))
    }

    /// Open a specific project directory.
    pub fn at(root: PathBuf, projects_home: PathBuf) -> Self {
        let mut model = Self {
            projects_home,
            index: Watched::new(root.join(INDEX_RELATIVE)),
            jobs_dir: root.join(JOBS_DIR),
            jobs_seen: None,
            jobs: Vec::new(),
            problem: None,
            project_root: root,
            last_poll: None,
        };
        model.reload_index();
        model.reload_jobs();
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

    /// An artifact's preview as an absolute path, when it has one.
    pub fn preview_path(&self, artifact: &Artifact) -> Option<PathBuf> {
        artifact
            .preview
            .as_ref()
            .map(|rel| self.project_root.join(rel))
            .filter(|p| p.is_file())
    }

    /// Every project under the projects home, by name, with what its own
    /// index says about it. Read fresh on each call; the Projects page is
    /// visited, not polled.
    pub fn projects(&self) -> Vec<ProjectSummary> {
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
                    stages: index.as_ref().map_or(0, |i| i.states.len()),
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
        let path = self.project_root.join(INTENT_RELATIVE);
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let body = serde_json::json!({ "stamp": stamp }).to_string();
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, body)?;
        std::fs::rename(tmp, path)
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
        if self.index.changed() {
            self.reload_index();
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
    }

    fn reload_index(&mut self) {
        match std::fs::read_to_string(&self.index.path) {
            Ok(text) => match serde_json::from_str::<Index>(&text) {
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
        .filter(|(k, _)| *k != "files")
        .map(|(k, v)| format!("{k} {}", render_value(v)))
        .collect();
    (!parts.is_empty()).then(|| parts.join(" · "))
}

pub fn render_value(value: &serde_json::Value) -> String {
    match value {
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

/// Seconds since the epoch → `HH:MM` local-agnostic UTC, for the
/// activity list; the date is dropped because activity is today's.
pub fn clock(epoch_seconds: f64) -> String {
    let secs = epoch_seconds.max(0.0) as u64;
    let h = (secs / 3600) % 24;
    let m = (secs / 60) % 60;
    format!("{h:02}:{m:02}")
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
    fn a_job_record_parses_and_is_running_without_an_exit() {
        let text = r#"{"id":"generate-demos-1a2b","tool":"generate-demos",
            "argv":["uv","run"],"cwd":"/p","log":"/p/x.log","pid":4,"started":1757000000.5}"#;
        let job: Job = serde_json::from_str(text).expect("parses");
        assert!(job.running());
        assert_eq!(job.tool, "generate-demos");
        assert_eq!(clock(job.started), "15:33");
    }

    #[test]
    fn summary_lines_skip_file_lists() {
        let mut m = serde_json::Map::new();
        m.insert("files".into(), serde_json::json!(["a", "b"]));
        assert!(summary_line(&m).is_none());
        m.insert("episodes".into(), serde_json::json!(2));
        assert_eq!(summary_line(&m).as_deref(), Some("episodes 2"));
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
