//! What is running now, whoever started it (2026-09-25: the operator "I
//! want to see in real time some information about what is currently
//! running in the studio"; a gate from a terminal left the window "idle").
//!
//! The job table (`<project>/mcp-jobs/`) holds every run: a door's (the
//! agent's MCP call), a terminal tool's and an agent's fork's, each
//! `<id>.json` (`rq_pipeline.mcp_jobs.JobRecord`), `<id>.status` (its
//! stage and progress, `RunStatus`), `<id>.log` and, once ended,
//! `<id>.exit`. A thread polls the table off the UI thread (a known
//! leftover: no file I/O on the UI thread) and hands the frame a
//! snapshot; the panel lists the running first, then what ended within
//! `FINISHED_KEPT`, with a progress bar, the stage line, a live log tail,
//! and the moves that belong to a run: watch it in the viewport, see its
//! stream in the viewer, stop it.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use serde::Deserialize;

/// Mirrored from `rq_pipeline/mcp_jobs.py` (`STATUS_SUFFIX`,
/// `STATUS_SCHEMA`, `JOB_SOURCES`), pinned by `tests/test_studio_mirrors.py`.
pub const STATUS_SUFFIX: &str = ".status";
pub const STATUS_SCHEMA: &str = "trainnr-job-status/1";
pub const JOB_SOURCES: [&str; 3] = ["door", "tool", "agent"];
/// The first source: a door's job runs in its own session, so a stop
/// signals its whole process group; any other stops the recorded pid.
const SOURCE_DOOR: &str = JOB_SOURCES[0];
/// How long an ended run stays in the panel with its exit code.
pub const FINISHED_KEPT: Duration = Duration::from_secs(300);
/// How much of a run's log the panel shows when opened.
pub const LOG_TAIL_LINES: usize = 12;
/// How often the thread re-reads the table.
const POLL_EVERY: Duration = Duration::from_secs(1);

/// `RunStatus`, as `rq_pipeline.mcp_jobs.write_status` writes it.
#[derive(Deserialize, Clone, Default, Debug, PartialEq)]
pub struct RunStatus {
    #[serde(default)]
    pub stage: String,
    #[serde(default)]
    pub done: u64,
    #[serde(default)]
    pub total: u64,
    #[serde(default)]
    pub unit: String,
    #[serde(default)]
    pub updated: f64,
    #[serde(default)]
    pub schema: String,
}

impl RunStatus {
    /// Read when its schema's family (before the `/`) is ours; a newer
    /// minor version's unknown fields are ignored.
    pub fn readable(&self) -> bool {
        let family = |s: &str| s.split('/').next().unwrap_or_default().to_owned();
        family(&self.schema) == family(STATUS_SCHEMA)
    }

    /// `0.35`, or None when no count is known.
    pub fn fraction(&self) -> Option<f32> {
        (self.total > 0).then(|| (self.done as f32 / self.total as f32).clamp(0.0, 1.0))
    }

    /// `7/20 trials`, or empty when no count is known.
    pub fn count_text(&self) -> String {
        if self.total == 0 {
            String::new()
        } else {
            format!("{}/{} {}", self.done, self.total, self.unit)
        }
    }
}

/// `JobRecord`, as `rq_pipeline.mcp_jobs` writes it, with what the poll
/// learned beside it: the exit code, whether the pid is alive, the live
/// status, and when it ended.
#[derive(Deserialize, Clone, Default)]
pub struct Job {
    pub id: String,
    pub tool: String,
    #[serde(default)]
    pub argv: Vec<String>,
    pub started: f64,
    #[serde(default)]
    pub pid: u32,
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub viewport: String,
    #[serde(default)]
    pub viewer: String,
    #[serde(default)]
    pub log: String,
    #[serde(skip)]
    pub exit: Option<i32>,
    #[serde(skip)]
    pub alive: bool,
    #[serde(skip)]
    pub status: Option<RunStatus>,
    /// When the exit file landed (epoch seconds): how long it has been
    /// over, for `FINISHED_KEPT`.
    #[serde(skip)]
    pub ended: Option<f64>,
}

impl Job {
    /// Running means no exit recorded AND the process still there: a
    /// job whose runner was killed before it could write the exit file
    /// showed as running for an hour after it ended (2026-09-10).
    pub fn running(&self) -> bool {
        self.exit.is_none() && self.alive
    }

    /// Gone with no exit recorded: killed, crashed, its watcher gone.
    pub fn died(&self) -> bool {
        self.exit.is_none() && !self.alive
    }

    /// The word for its state, as the Python side says it
    /// (`mcp_jobs.STATE_RUNNING`, `STATE_DIED`).
    pub fn state_word(&self) -> String {
        match self.exit {
            Some(0) => "done".to_owned(),
            Some(code) => format!("failed (exit {code})"),
            None if self.died() => "died (no exit recorded)".to_owned(),
            None => "running".to_owned(),
        }
    }

    /// When it last showed a sign of life: the exit, else its status,
    /// else its start.
    fn last_seen(&self) -> f64 {
        self.ended
            .or_else(|| self.status.as_ref().map(|s| s.updated).filter(|u| *u > 0.0))
            .unwrap_or(self.started)
    }

    /// The line the top bar shows for it: its name (or kind) and stage.
    pub fn short_line(&self) -> String {
        let who = if self.name.is_empty() {
            &self.tool
        } else {
            &self.name
        };
        match self.status.as_ref().filter(|s| !s.stage.is_empty()) {
            Some(status) => format!("{who}: {}", status.stage),
            None => who.clone(),
        }
    }
}

/// The panel's rows: every running job, newest first, then what ended
/// (or died) within `kept` of `now`, most recent first.
pub fn panel_rows(jobs: &[Job], now: f64, kept: Duration) -> Vec<&Job> {
    let mut running: Vec<&Job> = jobs.iter().filter(|j| j.running()).collect();
    running.sort_by(|a, b| b.started.total_cmp(&a.started));
    let mut ended: Vec<&Job> = jobs
        .iter()
        .filter(|j| !j.running() && now - j.last_seen() <= kept.as_secs_f64())
        .collect();
    ended.sort_by(|a, b| b.last_seen().total_cmp(&a.last_seen()));
    running.extend(ended);
    running
}

/// Whether a process id is alive. Linux answers through `/proc`; any
/// other unix (macOS) through `kill -0`, the same binary `spawn.rs`
/// signals with; Windows has no dependency-free check here and answers
/// "alive", so there the exit file alone decides (the runner writes it).
pub fn pid_alive(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    if cfg!(target_os = "linux") {
        return Path::new(&format!("/proc/{pid}")).exists();
    }
    #[cfg(unix)]
    {
        std::process::Command::new("kill")
            .args(["-0", "--", &pid.to_string()])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|status| status.success())
            .unwrap_or(true)
    }
    #[cfg(not(unix))]
    {
        true
    }
}

/// Stop a run: a door's job signals its process group (it runs in its
/// own session, `mcp_jobs._spawn`), a tool's or an agent's its pid. The
/// same gesture as `project.control.terminate_group`; the run's own
/// tracker records the exit it ends with.
pub fn stop(job: &Job) -> std::io::Result<()> {
    if job.pid == 0 {
        return Err(std::io::Error::other("no pid recorded"));
    }
    #[cfg(unix)]
    {
        let target = if job.source == SOURCE_DOOR {
            format!("-{}", job.pid)
        } else {
            job.pid.to_string()
        };
        let status = std::process::Command::new("kill")
            .args(["-TERM", "--", &target])
            .status()?;
        if status.success() {
            Ok(())
        } else {
            Err(std::io::Error::other(format!("kill -TERM {target} failed")))
        }
    }
    #[cfg(not(unix))]
    {
        let status = std::process::Command::new("taskkill")
            .args(["/PID", &job.pid.to_string(), "/T"])
            .status()?;
        if status.success() {
            Ok(())
        } else {
            Err(std::io::Error::other("taskkill failed"))
        }
    }
}

/// What one poll found: every job, and the watched job's log tail.
#[derive(Default, Clone)]
pub struct Snapshot {
    pub jobs: Vec<Job>,
    pub log_tail: Option<(String, Vec<String>)>,
}

/// The poll of one project's job table, off the UI thread. Dropped with
/// the model (a project switch opens a new one); its thread then ends.
pub struct JobsWatch {
    latest: Arc<Mutex<Option<Snapshot>>>,
    watched_log: Arc<Mutex<Option<String>>>,
    stop: Arc<AtomicBool>,
    generation: Arc<AtomicU64>,
    seen: u64,
}

impl JobsWatch {
    /// Start polling `jobs_dir`; the first snapshot is taken here, so a
    /// fresh window shows what already runs.
    pub fn start(jobs_dir: PathBuf) -> Self {
        let latest = Arc::new(Mutex::new(Some(read_snapshot(&jobs_dir, None))));
        let watched_log = Arc::new(Mutex::new(None::<String>));
        let stop = Arc::new(AtomicBool::new(false));
        let generation = Arc::new(AtomicU64::new(1));
        {
            let latest = Arc::clone(&latest);
            let watched_log = Arc::clone(&watched_log);
            let stop = Arc::clone(&stop);
            let generation = Arc::clone(&generation);
            let spawned = std::thread::Builder::new()
                .name("jobs-watch".to_owned())
                .spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        std::thread::sleep(POLL_EVERY);
                        let log = watched_log.lock().ok().and_then(|w| w.clone());
                        let snapshot = read_snapshot(&jobs_dir, log.as_deref());
                        if let Ok(mut slot) = latest.lock() {
                            *slot = Some(snapshot);
                            generation.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                });
            if let Err(err) = spawned {
                eprintln!("jobs-watch thread did not start: {err}");
            }
        }
        Self {
            latest,
            watched_log,
            stop,
            generation,
            seen: 0,
        }
    }

    /// The newest snapshot, once per poll (None when nothing new came).
    pub fn take(&mut self) -> Option<Snapshot> {
        let now = self.generation.load(Ordering::Relaxed);
        if now == self.seen {
            return None;
        }
        self.seen = now;
        self.latest.lock().ok().and_then(|slot| slot.clone())
    }

    /// Which job's log the next polls tail (None: none).
    pub fn watch_log(&self, job_id: Option<&str>) {
        if let Ok(mut watched) = self.watched_log.lock() {
            *watched = job_id.map(str::to_owned);
        }
    }
}

impl Drop for JobsWatch {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

fn epoch_of(time: SystemTime) -> Option<f64> {
    time.duration_since(SystemTime::UNIX_EPOCH)
        .ok()
        .map(|d| d.as_secs_f64())
}

/// Read the table once: every record, its exit and when it landed, its
/// status, its pid's liveness; and the watched job's log tail.
pub fn read_snapshot(jobs_dir: &Path, watched: Option<&str>) -> Snapshot {
    let Ok(entries) = std::fs::read_dir(jobs_dir) else {
        return Snapshot::default();
    };
    let mut jobs: Vec<Job> = entries
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .filter_map(|p| {
            let text = std::fs::read_to_string(&p).ok()?;
            let mut job: Job = serde_json::from_str(&text).ok()?;
            let exit_path = p.with_extension("exit");
            job.exit = std::fs::read_to_string(&exit_path)
                .ok()
                .and_then(|s| s.trim().parse().ok());
            job.ended = job
                .exit
                .and_then(|_| std::fs::metadata(&exit_path).ok())
                .and_then(|m| m.modified().ok())
                .and_then(epoch_of);
            job.status = std::fs::read_to_string(p.with_extension(&STATUS_SUFFIX[1..]))
                .ok()
                .and_then(|s| serde_json::from_str::<RunStatus>(&s).ok())
                .filter(RunStatus::readable);
            job.alive = job.exit.is_none() && pid_alive(job.pid);
            Some(job)
        })
        .collect();
    jobs.sort_by(|a, b| b.started.total_cmp(&a.started));
    let log_tail = watched.and_then(|id| {
        let job = jobs.iter().find(|j| j.id == id)?;
        let path = if job.log.is_empty() {
            jobs_dir.join(format!("{id}.log"))
        } else {
            PathBuf::from(&job.log)
        };
        Some((id.to_owned(), tail_lines(&path, LOG_TAIL_LINES)))
    });
    Snapshot { jobs, log_tail }
}

/// The last `n` lines of a text file (a bounded read from the end).
pub fn tail_lines(path: &Path, n: usize) -> Vec<String> {
    use std::io::{Read, Seek, SeekFrom};
    const TAIL_BYTES: u64 = 64 * 1024;
    let Ok(mut file) = std::fs::File::open(path) else {
        return Vec::new();
    };
    let len = file.metadata().map(|m| m.len()).unwrap_or(0);
    let _ = file.seek(SeekFrom::Start(len.saturating_sub(TAIL_BYTES)));
    let mut bytes = Vec::new();
    if file.read_to_end(&mut bytes).is_err() {
        return Vec::new();
    }
    let text = String::from_utf8_lossy(&bytes);
    let lines: Vec<&str> = text.lines().collect();
    lines[lines.len().saturating_sub(n)..]
        .iter()
        .map(|l| (*l).to_owned())
        .collect()
}

/// The top bar's words for what runs: `2 running · go2-c2 (mujoco):
/// trial 3 of 4: tracked`, cut to `SHORT_LINE_CHARS`; None when idle.
pub fn indicator_line(jobs: &[Job]) -> Option<String> {
    const SHORT_LINE_CHARS: usize = 56;
    let running: Vec<&Job> = panel_rows(jobs, 0.0, Duration::ZERO)
        .into_iter()
        .filter(|j| j.running())
        .collect();
    let first = running.first()?;
    let mut line = first.short_line();
    if line.chars().count() > SHORT_LINE_CHARS {
        line = line.chars().take(SHORT_LINE_CHARS - 1).collect::<String>() + "…";
    }
    Some(format!("{} running · {line}", running.len()))
}

/// What a click on a row asks for; the shell carries it out.
pub enum RowAction {
    /// Open (or close) the row's live log tail.
    Log,
    /// Load the run's viewer file into the viewer and turn to it.
    Viewer,
    /// Run the row's scene in the viewport.
    Viewport,
    /// First click on stop: arm it (the confirmation).
    ArmStop,
    /// Second click: stop the run.
    Stop,
    /// Changed its mind.
    Disarm,
}

/// One run, read-only: who and what, its state, the progress bar with
/// its count, the stage line, how long, which pid. The Overview's
/// Compute card shows exactly this; the panel adds the buttons.
pub fn summary(ui: &mut egui::Ui, job: &Job, now: f64) {
    use re_ui::UiExt as _;
    ui.horizontal_wrapped(|ui| {
        let who = if job.name.is_empty() {
            &job.tool
        } else {
            &job.name
        };
        ui.label(egui::RichText::new(who).strong());
        crate::widgets::tag(ui, &job.tool);
        if !job.source.is_empty() {
            crate::widgets::tag(ui, &job.source);
        }
        let color = if job.running() {
            ui.tokens().highlight_color
        } else if job.exit == Some(0) {
            ui.visuals().weak_text_color()
        } else {
            ui.visuals().warn_fg_color
        };
        ui.label(egui::RichText::new(job.state_word()).small().color(color));
    });
    let status = job.status.clone().unwrap_or_default();
    if let Some(fraction) = status.fraction() {
        ui.add(
            egui::ProgressBar::new(fraction)
                .desired_height(6.0)
                .fill(ui.tokens().highlight_color.linear_multiply(0.8)),
        );
    }
    let until = job.ended.unwrap_or(now);
    let mut facts = vec![crate::model::elapsed(until - job.started)];
    let count = status.count_text();
    if !count.is_empty() {
        facts.push(count);
    }
    if job.pid != 0 {
        facts.push(format!("pid {}", job.pid));
    }
    let line = if status.stage.is_empty() {
        facts.join(" · ")
    } else {
        format!("{} — {}", status.stage, facts.join(" · "))
    };
    ui.label(
        egui::RichText::new(line)
            .small()
            .color(ui.visuals().weak_text_color()),
    );
}

/// A row with its buttons: the log, the viewer, the viewport, stop
/// (asked twice: `armed` says the first click came).
pub fn row(
    ui: &mut egui::Ui,
    job: &Job,
    now: f64,
    log_open: bool,
    armed: bool,
) -> Option<RowAction> {
    let mut action = None;
    summary(ui, job, now);
    ui.horizontal(|ui| {
        let log_word = if log_open { "close log" } else { "open log" };
        if ui.small_button(log_word).clicked() {
            action = Some(RowAction::Log);
        }
        if !job.viewer.is_empty()
            && ui
                .small_button("show in viewer")
                .on_hover_text(&job.viewer)
                .clicked()
        {
            action = Some(RowAction::Viewer);
        }
        if crate::viewport::is_known_scene(&job.viewport)
            && ui
                .small_button("watch in viewport")
                .on_hover_text(&job.viewport)
                .clicked()
        {
            action = Some(RowAction::Viewport);
        }
        if job.running() {
            if armed {
                let warn = ui.visuals().warn_fg_color;
                if ui
                    .small_button(egui::RichText::new(format!("stop pid {}?", job.pid)).color(warn))
                    .clicked()
                {
                    action = Some(RowAction::Stop);
                }
                if ui.small_button("keep running").clicked() {
                    action = Some(RowAction::Disarm);
                }
            } else if ui.small_button("stop").clicked() {
                action = Some(RowAction::ArmStop);
            }
        }
    });
    action
}

/// A log tail, monospace, in a bounded scroll.
pub fn log_tail(ui: &mut egui::Ui, lines: &[String]) {
    egui::ScrollArea::vertical()
        .max_height(160.0)
        .stick_to_bottom(true)
        .show(ui, |ui| {
            for line in lines {
                ui.label(egui::RichText::new(line).monospace().small());
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job(id: &str, started: f64) -> Job {
        Job {
            id: id.to_owned(),
            tool: "gate-deployment".to_owned(),
            started,
            ..Job::default()
        }
    }

    #[test]
    fn running_first_newest_first_then_the_recently_ended() {
        let now = 1_000.0;
        let mut a = job("a", 10.0);
        a.alive = true;
        let mut b = job("b", 20.0);
        b.alive = true;
        let mut done = job("done", 5.0);
        done.exit = Some(0);
        done.ended = Some(now - 10.0);
        let mut old = job("old", 1.0);
        old.exit = Some(1);
        old.ended = Some(now - 10_000.0);
        let jobs = [a, b, done, old];
        let rows: Vec<&str> = panel_rows(&jobs, now, FINISHED_KEPT)
            .into_iter()
            .map(|j| j.id.as_str())
            .collect();
        assert_eq!(rows, ["b", "a", "done"]);
    }

    #[test]
    fn a_run_without_a_pid_or_an_exit_has_died_and_says_so() {
        let dead = job("dead", 1.0);
        assert!(dead.died() && !dead.running());
        assert_eq!(dead.state_word(), "died (no exit recorded)");
        let mut failed = job("failed", 1.0);
        failed.exit = Some(2);
        assert_eq!(failed.state_word(), "failed (exit 2)");
    }

    #[test]
    fn the_table_is_read_with_its_status_its_exit_and_the_watched_tail() {
        let dir = std::env::temp_dir().join(format!("rq-running-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let record = serde_json::json!({
            "id": "gate-1", "tool": "gate-deployment", "argv": [], "cwd": "/",
            "log": dir.join("gate-1.log").display().to_string(),
            "pid": std::process::id(), "started": 1.0, "source": "tool",
            "name": "go2-c2 (mujoco)", "viewport": "deploy:go2-c2", "viewer": ""
        });
        std::fs::write(dir.join("gate-1.json"), record.to_string()).expect("record");
        std::fs::write(
            dir.join("gate-1.status"),
            r#"{"stage": "trial 7 of 20: tracked", "done": 7, "total": 20, "unit": "trials", "updated": 2.0, "schema": "trainnr-job-status/1"}"#,
        )
        .expect("status");
        std::fs::write(dir.join("gate-1.log"), "one\ntwo\nthree\n").expect("log");
        let snap = read_snapshot(&dir, Some("gate-1"));
        let _ = std::fs::remove_dir_all(&dir);
        let job = &snap.jobs[0];
        assert!(job.running(), "this test process is the run's pid");
        let status = job.status.clone().expect("status read");
        assert!(status.readable());
        assert_eq!(status.count_text(), "7/20 trials");
        assert_eq!(job.short_line(), "go2-c2 (mujoco): trial 7 of 20: tracked");
        assert_eq!(
            snap.log_tail,
            Some((
                "gate-1".to_owned(),
                vec!["one".into(), "two".into(), "three".into()]
            ))
        );
    }

    #[test]
    fn a_status_of_another_family_is_not_read() {
        let other: RunStatus =
            serde_json::from_str(r#"{"stage": "x", "schema": "other/1"}"#).expect("parses");
        assert!(!other.readable());
        let newer: RunStatus =
            serde_json::from_str(r#"{"stage": "x", "schema": "trainnr-job-status/2", "extra": 1}"#)
                .expect("parses");
        assert!(newer.readable());
    }

    #[test]
    fn the_indicator_names_the_newest_run_and_says_idle_as_none() {
        let mut a = job("a", 10.0);
        a.alive = true;
        a.name = "go2-c2 (mujoco)".to_owned();
        a.status = Some(RunStatus {
            stage: "trial 3 of 4: tracked".to_owned(),
            ..RunStatus::default()
        });
        let mut b = job("b", 5.0);
        b.alive = true;
        assert_eq!(
            indicator_line(&[b.clone(), a]).as_deref(),
            Some("2 running · go2-c2 (mujoco): trial 3 of 4: tracked")
        );
        b.alive = false;
        assert_eq!(indicator_line(&[b]), None);
    }

    #[test]
    fn the_sources_start_with_the_door() {
        assert_eq!(SOURCE_DOOR, "door");
    }
}
