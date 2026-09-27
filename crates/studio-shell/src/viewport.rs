//! The 3D viewport: not our own renderer, MuJoCo's.
//!
//! `tools/studio-render-stream.py` runs the real `mujoco.Renderer` this repo
//! already trusts (pinned to the same `mujoco~=3.11.0` the pipeline uses)
//! and publishes frames into a shared-memory ring THIS side creates
//! (`ShmReader`), waking us with a one-byte stdout token per frame. No
//! MuJoCo C API is linked into this crate — see the module doc comment in
//! the Python script for why (the FFI route needs a patched fork of
//! `glutin` on macOS plus a second, older MuJoCo install).
//!
//! Input goes the other way as TAGGED stdin messages (the script's
//! `TAG_*` table): camera deltas (drag orbits, scroll zooms — deltas,
//! deliberately: the Python side owns every absolute camera fact, because
//! an absolute-valued wire needed those facts duplicated here and the
//! copies drifted), and the perturbation gesture — Ctrl+drag selects the
//! body under the pointer and shoves it through MuJoCo's own
//! mjv_select/mjvPerturb, exactly what the native viewer does in-process.
//! No pan yet.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;

use egui::{ColorImage, TextureHandle, TextureOptions};
use re_ui::UiExt as _;

use crate::spawn::{end_tree, kill_tree, pipeline_command, walk_command, RENDER_STREAM_SCRIPT};

/// The shared-memory frame ring's layout — the Python side's mirror
/// (`SHM_HEADER`/`SHM_MAGIC` in studio-render-stream.py): magic u32,
/// seq u32 (seqlock: odd while writing, even when stable), width u32,
/// height u32, then raw RGB8 pixels. Replaces raw frames over the
/// stdout pipe (~6 MB each, the measured lag, 2026-09-02); the pipe now
/// carries only a 1-byte token per published frame as the wake-up.
const SHM_HEADER_BYTES: u64 = 16;
const SHM_MAGIC: u32 = 0x524A_4D51;

/// One reader over the ring file: plain file I/O, not a mapping — the
/// page cache makes a 6 MB read ~1 ms, and std::fs works identically on
/// all three platforms with zero unsafe and zero dependencies.
struct ShmReader {
    file: File,
    path: PathBuf,
    last_seq: u32,
    /// The frame's bytes, kept between reads: a 6 MB allocation per
    /// frame was the hot path's own cost (2026-09-13). `ColorImage` still
    /// owns its pixels, so one copy remains.
    rgb: Vec<u8>,
}

impl ShmReader {
    /// The newest STABLE frame, or None when nothing new (or a write
    /// raced the copy — the next repaint retries).
    fn latest(&mut self) -> Option<ColorImage> {
        let mut header = [0u8; SHM_HEADER_BYTES as usize];
        self.file.seek(SeekFrom::Start(0)).ok()?;
        self.file.read_exact(&mut header).ok()?;
        let magic = u32::from_le_bytes(header[0..4].try_into().expect("4 bytes"));
        let seq = u32::from_le_bytes(header[4..8].try_into().expect("4 bytes"));
        let width = u32::from_le_bytes(header[8..12].try_into().expect("4 bytes")) as usize;
        let height = u32::from_le_bytes(header[12..16].try_into().expect("4 bytes")) as usize;
        if magic != SHM_MAGIC || seq % 2 == 1 || seq == self.last_seq {
            return None;
        }
        if !(1..=MAX_RENDER_SIDE as usize).contains(&width)
            || !(1..=MAX_RENDER_SIDE as usize).contains(&height)
        {
            return None;
        }
        self.rgb.resize(width * height * 3, 0);
        self.file.read_exact(&mut self.rgb).ok()?;
        // Seqlock close: a write that landed mid-copy moved the counter;
        // discard the torn frame and let the next repaint pick it up.
        self.file.seek(SeekFrom::Start(4)).ok()?;
        let mut seq_after = [0u8; 4];
        self.file.read_exact(&mut seq_after).ok()?;
        if u32::from_le_bytes(seq_after) != seq {
            return None;
        }
        self.last_seq = seq;
        Some(ColorImage::from_rgb([width, height], &self.rgb))
    }
}

/// One number per spawned stream: a restart of the SAME scene gets its own
/// frame file, so the old viewport's cleanup (`Drop for ShmReader`) cannot
/// delete the new one's (2026-09-24: restarting `deploy:<name>` found its
/// file gone and the render stream died at open).
static SHM_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// A scene name as a file-name part on every platform: `deploy:` scenes
/// carry colons, which Windows forbids in a file name.
fn file_safe(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect()
}

impl Drop for ShmReader {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Degrees of orbit per point of drag, and metres of distance per point of
/// scroll — tuned by feel against a 1024x576 viewport, not measured.
/// These stay HERE (pixel-domain input scaling is this side's fact);
/// every absolute camera fact — defaults, clamps — lives Python-side.
/// What a slider in the drawer sets: a joint (by qpos address), an
/// actuator, or one axis of a walk's commanded twist.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum SliderKind {
    Joint,
    Actuator,
    Twist,
}

const DRAG_DEGREES_PER_POINT: f32 = 0.4;
const ZOOM_METRES_PER_SCROLL_POINT: f32 = 0.004;

/// The render size the Python side starts at (its `WIDTH`/`HEIGHT`) —
/// the one wire fact genuinely shared across the language boundary, so a
/// resize message only fires when the panel actually differs from it.
const INITIAL_RENDER_SIZE: (u32, u32) = (1024, 576);

/// Render resolution bounds: below this a panel sliver isn't worth a
/// render call, above this the render cost isn't worth the extra pixels
/// on a viewport this size.
const MIN_RENDER_SIDE: u32 = 128;
const MAX_RENDER_SIDE: u32 = 1920;

/// A size change makes the Python side throw away and rebuild its
/// `mujoco.Renderer` (fixed-size once constructed), so during a live
/// window drag — where the panel size changes every frame — new sizes
/// are held back until this long has passed since the last one sent.
/// Orbit/zoom are exempt: they're cheap per-frame camera fields, and
/// holding them back would make dragging feel laggy.
const RESIZE_DEBOUNCE: std::time::Duration = std::time::Duration::from_millis(250);
/// How long the bar shows what the agent just did.
const AGENT_FLASH: std::time::Duration = std::time::Duration::from_millis(1800);
/// The longest status message the reader accepts. The stream's status is
/// a few kilobytes (the model description once, then the clock and the
/// inputs); a length past this is a torn pipe, not a message, and ends
/// the stream loudly rather than allocating whatever four bytes say.
const MAX_STATUS_BYTES: usize = 16 << 20;
/// The speed factors the stream accepts (`SPEED_MIN`/`SPEED_MAX` in
/// studio-render-stream.py, mirrored): what the agent may ask for.
pub const SPEED_RANGE: std::ops::RangeInclusive<f32> = 0.01..=100.0;

/// A named camera view: the name the agent and the menu use, the label
/// the menu shows, and the wire's `TAG_VIEW` index (`VIEW_PRESETS` in
/// the stream, in its order).
pub struct ViewPreset {
    pub name: &'static str,
    pub label: &'static str,
    pub index: u8,
}

/// The camera's named views, one table for the menu and the door.
pub const VIEW_PRESETS: &[ViewPreset] = &[
    ViewPreset {
        name: "front",
        label: "Front",
        index: 1,
    },
    ViewPreset {
        name: "side",
        label: "Side",
        index: 2,
    },
    ViewPreset {
        name: "top",
        label: "Top",
        index: 3,
    },
    ViewPreset {
        name: "reset",
        label: "Reset view",
        index: 0,
    },
];

/// The wire index of a view by its name.
pub fn view_preset(name: &str) -> Option<u8> {
    let wanted = name.trim().to_lowercase();
    VIEW_PRESETS
        .iter()
        .find(|v| v.name == wanted)
        .map(|v| v.index)
}

/// The names `view_preset` accepts, for a refusal.
pub fn view_preset_names() -> Vec<&'static str> {
    VIEW_PRESETS.iter().map(|v| v.name).collect()
}

/// How the RL view is spawned: the walk package's module, and how many
/// policy-driven worlds it rolls (a 3 × 3 tile of the batched sim).
struct WalkSpawn {
    module: &'static str,
    envs: u32,
}

const WALK_SPAWN: WalkSpawn = WalkSpawn {
    module: "rq_mjlab.walk_view",
    envs: 4, // what the operator compared against mjlab's own viewer (2026-09-12)
};

/// Owns the render-stream subprocess, the shared-memory frame ring, and
/// the orbit/perturb state the user drives with the mouse over the image.
pub struct ViewportFeed {
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    shm: Option<ShmReader>,
    /// Bumped by the token-reader thread per published frame; `show`
    /// reads the ring only when this moved (no per-repaint file I/O).
    frames_published: Arc<AtomicU64>,
    frames_drawn: u64,
    texture: Option<TextureHandle>,
    spawn_error: Option<String>,
    /// The render size last sent (absolute — the Python `Renderer` is
    /// rebuilt to match it).
    sent_size: (u32, u32),
    /// When the last size change went out — see `RESIZE_DEBOUNCE`.
    last_resize_sent: Option<std::time::Instant>,
    /// A Ctrl+drag perturbation is in flight (select sent, release owed).
    perturbing: bool,
    /// Set by the reader thread when the frame stream ends. A dead
    /// subprocess used to leave its last frame frozen on screen with no
    /// indication anything was wrong (measured: an oversized resize
    /// request killed the renderer and the panel just... stopped) —
    /// `show` turns this into a visible error instead.
    stream_ended: Arc<AtomicBool>,
    /// The scene this feed runs (a preview task name), for the state file.
    task: Option<String>,
    /// When each of the last frames was drawn, for the on-screen rate.
    drawn_at: std::collections::VecDeque<std::time::Instant>,
    /// The rate as last shown to a panel, and when (see `fps_settled`).
    fps_shown: std::cell::Cell<(Option<f32>, Option<std::time::Instant>)>,
    /// The agent's last action on the simulator, and when (see `flash`).
    agent_flash: Option<(String, std::time::Instant)>,
    /// The stream's status and model description (reader thread writes).
    report: Arc<std::sync::Mutex<SimReport>>,
    /// Slider values the human is editing, so a drag does not fight
    /// the 100 ms status echo: keyed by qpos address / actuator index.
    editing: std::collections::HashMap<(SliderKind, usize), f64>,
    /// The twist last sent (world, values) while the human holds the
    /// commands: the source the next axis composes from, ahead of the
    /// status echo (two door calls in a row raced on the echo, 2026-09-12).
    twist_held: Option<(i32, [f32; 3])>,
    /// Whether the picture had the pointer over it, and whether it had
    /// keyboard focus, on its last frame: where the keys go (`wants_keys`).
    hovered: bool,
    focused: bool,
}

/// One joint the Joint panel can slide (hinge or slide; free and ball
/// joints have no scalar, simulate's own rule) — from the stream's
/// `model` status.
#[derive(serde::Deserialize, Clone, Debug, PartialEq)]
pub struct SimJoint {
    pub name: String,
    pub qpos: usize,
    pub range: [f64; 2],
    pub limited: bool,
    #[serde(rename = "type")]
    pub kind: String,
}

#[derive(serde::Deserialize, Clone, Debug, PartialEq)]
pub struct SimActuator {
    pub name: String,
    pub range: [f64; 2],
    pub limited: bool,
}

/// The model as the stream describes it once: what the panels need.
#[derive(serde::Deserialize, Clone, Debug, PartialEq, Default)]
pub struct SimModel {
    #[serde(default)]
    pub joints: Vec<SimJoint>,
    #[serde(default)]
    pub actuators: Vec<SimActuator>,
    #[serde(default)]
    pub keyframes: Vec<String>,
    #[serde(default)]
    pub timestep: f64,
    #[serde(default)]
    pub integrator: String,
    #[serde(default)]
    pub solver: String,
    #[serde(default)]
    pub iterations: u32,
    #[serde(default)]
    pub gravity: [f64; 3],
    #[serde(default)]
    pub nbody: u32,
    #[serde(default)]
    pub ngeom: u32,
    /// MuJoCo's own flag names in index order (`mjtVisFlag`, `mjtRndFlag`).
    #[serde(default)]
    pub vis_flags: Vec<String>,
    #[serde(default)]
    pub rnd_flags: Vec<String>,
    /// The group-mask kinds the stream offers (geom, site, joint, …),
    /// in wire order, and how many groups each has (MuJoCo's mjNGROUP).
    #[serde(default)]
    pub groups: Vec<String>,
    #[serde(default)]
    pub ngroup: u32,
    /// Worlds in a many-worlds scene (0 for a single world).
    #[serde(default)]
    pub nworld: u32,
    /// A walk scene's command bounds - forward, left, turn - from the
    /// task that runs it; absent for a scene without commands.
    #[serde(default)]
    pub twist_ranges: Option<[[f64; 2]; 3]>,
    /// What the scene is, in a line for the bar (a deployment's trial,
    /// its outcome, re-run or replayed); absent: the scene's name says it.
    #[serde(default)]
    pub caption: Option<String>,
}

/// The twist the human commands in a walk scene: which world, and the
/// forward, left and turn values.
#[derive(serde::Deserialize, Clone, Debug, PartialEq, Default)]
pub struct SimTwist {
    #[serde(default)]
    pub world: i64,
    #[serde(default)]
    pub value: [f64; 3],
}

/// The clock and the inputs, every 100 ms (the stream's status message).
#[derive(serde::Deserialize, Clone, Debug, PartialEq, Default)]
pub struct SimStatus {
    #[serde(default)]
    pub time: f64,
    #[serde(default)]
    pub rtf: f64,
    #[serde(default)]
    pub paused: bool,
    #[serde(default)]
    pub manual: bool,
    #[serde(default)]
    pub speed: f64,
    #[serde(default)]
    pub qpos: Vec<f64>,
    #[serde(default)]
    pub ctrl: Vec<f64>,
    #[serde(default)]
    pub shadows: bool,
    #[serde(default)]
    pub render_ms: f64,
    #[serde(default)]
    pub vis: std::collections::BTreeMap<String, bool>,
    /// The same flags by index, filled once when the status lands (the
    /// drawer asked the map by a formatted index fifty times a frame).
    #[serde(skip)]
    pub vis_table: Vec<Option<bool>>,
    #[serde(default)]
    pub rnd: std::collections::BTreeMap<String, bool>,
    #[serde(skip)]
    pub rnd_table: Vec<Option<bool>>,
    /// Each group mask as rendered, by kind name.
    #[serde(default)]
    pub groups: std::collections::BTreeMap<String, Vec<bool>>,
    /// The commanded twist while the human holds it; None while the
    /// task commands.
    #[serde(default)]
    pub twist: Option<SimTwist>,
    /// Where the camera is: azimuth, elevation, distance, lookat x y z.
    #[serde(default)]
    pub camera: Vec<f64>,
    #[serde(default)]
    pub follow: i64,
    #[serde(default)]
    pub worlds: Vec<SimWorld>,
    #[serde(default)]
    pub model: Option<SimModel>,
}

/// One world of a many-worlds scene, as the status reports it.
#[derive(serde::Deserialize, Clone, Debug, PartialEq, Default)]
pub struct SimWorld {
    #[serde(default)]
    pub reward: f64,
    #[serde(default)]
    pub done: bool,
}

/// What the reader thread learned from the stream's status messages —
/// shared, not copied: the panels ask for it several times a frame, and
/// a status carries every joint and control value.
#[derive(Default)]
pub struct SimReport {
    pub status: Option<Arc<SimStatus>>,
    pub model: Option<Arc<SimModel>>,
}

/// The scene previews the idle strip offers — pipeline-registry tasks
/// `studio-render-stream.py` can run. The LIVE scene never comes from
/// here: a training run mirrors itself into the Rerun 3D view below
/// (`world/robot`, the recorder's mirror).
pub const PREVIEW_TASKS: &[&str] = &["kitting", "lift", "duck"];

/// The RL view: `rq_mjlab.walk_view` rolls the newest trained walk
/// checkpoint (policy-driven worlds on the GPU, CPU mirror into the
/// same frame ring, Ctrl+drag shoves land in the batched sim). Its own
/// spawn shape: the rq_mjlab venv, not the pipeline's.
pub const WALK_TASK: &str = "walk";

/// A deployment in the viewport: `deploy:<name>` live, or
/// `deploy:<name>:gate:<runtime>:<trial>`,
/// `deploy:<name>:preflight:<segment>` and
/// `deploy:<name>:attribution:<knob>:<rung>` replayed (the modes are the
/// Python registry's; this side knows only the prefix)
/// (`rq_pipeline/deploy/viewport_source.py::DEPLOY_PREFIX`, pinned by
/// `tests/test_studio_mirrors.py`). Run by the render stream, in the
/// pipeline's venv, inside the open project.
pub const DEPLOY_PREFIX: &str = "deploy:";
/// The application id the viewport's physics twin streams under, before
/// the scene's name (`tools/studio-render-stream.py` mirrors it): what the
/// shell closes before a new viewport starts, so twins never stack.
pub const TWIN_APP_PREFIX: &str = "robotiq-sim-";

/// Whether the viewport can run a scene by this name: a preview task,
/// the walk, or a deployment.
pub fn is_known_scene(name: &str) -> bool {
    PREVIEW_TASKS.contains(&name) || name == WALK_TASK || name.starts_with(DEPLOY_PREFIX)
}

impl ViewportFeed {
    /// No subprocess, no canned scene: the panel starts as a slim strip
    /// offering previews. The operator's rule (2026-09-01): the window's
    /// 3D follows what is actually happening — and what is actually
    /// happening streams into the viewer below, not into this panel.
    pub fn idle() -> Self {
        Self {
            child: None,
            stdin: None,
            shm: None,
            frames_published: Arc::new(AtomicU64::new(0)),
            frames_drawn: 0,
            texture: None,
            spawn_error: None,
            sent_size: INITIAL_RENDER_SIZE,
            last_resize_sent: None,
            perturbing: false,
            stream_ended: Arc::new(AtomicBool::new(false)),
            task: None,
            drawn_at: std::collections::VecDeque::new(),
            fps_shown: std::cell::Cell::new((None, None)),
            agent_flash: None,
            report: Arc::new(std::sync::Mutex::new(SimReport::default())),
            editing: std::collections::HashMap::new(),
            twist_held: None,
            hovered: false,
            focused: false,
        }
    }

    /// Whether a preview is running (or died trying) — the panel sizes
    /// itself by this.
    /// The preview task running, if any.
    pub fn task(&self) -> Option<&str> {
        self.task.as_deref()
    }

    /// The frame rate for a panel: refreshed once a second, so the
    /// number does not change under the reader's eyes every frame.
    pub fn fps_settled(&self) -> Option<f32> {
        let now = std::time::Instant::now();
        let (shown, at) = self.fps_shown.get();
        if at.is_some_and(|t| now.duration_since(t) < std::time::Duration::from_secs(1)) {
            return shown;
        }
        let fresh = self.fps().map(f32::round);
        self.fps_shown.set((fresh, Some(now)));
        fresh
    }

    /// Frames drawn to the screen in the last second, once any were.
    pub fn fps(&self) -> Option<f32> {
        let now = std::time::Instant::now();
        let recent = self
            .drawn_at
            .iter()
            .filter(|t| now.duration_since(**t) <= std::time::Duration::from_secs(1))
            .count();
        (!self.drawn_at.is_empty()).then_some(recent as f32)
    }

    pub fn is_active(&self) -> bool {
        self.child.is_some() || self.spawn_error.is_some() || self.texture.is_some()
    }

    /// Spawns `tools/studio-render-stream.py` under the pipeline's `uv`
    /// environment. `ctx` is cloned into the reader thread so it can wake
    /// the UI (`request_repaint`) the moment a new frame lands — egui does
    /// not otherwise know that a background thread produced fresh pixels.
    pub fn spawn(ctx: &egui::Context, task_name: &str, project_root: &std::path::Path) -> Self {
        // The RL view runs in the rq_mjlab venv (torch + warp + mjlab);
        // the previews in the pipeline's. Same ring, same stdin
        // protocol, different door (`spawn.rs` builds both).
        let mut command = if task_name == WALK_TASK {
            walk_command(WALK_SPAWN.module)
        } else {
            pipeline_command(RENDER_STREAM_SCRIPT)
        };
        // The frame ring: created HERE (the reader's lifetime owns it),
        // sized for the largest frame the wire allows, handed to the
        // script by path. See ShmReader for the layout.
        let shm_path = std::env::temp_dir().join(format!(
            "rq-viewport-{}-{}-{}.rgb",
            std::process::id(),
            SHM_SEQUENCE.fetch_add(1, Ordering::Relaxed),
            file_safe(task_name),
        ));
        let shm = File::create(&shm_path)
            .and_then(|file| {
                file.set_len(
                    SHM_HEADER_BYTES + u64::from(MAX_RENDER_SIDE) * u64::from(MAX_RENDER_SIDE) * 3,
                )?;
                Ok(())
            })
            .and_then(|()| File::open(&shm_path))
            .map(|file| ShmReader {
                file,
                path: shm_path.clone(),
                last_seq: 0,
                rgb: Vec::new(),
            });

        if task_name == WALK_TASK {
            // The project's own walk: its robot, its latest checkpoint.
            command
                .arg("--latest")
                .arg(format!("--envs={}", WALK_SPAWN.envs))
                .arg(format!("--project={}", project_root.display()));
        } else if task_name.starts_with(DEPLOY_PREFIX) {
            // A deployment lives in the open project.
            command
                .arg(task_name)
                .arg(format!("--project={}", project_root.display()));
        } else {
            command.arg(task_name);
        }
        command.arg(format!("--shm={}", shm_path.display()));
        let spawned = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn();

        let frames_published = Arc::new(AtomicU64::new(0));
        let stream_ended = Arc::new(AtomicBool::new(false));

        match (spawned, shm) {
            (Ok(mut child), Ok(shm)) => {
                let stdout = child.stdout.take().expect("piped stdout, always present");
                let stdin = child.stdin.take().expect("piped stdin, always present");
                let report = Arc::new(std::sync::Mutex::new(SimReport::default()));
                spawn_token_reader(
                    stdout,
                    Arc::clone(&frames_published),
                    Arc::clone(&stream_ended),
                    Arc::clone(&report),
                    ctx.clone(),
                );
                Self {
                    child: Some(child),
                    stdin: Some(stdin),
                    shm: Some(shm),
                    frames_published,
                    frames_drawn: 0,
                    texture: None,
                    spawn_error: None,
                    sent_size: INITIAL_RENDER_SIZE,
                    last_resize_sent: None,
                    perturbing: false,
                    stream_ended,
                    task: Some(task_name.to_owned()),
                    drawn_at: std::collections::VecDeque::new(),
                    fps_shown: std::cell::Cell::new((None, None)),
                    agent_flash: None,
                    report,
                    editing: std::collections::HashMap::new(),
                    twist_held: None,
                    hovered: false,
                    focused: false,
                }
            }
            (Ok(mut child), Err(err)) => {
                kill_tree(&mut child);
                Self::errored(format!(
                    "could not create the frame ring at {}: {err}",
                    shm_path.display()
                ))
            }
            (Err(err), _) => Self::errored(format!(
                "could not start the {task_name} scene: {err} (is `uv` on PATH?)"
            )),
        }
    }

    fn errored(message: String) -> Self {
        let mut feed = Self::idle();
        feed.spawn_error = Some(message);
        feed
    }

    /// Draws the newest frame into `ui`, or a placeholder/error message
    /// before the first frame arrives or if the subprocess never started.
    /// Dragging orbits the camera; scrolling while hovered zooms it.
    /// Returns the rect the picture occupies, for overlays drawn on it.
    pub fn show(&mut self, ui: &mut egui::Ui) -> Option<egui::Rect> {
        if !self.is_active() {
            ui.centered_and_justified(|ui| {
                ui.label(
                    egui::RichText::new(
                        "No scene is running. Pick one from the bar below — a training \
                         run streams into the viewer underneath on its own.",
                    )
                    .color(ui.visuals().weak_text_color()),
                );
            });
            return None;
        }
        let published = self.frames_published.load(Ordering::Relaxed);
        if published != self.frames_drawn {
            if let Some(image) = self.shm.as_mut().and_then(ShmReader::latest) {
                self.frames_drawn = published;
                let now = std::time::Instant::now();
                self.drawn_at.push_back(now);
                while self
                    .drawn_at
                    .front()
                    .is_some_and(|t| now.duration_since(*t) > std::time::Duration::from_secs(2))
                {
                    self.drawn_at.pop_front();
                }
                match &mut self.texture {
                    // Update in place: `load_texture` allocates a brand-new
                    // texture every call, and this runs at frame rate.
                    Some(texture) => texture.set(image, TextureOptions::LINEAR),
                    None => {
                        self.texture = Some(ui.ctx().load_texture(
                            "studio-viewport",
                            image,
                            TextureOptions::LINEAR,
                        ));
                    }
                }
            }
        }

        if self.stream_ended.load(Ordering::Relaxed) {
            ui.error_label(
                "The simulator process for this scene exited; the frame below is its \
                 last. Its output is in the project's .index/studio.log. Pick the \
                 scene again to restart it.",
            );
        }

        let Some(texture) = &self.texture else {
            if let Some(err) = &self.spawn_error {
                ui.error_label(err);
            } else {
                ui.info_label("Waiting for the first frame from MuJoCo…");
            }
            return None;
        };

        // `max_size` only ever caps — it never grows an image up to fill
        // the panel. `fit_to_exact_size` is the one that scales to fill;
        // `maintain_aspect_ratio` keeps MuJoCo's frame from stretching to
        // match a panel of a different aspect ratio (it will letterbox
        // instead, which is correct when the two aspect ratios differ).
        let available = ui.available_size();
        // Centered: when aspects differ the fitted image is smaller than
        // the panel, and a left-hugging frame reads as broken (seen live
        // when a clamp bug made the mismatch large).
        let response = ui
            .with_layout(
                egui::Layout::centered_and_justified(egui::Direction::LeftToRight),
                |ui| {
                    ui.add(
                        egui::Image::new(texture)
                            .fit_to_exact_size(available)
                            .maintain_aspect_ratio(true)
                            .sense(egui::Sense::click_and_drag()),
                    )
                },
            )
            .inner;

        // Input over the image. Plain drag orbits; Ctrl+drag PERTURBS —
        // a select on press (the body under the pointer), move messages
        // while held, a release when the button or Ctrl lets go. The
        // physics side does the real work with MuJoCo's own mjv_select /
        // mjvPerturb; this side only names the gesture.
        let ctrl_held = ui.input(|i| i.modifiers.ctrl);
        let image_rect = fitted_rect(response.rect, texture.aspect_ratio());
        // A click on the picture takes the keyboard from whatever held
        // it (Rerun's own views take focus when clicked, and every
        // shortcut went silent after one, 2026-09-12); hovering is
        // enough too - keys go where the pointer is, as in every 3D tool.
        if response.clicked() || response.drag_started() {
            response.request_focus();
        }
        self.hovered = response.hovered();
        self.focused = response.has_focus();
        let mut d_azimuth = 0.0f32;
        let mut d_elevation = 0.0f32;
        let mut d_distance = 0.0f32;
        if response.drag_started() && ctrl_held {
            if let Some(pointer) = response.interact_pointer_pos() {
                let x = ((pointer.x - image_rect.min.x) / image_rect.width()).clamp(0.0, 1.0);
                let y = ((pointer.y - image_rect.min.y) / image_rect.height()).clamp(0.0, 1.0);
                self.send_message(&encode_select(x, y));
                self.perturbing = true;
            }
        }
        if response.dragged() {
            let delta = response.drag_delta();
            if self.perturbing {
                // Normalized by the image height — mjv_movePerturb's own
                // convention (simulate.cc divides both axes by height).
                self.send_message(&encode_perturb_drag(
                    delta.x / image_rect.height(),
                    delta.y / image_rect.height(),
                ));
            } else {
                d_azimuth = -delta.x * DRAG_DEGREES_PER_POINT;
                d_elevation = -delta.y * DRAG_DEGREES_PER_POINT;
            }
        }
        if self.perturbing && (response.drag_stopped() || !ctrl_held) {
            self.send_message(&[TAG_RELEASE]);
            self.perturbing = false;
        }
        if response.hovered() {
            d_distance = -ui.input(|i| i.smooth_scroll_delta.y) * ZOOM_METRES_PER_SCROLL_POINT;
        }

        // Track the panel's actual size so the render resolution follows
        // the window — debounced (see RESIZE_DEBOUNCE): mid-drag the size
        // changes every frame, and each change costs a Renderer rebuild
        // on the Python side. The final size always lands, because the
        // panel keeps rendering frames after the drag ends and the ripe
        // check passes then.
        //
        // PHYSICAL pixels, not egui points: the wire's size field feeds
        // `mujoco.Renderer`, which counts pixels. Requesting the point
        // size rendered every Retina (2×) frame at half resolution and
        // let egui upscale the difference — visibly soft next to the
        // Rerun viewer beside it.
        let pixels_per_point = ui.ctx().pixels_per_point();
        let want = render_size(available * pixels_per_point);
        let size_changed = want != self.sent_size;
        let resize_ripe = self
            .last_resize_sent
            .is_none_or(|at| at.elapsed() >= RESIZE_DEBOUNCE);
        if size_changed && resize_ripe {
            self.sent_size = want;
            self.last_resize_sent = Some(std::time::Instant::now());
        } else if size_changed {
            // Not ripe yet — repaint again soon so the settled size goes
            // out even if nothing else triggers a frame.
            ui.ctx().request_repaint_after(RESIZE_DEBOUNCE);
        }

        if d_azimuth != 0.0
            || d_elevation != 0.0
            || d_distance != 0.0
            || size_changed && resize_ripe
        {
            self.send_update(d_azimuth, d_elevation, d_distance);
        }
        Some(image_rect)
    }

    /// Whether the keys belong to the picture this frame: the pointer is
    /// over it or it was clicked last, and no text field is being typed
    /// in. With nothing focused at all the keys are the picture's too
    /// (the transport bar's shortcuts always worked that way).
    pub fn wants_keys(&self, ctx: &egui::Context) -> bool {
        if !self.is_active() || ctx.text_edit_focused() {
            return false;
        }
        self.hovered || self.focused || !ctx.egui_wants_keyboard_input()
    }

    /// Who has the keys, for the event log: none, picture, other, text.
    pub fn focus_owner(&self, ctx: &egui::Context) -> &'static str {
        if ctx.text_edit_focused() {
            "text"
        } else if self.focused {
            "picture"
        } else if ctx.egui_wants_keyboard_input() {
            "other"
        } else {
            "none"
        }
    }

    /// The stream's latest status and model, for the panel and the state
    /// file: two reference counts bumped, nothing copied.
    pub fn report(&self) -> (Option<Arc<SimStatus>>, Option<Arc<SimModel>>) {
        self.report
            .lock()
            .map(|r| (r.status.clone(), r.model.clone()))
            .unwrap_or((None, None))
    }

    pub fn send_run(&mut self, run: bool) {
        self.send_message(&[TAG_RUN, u8::from(run)]);
    }

    pub fn send_step(&mut self, steps: u32) {
        self.send_message(&encode_u32(TAG_STEP, steps));
    }

    /// Reset to a keyframe, or to the model's initial state with `None`.
    pub fn send_reset(&mut self, keyframe: Option<u32>) {
        let key = keyframe.map_or(-1i32, |k| k as i32);
        let mut bytes = [0u8; 5];
        bytes[0] = TAG_RESET;
        bytes[1..5].copy_from_slice(&key.to_le_bytes());
        self.send_message(&bytes);
    }

    pub fn send_speed(&mut self, factor: f32) {
        let mut bytes = [0u8; 5];
        bytes[0] = TAG_SPEED;
        bytes[1..5].copy_from_slice(&factor.to_le_bytes());
        self.send_message(&bytes);
    }

    pub fn send_manual(&mut self, on: bool) {
        self.send_message(&[TAG_MANUAL, u8::from(on)]);
    }

    pub fn send_ctrl(&mut self, actuator: u32, value: f32) {
        self.editing
            .insert((SliderKind::Actuator, actuator as usize), f64::from(value));
        self.send_message(&encode_index_value(TAG_CTRL, actuator, value));
    }

    pub fn send_qpos(&mut self, qpos_address: u32, value: f32) {
        self.editing
            .insert((SliderKind::Joint, qpos_address as usize), f64::from(value));
        self.send_message(&encode_index_value(TAG_QPOS, qpos_address, value));
    }

    pub fn send_vis(&mut self, flag: u32, on: bool) {
        self.send_message(&encode_flag(TAG_VIS, flag, on));
    }

    pub fn send_rnd(&mut self, flag: u32, on: bool) {
        self.send_message(&encode_flag(TAG_RND, flag, on));
    }

    /// The commanded twist for one world of a walk scene; world -1 hands
    /// the commands back to the task.
    pub fn send_twist(&mut self, world: i32, twist: [f32; 3]) {
        self.twist_held = (world >= 0).then_some((world, twist));
        self.send_message(&encode_twist(world, twist));
    }

    /// The twist the human holds, as last sent (None: the task's own).
    pub fn twist_held(&self) -> Option<(i32, [f32; 3])> {
        self.twist_held
    }

    /// One bit of a group mask: the kind by its wire index (the model's
    /// `groups` order), the group 0..ngroup.
    pub fn send_group(&mut self, kind: u8, group: u8, on: bool) {
        self.send_message(&[TAG_GROUP, kind, group, u8::from(on)]);
    }

    /// The camera to a named view, by its wire index ([`VIEW_PRESETS`]).
    pub fn send_view(&mut self, preset: u8) {
        self.send_message(&[TAG_VIEW, preset]);
    }

    /// Keep the camera on one world of a many-worlds scene (-1: none).
    pub fn send_follow(&mut self, world: i32) {
        let mut bytes = [0u8; 5];
        bytes[0] = TAG_FOLLOW;
        bytes[1..5].copy_from_slice(&world.to_le_bytes());
        self.send_message(&bytes);
    }

    /// Move the camera's lookat in its own frame — forward, right, up —
    /// by seconds of key held per axis (signed). The metres per second
    /// are the stream's fact (it knows the distance); this side only
    /// says how long a key was down.
    pub fn send_pan(&mut self, forward_s: f32, right_s: f32, up_s: f32) {
        self.send_message(&encode_pan(forward_s, right_s, up_s));
    }

    /// The agent pressed something: the bar shows it for a moment.
    pub fn flash(&mut self, what: &str) {
        self.agent_flash = Some((what.to_owned(), std::time::Instant::now()));
    }

    /// What the agent last did, while it is worth showing.
    pub fn flashing(&self) -> Option<&str> {
        self.agent_flash
            .as_ref()
            .filter(|(_, at)| at.elapsed() < AGENT_FLASH)
            .map(|(what, _)| what.as_str())
    }

    /// The value a slider shows: what the human is dragging, else the
    /// stream's echo.
    pub fn slider_value(&self, kind: SliderKind, index: usize, echoed: f64) -> f64 {
        self.editing.get(&(kind, index)).copied().unwrap_or(echoed)
    }

    /// A slider the human is dragging shows this until the drag ends
    /// (the stream's echo lags a frame or two behind the hand).
    pub fn start_editing(&mut self, kind: SliderKind, index: usize, value: f64) {
        self.editing.insert((kind, index), value);
    }

    /// The drag ended: the stream's echo is the truth again.
    pub fn stop_editing(&mut self, kind: SliderKind, index: usize) {
        self.editing.remove(&(kind, index));
    }

    fn send_update(&mut self, d_azimuth: f32, d_elevation: f32, d_distance: f32) {
        let message = encode_camera_update(d_azimuth, d_elevation, d_distance, self.sent_size);
        self.send_message(&message);
    }

    fn send_message(&mut self, message: &[u8]) {
        let Some(stdin) = &mut self.stdin else {
            return;
        };
        // A closed pipe here means the render subprocess died; `show`'s
        // next call will already be reporting `spawn_error`-shaped state
        // via an absent texture, so silently dropping this write is fine.
        let _ = stdin.write_all(message);
    }
}

impl Drop for ViewportFeed {
    fn drop(&mut self) {
        // `Child` does not kill on drop (the standard library leaves that
        // to the caller); an orphaned render-stream process is exactly the
        // kind of leak the crate's own smoke test already checked for by
        // hand — do it here so every caller gets it for free. The child
        // leads its own process group (see `spawn.rs`); the whole tree goes.
        if let Some(child) = self.child.take() {
            end_tree(child);
        }
    }
}

/// The length prefix of a status message, or `None` when it is past
/// [`MAX_STATUS_BYTES`] — a torn pipe, not a message.
fn status_length(prefix: [u8; 4]) -> Option<usize> {
    let len = u32::from_le_bytes(prefix) as usize;
    (len <= MAX_STATUS_BYTES).then_some(len)
}

/// The wake-up channel: the script writes one byte per frame published
/// into the ring; this thread turns each into a repaint. The pixels
/// themselves never touch the pipe (see ShmReader).
fn spawn_token_reader(
    mut stdout: ChildStdout,
    frames_published: Arc<AtomicU64>,
    stream_ended: Arc<AtomicBool>,
    report: Arc<std::sync::Mutex<SimReport>>,
    ctx: egui::Context,
) {
    thread::spawn(move || {
        let mut token = [0u8; 1];
        // One body buffer for the stream's life: a status arrives many
        // times a second, and a fresh allocation per message is waste.
        let mut body: Vec<u8> = Vec::new();
        while stdout.read_exact(&mut token).is_ok() {
            match token[0] {
                FRAME_TOKEN => {
                    frames_published.fetch_add(1, Ordering::Relaxed);
                    ctx.request_repaint();
                }
                STATUS_TOKEN => {
                    // u32 LE length, then JSON (the stream's `SimControl.status`).
                    let mut len = [0u8; 4];
                    if stdout.read_exact(&mut len).is_err() {
                        break;
                    }
                    let Some(len) = status_length(len) else {
                        break; // corruption: end the stream, loudly (below)
                    };
                    body.clear();
                    body.resize(len, 0);
                    if stdout.read_exact(&mut body).is_err() {
                        break;
                    }
                    if let Ok(mut status) = serde_json::from_slice::<SimStatus>(&body) {
                        if let Ok(mut slot) = report.lock() {
                            if let Some(model) = status.model.take() {
                                slot.model = Some(Arc::new(model));
                            }
                            if let Some(model) = &slot.model {
                                status.vis_table = flag_table(&status.vis, model.vis_flags.len());
                                status.rnd_table = flag_table(&status.rnd, model.rnd_flags.len());
                            }
                            slot.status = Some(Arc::new(status));
                        }
                    }
                }
                _ => {} // a token this build does not know: skip it
            }
        }
        // Loud, not quiet: a dead stream shows as an error in the panel
        // rather than a frame silently frozen mid-motion.
        stream_ended.store(true, Ordering::Relaxed);
        ctx.request_repaint();
    });
}

/// A flag map keyed by formatted index (the stream's JSON) as a table
/// by index: `None` where the stream said nothing.
fn flag_table(map: &std::collections::BTreeMap<String, bool>, len: usize) -> Vec<Option<bool>> {
    let mut table = vec![None; len];
    for (key, on) in map {
        if let Ok(index) = key.parse::<usize>() {
            if index < len {
                table[index] = Some(*on);
            }
        }
    }
    table
}

/// The largest aspect-preserving sub-rect of `outer` matching the
/// texture's ratio, centered — where the letterboxed image actually
/// sits, so pointer positions normalize against the PICTURE, not the
/// panel around it.
fn fitted_rect(outer: egui::Rect, aspect: f32) -> egui::Rect {
    let outer_aspect = outer.width() / outer.height();
    let size = if outer_aspect > aspect {
        egui::vec2(outer.height() * aspect, outer.height())
    } else {
        egui::vec2(outer.width(), outer.width() / aspect)
    };
    egui::Rect::from_center_size(outer.center(), size)
}

/// Desired physical pixels → the render size to request, bounds applied
/// by SCALING BOTH dimensions together, never clamping one alone: the
/// first Retina pass clamped per-dimension, a wide panel hit the 1920
/// width cap with its height untouched, and the requested aspect no
/// longer matched the panel's — the frame letterboxed to a strip
/// (seen live). Aspect is the invariant; resolution is the variable.
fn render_size(desired: egui::Vec2) -> (u32, u32) {
    let (w, h) = (desired.x.max(1.0), desired.y.max(1.0));
    let max = MAX_RENDER_SIDE as f32;
    let min = MIN_RENDER_SIDE as f32;
    let mut scale = (max / w).min(max / h).min(1.0);
    scale = scale.max(min / w).max(min / h);
    // The final per-dimension clamp only bites on degenerate aspects
    // (a sliver so extreme both bounds can't hold at once) — there,
    // staying inside the compiled framebuffer wins over exact aspect.
    (
        ((w * scale).round() as u32).clamp(MIN_RENDER_SIDE, MAX_RENDER_SIDE),
        ((h * scale).round() as u32).clamp(MIN_RENDER_SIDE, MAX_RENDER_SIDE),
    )
}

/// The stdin protocol's tags — studio-render-stream.py's
/// `TAG_*` constants, mirrored (a u8 tag then a fixed payload).
const TAG_CAMERA: u8 = 1;
const TAG_SELECT: u8 = 2;
const TAG_DRAG: u8 = 3;
const TAG_RELEASE: u8 = 4;
// The simulate controls (studio-render-stream.py `TAG_RUN` …): run,
// step, reset, speed, manual, an actuator value, a joint value, a
// visualization flag, a rendering flag.
const TAG_RUN: u8 = 6;
const TAG_STEP: u8 = 7;
const TAG_RESET: u8 = 8;
const TAG_SPEED: u8 = 9;
const TAG_MANUAL: u8 = 10;
const TAG_CTRL: u8 = 11;
const TAG_QPOS: u8 = 12;
const TAG_VIS: u8 = 13;
const TAG_RND: u8 = 14;
const TAG_VIEW: u8 = 15;
const TAG_FOLLOW: u8 = 16;
const TAG_PAN: u8 = 17;
const TAG_GROUP: u8 = 18;
const TAG_TWIST: u8 = 19;
/// The stdout tokens: a frame published, a status message follows.
const FRAME_TOKEN: u8 = 0xF7;
const STATUS_TOKEN: u8 = 0xF8;

/// One camera+size update: tag then Python's `struct.unpack("<fffII", …)`
/// exactly — three little-endian f32 deltas, two little-endian u32
/// absolute pixels. Tested below against bytes Python's own struct
/// module would produce.
fn encode_camera_update(
    d_azimuth: f32,
    d_elevation: f32,
    d_distance: f32,
    size: (u32, u32),
) -> [u8; 21] {
    let mut bytes = [0u8; 21];
    bytes[0] = TAG_CAMERA;
    bytes[1..5].copy_from_slice(&d_azimuth.to_le_bytes());
    bytes[5..9].copy_from_slice(&d_elevation.to_le_bytes());
    bytes[9..13].copy_from_slice(&d_distance.to_le_bytes());
    bytes[13..17].copy_from_slice(&size.0.to_le_bytes());
    bytes[17..21].copy_from_slice(&size.1.to_le_bytes());
    bytes
}

/// One pan: tag then Python's `struct.unpack("<fff", …)` — seconds a
/// key was held along each camera axis, signed.
fn encode_pan(forward_s: f32, right_s: f32, up_s: f32) -> [u8; 13] {
    let mut bytes = [0u8; 13];
    bytes[0] = TAG_PAN;
    bytes[1..5].copy_from_slice(&forward_s.to_le_bytes());
    bytes[5..9].copy_from_slice(&right_s.to_le_bytes());
    bytes[9..13].copy_from_slice(&up_s.to_le_bytes());
    bytes
}

/// One twist: tag then Python's `struct.unpack("<ifff", …)` — the world
/// and the forward, left, turn values.
fn encode_twist(world: i32, twist: [f32; 3]) -> [u8; 17] {
    let mut bytes = [0u8; 17];
    bytes[0] = TAG_TWIST;
    bytes[1..5].copy_from_slice(&world.to_le_bytes());
    for (i, v) in twist.iter().enumerate() {
        bytes[5 + 4 * i..9 + 4 * i].copy_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// Start a perturbation on the body under the pointer — coordinates
/// normalized in the IMAGE, top-left origin (Python flips y for
/// mjv_select's bottom-up convention).
fn encode_select(x: f32, y: f32) -> [u8; 9] {
    let mut bytes = [0u8; 9];
    bytes[0] = TAG_SELECT;
    bytes[1..5].copy_from_slice(&x.to_le_bytes());
    bytes[5..9].copy_from_slice(&y.to_le_bytes());
    bytes
}

/// Move the active perturbation — deltas normalized by the image
/// height (mjv_movePerturb's own convention).
fn encode_perturb_drag(dx: f32, dy: f32) -> [u8; 9] {
    let mut bytes = [0u8; 9];
    bytes[0] = TAG_DRAG;
    bytes[1..5].copy_from_slice(&dx.to_le_bytes());
    bytes[5..9].copy_from_slice(&dy.to_le_bytes());
    bytes
}

fn encode_u32(tag: u8, value: u32) -> [u8; 5] {
    let mut bytes = [0u8; 5];
    bytes[0] = tag;
    bytes[1..5].copy_from_slice(&value.to_le_bytes());
    bytes
}

/// `struct.pack("<If", index, value)` after the tag.
fn encode_index_value(tag: u8, index: u32, value: f32) -> [u8; 9] {
    let mut bytes = [0u8; 9];
    bytes[0] = tag;
    bytes[1..5].copy_from_slice(&index.to_le_bytes());
    bytes[5..9].copy_from_slice(&value.to_le_bytes());
    bytes
}

/// `struct.pack("<IB", flag, on)` after the tag.
fn encode_flag(tag: u8, flag: u32, on: bool) -> [u8; 6] {
    let mut bytes = [0u8; 6];
    bytes[0] = tag;
    bytes[1..5].copy_from_slice(&flag.to_le_bytes());
    bytes[5] = u8::from(on);
    bytes
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_scene_name_becomes_a_file_name_part_on_every_platform() {
        assert_eq!(
            super::file_safe("deploy:go2-c2:gate:dds:3"),
            "deploy-go2-c2-gate-dds-3"
        );
        assert_eq!(super::file_safe("walk"), "walk");
    }

    #[test]
    fn two_spawns_of_one_scene_get_two_frame_files() {
        use std::sync::atomic::Ordering;
        let a = super::SHM_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let b = super::SHM_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        assert_ne!(a, b);
    }

    use super::*;

    #[test]
    fn a_deployment_is_a_known_scene_by_its_prefix() {
        assert!(is_known_scene("kitting"));
        assert!(is_known_scene(WALK_TASK));
        assert!(is_known_scene("deploy:go2-c2"));
        assert!(is_known_scene("deploy:go2-c2:gate:dds:3"));
        assert!(!is_known_scene("go2-c2"));
        assert!(!is_known_scene("nothing"));
    }

    #[test]
    fn the_simulate_controls_match_pythons_struct_formats() {
        // struct.pack("<If", 3, 0.5) and struct.pack("<IB", 14, 1), tag first.
        assert_eq!(
            encode_index_value(TAG_CTRL, 3, 0.5),
            [11, 3, 0, 0, 0, 0, 0, 0, 0x3F]
        );
        assert_eq!(encode_flag(TAG_VIS, 14, true), [13, 14, 0, 0, 0, 1]);
        assert_eq!(encode_u32(TAG_STEP, 10), [7, 10, 0, 0, 0]);
        // struct.pack("<fff", 0.5, -1.0, 0.0), tag first.
        assert_eq!(
            encode_pan(0.5, -1.0, 0.0),
            [17, 0, 0, 0, 0x3F, 0, 0, 0x80, 0xBF, 0, 0, 0, 0]
        );
        // struct.pack("<ifff", -1, 1.0, 0.0, 0.0), tag first.
        assert_eq!(
            encode_twist(-1, [1.0, 0.0, 0.0]),
            [19, 0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0x80, 0x3F, 0, 0, 0, 0, 0, 0, 0, 0]
        );
    }

    #[test]
    fn a_status_message_parses_with_its_model_once() {
        let text = r#"{"time":1.5,"rtf":0.99,"paused":false,"manual":true,"speed":1.0,
            "qpos":[0.1],"ctrl":[0.2],"shadows":false,"render_ms":10.4,"vis":{"14":true},"rnd":{},
            "model":{"joints":[{"name":"hip","qpos":7,"range":[-1.0,1.0],"limited":true,"type":"hinge"}],
            "actuators":[{"name":"hip","range":[-1.0,1.0],"limited":true}],"keyframes":["home"],
            "timestep":0.002,"integrator":"EULER","solver":"NEWTON","iterations":100,
            "gravity":[0.0,0.0,-9.81],"nbody":2,"ngeom":3,"vis_flags":["convexhull"],"rnd_flags":["shadow"]}}"#;
        let status: SimStatus = serde_json::from_str(text).expect("parses");
        assert!(status.manual);
        let model = status.model.expect("model once");
        assert_eq!(model.joints[0].qpos, 7);
        assert_eq!(model.keyframes, vec!["home"]);
        assert_eq!(model.integrator, "EULER");
        let bare: SimStatus = serde_json::from_str(r#"{"time":2.0}"#).expect("parses");
        assert!(bare.model.is_none());
    }

    #[test]
    fn the_wire_matches_pythons_struct_format() {
        // 0x01 then struct.pack("<fffII", 1.0, -2.5, 0.25, 1024, 576) —
        // computed with CPython and pinned here, so both ends of the
        // wire are held to the same bytes.
        let expected: [u8; 21] = [
            0x01, // TAG_CAMERA
            0x00, 0x00, 0x80, 0x3f, // 1.0f32 LE
            0x00, 0x00, 0x20, 0xc0, // -2.5f32 LE
            0x00, 0x00, 0x80, 0x3e, // 0.25f32 LE
            0x00, 0x04, 0x00, 0x00, // 1024u32 LE
            0x40, 0x02, 0x00, 0x00, // 576u32 LE
        ];
        assert_eq!(encode_camera_update(1.0, -2.5, 0.25, (1024, 576)), expected);
    }

    #[test]
    fn the_perturb_messages_carry_their_tags_and_floats() {
        let select = encode_select(0.5, 0.25);
        assert_eq!(select[0], TAG_SELECT);
        assert_eq!(&select[1..5], &0.5f32.to_le_bytes());
        assert_eq!(&select[5..9], &0.25f32.to_le_bytes());
        let drag = encode_perturb_drag(-0.01, 0.02);
        assert_eq!(drag[0], TAG_DRAG);
        assert_eq!(&drag[1..5], &(-0.01f32).to_le_bytes());
        assert_eq!(TAG_RELEASE, 4); // the one-byte message is the tag itself
    }

    #[test]
    fn the_fitted_rect_letterboxes_around_the_picture() {
        // A 2:1 texture in a square panel: full width, half height,
        // centered — pointer math must normalize against THAT rect.
        let outer = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(100.0, 100.0));
        let fitted = fitted_rect(outer, 2.0);
        assert_eq!(fitted.width(), 100.0);
        assert_eq!(fitted.height(), 50.0);
        assert_eq!(fitted.center(), outer.center());
    }

    #[test]
    fn oversized_requests_scale_both_dimensions_preserving_aspect() {
        // The regression this exists for: a 3000×380 Retina request was
        // clamped to 1920×380 — aspect 7.9:1 became 5.05:1 and the
        // frame letterboxed to a strip. Scaling keeps the ratio.
        let (w, h) = render_size(egui::vec2(3000.0, 380.0));
        assert_eq!((w, h), (1920, 243));
        let requested = 3000.0 / 380.0;
        let got = f64::from(w) / f64::from(h);
        assert!(
            (got - requested).abs() / requested < 0.01,
            "{got} vs {requested}"
        );
    }

    #[test]
    fn small_and_degenerate_sizes_stay_inside_bounds() {
        // A sliver scales UP to the floor…
        let (w, h) = render_size(egui::vec2(64.0, 40.0));
        assert!(w >= MIN_RENDER_SIDE && h >= MIN_RENDER_SIDE);
        // …and an aspect too extreme for both bounds still lands inside
        // them (framebuffer safety beats exact aspect there).
        let (w, h) = render_size(egui::vec2(10_000.0, 10.0));
        assert!(w <= MAX_RENDER_SIDE && h >= MIN_RENDER_SIDE);
    }

    #[test]
    fn a_status_length_past_the_cap_is_corruption() {
        assert_eq!(status_length(512u32.to_le_bytes()), Some(512));
        assert_eq!(
            status_length((MAX_STATUS_BYTES as u32).to_le_bytes()),
            Some(MAX_STATUS_BYTES)
        );
        assert_eq!(status_length(u32::MAX.to_le_bytes()), None);
    }

    #[test]
    fn views_are_named_once_for_the_menu_and_the_door() {
        assert_eq!(view_preset("front"), Some(1));
        assert_eq!(view_preset(" Reset "), Some(0));
        assert_eq!(view_preset("behind"), None);
        assert_eq!(view_preset_names(), vec!["front", "side", "top", "reset"]);
        assert!(SPEED_RANGE.contains(&1.0) && !SPEED_RANGE.contains(&0.0));
    }

    #[test]
    fn a_no_op_update_still_carries_the_size() {
        let bytes = encode_camera_update(0.0, 0.0, 0.0, (1920, 128));
        assert_eq!(bytes[0], TAG_CAMERA);
        assert_eq!(&bytes[1..13], &[0u8; 12]);
        assert_eq!(u32::from_le_bytes(bytes[13..17].try_into().unwrap()), 1920);
        assert_eq!(u32::from_le_bytes(bytes[17..21].try_into().unwrap()), 128);
    }
}
