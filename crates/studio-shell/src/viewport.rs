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
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;

use egui::{ColorImage, TextureHandle, TextureOptions};
use re_ui::UiExt as _;

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
        let mut rgb = vec![0u8; width * height * 3];
        self.file.read_exact(&mut rgb).ok()?;
        // Seqlock close: a write that landed mid-copy moved the counter;
        // discard the torn frame and let the next repaint pick it up.
        self.file.seek(SeekFrom::Start(4)).ok()?;
        let mut seq_after = [0u8; 4];
        self.file.read_exact(&mut seq_after).ok()?;
        if u32::from_le_bytes(seq_after) != seq {
            return None;
        }
        self.last_seq = seq;
        Some(ColorImage::from_rgb([width, height], &rgb))
    }
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
        }
    }

    /// Whether a preview is running (or died trying) — the panel sizes
    /// itself by this.
    /// The preview task running, if any.
    pub fn task(&self) -> Option<&str> {
        self.task.as_deref()
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
    pub fn spawn(ctx: &egui::Context, task_name: &str) -> Self {
        let repo_root = crate::repo_root();
        let pipeline_dir = repo_root.join("pipeline");
        let script = repo_root.join("tools").join("studio-render-stream.py");

        // `viz` brings rerun-sdk: the script narrates the physics into
        // the app's own embedded viewer (best-effort — see the script).
        let mut command = Command::new("uv");
        // Its own process group, so stop/drop can reap the WHOLE tree:
        // killing only the `uv` wrapper left the python grandchild
        // alive and flooding the ingest channel (the zombie stream,
        // 2026-09-01).
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt as _;
            command.process_group(0);
        }
        // The frame ring: created HERE (the reader's lifetime owns it),
        // sized for the largest frame the wire allows, handed to the
        // script by path. See ShmReader for the layout.
        let shm_path = std::env::temp_dir().join(format!(
            "rq-viewport-{}-{task_name}.rgb",
            std::process::id()
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
            });

        if task_name == WALK_TASK {
            // The RL view runs in the rq_mjlab venv (torch + warp +
            // mjlab); same ring, same stdin protocol, different door.
            command
                .args(["run", "--offline", "python", "-m", "rq_mjlab.walk_view"])
                .args(["--latest", "--envs", "9"])
                .arg(format!("--shm={}", shm_path.display()))
                .current_dir(repo_root.join("rq_mjlab"));
        } else {
            command
                .args(["run", "--extra", "sim", "--extra", "viz", "python"])
                .arg(&script)
                .arg(task_name)
                .arg(format!("--shm={}", shm_path.display()))
                .current_dir(&pipeline_dir);
        }
        // The pipeline's own wsl.env, spelled here because this spawn
        // does not go through a tool wrapper: without these, MuJoCo's
        // offscreen GL on WSL falls back to llvmpipe — the SOFTWARE
        // rasterizer at ~300 ms/frame and ~300% CPU (measured on the
        // kitting preview, 2026-09-01; the box's documented gotcha).
        // Harmless on native Linux; macOS must not get MUJOCO_GL=egl.
        #[cfg(target_os = "linux")]
        {
            command
                .env("MUJOCO_GL", "egl")
                .env("GALLIUM_DRIVER", "d3d12")
                .env("LD_LIBRARY_PATH", "/usr/lib/wsl/lib")
                .env("OMP_NUM_THREADS", "1");
        }
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
                spawn_token_reader(
                    stdout,
                    Arc::clone(&frames_published),
                    Arc::clone(&stream_ended),
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
                }
            }
            (Ok(mut child), Err(err)) => {
                let _ = child.kill();
                let _ = child.wait();
                Self::errored(format!(
                    "could not create the frame ring at {}: {err}",
                    shm_path.display()
                ))
            }
            (Err(err), _) => Self::errored(format!(
                "could not start {}: {err} (is `uv` on PATH?)",
                script.display()
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
    pub fn show(&mut self, ui: &mut egui::Ui) {
        if !self.is_active() {
            ui.horizontal(|ui| {
                ui.label(
                    "Live physics streams into the 3D view below while a run is on. \
                     Preview a pipeline scene:",
                );
                for task in PREVIEW_TASKS {
                    if ui.button(*task).clicked() {
                        *self = Self::spawn(ui.ctx(), task);
                    }
                }
                if ui
                    .button(WALK_TASK)
                    .on_hover_text(
                        "the newest trained walk checkpoint, live — \
                         Ctrl+drag shoves a duck and the policy recovers",
                    )
                    .clicked()
                {
                    *self = Self::spawn(ui.ctx(), WALK_TASK);
                }
            });
            return;
        }
        ui.horizontal(|ui| {
            ui.label("scene preview (not the live run)");
            if ui.button("✕ stop").clicked() {
                *self = Self::idle();
            }
        });
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
                "The MuJoCo render stream ended — the frame below is the last one \
                 received. Restart the app; the cause is in its terminal output.",
            );
        }

        let Some(texture) = &self.texture else {
            if let Some(err) = &self.spawn_error {
                ui.error_label(err);
            } else {
                ui.info_label("Waiting for the first frame from MuJoCo…");
            }
            return;
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
        // hand — do it here so every caller gets it for free.
        if let Some(mut child) = self.child.take() {
            // The child leads its own process group (see spawn); kill
            // the group so the python grandchild dies with the wrapper.
            #[cfg(unix)]
            {
                // `-s TERM -- -PGID`: without the `--`, procps kill can
                // re-parse a negative pgid as a signal spec plus a DIFFERENT
                // pid — measured 2026-09-01, and the mis-signaled process
                // was the Studio itself (stop closed the whole app).
                let _ = Command::new("kill")
                    .args(["-s", "TERM", "--", &format!("-{}", child.id())])
                    .status();
            }
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// The wake-up channel: the script writes one byte per frame published
/// into the ring; this thread turns each into a repaint. The pixels
/// themselves never touch the pipe (see ShmReader).
fn spawn_token_reader(
    mut stdout: ChildStdout,
    frames_published: Arc<AtomicU64>,
    stream_ended: Arc<AtomicBool>,
    ctx: egui::Context,
) {
    thread::spawn(move || {
        let mut token = [0u8; 1];
        while stdout.read_exact(&mut token).is_ok() {
            frames_published.fetch_add(1, Ordering::Relaxed);
            ctx.request_repaint();
        }
        // Loud, not quiet: a dead stream shows as an error in the panel
        // rather than a frame silently frozen mid-motion.
        stream_ended.store(true, Ordering::Relaxed);
        ctx.request_repaint();
    });
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn a_no_op_update_still_carries_the_size() {
        let bytes = encode_camera_update(0.0, 0.0, 0.0, (1920, 128));
        assert_eq!(bytes[0], TAG_CAMERA);
        assert_eq!(&bytes[1..13], &[0u8; 12]);
        assert_eq!(u32::from_le_bytes(bytes[13..17].try_into().unwrap()), 1920);
        assert_eq!(u32::from_le_bytes(bytes[17..21].try_into().unwrap()), 128);
    }
}
