//! The 3D viewport: not our own renderer, MuJoCo's.
//!
//! `tools/studio-render-stream.py` runs the real `mujoco.Renderer` this repo
//! already trusts (pinned to the same `mujoco~=3.11.0` the pipeline uses)
//! and frames its output onto stdout. This module spawns that script,
//! reads the frames on a background thread, and hands the newest one to
//! egui as a texture. No MuJoCo C API is linked into this crate — see the
//! module doc comment in the Python script for why (the FFI route needs a
//! patched fork of `glutin` on macOS plus a second, older MuJoCo install).
//!
//! Orbit and zoom work the same way in reverse: a drag/scroll on the image
//! updates `Orbit` here, which gets written back to the subprocess's stdin
//! in the script's wire format (3 `f32`s, then 2 `u32`s). There is no pan
//! yet — see `Orbit`. The same message also carries the panel's current
//! pixel size, so the render resolution tracks the window instead of
//! staying a fixed 1024x576 letterboxed into whatever shape the panel is.

use std::io::{Read, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use egui::{ColorImage, TextureHandle, TextureOptions};
use re_ui::UiExt as _;

struct RawFrame {
    width: usize,
    height: usize,
    rgb: Vec<u8>,
}

/// Azimuth/elevation/distance around a fixed look-at point, plus the
/// render's pixel size. Matches `studio-render-stream.py`'s `OrbitCamera`
/// and `WIDTH`/`HEIGHT` defaults field-for-field — the two are duplicated,
/// not shared, because they cross a language boundary; if one changes,
/// change the other.
#[derive(Clone, Copy, PartialEq)]
struct Orbit {
    azimuth_deg: f32,
    elevation_deg: f32,
    distance_m: f32,
    width_px: u32,
    height_px: u32,
}

impl Default for Orbit {
    fn default() -> Self {
        Self {
            azimuth_deg: 90.0,
            elevation_deg: -20.0,
            distance_m: 1.0,
            width_px: 1024,
            height_px: 576,
        }
    }
}

/// Degrees of orbit per point of drag, and metres of distance per point of
/// scroll — tuned by feel against a 1024x576 viewport, not measured.
const DRAG_DEGREES_PER_POINT: f32 = 0.4;
const ZOOM_METRES_PER_SCROLL_POINT: f32 = 0.004;
const MIN_DISTANCE_M: f32 = 0.15;
const MAX_DISTANCE_M: f32 = 4.0;
const MAX_ELEVATION_DEG: f32 = 89.0;

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

/// Owns the render-stream subprocess, the latest decoded frame, and the
/// orbit state the user is driving by dragging/scrolling over the image.
pub struct ViewportFeed {
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    latest: Arc<Mutex<Option<RawFrame>>>,
    texture: Option<TextureHandle>,
    spawn_error: Option<String>,
    orbit: Orbit,
    /// When the last size change went out — see `RESIZE_DEBOUNCE`.
    last_resize_sent: Option<std::time::Instant>,
    /// Set by the reader thread when the frame stream ends. A dead
    /// subprocess used to leave its last frame frozen on screen with no
    /// indication anything was wrong (measured: an oversized resize
    /// request killed the renderer and the panel just... stopped) —
    /// `show` turns this into a visible error instead.
    stream_ended: Arc<AtomicBool>,
}

impl ViewportFeed {
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
        let spawned = Command::new("uv")
            .args(["run", "--extra", "sim", "--extra", "viz", "python"])
            .arg(&script)
            .arg(task_name)
            .current_dir(&pipeline_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn();

        let latest = Arc::new(Mutex::new(None));
        let stream_ended = Arc::new(AtomicBool::new(false));

        match spawned {
            Ok(mut child) => {
                let stdout = child.stdout.take().expect("piped stdout, always present");
                let stdin = child.stdin.take().expect("piped stdin, always present");
                spawn_reader(
                    stdout,
                    Arc::clone(&latest),
                    Arc::clone(&stream_ended),
                    ctx.clone(),
                );
                Self {
                    child: Some(child),
                    stdin: Some(stdin),
                    latest,
                    texture: None,
                    spawn_error: None,
                    orbit: Orbit::default(),
                    last_resize_sent: None,
                    stream_ended,
                }
            }
            Err(err) => Self {
                child: None,
                stdin: None,
                latest,
                texture: None,
                spawn_error: Some(format!(
                    "could not start {}: {err} (is `uv` on PATH?)",
                    script.display()
                )),
                orbit: Orbit::default(),
                last_resize_sent: None,
                stream_ended,
            },
        }
    }

    /// Draws the newest frame into `ui`, or a placeholder/error message
    /// before the first frame arrives or if the subprocess never started.
    /// Dragging orbits the camera; scrolling while hovered zooms it.
    pub fn show(&mut self, ui: &mut egui::Ui) {
        if let Some(frame) = self.latest.lock().expect("not poisoned").take() {
            let image = ColorImage::from_rgb([frame.width, frame.height], &frame.rgb);
            self.texture = Some(ui.ctx().load_texture(
                "studio-viewport",
                image,
                TextureOptions::LINEAR,
            ));
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
        let response = ui.add(
            egui::Image::new(texture)
                .fit_to_exact_size(available)
                .maintain_aspect_ratio(true)
                .sense(egui::Sense::click_and_drag()),
        );

        let before = self.orbit;
        if response.dragged() {
            let delta = response.drag_delta();
            self.orbit.azimuth_deg -= delta.x * DRAG_DEGREES_PER_POINT;
            self.orbit.elevation_deg = (self.orbit.elevation_deg
                - delta.y * DRAG_DEGREES_PER_POINT)
                .clamp(-MAX_ELEVATION_DEG, MAX_ELEVATION_DEG);
        }
        if response.hovered() {
            let scroll_y = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll_y != 0.0 {
                self.orbit.distance_m = (self.orbit.distance_m
                    - scroll_y * ZOOM_METRES_PER_SCROLL_POINT)
                    .clamp(MIN_DISTANCE_M, MAX_DISTANCE_M);
            }
        }
        // Track the panel's actual size so the render resolution follows
        // the window instead of staying a fixed 1024x576 letterboxed into
        // whatever shape the panel happens to be — but debounced (see
        // RESIZE_DEBOUNCE): mid-drag the size changes every frame, and
        // each change costs a Renderer rebuild on the Python side. The
        // final size always lands, because the panel keeps rendering
        // frames after the drag ends and the ripe check passes then.
        let want_width = (available.x.round() as u32).clamp(MIN_RENDER_SIDE, MAX_RENDER_SIDE);
        let want_height = (available.y.round() as u32).clamp(MIN_RENDER_SIDE, MAX_RENDER_SIDE);
        let size_changed = (want_width, want_height) != (self.orbit.width_px, self.orbit.height_px);
        let resize_ripe = self
            .last_resize_sent
            .is_none_or(|at| at.elapsed() >= RESIZE_DEBOUNCE);
        if size_changed && resize_ripe {
            self.orbit.width_px = want_width;
            self.orbit.height_px = want_height;
            self.last_resize_sent = Some(std::time::Instant::now());
        } else if size_changed {
            // Not ripe yet — repaint again soon so the settled size goes
            // out even if nothing else triggers a frame.
            ui.ctx().request_repaint_after(RESIZE_DEBOUNCE);
        }

        if self.orbit != before {
            self.send_orbit();
        }
    }

    fn send_orbit(&mut self) {
        let Some(stdin) = &mut self.stdin else {
            return;
        };
        let mut bytes = [0u8; 20];
        bytes[0..4].copy_from_slice(&self.orbit.azimuth_deg.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.orbit.elevation_deg.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.orbit.distance_m.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.orbit.width_px.to_le_bytes());
        bytes[16..20].copy_from_slice(&self.orbit.height_px.to_le_bytes());
        // A closed pipe here means the render subprocess died; `show`'s
        // next call will already be reporting `spawn_error`-shaped state
        // via an absent texture, so silently dropping this write is fine.
        let _ = stdin.write_all(&bytes);
    }
}

impl Drop for ViewportFeed {
    fn drop(&mut self) {
        // `Child` does not kill on drop (the standard library leaves that
        // to the caller); an orphaned render-stream process is exactly the
        // kind of leak the crate's own smoke test already checked for by
        // hand — do it here so every caller gets it for free.
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn spawn_reader(
    mut stdout: ChildStdout,
    latest: Arc<Mutex<Option<RawFrame>>>,
    stream_ended: Arc<AtomicBool>,
    ctx: egui::Context,
) {
    thread::spawn(move || {
        let mut header = [0u8; 8];
        loop {
            if stdout.read_exact(&mut header).is_err() {
                break; // subprocess exited or pipe closed
            }
            let width = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
            let height = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;

            let mut rgb = vec![0u8; width * height * 3];
            if stdout.read_exact(&mut rgb).is_err() {
                break;
            }

            *latest.lock().expect("not poisoned") = Some(RawFrame { width, height, rgb });
            ctx.request_repaint();
        }
        // Loud, not quiet: a dead stream shows as an error in the panel
        // rather than a frame silently frozen mid-motion.
        stream_ended.store(true, Ordering::Relaxed);
        ctx.request_repaint();
    });
}
