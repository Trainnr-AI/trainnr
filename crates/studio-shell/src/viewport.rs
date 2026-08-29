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
//! in the script's 3-`f32` wire format. There is no pan yet — see `Orbit`.

use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;

use egui::{ColorImage, TextureHandle, TextureOptions};

struct RawFrame {
    width: usize,
    height: usize,
    rgb: Vec<u8>,
}

/// Azimuth/elevation/distance around a fixed look-at point. Matches
/// `studio-render-stream.py`'s `OrbitCamera` field-for-field and default-
/// for-default — the two are duplicated, not shared, because they cross a
/// language boundary; if one changes, change the other.
#[derive(Clone, Copy, PartialEq)]
struct Orbit {
    azimuth_deg: f32,
    elevation_deg: f32,
    distance_m: f32,
}

impl Default for Orbit {
    fn default() -> Self {
        Self {
            azimuth_deg: 90.0,
            elevation_deg: -20.0,
            distance_m: 1.0,
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

/// Owns the render-stream subprocess, the latest decoded frame, and the
/// orbit state the user is driving by dragging/scrolling over the image.
pub struct ViewportFeed {
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    latest: Arc<Mutex<Option<RawFrame>>>,
    texture: Option<TextureHandle>,
    spawn_error: Option<String>,
    orbit: Orbit,
}

impl ViewportFeed {
    /// Spawns `tools/studio-render-stream.py` under the pipeline's `uv`
    /// environment. `ctx` is cloned into the reader thread so it can wake
    /// the UI (`request_repaint`) the moment a new frame lands — egui does
    /// not otherwise know that a background thread produced fresh pixels.
    pub fn spawn(ctx: &egui::Context, task_name: &str) -> Self {
        let repo_root = repo_root();
        let pipeline_dir = repo_root.join("pipeline");
        let script = repo_root.join("tools").join("studio-render-stream.py");

        let spawned = Command::new("uv")
            .args(["run", "--extra", "sim", "python"])
            .arg(&script)
            .arg(task_name)
            .current_dir(&pipeline_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn();

        let latest = Arc::new(Mutex::new(None));

        match spawned {
            Ok(mut child) => {
                let stdout = child.stdout.take().expect("piped stdout, always present");
                let stdin = child.stdin.take().expect("piped stdin, always present");
                spawn_reader(stdout, Arc::clone(&latest), ctx.clone());
                Self {
                    child: Some(child),
                    stdin: Some(stdin),
                    latest,
                    texture: None,
                    spawn_error: None,
                    orbit: Orbit::default(),
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

        let Some(texture) = &self.texture else {
            if let Some(err) = &self.spawn_error {
                ui.colored_label(egui::Color32::RED, err);
            } else {
                ui.label("Waiting for the first frame from MuJoCo…");
            }
            return;
        };

        // `max_size` only ever caps — it never grows an image up to fill
        // the panel. `fit_to_exact_size` is the one that scales to fill;
        // `maintain_aspect_ratio` keeps MuJoCo's frame from stretching to
        // match a panel of a different aspect ratio (it will letterbox
        // instead, which is correct when the two aspect ratios differ).
        let response = ui.add(
            egui::Image::new(texture)
                .fit_to_exact_size(ui.available_size())
                .maintain_aspect_ratio(true)
                .sense(egui::Sense::click_and_drag()),
        );

        let before = self.orbit;
        if response.dragged() {
            let delta = response.drag_delta();
            self.orbit.azimuth_deg -= delta.x * DRAG_DEGREES_PER_POINT;
            self.orbit.elevation_deg = (self.orbit.elevation_deg - delta.y * DRAG_DEGREES_PER_POINT)
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

        if self.orbit != before {
            self.send_orbit();
        }
    }

    fn send_orbit(&mut self) {
        let Some(stdin) = &mut self.stdin else {
            return;
        };
        let mut bytes = [0u8; 12];
        bytes[0..4].copy_from_slice(&self.orbit.azimuth_deg.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.orbit.elevation_deg.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.orbit.distance_m.to_le_bytes());
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

fn spawn_reader(mut stdout: ChildStdout, latest: Arc<Mutex<Option<RawFrame>>>, ctx: egui::Context) {
    thread::spawn(move || {
        let mut header = [0u8; 8];
        loop {
            if stdout.read_exact(&mut header).is_err() {
                break; // subprocess exited or pipe closed — stop quietly
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
    });
}

/// `crates/studio-shell` is always two directories under the repo root —
/// true regardless of the shell's own current working directory, unlike
/// relying on `std::env::current_dir()`.
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("crates/studio-shell is two directories under the repo root")
        .to_path_buf()
}
