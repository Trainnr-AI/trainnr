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
//! Orbit and zoom work the same way in reverse, as DELTAS: a drag/scroll
//! becomes camera-angle/distance increments on the subprocess's stdin
//! (3 `f32` deltas, then the panel's absolute pixel size as 2 `u32`s).
//! Deltas, deliberately: the Python side owns every absolute camera fact
//! — per-rig defaults, clamps — because an absolute-valued wire needed
//! those facts duplicated here, and the copies drifted the day per-rig
//! defaults landed (first drag in the kitting scene snapped the camera
//! from its 2.2 m frame to this side's stale 1.0 m). No pan yet.

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

/// Owns the render-stream subprocess, the latest decoded frame, and the
/// orbit state the user is driving by dragging/scrolling over the image.
pub struct ViewportFeed {
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    latest: Arc<Mutex<Option<RawFrame>>>,
    texture: Option<TextureHandle>,
    spawn_error: Option<String>,
    /// The render size last sent (absolute — the Python `Renderer` is
    /// rebuilt to match it).
    sent_size: (u32, u32),
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
                    sent_size: INITIAL_RENDER_SIZE,
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
                sent_size: INITIAL_RENDER_SIZE,
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

        // Camera input as deltas — the Python side integrates and clamps.
        let mut d_azimuth = 0.0f32;
        let mut d_elevation = 0.0f32;
        let mut d_distance = 0.0f32;
        if response.dragged() {
            let delta = response.drag_delta();
            d_azimuth = -delta.x * DRAG_DEGREES_PER_POINT;
            d_elevation = -delta.y * DRAG_DEGREES_PER_POINT;
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
        let want = (
            ((available.x * pixels_per_point).round() as u32)
                .clamp(MIN_RENDER_SIDE, MAX_RENDER_SIDE),
            ((available.y * pixels_per_point).round() as u32)
                .clamp(MIN_RENDER_SIDE, MAX_RENDER_SIDE),
        );
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
        let Some(stdin) = &mut self.stdin else {
            return;
        };
        // A closed pipe here means the render subprocess died; `show`'s
        // next call will already be reporting `spawn_error`-shaped state
        // via an absent texture, so silently dropping this write is fine.
        let _ = stdin.write_all(&encode_camera_update(
            d_azimuth,
            d_elevation,
            d_distance,
            self.sent_size,
        ));
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

/// One camera+size update in the script's stdin wire format — Python's
/// `struct.unpack("<fffII", …)` exactly: three little-endian f32 deltas,
/// two little-endian u32 absolute pixels. Tested below against bytes
/// Python's own struct module would produce.
fn encode_camera_update(
    d_azimuth: f32,
    d_elevation: f32,
    d_distance: f32,
    size: (u32, u32),
) -> [u8; 20] {
    let mut bytes = [0u8; 20];
    bytes[0..4].copy_from_slice(&d_azimuth.to_le_bytes());
    bytes[4..8].copy_from_slice(&d_elevation.to_le_bytes());
    bytes[8..12].copy_from_slice(&d_distance.to_le_bytes());
    bytes[12..16].copy_from_slice(&size.0.to_le_bytes());
    bytes[16..20].copy_from_slice(&size.1.to_le_bytes());
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_wire_matches_pythons_struct_format() {
        // struct.pack("<fffII", 1.0, -2.5, 0.25, 1024, 576) — computed
        // with CPython and pinned here, so both ends of the wire are
        // held to the same bytes.
        let expected: [u8; 20] = [
            0x00, 0x00, 0x80, 0x3f, // 1.0f32 LE
            0x00, 0x00, 0x20, 0xc0, // -2.5f32 LE
            0x00, 0x00, 0x80, 0x3e, // 0.25f32 LE
            0x00, 0x04, 0x00, 0x00, // 1024u32 LE
            0x40, 0x02, 0x00, 0x00, // 576u32 LE
        ];
        assert_eq!(encode_camera_update(1.0, -2.5, 0.25, (1024, 576)), expected);
    }

    #[test]
    fn a_no_op_update_still_carries_the_size() {
        let bytes = encode_camera_update(0.0, 0.0, 0.0, (1920, 128));
        assert_eq!(&bytes[0..12], &[0u8; 12]);
        assert_eq!(u32::from_le_bytes(bytes[12..16].try_into().unwrap()), 1920);
        assert_eq!(u32::from_le_bytes(bytes[16..20].try_into().unwrap()), 128);
    }
}
