//! Object detection, behind a trait.
//!
//! Same discipline as [`crate::camera`]: downstream code depends on
//! `Detector`, never on `usls`. That matters more here than it looks,
//! because the model underneath is *expected* to change —
//! D-FINE (fixed 80 classes) today, an open-vocabulary detector when the
//! project wants "find the red cube", and eventually nothing at all when a
//! VLA subsumes perception. See docs/12-model-choice.md.
//!
//! # Why D-FINE and not YOLO
//!
//! Ultralytics models are **AGPL-3.0**, which their licence asserts covers
//! "code, models, and architectures" even for internal use, and names
//! robotics and edge devices as commercial triggers. This repo is meant to
//! be publishable, so copyleft is out of the default build.
//!
//! It costs nothing: **D-FINE-N is 42.8 mAP at 4M params vs YOLO26n's
//! 40.9 at 2.4M**, and D-FINE-S is 48.5 mAP. Apache-2.0, ICLR 2025
//! Spotlight, and NMS-free like every DETR — so there is no IoU loop to
//! write, no confidence/IoU threshold pair to tune, and no in-graph NMS to
//! wreck CoreML partitioning.

use anyhow::{Context, Result};
use usls::models::{Model, RTDETR};
use usls::{Config, Image};

use crate::camera::Frame;

/// One detected object, in pixel coordinates of the source frame.
#[derive(Debug, Clone)]
pub struct Detection {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub confidence: f32,
    pub label: String,
    pub class_id: u16,
}

impl Detection {
    /// Horizontal centre of the box, in pixels.
    pub fn center_x(&self) -> f32 {
        self.x + self.width / 2.0
    }

    /// Bearing of this object relative to where the camera points, in
    /// radians: negative = left of centre, positive = right.
    ///
    /// This is the bridge from perception to control — feed it to
    /// `shortest_turn` + `Pid` and the robot turns toward the object.
    /// See docs/learning/math-05-bearing-atan2.md.
    pub fn bearing(&self, frame_width: u32, horizontal_fov: f32) -> f32 {
        (self.center_x() / frame_width as f32 - 0.5) * horizontal_fov
    }
}

/// Anything that turns a frame into detections.
pub trait Detector {
    fn detect(&mut self, frame: &Frame) -> Result<Vec<Detection>>;
}

/// D-FINE via `usls` (ONNX Runtime underneath).
pub struct DFineDetector {
    model: usls::Runtime<RTDETR>,
    min_confidence: f32,
}

/// Where inference runs.
///
/// On Apple Silicon this is not an obvious win either way — the research
/// found a documented case where `CPUOnly` beat `CPUAndGPU` on a
/// conv-heavy model, because CoreML's benefit comes from graph
/// *partitioning*, and a model whose ops don't all map cleanly gets split
/// and shuttled between processors. **Measure, don't assume**:
/// `cargo run --release -p vision --bin bench`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Backend {
    /// ONNX Runtime CPU provider. Always available, predictable.
    Cpu,
    /// Apple CoreML. usls defaults to MLProgram format + static input
    /// shapes, which is exactly what the research says to use.
    CoreMl,
}

impl DFineDetector {
    /// Load D-FINE-N (COCO-80) on the CPU provider. Weights are
    /// **downloaded automatically** on first run and cached — no Python
    /// export step, ever.
    ///
    /// `min_confidence` filters detections; 0.35–0.5 is a sane range for a
    /// control loop, where a false positive costs more than a missed frame.
    pub fn new(min_confidence: f32) -> Result<Self> {
        Self::with_backend(min_confidence, Backend::Cpu)
    }

    pub fn with_backend(min_confidence: f32, backend: Backend) -> Result<Self> {
        let config = Config::d_fine_n_coco();
        let config = match backend {
            Backend::Cpu => config,
            Backend::CoreMl => config.with_model_device(usls::Device::CoreMl),
        };
        let config = config
            .commit()
            .context("building the D-FINE config (first run downloads weights)")?;
        let model = RTDETR::new(config).context("loading the D-FINE model")?;
        Ok(DFineDetector {
            model,
            min_confidence,
        })
    }
}

impl Detector for DFineDetector {
    fn detect(&mut self, frame: &Frame) -> Result<Vec<Detection>> {
        let image = Image::from_u8s(&frame.rgb, frame.width, frame.height)
            .context("wrapping the camera frame for inference")?;

        let results = self.model.run(&[image]).context("running detection")?;

        let mut detections = Vec::new();
        if let Some(y) = results.first() {
            for hbb in &y.hbbs {
                let confidence = hbb.confidence().unwrap_or(0.0);
                if confidence < self.min_confidence {
                    continue;
                }
                let (x, y_min, w, h) = hbb.xywh();
                detections.push(Detection {
                    x,
                    y: y_min,
                    width: w,
                    height: h,
                    confidence,
                    label: hbb.name().unwrap_or("object").to_string(),
                    class_id: hbb.id().unwrap_or(0) as u16,
                });
            }
        }
        Ok(detections)
    }
}
