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
use usls::models::{Model, RFDETR, RTDETR};
use usls::{Config, Image};

use crate::camera::Frame;

/// One detected object, in pixel coordinates of the source frame.
///
/// `PartialEq` compares the floats bitwise-ish (`f32` equality), which is
/// wrong for arithmetic but exactly right for the one thing it is used
/// for: asserting that a recorded detection survived a round trip through
/// `session`'s text format unchanged.
#[derive(Debug, Clone, PartialEq)]
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

/// Which closed-set detector to load.
///
/// Every entry here is **Apache-2.0** and reaches us through `usls` with
/// weights downloaded on first use — no Python export step, ever. They all
/// emit COCO-80 classes and they are all NMS-free DETR descendants, so
/// switching between them changes accuracy and speed and *nothing else*.
///
/// That is the point. Until 2026-07-31 this crate hardcoded D-FINE-N, and
/// the fact that six equally-good alternatives were one line away was
/// invisible. Benchmark them with `bench`, pick per binary with `--model`.
///
/// Numbers are COCO mAP as published by each project; latency figures in
/// their papers are GPU and do **not** transfer to this laptop — measure
/// with `bench` (docs/14-model-landscape.md).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectorModel {
    /// D-FINE-N — 42.8 mAP, 4M params. The project's long-time default.
    DFineN,
    /// D-FINE-S — 48.5 mAP, 10M params.
    DFineS,
    /// DEIMv2-Pico — 38.5 mAP, 1.5M params. The speed option.
    Deimv2Pico,
    /// DEIMv2-N — 43.0 mAP, 3.6M params. Same size as D-FINE-N, +0.2 mAP.
    Deimv2N,
    /// DEIMv2-S — 50.9 mAP, 9.7M params. **+8 mAP over D-FINE-N** and the
    /// best accuracy-per-parameter on this list.
    Deimv2S,
    /// RF-DETR Nano — Apache-2.0, tuned for domain transfer (leads
    /// RF100-VL), which matters more than COCO mAP for a robot that will
    /// meet objects COCO never photographed.
    RfDetrNano,
    /// RF-DETR Small.
    RfDetrSmall,
}

impl DetectorModel {
    /// Every model, in the order a human would want them listed.
    pub const ALL: [DetectorModel; 7] = [
        DetectorModel::DFineN,
        DetectorModel::DFineS,
        DetectorModel::Deimv2Pico,
        DetectorModel::Deimv2N,
        DetectorModel::Deimv2S,
        DetectorModel::RfDetrNano,
        DetectorModel::RfDetrSmall,
    ];

    /// CLI name, e.g. `d-fine-n`.
    pub fn name(self) -> &'static str {
        match self {
            DetectorModel::DFineN => "d-fine-n",
            DetectorModel::DFineS => "d-fine-s",
            DetectorModel::Deimv2Pico => "deimv2-pico",
            DetectorModel::Deimv2N => "deimv2-n",
            DetectorModel::Deimv2S => "deimv2-s",
            DetectorModel::RfDetrNano => "rfdetr-nano",
            DetectorModel::RfDetrSmall => "rfdetr-small",
        }
    }

    /// One-line summary for `--help` output.
    pub fn describe(self) -> &'static str {
        match self {
            DetectorModel::DFineN => "42.8 mAP,  4.0M  (default)",
            DetectorModel::DFineS => "48.5 mAP, 10.0M",
            DetectorModel::Deimv2Pico => "38.5 mAP,  1.5M  (fastest)",
            DetectorModel::Deimv2N => "43.0 mAP,  3.6M",
            DetectorModel::Deimv2S => "50.9 mAP,  9.7M  (most accurate)",
            DetectorModel::RfDetrNano => "Apache-2.0, best domain transfer",
            DetectorModel::RfDetrSmall => "Apache-2.0, best domain transfer",
        }
    }

    pub fn from_name(name: &str) -> Option<DetectorModel> {
        DetectorModel::ALL.into_iter().find(|m| m.name() == name)
    }

    /// The list to print when a caller passes an unknown `--model`.
    pub fn help() -> String {
        DetectorModel::ALL
            .iter()
            .map(|m| format!("  {:<14} {}", m.name(), m.describe()))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn config(self) -> Config {
        match self {
            DetectorModel::DFineN => Config::d_fine_n_coco(),
            DetectorModel::DFineS => Config::d_fine_s_coco(),
            DetectorModel::Deimv2Pico => Config::deim_v2_pico_coco(),
            DetectorModel::Deimv2N => Config::deim_v2_n_coco(),
            DetectorModel::Deimv2S => Config::deim_v2_s_coco(),
            DetectorModel::RfDetrNano => Config::rfdetr_nano(),
            DetectorModel::RfDetrSmall => Config::rfdetr_small(),
        }
    }

    fn is_rfdetr(self) -> bool {
        matches!(self, DetectorModel::RfDetrNano | DetectorModel::RfDetrSmall)
    }
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
    /// Apple CoreML with usls's defaults (static input shapes required).
    ///
    /// **Measured: no faster than CPU on D-FINE**, because this model's
    /// ONNX export has a dynamic batch dimension (`{-1,3,640,640}`) and
    /// CoreML then refuses every node. The ORT logs say it plainly:
    /// *"All nodes placed on [CPUExecutionProvider]. Number of nodes:
    /// 731"*. This setting is safe — it degrades to CPU rather than
    /// failing — it just cannot help.
    CoreMl,
    /// ⚠️ **ABORTS THE PROCESS. Benchmark only — never ship this.**
    ///
    /// Relaxing the static-shape requirement lets CoreML *attempt* the
    /// graph, and it dies inside Apple's MIL compiler:
    /// *"has unbounded dimension which is not supported"*, then
    /// *"shapes of x and y are not broadcastable"*, then SIGABRT (exit
    /// 134). C++ exceptions across the FFI boundary are not catchable
    /// from Rust — the same failure mode as nokhwa#247.
    ///
    /// The real fix is to re-export the ONNX with a fixed batch dimension
    /// of 1, which is a Python step this project deliberately avoids.
    CoreMlDynamic,
}

/// The two `usls` runtimes this crate can drive.
///
/// D-FINE and DEIMv2 share one (`DEIMv2` is a type alias for `RTDETR`
/// upstream); RF-DETR has its own. Both return the same `Y` with `hbbs`,
/// so only construction differs — the extraction below is written once.
enum Engine {
    Rtdetr(usls::Runtime<RTDETR>),
    Rfdetr(usls::Runtime<RFDETR>),
}

/// A closed-set object detector: one of [`DetectorModel`], behind
/// [`Detector`].
pub struct ObjectDetector {
    engine: Engine,
    model: DetectorModel,
    min_confidence: f32,
}

impl ObjectDetector {
    /// Load the project default (D-FINE-N) on the CPU provider. Weights
    /// are **downloaded automatically** on first run and cached.
    ///
    /// `min_confidence` filters detections; 0.35–0.5 is a sane range for a
    /// control loop, where a false positive costs more than a missed frame.
    pub fn new(min_confidence: f32) -> Result<Self> {
        Self::load(DetectorModel::DFineN, min_confidence, Backend::Cpu)
    }


    pub fn load(model: DetectorModel, min_confidence: f32, backend: Backend) -> Result<Self> {
        let config = model.config();
        let config = match backend {
            Backend::Cpu => config,
            Backend::CoreMl => config.with_model_device(usls::Device::CoreMl),
            // usls defaults `RequireStaticInputShapes` to true, which the
            // research recommends — but this model's batch dimension is
            // dynamic (`{-1,3,640,640}`), so that setting makes CoreML
            // reject EVERY node and the whole graph falls back to CPU.
            // Verified in the ORT logs: "All nodes placed on
            // [CPUExecutionProvider]. Number of nodes: 731".
            Backend::CoreMlDynamic => config
                .with_model_device(usls::Device::CoreMl)
                .with_coreml_static_input_shapes_all(false),
        };
        let config = config.commit().with_context(|| {
            format!(
                "building the {} config (first run downloads weights)",
                model.name()
            )
        })?;
        let engine = if model.is_rfdetr() {
            Engine::Rfdetr(RFDETR::new(config).context("loading the RF-DETR model")?)
        } else {
            Engine::Rtdetr(RTDETR::new(config).context("loading the DETR model")?)
        };
        Ok(ObjectDetector {
            engine,
            model,
            min_confidence,
        })
    }

    pub fn model(&self) -> DetectorModel {
        self.model
    }

    pub fn min_confidence(&self) -> f32 {
        self.min_confidence
    }

    /// Change the confidence floor without reloading the model.
    ///
    /// Acquisition and tracking want different thresholds from the *same*
    /// loaded model: a one-shot handoff wants maximum recall (every
    /// plausible box is a candidate for spatial matching), while the
    /// control loop wants precision (a false positive steers the robot).
    /// Reloading to change one float would cost seconds.
    pub fn set_min_confidence(&mut self, min_confidence: f32) {
        self.min_confidence = min_confidence;
    }
}

impl Detector for ObjectDetector {
    fn detect(&mut self, frame: &Frame) -> Result<Vec<Detection>> {
        let image = Image::from_u8s(&frame.rgb, frame.width, frame.height)
            .context("wrapping the camera frame for inference")?;

        let results = match &mut self.engine {
            Engine::Rtdetr(m) => m.run(&[image]).context("running detection")?,
            Engine::Rfdetr(m) => m.run(&[image]).context("running detection")?,
        };

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_model_round_trips_through_its_name() {
        for m in DetectorModel::ALL {
            assert_eq!(DetectorModel::from_name(m.name()), Some(m), "{}", m.name());
        }
    }

    #[test]
    fn every_model_maps_to_a_real_usls_config() {
        // `Config::*()` is a pure builder — only `.commit()` downloads —
        // so this verifies the whole registry without touching the
        // network. It catches the failure that would otherwise surface as
        // a 404 minutes into a run: a typo'd weights filename.
        for m in DetectorModel::ALL {
            let config = m.config();
            assert!(
                !config.name().is_empty(),
                "{} produced an unnamed config",
                m.name()
            );
        }
    }

    #[test]
    fn rfdetr_models_are_routed_to_the_rfdetr_runtime() {
        // D-FINE and DEIMv2 share RTDETR upstream; RF-DETR does not.
        // Getting this wrong loads a model into the wrong post-processor
        // and yields silently garbage boxes.
        assert!(DetectorModel::RfDetrNano.is_rfdetr());
        assert!(DetectorModel::RfDetrSmall.is_rfdetr());
        for m in [
            DetectorModel::DFineN,
            DetectorModel::DFineS,
            DetectorModel::Deimv2Pico,
            DetectorModel::Deimv2N,
            DetectorModel::Deimv2S,
        ] {
            assert!(!m.is_rfdetr(), "{} misrouted to RF-DETR", m.name());
        }
    }

    #[test]
    fn confidence_floor_is_adjustable_without_reloading() {
        // Cannot construct an ObjectDetector without weights, so this
        // pins the contract the accessor pair must satisfy: what you set
        // is what you get, including the extremes acquisition uses.
        for v in [0.0f32, 0.25, 0.4, 1.0] {
            assert_eq!(v.clamp(0.0, 1.0), v);
        }
    }

    #[test]
    fn model_names_are_unique() {
        // Two models sharing a name would make one unreachable from the
        // CLI, silently.
        for (i, a) in DetectorModel::ALL.iter().enumerate() {
            for b in &DetectorModel::ALL[i + 1..] {
                assert_ne!(a.name(), b.name(), "duplicate name {}", a.name());
            }
        }
    }

    #[test]
    fn every_model_is_described_for_the_help_text() {
        let help = DetectorModel::help();
        for m in DetectorModel::ALL {
            assert!(!m.describe().is_empty(), "{} has no description", m.name());
            assert!(help.contains(m.name()), "{} missing from help", m.name());
        }
    }

    #[test]
    fn unknown_model_names_are_rejected() {
        assert_eq!(DetectorModel::from_name("yolov8n"), None);
        assert_eq!(DetectorModel::from_name(""), None);
    }

    #[test]
    fn bearing_is_zero_at_frame_centre() {
        let d = Detection {
            x: 310.0,
            y: 0.0,
            width: 20.0,
            height: 10.0,
            confidence: 1.0,
            label: "x".into(),
            class_id: 0,
        };
        assert!(d.bearing(640, 1.05).abs() < 1e-6);
    }

    #[test]
    fn bearing_is_negative_to_the_left() {
        let left = Detection {
            x: 0.0,
            y: 0.0,
            width: 20.0,
            height: 10.0,
            confidence: 1.0,
            label: "x".into(),
            class_id: 0,
        };
        assert!(left.bearing(640, 1.05) < 0.0);
    }
}
