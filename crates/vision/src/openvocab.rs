//! Naming what to look for, instead of accepting COCO's 80 classes.
//!
//! This is where the project's language ambition first reaches the code.
//! Until now the detector could find a "cup" because someone decided
//! decades ago that cups were class 41. Here you type the phrase.
//!
//! # Why the vocabulary change is a separate, expensive method
//!
//! [`Promptable::set_vocabulary`] is deliberately *not* a parameter to
//! `detect()`, because the cost is not per-frame — it is per-vocabulary.
//! Open-vocabulary detectors are **"prompt-then-detect"**: the text is
//! encoded once and folded into the model, after which inference costs the
//! same as a closed-set detector. Making that a separate call encodes the
//! architecture in the type system, so no caller can assume prompts are
//! free every frame.
//!
//! # Licence: Apache-2.0 only
//!
//! The fast open-vocab options are copyleft — YOLOE is AGPL-3.0 (an
//! Ultralytics fork), YOLO-World is GPL-3.0. Per the rule adopted in
//! docs/12-model-choice.md, neither belongs in the default build. Grounding
//! DINO is **Apache-2.0**, and pays for that with speed.
//!
//! # The honest limitation: attributes
//!
//! "red cube" is an *attribute* query, and attributes are precisely where
//! CLIP-style text embeddings are weakest. Expect these models to find *a
//! cube* reliably and to confuse *which colour* far more often than you
//! would like. [`dominant_hue`] exists for that: detect the object with the
//! model, then decide its colour with ten lines of arithmetic that are
//! deterministic, free, and correct.

use anyhow::{Context, Result};
use usls::models::{GroundingDINO, Model};
use usls::{Config, Image};

use crate::camera::Frame;
use crate::detect::{Detection, Detector};

/// A detector whose target set can be changed at runtime — at a cost.
pub trait Promptable {
    /// Re-target the detector. **Expensive**: re-encodes the text and
    /// rebuilds the model. Call it when the goal changes, never per frame.
    fn set_vocabulary(&mut self, phrases: &[&str]) -> Result<()>;

    /// What it is currently looking for.
    fn vocabulary(&self) -> &[String];
}

/// Grounding DINO (tiny) via usls — Apache-2.0, text-prompted detection.
pub struct OpenVocabDetector {
    model: usls::Runtime<GroundingDINO>,
    vocabulary: Vec<String>,
    min_confidence: f32,
}

impl OpenVocabDetector {
    /// Build a detector looking for `phrases`.
    ///
    /// First call downloads weights (Grounding DINO tiny is considerably
    /// larger than D-FINE-N — expect a wait, then caching).
    pub fn new(phrases: &[&str], min_confidence: f32) -> Result<Self> {
        let vocabulary: Vec<String> = phrases.iter().map(|s| s.to_string()).collect();
        let model = Self::build(&vocabulary)?;
        Ok(OpenVocabDetector {
            model,
            vocabulary,
            min_confidence,
        })
    }

    fn build(vocabulary: &[String]) -> Result<usls::Runtime<GroundingDINO>> {
        anyhow::ensure!(!vocabulary.is_empty(), "vocabulary cannot be empty");
        let config = Config::grounding_dino_tiny()
            .with_text_names_owned(vocabulary.to_vec())
            .commit()
            .context("building the Grounding DINO config")?;
        GroundingDINO::new(config).context("loading Grounding DINO")
    }
}

impl Promptable for OpenVocabDetector {
    fn set_vocabulary(&mut self, phrases: &[&str]) -> Result<()> {
        let vocabulary: Vec<String> = phrases.iter().map(|s| s.to_string()).collect();
        // Rebuild. The weights are cached, so this is re-encoding the text
        // and re-committing the session, not a download.
        self.model = Self::build(&vocabulary)?;
        self.vocabulary = vocabulary;
        Ok(())
    }

    fn vocabulary(&self) -> &[String] {
        &self.vocabulary
    }
}

impl Detector for OpenVocabDetector {
    fn detect(&mut self, frame: &Frame) -> Result<Vec<Detection>> {
        let image = Image::from_u8s(&frame.rgb, frame.width, frame.height)
            .context("wrapping frame for open-vocab inference")?;
        let results = self.model.run(&[image]).context("running Grounding DINO")?;

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

/// Mean hue (degrees, 0–360) and saturation (0–1) inside a box.
///
/// The pragmatic answer to "red cube": let the model find *a cube*, then
/// decide the colour here. Deterministic, ~0 ms, and correct for saturated
/// primaries — where an open-vocabulary detector is guessing.
///
/// Low-saturation pixels are skipped: hue is meaningless for grey, and
/// averaging it in would drag every reading toward whatever noise the
/// background contributes.
pub fn dominant_hue(frame: &Frame, det: &Detection) -> Option<(f32, f32)> {
    let x0 = det.x.max(0.0) as u32;
    let y0 = det.y.max(0.0) as u32;
    let x1 = ((det.x + det.width) as u32).min(frame.width);
    let y1 = ((det.y + det.height) as u32).min(frame.height);
    if x1 <= x0 || y1 <= y0 {
        return None;
    }

    // Hue is circular, so it cannot be averaged directly — 359° and 1° are
    // 2° apart, but their arithmetic mean is 180°, the opposite colour.
    // Accumulate unit vectors instead and take the angle of the sum.
    let (mut sin_sum, mut cos_sum, mut sat_sum) = (0.0f32, 0.0f32, 0.0f32);
    let mut counted = 0u32;

    for y in y0..y1 {
        for x in x0..x1 {
            let i = ((y * frame.width + x) * 3) as usize;
            let (r, g, b) = (
                frame.rgb[i] as f32 / 255.0,
                frame.rgb[i + 1] as f32 / 255.0,
                frame.rgb[i + 2] as f32 / 255.0,
            );
            let max = r.max(g).max(b);
            let min = r.min(g).min(b);
            let delta = max - min;
            let sat = if max <= 0.0 { 0.0 } else { delta / max };
            if sat < 0.25 || delta <= 0.0 {
                continue; // grey: hue carries no information
            }
            let hue = 60.0
                * if max == r {
                    ((g - b) / delta).rem_euclid(6.0)
                } else if max == g {
                    (b - r) / delta + 2.0
                } else {
                    (r - g) / delta + 4.0
                };
            let rad = hue.to_radians();
            sin_sum += rad.sin();
            cos_sum += rad.cos();
            sat_sum += sat;
            counted += 1;
        }
    }

    if counted == 0 {
        return None;
    }
    let hue = sin_sum.atan2(cos_sum).to_degrees().rem_euclid(360.0);
    Some((hue, sat_sum / counted as f32))
}

/// Coarse colour name from a hue, for printing.
pub fn hue_name(hue: f32) -> &'static str {
    match hue {
        h if !(15.0..345.0).contains(&h) => "red",
        h if h < 45.0 => "orange",
        h if h < 70.0 => "yellow",
        h if h < 165.0 => "green",
        h if h < 200.0 => "cyan",
        h if h < 255.0 => "blue",
        h if h < 290.0 => "purple",
        _ => "magenta",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32, rgb: [u8; 3]) -> Frame {
        Frame {
            width: w,
            height: h,
            rgb: rgb
                .iter()
                .cycle()
                .take((w * h * 3) as usize)
                .copied()
                .collect(),
        }
    }

    fn whole(f: &Frame) -> Detection {
        Detection {
            x: 0.0,
            y: 0.0,
            width: f.width as f32,
            height: f.height as f32,
            confidence: 1.0,
            label: "test".into(),
            class_id: 0,
        }
    }

    #[test]
    fn primaries_are_identified() {
        for (rgb, expected) in [
            ([255u8, 0, 0], "red"),
            ([0, 255, 0], "green"),
            ([0, 0, 255], "blue"),
            ([255, 255, 0], "yellow"),
        ] {
            let f = solid(8, 8, rgb);
            let (hue, sat) = dominant_hue(&f, &whole(&f)).expect("saturated colour");
            assert!(sat > 0.9, "{expected}: saturation {sat}");
            assert_eq!(hue_name(hue), expected, "hue was {hue}");
        }
    }

    #[test]
    fn grey_has_no_hue() {
        let f = solid(8, 8, [128, 128, 128]);
        assert_eq!(dominant_hue(&f, &whole(&f)), None);
    }

    #[test]
    fn hue_averaging_wraps_correctly() {
        // Half the pixels at hue ~5 deg, half at ~355 deg. The circular
        // mean is 0 (red). A naive arithmetic mean would give 180 (cyan) —
        // the exact opposite colour.
        let mut f = solid(4, 2, [255, 0, 0]);
        for x in 0..4u32 {
            let i = ((1 * f.width + x) * 3) as usize;
            f.rgb[i] = 255;
            f.rgb[i + 1] = 0;
            f.rgb[i + 2] = 20; // slightly magenta -> hue near 355
        }
        let (hue, _) = dominant_hue(&f, &whole(&f)).unwrap();
        assert_eq!(hue_name(hue), "red", "circular mean failed, hue = {hue}");
    }

    #[test]
    fn box_outside_the_frame_is_rejected() {
        let f = solid(4, 4, [255, 0, 0]);
        let d = Detection {
            x: 100.0,
            y: 100.0,
            width: 10.0,
            height: 10.0,
            confidence: 1.0,
            label: "x".into(),
            class_id: 0,
        };
        assert_eq!(dominant_hue(&f, &d), None);
    }
}
