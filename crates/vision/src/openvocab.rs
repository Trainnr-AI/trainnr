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

/// EXERCISE 7 — centre-weighted hue. **Prakhar implements this.**
///
/// Fixes a limitation found on camera on 2026-07-31: [`dominant_hue`]
/// averages over the *whole* box, so a translucent grey bottle on a wooden
/// desk reported as **"orange"** — it measured the desk showing through and
/// around the bottle, not the bottle. Detector boxes always contain some
/// background; near the edges they contain mostly background.
///
/// The fix: weight each pixel by how close it is to the box centre, so
/// centre pixels (almost certainly the object) count for much more than
/// corner pixels (often background).
///
/// # The recipe, line by line
///
/// 1. **Clip the box to the frame**, exactly as [`dominant_hue`] does —
///    copy those first six lines, including the `if x1 <= x0 || y1 <= y0`
///    early return. Detector boxes routinely hang off the frame edge.
///
/// 2. **Find the box centre and half-size**, in `f32`:
///    ```text
///    cx = (x0 + x1) as f32 / 2.0        hw = (x1 - x0) as f32 / 2.0
///    cy = (y0 + y1) as f32 / 2.0        hh = (y1 - y0) as f32 / 2.0
///    ```
///    `hw`/`hh` are half the box width/height. Dividing by them below is
///    what makes the weight independent of box size — a normalised
///    distance, so a tall thin box behaves like a square one.
///
/// 3. **Per pixel, compute the normalised distance from the centre.** Use
///    the pixel's *centre* (`x as f32 + 0.5`), not its corner:
///    ```text
///    dx = (x as f32 + 0.5 - cx) / hw
///    dy = (y as f32 + 0.5 - cy) / hh
///    r  = dx.hypot(dy)
///    ```
///    `r` is 0 at the box centre and 1 at the middle of each edge.
///    `hypot(a, b)` is `sqrt(a² + b²)` — see docs/learning/math-01.
///
/// 4. **Turn distance into a weight** that falls to zero at the edge:
///    ```text
///    w = (1.0 - r).max(0.0)
///    w = w * w                    // square it: sharper falloff
///    ```
///    `.max(0.0)` matters — past `r = 1` (the box corners) `1 - r` goes
///    *negative*, and a negative weight would actively pull the average
///    toward the corner colour. The same `.max(0.0)` floor you wrote in
///    the alignment throttle, for the same reason.
///
/// 5. **Skip unsaturated pixels** (`sat < 0.25`), as [`dominant_hue`] does.
///    Grey has no meaningful hue.
///
/// 6. **Accumulate, multiplied by `w`:**
///    ```text
///    sin_sum += w * rad.sin();     cos_sum += w * rad.cos();
///    sat_sum += w * sat;           weight_sum += w;
///    ```
///    Still a *circular* mean — you cannot average hue directly, because
///    359° and 1° are 2° apart but their arithmetic mean is 180°, the
///    opposite colour. Same trick as [`dominant_hue`].
///
/// 7. **Finish:** if `weight_sum <= 0.0`, return `None` (every pixel was
///    grey, or weighted to nothing). Otherwise:
///    ```text
///    hue = sin_sum.atan2(cos_sum).to_degrees().rem_euclid(360.0)
///    Some((hue, sat_sum / weight_sum))
///    ```
///    Note the saturation is divided by `weight_sum`, **not** by a pixel
///    count — it is a weighted average too.
///
/// Run the tests with:
/// ```sh
/// cargo test -p vision weighted -- --ignored
/// ```
pub fn dominant_hue_weighted(frame: &Frame, det: &Detection) -> Option<(f32, f32)> {
    let x0 = det.x.max(0.0) as u32;
    let y0 = det.y.max(0.0) as u32;
    let x1 = ((det.x + det.width) as u32).min(frame.width);
    let y1 = ((det.y + det.height) as u32).min(frame.height);
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    let cx = (x0 + x1) as f32 / 2.0;
    let cy = (y0 + y1) as f32 / 2.0;
    let hw = (x1 - x0) as f32 / 2.0;
    let hh = (y1 - y0) as f32 / 2.0;
    todo!("exercise 7 — see the recipe above")
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
            let i = ((f.width + x) * 3) as usize; // row 1
            f.rgb[i] = 255;
            f.rgb[i + 1] = 0;
            f.rgb[i + 2] = 20; // slightly magenta -> hue near 355
        }
        let (hue, _) = dominant_hue(&f, &whole(&f)).unwrap();
        assert_eq!(hue_name(hue), "red", "circular mean failed, hue = {hue}");
    }

    #[test]
    fn every_hue_bucket_is_reachable_and_ordered() {
        // Walks the wheel and checks each named band, including the
        // wrap-around at red which is the one that is easy to get wrong.
        for (hue, expected) in [
            (0.0, "red"),
            (10.0, "red"),
            (350.0, "red"),
            (359.9, "red"),
            (30.0, "orange"),
            (60.0, "yellow"),
            (120.0, "green"),
            (180.0, "cyan"),
            (225.0, "blue"),
            (270.0, "purple"),
            (320.0, "magenta"),
        ] {
            assert_eq!(hue_name(hue), expected, "hue {hue}");
        }
    }

    #[test]
    fn hue_bucket_boundaries_are_exact() {
        // Each boundary belongs to the band ABOVE it. Pinned because an
        // off-by-one here renames colours near the edges, which reads as
        // a model problem rather than an arithmetic one.
        assert_eq!(hue_name(15.0), "orange");
        assert_eq!(hue_name(14.9), "red");
        assert_eq!(hue_name(45.0), "yellow");
        assert_eq!(hue_name(70.0), "green");
        assert_eq!(hue_name(165.0), "cyan");
        assert_eq!(hue_name(200.0), "blue");
        assert_eq!(hue_name(255.0), "purple");
        assert_eq!(hue_name(290.0), "magenta");
        assert_eq!(hue_name(345.0), "red");
    }

    #[test]
    fn a_partially_overlapping_box_is_clipped_not_rejected() {
        // Detector boxes routinely hang off the frame edge. Clipping keeps
        // the colour reading; rejecting would lose it entirely.
        let f = solid(8, 8, [255, 0, 0]);
        let d = Detection {
            x: 6.0,
            y: 6.0,
            width: 10.0,
            height: 10.0,
            confidence: 1.0,
            label: "x".into(),
            class_id: 0,
        };
        let (hue, _) = dominant_hue(&f, &d).expect("clipped box still has pixels");
        assert_eq!(hue_name(hue), "red");
    }

    #[test]
    fn a_negative_origin_box_is_clamped_to_the_frame() {
        let f = solid(8, 8, [0, 0, 255]);
        let d = Detection {
            x: -5.0,
            y: -5.0,
            width: 8.0,
            height: 8.0,
            confidence: 1.0,
            label: "x".into(),
            class_id: 0,
        };
        let (hue, _) = dominant_hue(&f, &d).expect("clamped box has pixels");
        assert_eq!(hue_name(hue), "blue");
    }

    #[test]
    fn a_zero_area_box_has_no_hue() {
        let f = solid(8, 8, [255, 0, 0]);
        let d = Detection {
            x: 4.0,
            y: 4.0,
            width: 0.0,
            height: 0.0,
            confidence: 1.0,
            label: "x".into(),
            class_id: 0,
        };
        assert_eq!(dominant_hue(&f, &d), None);
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

    // ---- EXERCISE 7: centre-weighted hue ----

    /// A frame whose box centre is one colour and whose edges are another.
    /// This is the shape of the real failure: object in the middle, desk
    /// and wall around it.
    fn centre_on_background(n: u32, centre_px: u32, centre: [u8; 3], background: [u8; 3]) -> Frame {
        let mut rgb = vec![0u8; (n * n * 3) as usize];
        let lo = (n - centre_px) / 2;
        let hi = (n + centre_px) / 2;
        for y in 0..n {
            for x in 0..n {
                let c = if x >= lo && x < hi && y >= lo && y < hi {
                    centre
                } else {
                    background
                };
                let i = ((y * n + x) * 3) as usize;
                rgb[i..i + 3].copy_from_slice(&c);
            }
        }
        Frame {
            width: n,
            height: n,
            rgb,
        }
    }

    #[test]
    #[ignore = "exercise 7"]
    fn weighted_hue_matches_plain_hue_on_a_solid_colour() {
        // Sanity: with nothing to disagree about, weighting changes nothing.
        for rgb in [[255u8, 0, 0], [0, 255, 0], [0, 0, 255]] {
            let f = solid(16, 16, rgb);
            let plain = dominant_hue(&f, &whole(&f)).unwrap();
            let weighted = dominant_hue_weighted(&f, &whole(&f)).unwrap();
            assert!(
                (plain.0 - weighted.0).abs() < 1.0,
                "solid colour: plain {:.0} vs weighted {:.0}",
                plain.0,
                weighted.0
            );
        }
    }

    #[test]
    #[ignore = "exercise 7"]
    fn weighted_hue_recovers_the_centre_colour_from_a_busy_box() {
        // THE bug this fixes. Red object filling the middle of the box,
        // blue background around it. The plain average is dragged off the
        // object entirely; the weighted one should read the object.
        let f = centre_on_background(32, 18, [255, 0, 0], [0, 0, 255]);
        let b = whole(&f);

        let (plain, _) = dominant_hue(&f, &b).expect("saturated");
        let (weighted, _) = dominant_hue_weighted(&f, &b).expect("saturated");

        assert_ne!(
            hue_name(plain),
            "red",
            "plain average should NOT read red here (it read {:.0}deg) — \
             if it does, the test frame is not exercising the bug",
            plain
        );
        assert_eq!(
            hue_name(weighted),
            "red",
            "weighted average read {:.0}deg ({}), expected red",
            weighted,
            hue_name(weighted)
        );
    }

    #[test]
    #[ignore = "exercise 7"]
    fn weighted_hue_ignores_grey_like_the_plain_version() {
        let f = solid(16, 16, [128, 128, 128]);
        assert_eq!(dominant_hue_weighted(&f, &whole(&f)), None);
    }

    #[test]
    #[ignore = "exercise 7"]
    fn weighted_hue_rejects_a_box_outside_the_frame() {
        let f = solid(8, 8, [255, 0, 0]);
        let d = Detection {
            x: 100.0,
            y: 100.0,
            width: 10.0,
            height: 10.0,
            confidence: 1.0,
            label: "x".into(),
            class_id: 0,
        };
        assert_eq!(dominant_hue_weighted(&f, &d), None);
    }

    #[test]
    #[ignore = "exercise 7"]
    fn weighted_hue_still_wraps_correctly() {
        // The circular mean must survive the weighting: half the pixels at
        // hue ~5deg, half at ~355deg, mean must be red (0), not cyan (180).
        let mut f = solid(4, 2, [255, 0, 0]);
        for x in 0..4u32 {
            let i = ((f.width + x) * 3) as usize;
            f.rgb[i] = 255;
            f.rgb[i + 1] = 0;
            f.rgb[i + 2] = 20;
        }
        let (hue, _) = dominant_hue_weighted(&f, &whole(&f)).unwrap();
        assert_eq!(hue_name(hue), "red", "circular mean broke, hue = {hue}");
    }
}
